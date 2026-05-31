"""Generic per-case stage-1 clip-pass runner.

One ``run_case`` invocation handles one ``LabelCaseSpec``:

- video case: routes to ``LabelsGenerator.run(video_path, prompt)`` with
  the per-clip video file; fingerprint ``data`` slot is a stable hash of
  ``(clip_id, file_stat(video))`` per selected clip.
- every other case: routes to ``LabelsGenerator.run_text`` with the
  case's text adapter output prefixed by ``prompt + "\n\n"``; fingerprint
  ``data`` slot is ``hash_rows((clip_id, source_hash))`` where
  ``source_hash = hash_text(input_text or "")``. Cases that declare
  ``consumes_label_cases`` additionally compose
  ``stage_dependency_hash(LABELS, dep_case)`` into the dependency slot
  and consume the upstream case's ``ClipLabel.payload`` as part of their
  input.

Case-specific behaviour flows exclusively through ``LabelCaseSpec``;
this file does not branch on ``spec.name``.
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.config import LabelsSettings, Settings
from core.database import Clip, ClipLabel, clip_used_in_analysis
from core.log import event, item, scope
from core.pipeline import Stage
from modules.labels.cases import LabelCaseSpec
from modules.labels.prompts import prompt_for
from modules.labels.state import (
    STAGE_LABELS,
    clip_labels_config_payload,
    clip_scope_for,
)
from modules.labels.store import bump_failure, upsert_success, upsert_terminal_failure
from modules.labels.validation import format_failure_error, validate


def _selected_clip_ids(session: Session) -> list[int]:
    return list(
        session.execute(
            select(Clip.id).where(*clip_used_in_analysis()).order_by(Clip.id)
        )
        .scalars()
        .all()
    )


def _pending_clip_ids(
    session: Session, *, case: str, labels: LabelsSettings
) -> list[int]:
    rows = session.execute(
        select(Clip.id, ClipLabel.status, ClipLabel.attempts)
        .outerjoin(
            ClipLabel,
            (ClipLabel.clip_id == Clip.id) & (ClipLabel.label_case == case),
        )
        .where(*clip_used_in_analysis())
        .order_by(Clip.id)
    ).all()
    out: list[int] = []
    for clip_id, status, attempts in rows:
        if (
            status is None
            or status == "pending"
            or (status == "failed" and (attempts or 0) < labels.max_attempts)
        ):
            out.append(clip_id)
    return out


def _data_hash_for_video(session: Session, settings: Settings) -> str:
    rows = [
        (cid, fp.file_stat_for_hash(settings.paths.video_for(cid)))
        for cid in _selected_clip_ids(session)
    ]
    return fp.hash_rows(rows)


def _dependency_hash(session: Session, *, spec: LabelCaseSpec) -> str:
    parts: list[str] = [
        fp.stage_dependency_hash(session, st, sc)
        for st, sc in spec.stage1_dependency_stages
    ]
    for dep_case in spec.consumes_label_cases:
        parts.append(fp.stage_dependency_hash(session, Stage.LABELS, dep_case))
    return fp.compose_hashes(*parts) if parts else fp.hash_text("")


def _current_fingerprint(
    session: Session,
    *,
    settings: Settings,
    labels: LabelsSettings,
    spec: LabelCaseSpec,
) -> fp.Fingerprint:
    return fp.Fingerprint(
        data=_data_hash_for_video(session, settings),
        config=fp.hash_text(clip_labels_config_payload(labels, case=spec.name)),
        dependency=_dependency_hash(session, spec=spec),
    )


@scope("labels:clips:{spec.name}")
def run_case(
    *,
    session: Session,
    settings: Settings,
    labels: LabelsSettings,
    generator,
    spec: LabelCaseSpec,
) -> None:
    """Video-case stage-1 runner. Caller owns generator lifetime.

    Only the ``video`` case reaches stage 1: it is the sole place frames are
    reduced to text. Stage-1-skipped cases (``runs_clip_pass=False``) are
    synthesised straight from raw signals in :mod:`modules.labels.cluster_pass`
    and never enter this runner — ``pipeline.run`` filters them out upstream.
    """
    assert spec.clip_uses_video, (
        f"clip_pass.run_case only handles the video case; got {spec.name!r}. "
        "Stage-1-skipped cases are synthesised in cluster_pass."
    )
    paths = settings.paths

    def _wipe_case(s: Session) -> None:
        s.execute(delete(ClipLabel).where(ClipLabel.label_case == spec.name))

    current = _current_fingerprint(
        session,
        settings=settings,
        labels=labels,
        spec=spec,
    )
    fp.gate(
        session,
        STAGE_LABELS,
        clip_scope_for(spec.name),
        current,
        on_drift=_wipe_case,
        check_dependency=True,
        check_data=True,
    )
    session.commit()

    pending = _pending_clip_ids(session, case=spec.name, labels=labels)
    if pending:
        event(
            "GET",
            "qwen3-vl-instruct",
            stats={"case": spec.name, "clips": len(pending)},
        )
        prompt_body = prompt_for(labels, case=spec.name)
        batch_size = max(1, int(labels.batch_size))
        if batch_size > 1:
            _process_video_batches(
                session,
                pending=pending,
                prompt_body=prompt_body,
                labels=labels,
                generator=generator,
                spec=spec,
                paths=paths,
                batch_size=batch_size,
            )
        else:
            for clip_id in pending:
                key = (clip_id, spec.name)
                with item("EXTRACT", f"{spec.name}/clip_{clip_id}"):
                    _process_one(
                        session,
                        clip_id=clip_id,
                        key=key,
                        prompt_body=prompt_body,
                        labels=labels,
                        generator=generator,
                        spec=spec,
                        paths=paths,
                    )
                    session.commit()

    fp.mark_complete(session, STAGE_LABELS, clip_scope_for(spec.name), current)
    session.commit()


def _process_video_batches(
    session: Session,
    *,
    pending: list[int],
    prompt_body: str,
    labels: LabelsSettings,
    generator,
    spec: LabelCaseSpec,
    paths,
    batch_size: int,
) -> None:
    """Video-case batched extraction. Same prompt across a batch, one
    decoded string per clip. On a batch-wide generator exception every
    clip in the batch is marked failed; per-clip JSON-validation failures
    are isolated by ``_store_result``.

    CPU/GPU pipelining: the next batch's frame decode + tokenize runs on a
    single-slot background thread while the current batch generates on the
    GPU. ``torchcodec`` + HF tokenizer both release the GIL, so the worker
    progresses in parallel; the GPU side stays on the main thread to keep
    CUDA call ordering deterministic.
    """
    from concurrent.futures import ThreadPoolExecutor

    chunks = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    if not chunks:
        return
    paths_for: list[list] = [
        [paths.video_for(cid) for cid in chunk] for chunk in chunks
    ]

    def _fail_chunk(chunk: list[int], exc: BaseException) -> None:
        for cid in chunk:
            bump_failure(
                session,
                ClipLabel,
                key=(cid, spec.name),
                error=f"runtime:{exc}",
                max_attempts=labels.max_attempts,
            )
        session.commit()

    def _store_chunk(chunk: list[int], raws: list[str]) -> None:
        for cid, raw in zip(chunk, raws, strict=True):
            _store_result(
                session,
                raw=raw,
                key=(cid, spec.name),
                labels=labels,
                spec=spec,
            )
        session.commit()

    # Prefetch path needs both halves of the split interface. Fakes/older
    # generators that only expose ``run_many`` fall back to the synchronous
    # in-line call (no CPU/GPU overlap, same throughput floor as before).
    prepare = getattr(generator, "prepare_many", None)
    generate = getattr(generator, "generate_from_inputs", None)
    if prepare is None or generate is None:
        _process_video_batches_sync(
            chunks,
            paths_for,
            generator=generator,
            prompt_body=prompt_body,
            spec=spec,
            fail_chunk=_fail_chunk,
            store_chunk=_store_chunk,
        )
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="labels-prep") as ex:
        _process_video_batches_pipelined(
            chunks,
            paths_for,
            ex=ex,
            prepare=prepare,
            generate=generate,
            prompt_body=prompt_body,
            spec=spec,
            fail_chunk=_fail_chunk,
            store_chunk=_store_chunk,
        )


def _batch_log_label(spec: LabelCaseSpec, chunk: list[int]) -> str:
    labels_for_log = ",".join(f"clip_{cid}" for cid in chunk)
    return f"{spec.name}/batch[{labels_for_log}]"


def _process_video_batches_sync(
    chunks: list[list[int]],
    paths_for: list[list],
    *,
    generator,
    prompt_body: str,
    spec: LabelCaseSpec,
    fail_chunk,
    store_chunk,
) -> None:
    """Synchronous fallback for generators without the split prepare/generate
    interface: decode + generate inline, one chunk at a time."""
    for chunk, video_paths in zip(chunks, paths_for, strict=True):
        with item("EXTRACT", _batch_log_label(spec, chunk)):
            try:
                raws = generator.run_many(video_paths, prompt_body)
            except Exception as exc:
                fail_chunk(chunk, exc)
                continue
            store_chunk(chunk, raws)


def _process_video_batches_pipelined(
    chunks: list[list[int]],
    paths_for: list[list],
    *,
    ex,
    prepare,
    generate,
    prompt_body: str,
    spec: LabelCaseSpec,
    fail_chunk,
    store_chunk,
) -> None:
    """CPU/GPU-overlapped path: prefetch the next chunk's inputs on ``ex``
    while the current chunk generates on the GPU."""
    prep_fut = ex.submit(prepare, paths_for[0], prompt_body)

    def _prefetch_next(i: int) -> None:
        nonlocal prep_fut
        if i + 1 < len(chunks):
            prep_fut = ex.submit(prepare, paths_for[i + 1], prompt_body)

    for i, chunk in enumerate(chunks):
        with item("EXTRACT", _batch_log_label(spec, chunk)):
            try:
                inputs = prep_fut.result()
            except Exception as exc:
                _prefetch_next(i)
                fail_chunk(chunk, exc)
                continue
            _prefetch_next(i)
            try:
                raws = generate(inputs)
            except Exception as exc:
                fail_chunk(chunk, exc)
                continue
            store_chunk(chunk, raws)


def _process_one(
    session: Session,
    *,
    clip_id: int,
    key: tuple,
    prompt_body: str,
    labels: LabelsSettings,
    generator,
    spec: LabelCaseSpec,
    paths,
) -> None:
    try:
        raw = generator.run(paths.video_for(clip_id), prompt_body)
    except Exception as exc:
        bump_failure(
            session,
            ClipLabel,
            key=key,
            error=f"runtime:{exc}",
            max_attempts=labels.max_attempts,
        )
        return
    _store_result(
        session,
        raw=raw,
        key=key,
        labels=labels,
        spec=spec,
    )


def _store_result(
    session: Session,
    *,
    raw: str,
    key: tuple,
    labels: LabelsSettings,
    spec: LabelCaseSpec,
) -> None:
    payload, status, warnings = validate(raw, labels, case=spec.name)
    if status == "failed":
        # Hard validation fails (H1=non-JSON, H2=wrong key set, H3=wrong shape)
        # are deterministic under the seeded generator — retrying the same
        # prompt with the same seed cannot change the output. Skip the retry
        # budget and write a terminal failure directly; runtime errors above
        # still go through ``bump_failure`` because they CAN be transient.
        code = warnings[0] if warnings else "validation"
        upsert_terminal_failure(
            session,
            ClipLabel,
            key=key,
            error=format_failure_error(code, raw),
            attempts=labels.max_attempts,
        )
        return
    assert payload is not None
    upsert_success(
        session,
        ClipLabel,
        key=key,
        validation=status,
        payload=payload,
        warnings=warnings,
    )
