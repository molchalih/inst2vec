"""Generic per-case stage-1 clip-pass runner.

One ``run_case`` invocation handles one ``LabelCaseSpec``:

- video case: routes to ``LabelsGenerator.run(video_path, prompt)`` with
  the per-clip video file; fingerprint ``data`` slot is a stable hash of
  ``(clip_id, file_stat(video))`` per selected clip.
- every other case: routes to ``LabelsGenerator.run_text`` with the
  case's text adapter output; fingerprint ``data`` slot is an empty-content
  hash (``hash_text("")``) because there is no per-clip video file to stat.
  Per-clip content drift is captured transitively through the ``dependency``
  slot, which folds in the upstream stage seals declared by
  ``spec.stage1_dependency_stages`` (speech / captions / MIR) and any
  consumed label-case seals from ``spec.consumes_label_cases`` (e.g. the
  visual case for sandwich).

Case-specific behaviour flows exclusively through ``LabelCaseSpec``;
this file does not branch on ``spec.name``.
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.config import LabelsSettings, Settings
from core.database import AudioMIR, Clip, ClipLabel, clip_used_in_analysis
from core.log import event, item, scope
from core.pipeline import Stage
from modules.labels.cases import LabelCaseSpec
from modules.labels.prompts import prompt_for
from modules.labels.schema import clip_schema
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


def _data_hash(session: Session, *, settings: Settings, spec: LabelCaseSpec) -> str:
    """Fingerprint ``data`` slot. Only the video case stats per-clip files;
    text cases carry no per-clip video data, so use the empty-content hash."""
    if spec.clip_uses_video:
        return _data_hash_for_video(session, settings)
    return fp.hash_text("")


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
        data=_data_hash(session, settings=settings, spec=spec),
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
    """Per-case stage-1 runner. Caller owns generator lifetime.

    The ``video`` case routes each clip through ``generator.run`` with the
    per-clip video file; the four text cases (spoken / textual / auditory /
    sandwich) build per-clip evidence via ``spec.clip_input`` and route it
    through ``generator.run_text``. Both paths validate and write
    ``ClipLabel`` rows under the same fingerprint gate.
    """
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
        if spec.clip_uses_video:
            schema = clip_schema(spec, labels)
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
                    schema=schema,
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
                            schema=schema,
                        )
                        session.commit()
        else:
            _process_text(
                session,
                pending=pending,
                prompt_body=prompt_body,
                labels=labels,
                generator=generator,
                spec=spec,
            )

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
    schema: dict,
) -> None:
    """Video-case batched extraction. Same prompt across a batch, one
    decoded string per clip. On a batch-wide generator exception every
    clip in the batch is marked failed; per-clip JSON-validation failures
    are isolated by ``_store_result``.

    Batches clips into chunks and calls ``_process_video_batches_sync`` for
    each. On a batch-wide generator exception every clip in the batch is
    marked failed; per-clip JSON-validation failures are isolated by
    ``_store_result``.
    """
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

    _process_video_batches_sync(
        chunks,
        paths_for,
        generator=generator,
        prompt_body=prompt_body,
        spec=spec,
        fail_chunk=_fail_chunk,
        store_chunk=_store_chunk,
        schema=schema,
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
    schema: dict,
) -> None:
    """Decode and generate inline, one chunk at a time."""
    for chunk, video_paths in zip(chunks, paths_for, strict=True):
        with item("EXTRACT", _batch_log_label(spec, chunk)):
            try:
                raws = generator.run_many(video_paths, prompt_body, schema=schema)
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
    schema: dict,
) -> None:
    try:
        raw = generator.run(paths.video_for(clip_id), prompt_body, schema=schema)
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


def _process_text(
    session: Session,
    *,
    pending: list[int],
    prompt_body: str,
    labels: LabelsSettings,
    generator,
    spec: LabelCaseSpec,
) -> None:
    """Stage-1 text path: one ``run_text`` per clip from ``spec.clip_input``."""
    schema = clip_schema(spec, labels)
    mir_by_clip, visual_by_clip, dep_terminal = _text_evidence_maps(
        session, pending=pending, spec=spec, labels=labels
    )
    for clip_id in pending:
        key = (clip_id, spec.name)
        with item("EXTRACT", f"{spec.name}/clip_{clip_id}"):
            clip = session.get(Clip, clip_id)
            visual_payload = (
                visual_by_clip.get(clip_id) if spec.consumes_label_cases else None
            )
            if spec.consumes_label_cases and visual_payload is None:
                if clip_id in dep_terminal:
                    # The upstream label is terminally failed (it will not
                    # succeed under the current fingerprint). Record a terminal
                    # failure; it recovers only when the upstream case drifts,
                    # which also drifts this case's dependency hash and re-runs it.
                    upsert_terminal_failure(
                        session,
                        ClipLabel,
                        key=key,
                        error=spec.none_input_error or "missing_dependency_label",
                        attempts=labels.max_attempts,
                    )
                    session.commit()
                # else: the upstream label is still pending / retryable — leave
                # this clip pending (write nothing) so a later run retries it
                # once the upstream label is available.
                continue
            evidence = spec.clip_input(clip, mir_by_clip.get(clip_id), visual_payload)
            if evidence is None:
                # Legitimately-empty input (no speech / no caption / no MIR)
                # also becomes a terminal failure row; the cluster pass ignores
                # any non-success member when aggregating cluster labels.
                upsert_terminal_failure(
                    session,
                    ClipLabel,
                    key=key,
                    error=spec.none_input_error or "no_input",
                    attempts=labels.max_attempts,
                )
                session.commit()
                continue
            prompt = f"{prompt_body}\n\n{evidence}"
            try:
                raw = generator.run_text(
                    prompt, max_new_tokens=labels.max_new_tokens, schema=schema
                )
            except Exception as exc:
                bump_failure(
                    session,
                    ClipLabel,
                    key=key,
                    error=f"runtime:{exc}",
                    max_attempts=labels.max_attempts,
                )
                session.commit()
                continue
            _store_result(session, raw=raw, key=key, labels=labels, spec=spec)
            session.commit()


def _text_evidence_maps(
    session: Session, *, pending: list[int], spec: LabelCaseSpec, labels: LabelsSettings
) -> tuple[dict[int, AudioMIR], dict[int, dict], set[int]]:
    mir_by_clip = {
        m.clip_id: m
        for m in session.execute(select(AudioMIR).where(AudioMIR.clip_id.in_(pending)))
        .scalars()
        .all()
    }
    visual_by_clip: dict[int, dict] = {}
    dep_terminal: set[int] = set()
    if spec.consumes_label_cases:
        dep_case = spec.consumes_label_cases[0]
        rows = (
            session.execute(
                select(ClipLabel).where(
                    ClipLabel.clip_id.in_(pending),
                    ClipLabel.label_case == dep_case,
                )
            )
            .scalars()
            .all()
        )
        for r in rows:
            if r.status == "success":
                visual_by_clip[r.clip_id] = r.payload or {}
            elif r.status == "failed" and (r.attempts or 0) >= labels.max_attempts:
                dep_terminal.add(r.clip_id)
    return mir_by_clip, visual_by_clip, dep_terminal


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
