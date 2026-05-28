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
from core.database import AudioMIR, Clip, ClipLabel
from core.log import event, item, scope
from core.pipeline import Stage
from modules.labels.cases import LabelCaseSpec
from modules.labels.prompts import prompt_for
from modules.labels.state import (
    STAGE_LABELS,
    clip_labels_config_payload,
    clip_scope_for,
)
from modules.labels.store import bump_failure, upsert_success
from modules.labels.validation import validate


def _selected_clip_ids(session: Session) -> list[int]:
    return list(
        session.execute(
            select(Clip.id)
            .where(Clip.is_selected.is_(True), Clip.is_downloaded.is_(True))
            .order_by(Clip.id)
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
        .where(Clip.is_selected.is_(True), Clip.is_downloaded.is_(True))
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


def _consumed_payload(
    session: Session, *, clip_id: int, spec: LabelCaseSpec
) -> dict | None:
    """Stage-1 payload from the (single) label case this case consumes.

    The current ``clip_input`` adapter contract takes one ``visual_payload``
    kwarg, so cases that consume multiple upstream label cases are not yet
    supported — assert that the spec declares zero or one such case.
    """
    deps = spec.consumes_label_cases
    if not deps:
        return None
    assert len(deps) == 1, f"{spec.name} consumes_label_cases must be 0 or 1"
    existing = session.get(ClipLabel, (clip_id, deps[0]))
    return existing.payload if existing is not None else None


def _build_text_inputs(
    session: Session, *, spec: LabelCaseSpec
) -> dict[int, str | None]:
    """Call ``spec.clip_input`` exactly once per selected clip.

    Returned mapping is consumed by both ``_data_hash_from_inputs`` (for the
    fingerprint ``data`` slot) and the per-clip generation loop. ``None`` is
    preserved verbatim so the runner can mark the row failed with
    ``spec.none_input_error``.
    """
    out: dict[int, str | None] = {}
    for cid in _selected_clip_ids(session):
        clip = session.get(Clip, cid)
        mir_row = session.execute(
            select(AudioMIR).where(AudioMIR.clip_id == cid)
        ).scalar_one_or_none()
        visual_payload = _consumed_payload(session, clip_id=cid, spec=spec)
        out[cid] = spec.clip_input(clip, mir_row, visual_payload)
    return out


def _data_hash_from_inputs(inputs: dict[int, str | None]) -> str:
    rows = [(cid, fp.hash_text(text or "")) for cid, text in sorted(inputs.items())]
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
    text_inputs: dict[int, str | None],
) -> fp.Fingerprint:
    data = (
        _data_hash_for_video(session, settings)
        if spec.clip_uses_video
        else _data_hash_from_inputs(text_inputs)
    )
    return fp.Fingerprint(
        data=data,
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
    """Generic per-case stage-1 runner. Caller owns generator lifetime."""
    paths = settings.paths

    def _wipe_case(s: Session) -> None:
        s.execute(delete(ClipLabel).where(ClipLabel.label_case == spec.name))

    text_inputs: dict[int, str | None] = (
        {} if spec.clip_uses_video else _build_text_inputs(session, spec=spec)
    )
    current = _current_fingerprint(
        session,
        settings=settings,
        labels=labels,
        spec=spec,
        text_inputs=text_inputs,
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
        if spec.clip_uses_video and batch_size > 1:
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
                        text_inputs=text_inputs,
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
    """
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        video_paths = [paths.video_for(cid) for cid in chunk]
        labels_for_log = ",".join(f"clip_{cid}" for cid in chunk)
        with item("EXTRACT", f"{spec.name}/batch[{labels_for_log}]"):
            try:
                raws = generator.run_many(video_paths, prompt_body)
            except Exception as exc:
                for cid in chunk:
                    bump_failure(
                        session,
                        ClipLabel,
                        key=(cid, spec.name),
                        error=f"runtime:{exc}",
                        max_attempts=labels.max_attempts,
                    )
                session.commit()
                continue
            for cid, raw in zip(chunk, raws, strict=True):
                _store_result(
                    session,
                    raw=raw,
                    key=(cid, spec.name),
                    labels=labels,
                    spec=spec,
                    source_text=None,
                )
            session.commit()


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
    text_inputs: dict[int, str | None],
) -> None:
    if spec.clip_uses_video:
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
            source_text=None,
        )
        return

    input_text = text_inputs.get(clip_id)
    if input_text is None:
        assert spec.none_input_error is not None, (
            f"{spec.name} adapter returned None but spec.none_input_error is None"
        )
        bump_failure(
            session,
            ClipLabel,
            key=key,
            error=spec.none_input_error,
            max_attempts=labels.max_attempts,
        )
        return

    prompt = prompt_body + "\n\n" + input_text
    try:
        raw = generator.run_text(prompt, max_new_tokens=labels.max_new_tokens)
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
        source_text=input_text,
    )


def _store_result(
    session: Session,
    *,
    raw: str,
    key: tuple,
    labels: LabelsSettings,
    spec: LabelCaseSpec,
    source_text: str | None,
) -> None:
    payload, status, warnings = validate(raw, labels, case=spec.name)
    if status == "failed":
        bump_failure(
            session,
            ClipLabel,
            key=key,
            error=warnings[0] if warnings else "validation",
            max_attempts=labels.max_attempts,
        )
        return
    assert payload is not None
    extras: dict[str, object] = {}
    if source_text is not None:
        extras["source_hash"] = fp.hash_text(source_text)
    upsert_success(
        session,
        ClipLabel,
        key=key,
        validation=status,
        payload=payload,
        warnings=warnings,
        extras=extras or None,
    )
