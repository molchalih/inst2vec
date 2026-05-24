"""Shared clip-embedding runner driven by EmbeddingCaseSpec entries.

Per case the stage:

  1. Picks candidates (selected + downloaded + optional eligibility).
  2. Computes Fingerprint(data, config, dependency) plus a per-clip
     source-hash map. Dependency rows are case-specific and mirror what
     the case's text+payload builders read.
  3. If the stage fingerprint matches, skips.
  4. If only config drifted, wipes every ClipEmbedding row for the case.
  5. Diffs per-clip source hashes against stored rows to pick the (re-)embed set.
  6. Embeds that subset, committing each row as it succeeds.
  7. Seals the fingerprint only when zero clips failed; partial failure
     leaves the stage unsealed so the next run retries only the missing ones.

Per-clip failures (e.g. ``process_vision_info`` raising on a corrupt
video, or a network error from a remote provider) are caught by the
worker, logged as a structured ERR line, and reported back to the broker
as a terminal failure once ``max_attempts`` is exhausted. The orchestrator
in ``modules/embeddings/distributed.py`` counts those failures so the
runner refuses to seal — the next run retries just those clips. Failed
clips are NOT replaced with placeholder embeddings.
"""

from __future__ import annotations

import os
import time

from core import fingerprint as fp
from core.console import log
from core.database import (
    Clip,
    ClipEmbedding,
    StageState,
    get_session,
)
from core.pipeline import Stage
from modules.embeddings.broker import make_job
from modules.embeddings.cases import (
    CASE_REGISTRY,
    EmbeddingCaseSpec,
    EmbeddingSecrets,
    case_config_identity,
    default_cases,
)
from modules.embeddings.sampling import (
    adaptive_sampling,
)
from modules.embeddings.state import (
    get_audio_mir_map,
    get_clip_embedding_candidates,
    get_embedded_source_hashes,
    per_clip_source_hashes_and_aggregate,
)

STAGE = Stage.CLIP_EMBEDDINGS


def embed_clip_embeddings(
    settings,
    secrets: EmbeddingSecrets | None = None,
    cases: list[str] | None = None,
) -> None:
    """Embed clips for the given cases (default: result of default_cases(settings)).

    ``secrets`` carries credentials for remote provider factories; local
    factories ignore it. Defaults to an empty ``EmbeddingSecrets()`` so
    callers that don't use remote providers can omit it.
    """
    if secrets is None:
        secrets = EmbeddingSecrets()
    case_names = list(cases) if cases is not None else list(default_cases(settings))
    for name in case_names:
        if name == "gemini" and not settings.embeddings.gemini_enabled:
            raise RuntimeError(
                "gemini case requested but embeddings.gemini_enabled=false"
            )
        spec = CASE_REGISTRY[name]
        _run_case(settings, secrets, spec)


def _video_path(clip_id: int, video_dir: str) -> str:
    return os.path.abspath(os.path.join(video_dir, f"{clip_id}.mp4"))


def build_jobs_for_case(
    spec,
    clips,
    *,
    texts: dict[int, str | None],
    video_dir: str,
    adaptive_max_frames: int,
    adaptive_default_fps: float,
) -> list[dict]:
    """One Job per clip. Probes the LOCAL video for fps/max_frames so pods
    don't have to. ``texts`` is the prebuilt text per clip. Clips with no
    local video file are skipped for video cases. ``audio_key`` is set for
    cases whose dependency columns include ``_audio_file_stat`` (maest /
    gemini); only the local worker leases those (served_remotely=False)."""
    needs_audio = "_audio_file_stat" in spec.dependency_columns
    jobs: list[dict] = []
    for clip in clips:
        if spec.requires_video:
            path = _video_path(clip.id, video_dir)
            if not os.path.exists(path):
                continue
            fps, max_frames = adaptive_sampling(
                path, adaptive_max_frames, adaptive_default_fps
            )
            video_key = f"videos/{clip.id}.mp4"
        else:
            fps, max_frames, video_key = None, None, None
        audio_key = f"audio/{clip.id}.mp3" if needs_audio else None
        jobs.append(
            make_job(
                clip_id=clip.id,
                case=spec.name,
                text=texts.get(clip.id),
                video_key=video_key,
                fps=fps,
                max_frames=max_frames,
                remote_eligible=bool(getattr(clip, "is_uploaded", False)),
                audio_key=audio_key,
            )
        )
    return jobs


def _compute_fingerprint_and_per_clip(
    session, spec: EmbeddingCaseSpec, settings, candidates: list[Clip]
) -> tuple[fp.Fingerprint, dict[int, str]]:
    """Return (Fingerprint, {clip_id: per_clip_source_hash}) for ``case``."""
    candidate_ids = sorted(c.id for c in candidates)
    per_clip, dep_agg = per_clip_source_hashes_and_aggregate(
        session, spec.name, candidate_ids, settings=settings
    )
    current = fp.Fingerprint(
        data=fp.hash_rows((cid,) for cid in candidate_ids),
        config=fp.hash_text(case_config_identity(spec, settings)),
        dependency=dep_agg,
    )
    return current, per_clip


def _wipe_case(session, case: str) -> None:
    session.query(ClipEmbedding).filter_by(embedding_case=case).delete()
    session.commit()


def _run_case(settings, secrets, spec: EmbeddingCaseSpec) -> None:
    log_tag = f"embed:{spec.name}"
    session = get_session()
    try:
        candidates = get_clip_embedding_candidates(
            session, settings.embeddings.exclude_disqualified_users
        )
        candidate_ids = {c.id for c in candidates}

        # Sweep orphan ClipEmbedding rows for this case.
        orphan_q = session.query(ClipEmbedding).filter(
            ClipEmbedding.embedding_case == spec.name,
        )
        if candidate_ids:
            orphan_q = orphan_q.filter(~ClipEmbedding.clip_id.in_(candidate_ids))
        deleted = orphan_q.delete(synchronize_session=False)
        if deleted:
            session.commit()
            log(log_tag, "WRITE", "orphans", "ok", stats={"deleted": deleted})

        current, per_clip = _compute_fingerprint_and_per_clip(
            session, spec, settings, candidates
        )
        if not fp.is_stale(session, STAGE, spec.name, current):
            log(log_tag, "SKIP", "fingerprint", "ok")
            return

        stored = session.get(StageState, (STAGE, spec.name))
        if stored is not None and stored.config_hash != current.config:
            diff = fp.describe_diff(session, STAGE, spec.name, current)
            log(log_tag, "SCAN", "fingerprint", "stale", stats={"diff": diff})
            _wipe_case(session, spec.name)

        embedded = get_embedded_source_hashes(session, spec.name)
        target_ids = fp.row_diff(per_clip, embedded)
        log(log_tag, "SCAN", "jobs", "ok", stats={"todo": len(target_ids)})

        _embed_targets(
            session,
            spec,
            settings,
            secrets,
            log_tag,
            candidates,
            target_ids,
            per_clip,
            embedded,
            current,
        )
    finally:
        session.close()


def _embed_targets(
    session,
    spec: EmbeddingCaseSpec,
    settings,
    secrets,
    log_tag: str,
    candidates: list[Clip],
    target_ids: set[int],
    per_clip: dict[int, str],
    embedded: dict[int, str | None],
    current: fp.Fingerprint,
) -> None:
    """Embed the subset of ``candidates`` whose ids are in ``target_ids``."""
    targets = [c for c in candidates if c.id in target_ids]

    audio_mir_map: dict = {}
    if spec.text_builder is not None and ("_audio_mir_row" in spec.dependency_columns):
        audio_mir_map = get_audio_mir_map(session)

    video_dir = settings.paths.video_dir
    needs_audio = "_audio_file_stat" in spec.dependency_columns
    audio_dir = settings.paths.audio_dir if needs_audio else None
    texts: dict[int, str | None] = {}
    targets_built: list[Clip] = []
    stale_skipped: list[int] = []
    fresh_skipped = 0
    for clip in targets:
        had_row = clip.id in embedded
        if spec.requires_video:
            path = _video_path(clip.id, video_dir)
            if not os.path.exists(path):
                if had_row:
                    stale_skipped.append(clip.id)
                else:
                    fresh_skipped += 1
                continue
        if audio_dir is not None:
            audio_path = os.path.abspath(os.path.join(audio_dir, f"{clip.id}.mp3"))
            if not os.path.exists(audio_path):
                if had_row:
                    stale_skipped.append(clip.id)
                else:
                    fresh_skipped += 1
                continue
        text: str | None = None
        if spec.text_builder is not None:
            text = spec.text_builder(clip, audio_mir_map.get(clip.id))
            if text is None:
                if had_row:
                    stale_skipped.append(clip.id)
                else:
                    fresh_skipped += 1
                continue
        texts[clip.id] = text
        targets_built.append(clip)

    if stale_skipped:
        session.query(ClipEmbedding).filter(
            ClipEmbedding.embedding_case == spec.name,
            ClipEmbedding.clip_id.in_(stale_skipped),
        ).delete(synchronize_session=False)
        session.commit()
        log(
            log_tag,
            "WRITE",
            "stale_rows",
            "ok",
            stats={"dropped": len(stale_skipped)},
        )

    t_stage = time.perf_counter()

    if not targets_built:
        if stale_skipped:
            log(
                log_tag,
                "SEAL",
                "embed",
                "stale",
                stats={"stale": len(stale_skipped)},
            )
            return  # do not seal — un-buildable stale targets remain unresolved
        if fresh_skipped:
            log(
                log_tag,
                "SCAN",
                "candidates",
                "ok",
                stats={"non_embeddable": fresh_skipped},
            )
        fp.mark_complete(session, STAGE, spec.name, current)
        session.commit()
        log(
            log_tag,
            "SEAL",
            "embed",
            "ok",
            stats={"done": 0, "time": time.perf_counter() - t_stage},
        )
        return

    from modules.embeddings.distributed import embed_jobs_distributed

    case_jobs = build_jobs_for_case(
        spec,
        targets_built,
        texts=texts,
        video_dir=video_dir,
        adaptive_max_frames=settings.embeddings.adaptive_max_frames,
        adaptive_default_fps=settings.embeddings.adaptive_default_fps,
    )
    succeeded, failures = embed_jobs_distributed(
        settings, secrets, session, spec, case_jobs, per_clip, log_tag
    )

    if failures or stale_skipped:
        log(
            log_tag,
            "SEAL",
            "embed",
            "stale",
            stats={
                "done": succeeded,
                "err": failures,
                "stale": len(stale_skipped),
                "time": time.perf_counter() - t_stage,
            },
        )
    else:
        fp.mark_complete(session, STAGE, spec.name, current)
        session.commit()
        log(
            log_tag,
            "SEAL",
            "embed",
            "ok",
            stats={
                "done": succeeded,
                "time": time.perf_counter() - t_stage,
            },
        )
