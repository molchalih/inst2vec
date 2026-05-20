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
video, or a network error from a remote provider) propagate from the
provider into ``_dispatch_embedding_jobs``, which logs at ``warn``,
counts the failure, and yields ``(clip_id, None, None)``. The runner skips
the row and refuses to seal — the next run will retry just those
clips. Failed clips are NOT replaced with placeholder embeddings.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from core import fingerprint as fp
from core.console import log, progress
from core.database import (
    Clip,
    ClipEmbedding,
    StageState,
    get_session,
)
from core.pipeline import Stage
from modules.embeddings.cases import (
    CASE_REGISTRY,
    EmbeddingCaseSpec,
    EmbeddingSecrets,
    case_config_identity,
    default_cases,
)
from modules.embeddings.sampling import (
    adaptive_sampling,
    frame_retry_schedule,
    is_token_mismatch_error,
)
from modules.embeddings.state import (
    get_clip_embedding_candidates,
    get_embedded_source_hashes,
    get_music_map,
    per_clip_source_hashes_and_aggregate,
)
from modules.embeddings.vectors import to_bytes

STAGE = Stage.CLIP_EMBEDDINGS


def _dispatch_embedding_jobs(
    jobs: list[dict],
    embed_fn: Callable[[dict], tuple[int, bytes | None]],
    inflight: int,
    log_tag: str = "embed",
) -> Iterator[tuple[int, bytes | None, float | None]]:
    """Run ``embed_fn`` over ``jobs`` with bounded concurrency.

    Yields ``(clip_id, blob_or_none, elapsed_seconds_or_none)``. On
    success, ``elapsed`` is the wall time spent inside ``embed_fn``.
    On exception, the dispatcher logs and yields ``(clip_id, None, None)``.
    """
    if inflight <= 1:
        for job in jobs:
            t0 = time.perf_counter()
            try:
                clip_id, blob = embed_fn(job)
            except Exception as exc:
                clip_id = job.get("clip_id", -1)
                log(
                    log_tag,
                    "EMB",
                    f"clip_{clip_id}",
                    "ERR",
                    stats={"err": repr(exc)},
                )
                yield (clip_id, None, None)
                continue
            yield (clip_id, blob, time.perf_counter() - t0)
        return

    def _timed(job: dict) -> tuple[int, bytes | None, float]:
        # Measure inside the worker so queue-wait under saturation
        # (len(jobs) > inflight) doesn't inflate the reported time.
        t0 = time.perf_counter()
        clip_id, blob = embed_fn(job)
        return clip_id, blob, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=inflight) as pool:
        futures: dict[Future, dict] = {pool.submit(_timed, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                clip_id, blob, elapsed = fut.result()
            except Exception as exc:
                clip_id = job.get("clip_id", -1)
                log(
                    log_tag,
                    "EMB",
                    f"clip_{clip_id}",
                    "ERR",
                    stats={"err": repr(exc)},
                )
                yield (clip_id, None, None)
                continue
            yield (clip_id, blob, elapsed)


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
            session,
            settings.embeddings.exclude_disqualified_users,
            require_uploaded=settings.embeddings.provider == "remote",
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

    music_map: dict = {}
    if spec.text_builder is not None:
        music_map = get_music_map(session)

    video_dir = settings.paths.video_dir
    audio_dir = settings.paths.audio_dir if spec.name == "gemini" else None
    jobs: list[tuple[Clip, str | None]] = []
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
        text: str | None = None
        if spec.text_builder is not None:
            text = spec.text_builder(clip, music_map)
            if text is None:
                if had_row:
                    stale_skipped.append(clip.id)
                else:
                    fresh_skipped += 1
                continue
        jobs.append((clip, text))

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

    if not jobs:
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
        else:
            log(log_tag, "SCAN", "jobs", "none")
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

    provider = spec.provider_factory(settings, secrets)

    job_specs: list[dict] = []
    clip_by_id: dict[int, Clip] = {}
    for clip, text in jobs:
        clip_by_id[clip.id] = clip
        if spec.requires_video:
            path = _video_path(clip.id, video_dir)
            fps_, max_frames = adaptive_sampling(
                path,
                settings.embeddings.adaptive_max_frames,
                settings.embeddings.adaptive_default_fps,
            )
        else:
            path, fps_, max_frames = None, None, None
        audio_path = (
            os.path.abspath(os.path.join(audio_dir, f"{clip.id}.mp3"))
            if audio_dir is not None
            else None
        )
        # Symmetric to the video-missing check above: a gemini job without
        # its audio file would surface as a generic provider error. Skip
        # instead so the runner reports a clean SKIP, not an EMB ERR.
        if audio_path is not None and not os.path.exists(audio_path):
            if clip.id in embedded:
                stale_skipped.append(clip.id)
            else:
                fresh_skipped += 1
            continue
        job_specs.append(
            {
                "clip_id": clip.id,
                "text": text,
                "path": path,
                "audio_path": audio_path,
                "fps": fps_,
                "max_frames": max_frames,
            }
        )

    def _embed_job(job: dict) -> tuple[int, bytes | None]:
        clip = clip_by_id[job["clip_id"]]
        blob = _embed_with_token_fallback(
            provider,
            spec,
            clip,
            job["text"],
            job["path"],
            job["audio_path"],
            job["fps"],
            job["max_frames"],
        )
        return clip.id, blob

    failures = 0
    succeeded = 0
    inflight = settings.embeddings.inflight
    with progress(len(jobs), f"Embedding {spec.name}") as advance:
        for clip_id, blob, elapsed in _dispatch_embedding_jobs(
            job_specs, _embed_job, inflight, log_tag
        ):
            if blob is None:
                failures += 1
                advance(detail=f"✗ {clip_id}")
                continue
            row = ClipEmbedding(
                clip_id=clip_id,
                embedding_case=spec.name,
                embedding=blob,
                source_hash=per_clip[clip_id],
            )
            session.merge(row)
            session.commit()
            succeeded += 1
            stats: dict = {"dim": len(blob) // 4}
            if elapsed is not None:
                stats["time"] = elapsed
            log(
                log_tag,
                "EMB",
                f"clip_{clip_id}",
                "ok",
                stats=stats,
            )
            advance(detail=f"✓ {clip_id}")

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


def _embed_with_token_fallback(
    provider,
    spec: EmbeddingCaseSpec,
    clip,
    text: str | None,
    video_path: str | None,
    audio_path: str | None,
    fps_: float | None,
    max_frames: int | None,
) -> bytes | None:
    """Run the provider once, with a descending frame-cap retry only for
    cases that opt into video token-budget fallback. Returns the float32
    blob on success.

    Non-token-budget exceptions propagate to the dispatcher so the
    failure cause is logged per-clip. Token-mismatch errors are caught
    only while a retry frame-cap remains; the final attempt's exception
    also propagates.
    """

    def _build(cap_: int | None) -> dict:
        p = spec.payload_builder(clip, text, video_path, audio_path, fps_, cap_)
        p["clip_id"] = clip.id
        p["case"] = spec.name
        return p

    if not spec.apply_video_token_fallback or max_frames is None:
        payload = _build(max_frames)
        out = provider.embed(payload)
        return to_bytes(out[0])

    caps = frame_retry_schedule(max_frames)
    for attempt_idx, cap in enumerate(caps):
        payload = _build(cap)
        try:
            out = provider.embed(payload)
        except Exception as e:
            if is_token_mismatch_error(e) and attempt_idx < len(caps) - 1:
                continue
            raise
        return to_bytes(out[0])
    return None
