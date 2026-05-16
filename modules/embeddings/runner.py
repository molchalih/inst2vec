"""Shared clip-embedding runner driven by EmbeddingCaseSpec entries.

Per case the stage:

  1. Picks candidates (selected + downloaded + optional eligibility).
  2. Computes Fingerprint(data, config, dependency). Dependency rows are
     case-specific and mirror what the case's text+payload builders read.
  3. If stale, deletes ClipEmbedding rows for the case, recomputes,
     mark_completes, and commits once at the end.
"""

from __future__ import annotations

import os

from modules import fingerprint as fp
from modules.console import log, progress
from modules.database import Base, Clip, ClipEmbedding, get_engine, get_session
from modules.embeddings.cases import (
    CASE_REGISTRY,
    DEFAULT_CASES,
    EmbeddingCaseSpec,
    case_config_identity,
)
from modules.embeddings.sampling import (
    adaptive_sampling,
    frame_retry_schedule,
    is_token_mismatch_error,
)
from modules.embeddings.state import (
    dependency_rows_for_case,
    get_clip_embedding_candidates,
    get_music_map,
)
from modules.embeddings.vectors import to_bytes

STAGE = "clip_embeddings"


def embed_clip_embeddings(settings, cases: list[str] | None = None) -> None:
    """Embed clips for the given cases (default: all DEFAULT_CASES)."""
    case_names = list(cases) if cases is not None else list(DEFAULT_CASES)
    for name in case_names:
        spec = CASE_REGISTRY[name]
        _run_case(settings, spec)


def _video_path(clip_id: int, video_dir: str) -> str:
    return os.path.abspath(os.path.join(video_dir, f"{clip_id}.mp4"))


def _compute_fingerprint(
    session, spec: EmbeddingCaseSpec, settings, candidates: list[Clip]
) -> fp.Fingerprint:
    candidate_ids = sorted(c.id for c in candidates)
    data = fp.hash_rows((cid,) for cid in candidate_ids)
    config = fp.hash_text(case_config_identity(spec, settings))
    dependency = fp.hash_rows(
        dependency_rows_for_case(session, spec.name, candidate_ids)
    )
    return fp.Fingerprint(data=data, config=config, dependency=dependency)


def _clear_case(session, case: str) -> None:
    session.query(ClipEmbedding).filter_by(embedding_case=case).delete()
    session.commit()


def _run_case(settings, spec: EmbeddingCaseSpec) -> None:
    log_tag = f"embed:{spec.name}"
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        candidates = get_clip_embedding_candidates(
            session, settings.embeddings.exclude_disqualified_users
        )

        current = _compute_fingerprint(session, spec, settings, candidates)
        if not fp.is_stale(session, STAGE, spec.name, current):
            log(log_tag, "fingerprint match — skipping")
            return

        diff = fp.describe_diff(session, STAGE, spec.name, current)
        log(log_tag, f"stale ({diff}) — recomputing")
        _clear_case(session, spec.name)

        # Materialize work list now that the case is cleared.
        music_map: dict = {}
        if spec.text_builder is not None:
            music_map = get_music_map(session)

        video_dir = settings.paths.video_dir
        jobs: list[tuple[Clip, str | None]] = []
        for clip in candidates:
            if spec.requires_video:
                path = _video_path(clip.id, video_dir)
                if not os.path.exists(path):
                    continue
            text: str | None = None
            if spec.text_builder is not None:
                text = spec.text_builder(clip, music_map)
                if text is None:
                    continue
            jobs.append((clip, text))

        if not jobs:
            log(log_tag, "nothing to embed (empty work set after filtering)")
            fp.mark_complete(session, STAGE, spec.name, current)
            session.commit()
            return

        log(
            log_tag,
            f"{len(jobs)} clips to embed",
        )

        provider = spec.provider_factory(settings)

        failures = 0
        with progress(len(jobs), f"Embedding {spec.name}") as advance:
            for clip, text in jobs:
                if spec.requires_video:
                    path = _video_path(clip.id, video_dir)
                    fps_, max_frames, _ = adaptive_sampling(
                        path,
                        settings.embeddings.adaptive_max_frames,
                        settings.embeddings.adaptive_default_fps,
                    )
                else:
                    path, fps_, max_frames = None, None, None

                blob = _embed_with_token_fallback(
                    provider, spec, clip, text, path, fps_, max_frames
                )
                if blob is None:
                    failures += 1
                    advance(detail=f"✗ {clip.id}")
                    continue

                row = ClipEmbedding(
                    clip_id=clip.id,
                    embedding_case=spec.name,
                    embedding=blob,
                )
                session.merge(row)
                session.commit()
                advance(detail=f"✓ {clip.id}")

        # Only seal the fingerprint when every intended job produced a row.
        # Otherwise the same data/config/dependency hashes would mark missing
        # rows as complete on the next run and they'd never be retried.
        if failures:
            log(
                log_tag,
                f"{failures}/{len(jobs)} failed — leaving stage stale for retry",
                level="warn",
            )
        else:
            fp.mark_complete(session, STAGE, spec.name, current)
            session.commit()
        log(log_tag, "done", level="ok")
    finally:
        session.close()


def _embed_with_token_fallback(
    provider,
    spec: EmbeddingCaseSpec,
    clip,
    text: str | None,
    video_path: str | None,
    fps_: float | None,
    max_frames: int | None,
) -> bytes | None:
    """Run the provider once, with a descending frame-cap retry only for
    cases that opt into video token-budget fallback. Returns the float32
    blob on success, or None if all attempts fail (next run will retry).
    """
    if not spec.apply_video_token_fallback or max_frames is None:
        payload = spec.payload_builder(clip, text, video_path, fps_, max_frames)
        try:
            out = provider.embed(payload)
        except Exception:
            return None
        return to_bytes(out[0])

    caps = frame_retry_schedule(max_frames)
    for attempt_idx, cap in enumerate(caps):
        payload = spec.payload_builder(clip, text, video_path, fps_, cap)
        try:
            out = provider.embed(payload)
            return to_bytes(out[0])
        except Exception as e:
            if is_token_mismatch_error(e) and attempt_idx < len(caps) - 1:
                continue
            return None
    return None
