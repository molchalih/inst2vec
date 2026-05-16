"""Shared clip-embedding runner driven by EmbeddingCaseSpec entries."""

from __future__ import annotations

import os

from modules.console import log, progress
from modules.database import Base, ClipEmbedding, get_engine, get_session
from modules.embeddings.cases import CASE_REGISTRY, DEFAULT_CASES, EmbeddingCaseSpec
from modules.embeddings.sampling import (
    adaptive_sampling,
    frame_retry_schedule,
    is_token_mismatch_error,
)
from modules.embeddings.state import (
    get_clip_embedding_candidates,
    get_embedded_clip_ids,
    get_music_map,
)
from modules.embeddings.vectors import to_bytes


def embed_clip_embeddings(settings, cases: list[str] | None = None) -> None:
    """Embed clips for the given cases (default: all DEFAULT_CASES)."""
    case_names = list(cases) if cases is not None else list(DEFAULT_CASES)
    for name in case_names:
        spec = CASE_REGISTRY[name]
        _run_case(settings, spec)


def _video_path(clip_id: int, video_dir: str) -> str:
    return os.path.abspath(os.path.join(video_dir, f"{clip_id}.mp4"))


def _run_case(settings, spec: EmbeddingCaseSpec) -> None:
    log_tag = f"embed:{spec.name}"
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        embedded = get_embedded_clip_ids(session, spec.name)
        candidates = get_clip_embedding_candidates(
            session, settings.embeddings.exclude_disqualified_users
        )

        music_map: dict = {}
        if spec.requires_text:
            music_map = get_music_map(session)

        video_dir = settings.paths.video_dir
        jobs: list[tuple[object, str | None]] = []
        for clip in candidates:
            if clip.id in embedded:
                continue
            if spec.requires_video:
                path = _video_path(clip.id, video_dir)
                if not os.path.exists(path):
                    continue
            text: str | None = None
            if spec.requires_text:
                assert spec.text_builder is not None
                text = spec.text_builder(clip, music_map)
                if text is None:
                    continue
            jobs.append((clip, text))

        if not jobs:
            log(log_tag, "nothing to do")
            return

        log(log_tag, f"{len(jobs)} clips to embed ({len(embedded)} already done)")

        provider = spec.provider_factory(settings)

        with progress(len(jobs), f"Embedding {spec.name}") as advance:
            for clip, text in jobs:
                if spec.requires_video:
                    path = _video_path(clip.id, video_dir)
                    fps, max_frames, _ = adaptive_sampling(
                        path,
                        settings.embeddings.adaptive_max_frames,
                        settings.embeddings.adaptive_default_fps,
                    )
                else:
                    path, fps, max_frames = None, None, None

                blob = _embed_with_token_fallback(
                    provider, spec, clip, text, path, fps, max_frames
                )
                if blob is None:
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

        log(log_tag, "done", level="ok")
    finally:
        session.close()


def _embed_with_token_fallback(
    provider,
    spec: EmbeddingCaseSpec,
    clip,
    text: str | None,
    video_path: str | None,
    fps: float | None,
    max_frames: int | None,
) -> bytes | None:
    """Run the provider once, with a descending frame-cap retry only for
    cases that opt into video token-budget fallback. Returns the float32
    blob on success, or None if all attempts fail (next run will retry).
    """
    if not spec.apply_video_token_fallback or max_frames is None:
        payload = spec.payload_builder(clip, text, video_path, fps, max_frames)
        try:
            out = provider.embed(payload)
        except Exception:
            return None
        return to_bytes(out[0])

    caps = frame_retry_schedule(max_frames)
    for attempt_idx, cap in enumerate(caps):
        payload = spec.payload_builder(clip, text, video_path, fps, cap)
        try:
            out = provider.embed(payload)
            return to_bytes(out[0])
        except Exception as e:
            if is_token_mismatch_error(e) and attempt_idx < len(caps) - 1:
                continue
            return None
    return None
