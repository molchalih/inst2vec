"""Local in-process clip embedding.

One ``LocalEmbedder`` is opened per clip-embedding stage. It builds a single
``ProviderRouter`` (so all Qwen-backbone cases share one model instance) and,
for each case, embeds that case's jobs — optionally coalescing same-case jobs
into one padded GPU forward — writing each vector to the DB as it completes.

A clip whose embed raises is logged as a structured ERR line and counted as a
failure so the runner refuses to seal the case; the next run retries only the
unresolved clips. Failed clips are never written as placeholder embeddings.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from core.database import ClipEmbedding
from core.log import item as log_item
from modules.embeddings.cases import EmbeddingCaseSpec, build_provider_router
from modules.embeddings.sampling import frame_retry_schedule, is_token_mismatch_error
from modules.embeddings.vectors import to_bytes


def _safe_join(root: str, key: str) -> str:
    # Jobs carry bare filenames ("{clip_id}.mp4"/".mp3") resolved against the
    # media root; reject anything with path separators or a parent ref.
    if key != os.path.basename(key) or key in (".", ".."):
        raise ValueError(f"unsafe media key (must be a bare filename): {key!r}")
    return os.path.join(root, key)


def _resolve_video_path(video_root: str, video_key: str | None) -> str | None:
    # qwen-vl-utils forms a "file://" URI from the path; abspath keeps it valid
    # and matches build_jobs_for_case's existence probe.
    if video_key is None:
        return None
    return os.path.abspath(_safe_join(video_root, video_key))


def _resolve_audio_path(audio_root: str | None, audio_key: str | None) -> str | None:
    if audio_key is None or audio_root is None:
        return None
    return os.path.abspath(_safe_join(audio_root, audio_key))


def embed_with_token_fallback(
    provider,
    spec: EmbeddingCaseSpec,
    *,
    clip_id: int,
    text: str | None,
    video_path: str | None,
    audio_path: str | None,
    fps: float | None,
    max_frames: int | None,
) -> bytes:
    """Embed one payload, retrying with smaller frame caps on a video
    token-budget mismatch for cases that opt in. Returns a float32 blob.
    Non-token errors and the final attempt's error propagate."""

    def _build(cap: int | None) -> dict:
        p = spec.payload_builder(None, text, video_path, audio_path, fps, cap)
        p["clip_id"] = clip_id
        p["case"] = spec.name
        return p

    if not spec.apply_video_token_fallback or max_frames is None:
        out = provider.embed(_build(max_frames))
        return to_bytes(out[0])

    caps = frame_retry_schedule(max_frames)
    for idx, cap in enumerate(caps):
        try:
            out = provider.embed(_build(cap))
        except Exception as e:
            if is_token_mismatch_error(e) and idx < len(caps) - 1:
                continue
            raise
        return to_bytes(out[0])
    raise RuntimeError("frame_retry_schedule exhausted")  # unreachable: caps non-empty


def _build_payload(
    spec: EmbeddingCaseSpec, job: dict, video_root: str, audio_root: str | None
) -> dict:
    p = spec.payload_builder(
        None,
        job["text"],
        _resolve_video_path(video_root, job["video_key"]),
        _resolve_audio_path(audio_root, job.get("audio_key")),
        job["fps"],
        job["max_frames"],
    )
    p["clip_id"] = job["clip_id"]
    p["case"] = spec.name
    return p


def _chunks(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class LocalEmbedder:
    """Open once per clip-embedding stage; embed each case via ``embed_case``."""

    def __init__(self, settings, cases: list[str]) -> None:
        self._settings = settings
        self._router = build_provider_router(settings, list(cases))
        self._video_root = settings.paths.video_dir
        self._audio_root = settings.paths.audio_dir
        self._batch_size = max(
            int(getattr(settings.embeddings, "embed_batch_size", 1)), 1
        )

    def __enter__(self) -> LocalEmbedder:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def _embed_one(
        self, spec: EmbeddingCaseSpec, job: dict
    ) -> tuple[int, bytes | None, str | None]:
        try:
            blob = embed_with_token_fallback(
                self._router,
                spec,
                clip_id=job["clip_id"],
                text=job["text"],
                video_path=_resolve_video_path(self._video_root, job["video_key"]),
                audio_path=_resolve_audio_path(self._audio_root, job.get("audio_key")),
                fps=job["fps"],
                max_frames=job["max_frames"],
            )
            return job["clip_id"], blob, None
        except Exception as e:  # logged + counted by embed_case; never seals
            return job["clip_id"], None, repr(e)

    def _embed_group(
        self, spec: EmbeddingCaseSpec, group: list[dict]
    ) -> list[tuple[int, bytes | None, str | None]]:
        """Embed a same-case group. Single-payload (or token-fallback) cases go
        per-clip; otherwise one padded ``embed_batch`` forward, falling back to
        per-clip on any batch error so one bad clip can't fail its neighbours."""
        if len(group) == 1 or spec.apply_video_token_fallback:
            return [self._embed_one(spec, job) for job in group]
        payloads = [
            _build_payload(spec, job, self._video_root, self._audio_root)
            for job in group
        ]
        try:
            outs = self._router.embed_batch(payloads)
        except Exception:
            return [self._embed_one(spec, job) for job in group]
        return [
            (job["clip_id"], to_bytes(vec), None)
            for job, vec in zip(group, outs, strict=True)
        ]

    def embed_case(
        self,
        session,
        spec: EmbeddingCaseSpec,
        jobs: list[dict],
        per_clip: dict[int, str],
    ) -> tuple[int, int]:
        """Embed ``jobs`` for ``spec`` and write each success to ``session``.

        Returns (succeeded, failures). The DB write is the single writer for
        the case, mirroring the per-clip source hash captured at scan time."""
        if not jobs:
            return 0, 0
        batch_size = 1 if spec.apply_video_token_fallback else self._batch_size
        succeeded = 0
        failures = 0
        for group in _chunks(jobs, batch_size):
            for clip_id, blob, err in self._embed_group(spec, group):
                with log_item("EXTRACT", f"clip_{clip_id}") as t:
                    if blob is None:
                        raise RuntimeError(err or "no blob returned")
                    session.merge(
                        ClipEmbedding(
                            clip_id=clip_id,
                            embedding_case=spec.name,
                            embedding=blob,
                            source_hash=per_clip[clip_id],
                        )
                    )
                    session.commit()
                    t.stats(dim=len(blob) // 4)
                if t.failed:
                    failures += 1
                else:
                    succeeded += 1
        return succeeded, failures
