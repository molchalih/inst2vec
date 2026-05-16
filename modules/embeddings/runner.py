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
"""

from __future__ import annotations

import os

from modules import fingerprint as fp
from modules.console import log, progress
from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    StageState,
    get_engine,
    get_session,
)
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
    get_clip_embedding_candidates,
    get_embedded_source_hashes,
    get_music_map,
    per_clip_source_hashes_and_aggregate,
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


def _compute_fingerprint_and_per_clip(
    session, spec: EmbeddingCaseSpec, settings, candidates: list[Clip]
) -> tuple[fp.Fingerprint, dict[int, str]]:
    """Return (Fingerprint, {clip_id: per_clip_source_hash}) for ``case``.

    Both share the same ``dependency_rows_for_case`` source of truth so the
    aggregate ``Fingerprint.dependency`` and the per-row hashes never drift.
    """
    candidate_ids = sorted(c.id for c in candidates)
    per_clip, dep_agg = per_clip_source_hashes_and_aggregate(
        session, spec.name, candidate_ids
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


def _diff_targets(
    per_clip: dict[int, str], embedded: dict[int, str | None]
) -> set[int]:
    """Clip ids that need (re-)embedding: missing rows or stored hash != desired."""
    return {cid for cid, want in per_clip.items() if embedded.get(cid) != want}


def _run_case(settings, spec: EmbeddingCaseSpec) -> None:
    log_tag = f"embed:{spec.name}"
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        candidates = get_clip_embedding_candidates(
            session, settings.embeddings.exclude_disqualified_users
        )

        current, per_clip = _compute_fingerprint_and_per_clip(
            session, spec, settings, candidates
        )
        if not fp.is_stale(session, STAGE, spec.name, current):
            log(log_tag, "fingerprint match — skipping")
            return

        stored = session.get(StageState, (STAGE, spec.name))
        if stored is not None and stored.config_hash != current.config:
            diff = fp.describe_diff(session, STAGE, spec.name, current)
            log(log_tag, f"config drift ({diff}) — wiping case")
            _wipe_case(session, spec.name)

        embedded = get_embedded_source_hashes(session, spec.name)
        target_ids = _diff_targets(per_clip, embedded)
        log(log_tag, f"{len(target_ids)} clip(s) to (re-)embed")

        _embed_targets(
            session, spec, settings, log_tag, candidates, target_ids, per_clip, current
        )
    finally:
        session.close()


def _embed_targets(
    session,
    spec: EmbeddingCaseSpec,
    settings,
    log_tag: str,
    candidates: list[Clip],
    target_ids: set[int],
    per_clip: dict[int, str],
    current: fp.Fingerprint,
) -> None:
    """Embed the subset of ``candidates`` whose ids are in ``target_ids``.

    Writes ``source_hash`` on every merged row so future runs can diff. On
    full success seals the stage; on any failure leaves stage unsealed so the
    next run retries only the still-missing/stale clips.
    """
    targets = [c for c in candidates if c.id in target_ids]

    # Materialize the work list (skip clips missing video files or text).
    music_map: dict = {}
    if spec.text_builder is not None:
        music_map = get_music_map(session)

    video_dir = settings.paths.video_dir
    jobs: list[tuple[Clip, str | None]] = []
    for clip in targets:
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

    log(log_tag, f"{len(jobs)} clips to embed")

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
                source_hash=per_clip[clip.id],
            )
            session.merge(row)
            session.commit()
            advance(detail=f"✓ {clip.id}")

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
