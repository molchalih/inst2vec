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
video) are caught by the ``LocalEmbedder`` in
``modules/embeddings/local.py``, logged as a structured ERR line, and
counted so the runner refuses to seal — the next run retries just those
clips. Failed clips are NOT replaced with placeholder embeddings.
"""

from __future__ import annotations

import os
import time

from core import fingerprint as fp
from core.config import Secrets, Settings
from core.database import (
    Clip,
    ClipEmbedding,
    StageState,
    get_session,
)
from core.log import StageResult, event, scope, stage
from core.pipeline import Stage
from modules.embeddings.cases import (
    CASE_REGISTRY,
    EmbeddingCaseSpec,
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


@stage("embed")
def run_clip(settings: Settings, secrets: Secrets) -> StageResult:
    """Clip-level embeddings across all configured cases."""
    del secrets  # local embedding needs no credentials
    embed_clip_embeddings(settings)
    return StageResult()


def embed_clip_embeddings(
    settings,
    secrets=None,  # retained for positional call-compat; embedding is local
    cases: list[str] | None = None,
) -> None:
    """Embed clips for the given cases (default: result of default_cases(settings)).

    Opens ONE ``LocalEmbedder`` for the whole stage so the Qwen-backbone cases
    share a single in-process model instance across cases."""
    del secrets
    case_names = list(cases) if cases is not None else list(default_cases(settings))

    from modules.embeddings.local import LocalEmbedder

    with LocalEmbedder(settings, case_names) as embedder:
        for name in case_names:
            _run_case(settings, name, embedder)


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
    """One job dict per clip. Probes the local video for fps/max_frames.
    ``texts`` is the prebuilt text per clip. Clips with no local video file are
    skipped for video cases. ``audio_key`` is set for cases whose dependency
    columns include ``_audio_file_stat`` (auditory)."""
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
            video_key = f"{clip.id}.mp4"
        else:
            fps, max_frames, video_key = None, None, None
        jobs.append(
            {
                "clip_id": clip.id,
                "case": spec.name,
                "text": texts.get(clip.id),
                "video_key": video_key,
                "audio_key": f"{clip.id}.mp3" if needs_audio else None,
                "fps": fps,
                "max_frames": max_frames,
            }
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


@scope("embed:{case}")
def _run_case(
    settings,
    case: str,
    embedder=None,  # None on the direct-call test path; body returns before embed_case when fp matches
) -> tuple[int, int]:
    spec = CASE_REGISTRY[case]
    session = get_session()
    try:
        candidates = get_clip_embedding_candidates(
            session, settings.embeddings.exclude_disqualified_users
        )
        candidate_ids = {c.id for c in candidates}

        # Sweep orphan ClipEmbedding rows for this case. An empty candidate set
        # is never a legitimate "delete every row" signal — it means upstream
        # (selection) produced nothing this run, e.g. a half-loaded DB on a
        # fresh box. Skip the sweep entirely rather than wipe the whole case.
        if candidate_ids:
            deleted = (
                session.query(ClipEmbedding)
                .filter(
                    ClipEmbedding.embedding_case == spec.name,
                    ~ClipEmbedding.clip_id.in_(candidate_ids),
                )
                .delete(synchronize_session=False)
            )
            if deleted:
                session.commit()
                event("DELETE", "orphans", stats={"count": deleted})

        current, per_clip = _compute_fingerprint_and_per_clip(
            session, spec, settings, candidates
        )
        if not fp.is_stale(session, STAGE, spec.name, current):
            event("SKIP", "fingerprint")
            return 0, 0

        stored = session.get(StageState, (STAGE, spec.name))
        if stored is not None and stored.config_hash != current.config:
            diff = fp.describe_diff(session, STAGE, spec.name, current)
            event("SCAN", "fingerprint", result="WARN", stats={"diff": diff})
            _wipe_case(session, spec.name)

        embedded = get_embedded_source_hashes(session, spec.name)
        target_ids = fp.row_diff(per_clip, embedded)
        event("SCAN", "clips", stats={"todo": len(target_ids)})

        return _embed_targets(
            session,
            spec,
            settings,
            candidates,
            target_ids,
            per_clip,
            embedded,
            current,
            embedder,
        )
    finally:
        session.close()


def _clip_embed_text(
    clip: Clip,
    spec: EmbeddingCaseSpec,
    *,
    video_dir: str,
    audio_dir: str | None,
    audio_mir_map: dict,
) -> tuple[bool, str | None]:
    """Resolve a clip's embed text or signal that it is not embeddable.

    Returns ``(True, text)`` when the clip can be embedded (``text`` may be
    ``None`` for non-text cases), or ``(False, None)`` when a required video /
    audio file is missing or the text builder yields ``None``.
    """
    if spec.requires_video and not os.path.exists(_video_path(clip.id, video_dir)):
        return False, None
    if audio_dir is not None:
        audio_path = os.path.abspath(os.path.join(audio_dir, f"{clip.id}.mp3"))
        if not os.path.exists(audio_path):
            return False, None
    if spec.text_builder is not None:
        text = spec.text_builder(clip, audio_mir_map.get(clip.id))
        if text is None:
            return False, None
        return True, text
    return True, None


def _build_target_texts(
    session,
    spec: EmbeddingCaseSpec,
    settings,
    targets: list[Clip],
    embedded: dict[int, str | None],
) -> tuple[dict[int, str | None], list[Clip], list[int], int]:
    """Partition ``targets`` into buildable vs. (stale|fresh) un-embeddable.

    Returns ``(texts, targets_built, stale_skipped, fresh_skipped)``. A target
    that cannot be built counts as ``stale`` when it already had an embedding
    row, else ``fresh``.
    """
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
        ok, text = _clip_embed_text(
            clip,
            spec,
            video_dir=video_dir,
            audio_dir=audio_dir,
            audio_mir_map=audio_mir_map,
        )
        if not ok:
            if clip.id in embedded:
                stale_skipped.append(clip.id)
            else:
                fresh_skipped += 1
            continue
        texts[clip.id] = text
        targets_built.append(clip)
    return texts, targets_built, stale_skipped, fresh_skipped


def _seal_no_targets(
    session,
    spec: EmbeddingCaseSpec,
    current: fp.Fingerprint,
    stale_skipped: list[int],
    fresh_skipped: int,
    t_stage: float,
) -> None:
    """Emit the no-buildable-targets SEAL outcome. Only seals when no stale
    rows remain unresolved."""
    if stale_skipped:
        # do not seal — un-buildable stale targets remain unresolved
        event(
            "SEAL",
            "embed",
            result="WARN",
            stats={"stale": len(stale_skipped)},
        )
        return
    if fresh_skipped:
        event("SCAN", "candidates", stats={"non_embeddable": fresh_skipped})
    fp.mark_complete(session, STAGE, spec.name, current)
    session.commit()
    event(
        "SEAL",
        "embed",
        stats={"done": 0, "time": time.perf_counter() - t_stage},
    )


def _embed_targets(
    session,
    spec: EmbeddingCaseSpec,
    settings,
    candidates: list[Clip],
    target_ids: set[int],
    per_clip: dict[int, str],
    embedded: dict[int, str | None],
    current: fp.Fingerprint,
    embedder,
) -> tuple[int, int]:
    """Embed the subset of ``candidates`` whose ids are in ``target_ids``."""
    targets = [c for c in candidates if c.id in target_ids]

    texts, targets_built, stale_skipped, fresh_skipped = _build_target_texts(
        session, spec, settings, targets, embedded
    )

    if stale_skipped:
        session.query(ClipEmbedding).filter(
            ClipEmbedding.embedding_case == spec.name,
            ClipEmbedding.clip_id.in_(stale_skipped),
        ).delete(synchronize_session=False)
        session.commit()
        event("DELETE", "stale_rows", stats={"count": len(stale_skipped)})

    t_stage = time.perf_counter()

    if not targets_built:
        _seal_no_targets(session, spec, current, stale_skipped, fresh_skipped, t_stage)
        return 0, 0

    video_dir = settings.paths.video_dir
    case_jobs = build_jobs_for_case(
        spec,
        targets_built,
        texts=texts,
        video_dir=video_dir,
        adaptive_max_frames=settings.embeddings.adaptive_max_frames,
        adaptive_default_fps=settings.embeddings.adaptive_default_fps,
    )
    succeeded, failures = embedder.embed_case(session, spec, case_jobs, per_clip)

    if failures or stale_skipped:
        event(
            "SEAL",
            "embed",
            result="WARN",
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
        event(
            "SEAL",
            "embed",
            stats={
                "done": succeeded,
                "time": time.perf_counter() - t_stage,
            },
        )
    return succeeded, failures
