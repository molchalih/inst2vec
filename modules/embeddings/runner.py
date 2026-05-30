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
from core.config import Secrets, Settings
from core.database import (
    Clip,
    ClipEmbedding,
    StageState,
    get_session,
)
from core.log import StageResult, event, scope, stage
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


def _preflight_warn_pods(settings, secrets) -> None:
    """Warn if pods are configured but no clip can be remote-eligible.

    remote_eligible mirrors Clip.is_uploaded; with no object-store bucket the
    upload stage is a no-op, so every job stays local and connected pods get
    no work. Guarded on the storage attribute so settings stubs are exempt."""
    storage = getattr(settings, "storage", None)
    if secrets.embedder_token and storage is not None and not storage.bucket:
        event(
            "SCAN",
            "pods",
            result="WARN",
            stats={
                "msg": "EMBEDDER_TOKEN set but storage.bucket empty: no clip is "
                "remote-eligible, so connected pods receive no work — set "
                "storage.bucket, or unset EMBEDDER_TOKEN for local-only embedding"
            },
        )


@stage("embed")
def run_clip(settings: Settings, secrets: Secrets) -> StageResult:
    """Clip-level embeddings across all configured cases."""
    emb_secrets = EmbeddingSecrets(
        gemini_api_key=secrets.gemini_api_key,
        embedder_token=secrets.embedder_token,
        runpod_api_key=secrets.runpod_api_key,
        coordinator_public_host=secrets.coordinator_public_host,
        huggingface_token=secrets.huggingface_token,
    )
    embed_clip_embeddings(settings, emb_secrets)
    return StageResult()


def embed_clip_embeddings(
    settings,
    secrets: EmbeddingSecrets | None = None,
    cases: list[str] | None = None,
) -> None:
    """Embed clips for the given cases (default: result of default_cases(settings)).

    Opens ONE StageEmbedder for the whole stage so a single coordinator +
    local worker span every case; pods stay connected across cases and exit
    only when the stage closes. ``secrets`` carries provider credentials +
    the embedder token; defaults to an empty ``EmbeddingSecrets()``."""
    if secrets is None:
        secrets = EmbeddingSecrets()
    case_names = list(cases) if cases is not None else list(default_cases(settings))
    for name in case_names:
        if name == "gemini" and not settings.embeddings.gemini_enabled:
            raise RuntimeError(
                "gemini case requested but embeddings.gemini_enabled=false"
            )

    _preflight_warn_pods(settings, secrets)

    from modules.embeddings.distributed import StageEmbedder
    from modules.embeddings.fleet import pod_fleet

    with (
        pod_fleet(settings, secrets) as fleet,
        StageEmbedder(settings, secrets, case_names, fleet=fleet) as stage_emb,
    ):
        for name in case_names:
            _run_case(settings, secrets, name, stage_emb)


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
            video_key = f"{clip.id}.mp4"
        else:
            fps, max_frames, video_key = None, None, None
        audio_key = f"{clip.id}.mp3" if needs_audio else None
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


@scope("embed:{case}")
def _run_case(
    settings,
    secrets,  # retained for positional compat with test_embeddings_backwards_compat.py; unused by body
    case: str,
    stage_emb=None,  # None on the direct-call test path; body returns before stage_emb.drain_case when fp matches
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
            stage_emb,
        )
    finally:
        session.close()


def _embed_targets(
    session,
    spec: EmbeddingCaseSpec,
    settings,
    candidates: list[Clip],
    target_ids: set[int],
    per_clip: dict[int, str],
    embedded: dict[int, str | None],
    current: fp.Fingerprint,
    stage_emb,
) -> tuple[int, int]:
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
        event("DELETE", "stale_rows", stats={"count": len(stale_skipped)})

    t_stage = time.perf_counter()

    if not targets_built:
        if stale_skipped:
            event(
                "SEAL",
                "embed",
                result="WARN",
                stats={"stale": len(stale_skipped)},
            )
            return 0, 0  # do not seal — un-buildable stale targets remain unresolved
        if fresh_skipped:
            event("SCAN", "candidates", stats={"non_embeddable": fresh_skipped})
        fp.mark_complete(session, STAGE, spec.name, current)
        session.commit()
        event(
            "SEAL",
            "embed",
            stats={"done": 0, "time": time.perf_counter() - t_stage},
        )
        return 0, 0

    # Pass active scope string as log_tag so distributed.py (pre-T24) can
    # emit its per-clip lines under the correct scope prefix.
    log_tag = f"embed:{spec.name}"
    case_jobs = build_jobs_for_case(
        spec,
        targets_built,
        texts=texts,
        video_dir=video_dir,
        adaptive_max_frames=settings.embeddings.adaptive_max_frames,
        adaptive_default_fps=settings.embeddings.adaptive_default_fps,
    )
    succeeded, failures = stage_emb.drain_case(
        session, spec, case_jobs, per_clip, log_tag
    )

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
