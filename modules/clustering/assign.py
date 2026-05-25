"""Per-case final clustering assignment.

Fingerprint-gated stage that materializes the selected best ClusterRun's
parameters into UserCluster rows. Mirrors the user-embeddings stage
pattern: fingerprint -> wipe scoped outputs on stale -> recompute in
memory -> short write block (delete + bulk merge + mark_complete +
commit). Empty UserClusters is a valid stable output.
"""

from __future__ import annotations

import hashlib
import json
import time

from core import fingerprint as fp
from core.config import Settings, ValidationSettings
from core.console import log
from core.database import (
    ClusterRun,
    StageState,
    UserCluster,
    get_session,
)
from core.pipeline import Stage
from modules.clustering.core import (
    CLUSTER_PARAM_COLS,
    compute_clusters,
    load_user_matrix,
)
from modules.clustering.results import select_best_cluster_run

STAGE = Stage.CLUSTER_ASSIGN
# Bump when assign-stage logic changes in a way the data/dependency
# fingerprints would not detect (e.g., changing how labels are derived
# from compute_clusters output).
_CONFIG_IDENTITY = "assign=v1"


def _best_params(best: ClusterRun) -> dict:
    return {col: getattr(best, col) for col in CLUSTER_PARAM_COLS}


def _fingerprint(session, case: str, settings: ValidationSettings) -> fp.Fingerprint:
    best = select_best_cluster_run(
        session, case, threshold=float(settings.plateau_drop_threshold)
    )
    if best is None:
        data_payload: list[tuple] = []
    else:
        params = _best_params(best)
        params_hash = hashlib.sha256(
            json.dumps(params, sort_keys=True, default=str).encode()
        ).hexdigest()
        data_payload = [(best.id, params_hash)]
    config_payload = json.dumps(
        {
            "identity": _CONFIG_IDENTITY,
            "plateau_drop_threshold": float(settings.plateau_drop_threshold),
        },
        sort_keys=True,
    )
    return fp.Fingerprint(
        data=fp.hash_rows(data_payload),
        config=fp.hash_text(config_payload),
        dependency=fp.stage_dependency_hash(session, Stage.CLUSTER_VALIDATION, case),
    )


def _assign_case(
    case: str,
    settings: ValidationSettings,
    preprocess: str = "none",
    max_cluster_frac: float = 0.0,
) -> None:
    scope = f"cluster:{case}"
    t_stage = time.perf_counter()
    # 1. gate on upstream validation state
    session = get_session()
    try:
        upstream = session.get(StageState, ("cluster_validation", case))
        if upstream is None:
            log(scope, "SKIP", "validation", "none")
            return
        current = _fingerprint(session, case, settings)
        stale = fp.is_stale(session, STAGE, case, current)
        diff = fp.describe_diff(session, STAGE, case, current) if stale else ""
        best = select_best_cluster_run(
            session, case, threshold=float(settings.plateau_drop_threshold)
        )
    finally:
        session.close()

    if not stale:
        log(scope, "SKIP", "fingerprint", "ok")
        return

    log(scope, "SCAN", "fingerprint", "stale", stats={"diff": diff})

    # 2. compute in memory (no DB lock)
    new_user_clusters: list[UserCluster] = []
    fit_stats: dict = {}
    if best is not None:
        matrix, user_ids = load_user_matrix(case, preprocess=preprocess)
        if matrix.shape[0] > 0:
            params = _best_params(best)
            t_fit = time.perf_counter()
            try:
                result = compute_clusters(
                    matrix, hdbscan_max_cluster_frac=max_cluster_frac, **params
                )
                new_user_clusters = [
                    UserCluster(
                        user_id=user_ids[i],
                        embedding_case=case,
                        cluster_id=int(result.labels[i]),
                        umap_x=float(result.coords_2d[i, 0]),
                        umap_y=float(result.coords_2d[i, 1]),
                    )
                    for i in range(len(user_ids))
                ]
                fit_stats = {
                    "time": time.perf_counter() - t_fit,
                    "k": result.n_clusters,
                    "noise": round(result.noise_ratio, 3),
                }
                log(scope, "FIT", "champion", "ok", stats=fit_stats)
            except ValueError as exc:
                log(
                    scope,
                    "FIT",
                    "champion",
                    "ERR",
                    stats={
                        "time": time.perf_counter() - t_fit,
                        "err": str(exc),
                    },
                )
                new_user_clusters = []

    # 3. short write section (open AFTER compute)
    session = get_session()
    try:
        session.query(UserCluster).filter_by(embedding_case=case).delete()
        if new_user_clusters:
            session.bulk_save_objects(new_user_clusters)
        fp.mark_complete(session, STAGE, case, current)
        session.commit()
    finally:
        session.close()
    log(
        scope,
        "WRITE",
        "user_clusters",
        "ok",
        stats={"rows": len(new_user_clusters)},
    )
    log(
        scope,
        "SEAL",
        "assign",
        "ok",
        stats={"time": time.perf_counter() - t_stage},
    )


def assign_clusters(settings: Settings, cases: tuple[str, ...]) -> None:
    """Per-case final clustering assignment, fingerprint-gated.

    ``cases`` is the tuple of embedding case names to assign clusters for
    (e.g. ``("video", "sandwich", "audio")``).  Best-run selection uses
    ``settings.validation.plateau_drop_threshold``.
    """
    validation_settings = settings.validation
    max_cluster_frac = float(settings.search.hdbscan_max_cluster_frac)
    for case in cases:
        preprocess = settings.search.embedding_preprocess.get(case, "none")
        _assign_case(case, validation_settings, preprocess, max_cluster_frac)
