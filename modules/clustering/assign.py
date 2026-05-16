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

from modules import fingerprint as fp
from modules.clustering.core import compute_clusters, load_user_matrix
from modules.clustering.results import DEFAULT_CASES, select_best_cluster_run
from modules.console import log
from modules.database import (
    Base,
    ClusterRun,
    StageState,
    UserCluster,
    get_engine,
    get_session,
)

STAGE = "cluster_assign"
_PARAM_COLS = (
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "umap2d_n_neighbors",
    "umap2d_min_dist",
    "umap2d_metric",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "hdbscan_cluster_selection_method",
    "hdbscan_metric",
    "random_state",
)
_CONFIG_IDENTITY = "assign=v1"


def _best_params(best: ClusterRun) -> dict:
    return {col: getattr(best, col) for col in _PARAM_COLS}


def _fingerprint(session, case: str) -> fp.Fingerprint:
    best = select_best_cluster_run(session, case)
    if best is None:
        data_payload: list[tuple] = []
    else:
        params = _best_params(best)
        params_hash = hashlib.sha256(
            json.dumps(params, sort_keys=True, default=str).encode()
        ).hexdigest()
        data_payload = [(best.id, params_hash)]
    return fp.Fingerprint(
        data=fp.hash_rows(data_payload),
        config=fp.hash_text(_CONFIG_IDENTITY),
        dependency=fp.stage_dependency_hash(session, "cluster_validation", case),
    )


def _assign_case(case: str) -> None:
    # 1. gate on upstream validation state
    session = get_session()
    try:
        upstream = session.get(StageState, ("cluster_validation", case))
        if upstream is None:
            log(f"assign:{case}", "no validation state — skipping")
            return
        current = _fingerprint(session, case)
        stale = fp.is_stale(session, STAGE, case, current)
        diff = fp.describe_diff(session, STAGE, case, current) if stale else ""
        best = select_best_cluster_run(session, case)
    finally:
        session.close()

    if not stale:
        log(f"assign:{case}", "fingerprint match — skipping")
        return

    log(f"assign:{case}", f"stale ({diff}) — recomputing")

    # 2. compute in memory (no DB lock)
    new_user_clusters: list[UserCluster] = []
    if best is not None:
        matrix, user_ids = load_user_matrix(case)
        if matrix.shape[0] > 0:
            params = _best_params(best)
            try:
                result = compute_clusters(matrix, **params)
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
            except ValueError as exc:
                log(
                    f"assign:{case}",
                    f"compute_clusters skipped — {exc}",
                    level="warn",
                )
                # treat as empty assignment; still seal so we don't retry
                new_user_clusters = []

    # 3. short write section (open AFTER compute)
    session = get_session()
    try:
        session.query(UserCluster).filter_by(embedding_case=case).delete()
        for uc in new_user_clusters:
            session.merge(uc)
        fp.mark_complete(session, STAGE, case, current)
        session.commit()
    finally:
        session.close()
    log(
        f"assign:{case}",
        f"sealed with {len(new_user_clusters)} user_clusters",
        level="ok",
    )


def assign_clusters() -> None:
    """Per-case final clustering assignment, fingerprint-gated."""
    Base.metadata.create_all(get_engine())
    for case in DEFAULT_CASES:
        _assign_case(case)
