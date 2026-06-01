"""Per-case final clustering assignment.

Fingerprint-gated stage that materializes the selected best ClusterRun's
parameters into UserCluster rows. Mirrors the user-embeddings stage
pattern: fingerprint → wipe scoped outputs on stale → recompute in
memory → short write block (delete + bulk merge + mark_complete +
commit). Empty UserClusters is a valid stable output.
"""

from __future__ import annotations

import hashlib
import json
import time

from core import fingerprint as fp
from core.config import Settings, ValidationSettings
from core.database import (
    ClusterRun,
    StageState,
    UserCluster,
    get_session,
)
from core.log import event, stage, warn
from core.pipeline import Stage
from modules.clustering.core import (
    CLUSTER_PARAM_COLS,
    ClusterParams,
    compute_clusters,
    load_user_matrix,
)
from modules.clustering.results import select_best_cluster_run

STAGE = Stage.CLUSTER_ASSIGN
# Bump when assign-stage logic changes in a way the data/dependency
# fingerprints would not detect (e.g., how labels are derived from
# compute_clusters output).
_CONFIG_IDENTITY = "assign=v2"


def _best_params(best: ClusterRun) -> dict:
    return {col: getattr(best, col) for col in CLUSTER_PARAM_COLS}


def _fingerprint(session, case: str, settings: ValidationSettings) -> fp.Fingerprint:
    best = select_best_cluster_run(
        session, case, threshold=float(settings.plateau_drop_threshold)
    )
    if best is None:
        data_payload: list[tuple] = []
    else:
        params_hash = hashlib.sha256(
            json.dumps(_best_params(best), sort_keys=True, default=str).encode()
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


def _fit_user_clusters(
    case: str, best: ClusterRun, max_cluster_frac: float, preprocess: str
) -> list[UserCluster]:
    """Run the champion's params on the user matrix; emit a UserCluster row per user."""
    matrix, user_ids = load_user_matrix(case, preprocess=preprocess)
    if matrix.shape[0] == 0:
        return []
    t_fit = time.perf_counter()
    combo = _best_params(best)
    try:
        result = compute_clusters(
            matrix,
            ClusterParams.from_combo(combo, max_cluster_frac=max_cluster_frac),
            random_state=int(combo["random_state"]),
        )
    except ValueError as exc:
        event(
            "EXTRACT",
            "champion",
            result="ERR",
            stats={"time": time.perf_counter() - t_fit, "err": str(exc)},
        )
        return []
    event(
        "EXTRACT",
        "champion",
        stats={
            "time": time.perf_counter() - t_fit,
            "k": result.n_clusters,
            "noise": round(result.noise_ratio, 3),
        },
    )
    centralities = result.centralities
    has_centrality = centralities.size == len(user_ids)
    return [
        UserCluster(
            user_id=user_ids[i],
            embedding_case=case,
            cluster_id=int(result.labels[i]),
            umap_x=float(result.coords_2d[i, 0]),
            umap_y=float(result.coords_2d[i, 1]),
            centrality=float(centralities[i]) if has_centrality else 0.0,
        )
        for i in range(len(user_ids))
    ]


def _seal(case: str, current: fp.Fingerprint, rows: list[UserCluster]) -> None:
    session = get_session()
    try:
        session.query(UserCluster).filter_by(embedding_case=case).delete()
        if rows:
            session.bulk_save_objects(rows)
        fp.mark_complete(session, STAGE, case, current)
        session.commit()
    finally:
        session.close()


def _check_fingerprint(
    case: str, settings: ValidationSettings
) -> tuple[fp.Fingerprint | None, bool, str, ClusterRun | None]:
    """Returns (current, stale, diff, best). current is None when upstream
    validation has not run for this case (then stale is False)."""
    session = get_session()
    try:
        if session.get(StageState, ("cluster_validation", case)) is None:
            return None, False, "", None
        current = _fingerprint(session, case, settings)
        stale = fp.is_stale(session, STAGE, case, current)
        diff = fp.describe_diff(session, STAGE, case, current) if stale else ""
        best = select_best_cluster_run(
            session, case, threshold=float(settings.plateau_drop_threshold)
        )
        return current, stale, diff, best
    finally:
        session.close()


def _assign_case(
    case: str,
    settings: ValidationSettings,
    preprocess: str = "none",
    max_cluster_frac: float = 0.0,
) -> None:
    current, stale, diff, best = _check_fingerprint(case, settings)

    if current is None:
        warn("SKIP", "validation")
        return
    if not stale:
        event("SKIP", "fingerprint")
        return

    warn("SCAN", "fingerprint", stats={"diff": diff})
    rows = (
        _fit_user_clusters(case, best, max_cluster_frac, preprocess)
        if best is not None
        else []
    )
    _seal(case, current, rows)
    event("WRITE", "user_clusters", stats={"rows": len(rows)})


@stage("clustering:assign")
def assign_clusters(settings: Settings, cases: tuple[str, ...]) -> None:
    """Per-case final clustering assignment, fingerprint-gated."""
    validation = settings.validation
    max_cluster_frac = float(settings.search.hdbscan_max_cluster_frac)
    for case in cases:
        preprocess = settings.search.embedding_preprocess.get(case, "none")
        _assign_case(case, validation, preprocess, max_cluster_frac)
