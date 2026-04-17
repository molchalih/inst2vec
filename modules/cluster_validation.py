"""Phase 6b — clustering validation: filter, score, composite, bootstrap, plateau."""
import math
import os

import numpy as np
import hdbscan.validity
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sqlalchemy.orm import Session

from modules.database import ClusterRun
from modules.clustering import compute_clusters, load_user_matrix


_PARAM_COLS = [
    "umap_n_components", "umap_n_neighbors", "umap_min_dist", "umap_metric",
    "umap2d_n_neighbors", "umap2d_min_dist", "umap2d_metric",
    "hdbscan_min_cluster_size", "hdbscan_min_samples",
    "hdbscan_cluster_selection_method", "hdbscan_metric",
    "random_state",
]


def _row_to_params(row: ClusterRun) -> dict:
    return {col: getattr(row, col) for col in _PARAM_COLS}


def _minmax(values: list[float]) -> list[float]:
    finite = [v for v in values if not math.isnan(v)]
    if not finite or max(finite) == min(finite):
        return [0.0 for _ in values]
    lo, hi = min(finite), max(finite)
    return [0.0 if math.isnan(v) else (v - lo) / (hi - lo) for v in values]


def _phase_filter(session: Session, case: str) -> None:
    max_noise = float(os.environ.get("VALIDATION_MAX_NOISE_RATIO", "0.3"))
    min_clusters = int(os.environ.get("VALIDATION_MIN_CLUSTERS", "3"))
    max_clusters = int(os.environ.get("VALIDATION_MAX_CLUSTERS", "20"))

    rows = (
        session.query(ClusterRun)
        .filter(ClusterRun.embedding_case == case, ClusterRun.disqualified.is_(None))
        .all()
    )
    n_pass = 0
    for row in rows:
        passes = (
            row.noise_ratio <= max_noise
            and min_clusters <= row.n_clusters <= max_clusters
        )
        row.disqualified = 0 if passes else 1
        n_pass += int(passes)
    session.commit()

    print(f"[validate:{case}] filter — {n_pass} passed, {len(rows) - n_pass} disqualified")


def _phase_score(session: Session, case: str, matrix: np.ndarray) -> None:
    rows = (
        session.query(ClusterRun)
        .filter(
            ClusterRun.embedding_case == case,
            ClusterRun.disqualified == 0,
            ClusterRun.dbcv.is_(None),
        )
        .all()
    )

    for i, row in enumerate(rows):
        params = _row_to_params(row)
        try:
            result = compute_clusters(matrix, return_nd_matrix=True, **params)
        except ValueError as exc:
            print(f"[validate:{case}] score skip id={row.id} — {exc}")
            continue

        X_nd = result.matrix_nd.astype(np.float64)
        labels = result.labels

        try:
            row.dbcv = float(hdbscan.validity.validity_index(
                X_nd, labels, metric=row.hdbscan_metric
            ))
        except Exception:
            row.dbcv = float("nan")

        non_noise = labels != -1
        unique_clusters = np.unique(labels[non_noise])
        if len(unique_clusters) >= 2:
            try:
                row.silhouette = float(silhouette_score(X_nd[non_noise], labels[non_noise]))
            except Exception:
                row.silhouette = float("nan")
        else:
            row.silhouette = float("nan")

        session.commit()
        dbcv_str = f"{row.dbcv:.4f}" if row.dbcv is not None else "nan"
        sil_str = f"{row.silhouette:.4f}" if row.silhouette is not None else "nan"
        print(
            f"[validate:{case}] scored {i + 1}/{len(rows)} id={row.id}"
            f" dbcv={dbcv_str} sil={sil_str}"
        )
