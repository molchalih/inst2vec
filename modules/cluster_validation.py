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
