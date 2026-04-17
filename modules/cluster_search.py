"""Grid search over UMAP + HDBSCAN hyperparameters; saves aggregate metrics to ClusterRun."""
import os
from itertools import product

import numpy as np
from sqlalchemy.exc import IntegrityError

from modules.database import Base, engine, get_session, ClusterRun
from modules.clustering import compute_clusters, load_user_matrix


def _parse_ints(val: str) -> list[int]:
    return [int(x) for x in val.split()]


def _parse_floats(val: str) -> list[float]:
    return [float(x) for x in val.split()]


def _parse_strs(val: str) -> list[str]:
    return val.split()


def _load_grid() -> list[dict]:
    """Build cartesian product of hyperparameter combos from env vars."""
    umap_n_components  = _parse_ints(os.environ.get("CLUSTERING_UMAP_N_COMPONENTS", "15"))
    umap_n_neighbors   = _parse_ints(os.environ.get("CLUSTERING_UMAP_N_NEIGHBORS", "15"))
    umap_min_dist      = _parse_floats(os.environ.get("CLUSTERING_UMAP_MIN_DIST", "0.0"))
    umap_metrics       = _parse_strs(os.environ.get("CLUSTERING_UMAP_METRICS", "cosine"))
    umap2d_n_neighbors = int(os.environ.get("CLUSTERING_UMAP2D_N_NEIGHBORS", "15"))
    umap2d_min_dist    = float(os.environ.get("CLUSTERING_UMAP2D_MIN_DIST", "0.1"))
    umap2d_metrics     = _parse_strs(os.environ.get("CLUSTERING_UMAP2D_METRICS", "cosine"))
    hdbscan_min_sizes  = _parse_ints(os.environ.get("CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE", "15"))
    hdbscan_selection  = _parse_strs(os.environ.get("CLUSTERING_HDBSCAN_SELECTION", "eom"))
    hdbscan_metrics    = _parse_strs(os.environ.get("CLUSTERING_HDBSCAN_METRICS", "euclidean"))
    random_state       = int(os.environ.get("CLUSTERING_RANDOM_STATE", "42"))
    cases              = ["video", "sandwich", "audio"]

    combos = []
    for case, nc, nn, md, um, u2m, mcs, sel, hm in product(
        cases, umap_n_components, umap_n_neighbors, umap_min_dist, umap_metrics,
        umap2d_metrics, hdbscan_min_sizes, hdbscan_selection, hdbscan_metrics,
    ):
        combos.append(dict(
            embedding_case=case,
            umap_n_components=nc,
            umap_n_neighbors=nn,
            umap_min_dist=md,
            umap_metric=um,
            umap2d_n_neighbors=umap2d_n_neighbors,
            umap2d_min_dist=umap2d_min_dist,
            umap2d_metric=u2m,
            hdbscan_min_cluster_size=mcs,
            hdbscan_min_samples=None,
            hdbscan_cluster_selection_method=sel,
            hdbscan_metric=hm,
            random_state=random_state,
        ))
    return combos


def run_cluster_search() -> None:
    pass  # implemented in Task 3


def validate_clustering() -> dict[str, dict]:
    pass  # implemented in Task 3
