"""Pure clustering logic: two-pass UMAP + HDBSCAN → ClusterResult."""

from dataclasses import dataclass, field
from typing import cast

import hdbscan
import numpy as np
from sqlalchemy import select
from umap import UMAP

from core.database import (
    Clip,
    UserEmbedding,
    clip_used_in_analysis,
    get_session,
)

DEFAULT_HDBSCAN_METRIC = "euclidean"

CLUSTER_PARAM_COLS = (
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


def resolve_hdbscan_metric(hdbscan_metric: str | None = None) -> str:
    """HDBSCAN runs on pass-1 UMAP coordinates; distance is always Euclidean in that space."""
    return DEFAULT_HDBSCAN_METRIC


@dataclass
class ClusterResult:
    labels: np.ndarray
    coords_2d: np.ndarray
    n_clusters: int
    noise_ratio: float
    cluster_sizes: list[int] = field(default_factory=list)
    matrix_nd: np.ndarray | None = None


def resolve_umap2d_params(
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    umap2d_n_neighbors: int | None,
    umap2d_min_dist: float | None,
    umap2d_metric: str | None,
) -> tuple[int, float, str]:
    """When a pass-2 UMAP arg is None, inherit from pass-1."""
    nn = umap2d_n_neighbors if umap2d_n_neighbors is not None else umap_n_neighbors
    md = umap2d_min_dist if umap2d_min_dist is not None else umap_min_dist
    met = umap2d_metric if umap2d_metric is not None else umap_metric
    return nn, md, met


def compute_clusters(
    matrix: np.ndarray,
    umap_n_components: int = 15,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    umap_metric: str = "cosine",
    umap2d_n_neighbors: int | None = None,
    umap2d_min_dist: float | None = None,
    umap2d_metric: str | None = None,
    hdbscan_min_cluster_size: int = 15,
    hdbscan_min_samples: int | None = None,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_metric: str = DEFAULT_HDBSCAN_METRIC,
    hdbscan_max_cluster_frac: float = 0.0,
    random_state: int = 42,
    return_nd_matrix: bool = False,
    umap_n_jobs: int = 1,
) -> ClusterResult:
    min_required = umap_n_components + 1
    if matrix.shape[0] < min_required:
        raise ValueError(
            f"compute_clusters requires at least {min_required} rows "
            f"(umap_n_components={umap_n_components} + 1), got {matrix.shape[0]}"
        )

    u2_n, u2_md, u2_metric = resolve_umap2d_params(
        umap_n_neighbors,
        umap_min_dist,
        umap_metric,
        umap2d_n_neighbors,
        umap2d_min_dist,
        umap2d_metric,
    )

    # Pass 1 — reduce to n_components for clustering
    # spectral init often warns/fails on tight eigengaps (then falls back to random); skip it
    reducer_nd = UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        init="random",
        random_state=random_state,
        n_jobs=umap_n_jobs,
    )
    matrix_nd = cast(np.ndarray, reducer_nd.fit_transform(matrix))

    effective_metric = resolve_hdbscan_metric(hdbscan_metric)
    # Cap the largest cluster as a fraction of the sample (only eom honors it;
    # HDBSCAN treats max_cluster_size=0 as "no limit").
    max_cluster_size = (
        round(hdbscan_max_cluster_frac * matrix.shape[0])
        if hdbscan_max_cluster_frac and hdbscan_max_cluster_frac > 0.0
        else 0
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_method=hdbscan_cluster_selection_method,
        metric=effective_metric,
        max_cluster_size=max_cluster_size,
    )
    labels = clusterer.fit_predict(matrix_nd)

    # Pass 2 — independent 2D reduction from the original matrix (not matrix_nd)
    reducer_2d = UMAP(
        n_components=2,
        n_neighbors=u2_n,
        min_dist=u2_md,
        metric=u2_metric,
        init="random",
        random_state=random_state,
        n_jobs=umap_n_jobs,
    )
    coords_2d = cast(np.ndarray, reducer_2d.fit_transform(matrix))

    unique_labels = [lbl for lbl in set(labels) if lbl >= 0]
    n_clusters = len(unique_labels)
    noise_ratio = float(np.sum(labels == -1)) / len(labels)
    cluster_sizes = sorted(
        [int(np.sum(labels == lbl)) for lbl in unique_labels],
        reverse=True,
    )

    return ClusterResult(
        labels=labels,
        coords_2d=coords_2d.astype(np.float32),
        n_clusters=n_clusters,
        noise_ratio=noise_ratio,
        cluster_sizes=cluster_sizes,
        matrix_nd=matrix_nd.astype(np.float32) if return_nd_matrix else None,
    )


def preprocess_matrix(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Strip the dominant shared component before clustering.

    "none" returns the matrix unchanged; "center" subtracts the column mean;
    "standardize" z-scores each column (zero-variance columns are left at 0).
    Anisotropic embeddings (e.g. raw MAEST at ~0.97 mean cosine) cluster into
    one mega-blob under cosine until this shared offset/scale is removed.
    """
    if mode in ("", "none") or matrix.shape[0] == 0:
        return matrix
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if mode == "center":
        return centered
    if mode == "standardize":
        std = matrix.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return centered / std
    raise ValueError(f"unknown embedding preprocess mode: {mode!r}")


def load_user_matrix(
    embedding_case: str, preprocess: str = "none"
) -> tuple[np.ndarray, list[int]]:
    """Load user embeddings for the analysis dataset. Returns (matrix, user_ids).

    Analysis scope is enforced by reusing ``clip_used_in_analysis()`` via a
    subquery on UserEmbedding.user_id; no clip-level WHERE clause is
    duplicated here. ``preprocess`` applies an optional per-case transform
    (see :func:`preprocess_matrix`) so search/validation/assign all share the
    same transformed matrix.
    """
    session = get_session()
    try:
        analysis_users = select(Clip.user_id).distinct().where(*clip_used_in_analysis())
        rows = (
            session.query(UserEmbedding.user_id, UserEmbedding.embedding)
            .filter(
                UserEmbedding.embedding_case == embedding_case,
                UserEmbedding.user_id.in_(analysis_users),
            )
            .order_by(UserEmbedding.user_id)
            .all()
        )
        if not rows:
            return np.empty((0, 0), dtype=np.float32), []
        user_ids = [r.user_id for r in rows]
        arrays = [np.frombuffer(r.embedding, dtype=np.float32).copy() for r in rows]
        return preprocess_matrix(np.stack(arrays), preprocess), user_ids
    finally:
        session.close()
