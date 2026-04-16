"""Pure clustering logic: two-pass UMAP + HDBSCAN → ClusterResult."""
from dataclasses import dataclass, field

import numpy as np
import hdbscan
from umap import UMAP


@dataclass
class ClusterResult:
    labels: np.ndarray
    coords_2d: np.ndarray
    n_clusters: int
    noise_ratio: float
    cluster_sizes: list[int] = field(default_factory=list)


def compute_clusters(
    matrix: np.ndarray,
    umap_n_components: int = 15,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    umap_metric: str = "cosine",
    umap2d_n_neighbors: int = 15,
    umap2d_min_dist: float = 0.1,
    umap2d_metric: str = "cosine",
    hdbscan_min_cluster_size: int = 15,
    hdbscan_min_samples: int | None = None,
    hdbscan_cluster_selection_method: str = "eom",
    hdbscan_metric: str = "euclidean",
    random_state: int = 42,
) -> ClusterResult:
    min_required = umap_n_components + 1
    if matrix.shape[0] < min_required:
        raise ValueError(
            f"compute_clusters requires at least {min_required} rows "
            f"(umap_n_components={umap_n_components} + 1), got {matrix.shape[0]}"
        )

    # Pass 1 — reduce to n_components for clustering
    reducer_nd = UMAP(
        n_components=umap_n_components,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        random_state=random_state,
    )
    matrix_nd = reducer_nd.fit_transform(matrix)

    # HDBSCAN on the reduced matrix
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_method=hdbscan_cluster_selection_method,
        metric=hdbscan_metric,
    )
    labels = clusterer.fit_predict(matrix_nd)

    # Pass 2 — independent 2D reduction from the original matrix (not matrix_nd)
    reducer_2d = UMAP(
        n_components=2,
        n_neighbors=umap2d_n_neighbors,
        min_dist=umap2d_min_dist,
        metric=umap2d_metric,
        random_state=random_state,
    )
    coords_2d = reducer_2d.fit_transform(matrix)

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
    )
