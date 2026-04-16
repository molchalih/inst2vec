"""Pure clustering logic: two-pass UMAP + HDBSCAN → ClusterResult."""
from dataclasses import dataclass, field

import numpy as np
import hdbscan
from umap import UMAP

from modules.database import Base, engine, get_session, UserEmbedding, UserCluster


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


def load_user_matrix(embedding_case: str) -> tuple[np.ndarray, list[int]]:
    """Load user embeddings from DB. Returns (matrix, user_pks) in matching order."""
    session = get_session()
    try:
        rows = (
            session.query(UserEmbedding.user_pk, UserEmbedding.embedding)
            .filter(UserEmbedding.embedding_case == embedding_case)
            .all()
        )
        if not rows:
            return np.empty((0, 0), dtype=np.float32), []
        user_pks = [r.user_pk for r in rows]
        arrays = [np.frombuffer(r.embedding, dtype=np.float32).copy() for r in rows]
        return np.stack(arrays), user_pks
    finally:
        session.close()


def cluster_users(embedding_case: str, **params) -> None:
    Base.metadata.create_all(engine)
    matrix, user_pks = load_user_matrix(embedding_case)

    if matrix.shape[0] == 0:
        print(f"[cluster:{embedding_case}] nothing to do")
        return

    min_required = params.get("umap_n_components", 15) + 1
    if matrix.shape[0] < min_required:
        print(f"[cluster:{embedding_case}] only {matrix.shape[0]} users — need at least {min_required} to cluster, skipping")
        return

    print(f"[cluster:{embedding_case}] {matrix.shape[0]} users — running UMAP + HDBSCAN")
    result = compute_clusters(matrix, **params)

    session = get_session()
    try:
        for i, user_pk in enumerate(user_pks):
            row = UserCluster(
                user_pk=user_pk,
                embedding_case=embedding_case,
                cluster_id=int(result.labels[i]),
                umap_x=float(result.coords_2d[i, 0]),
                umap_y=float(result.coords_2d[i, 1]),
            )
            session.merge(row)
        session.commit()
    finally:
        session.close()

    sizes_str = f"min={min(result.cluster_sizes)} median={int(np.median(result.cluster_sizes))} max={max(result.cluster_sizes)}" if result.cluster_sizes else "n/a"
    print(
        f"[cluster:{embedding_case}] {result.n_clusters} clusters, "
        f"{result.noise_ratio:.1%} noise, sizes: {sizes_str}"
    )
