import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from modules.clustering import (
    DEFAULT_HDBSCAN_METRIC,
    ClusterResult,
    compute_clusters,
    resolve_hdbscan_metric,
    resolve_umap2d_params,
)
from modules.database import UserCluster


def test_resolve_hdbscan_metric_always_returns_euclidean():
    assert DEFAULT_HDBSCAN_METRIC == "euclidean"
    assert resolve_hdbscan_metric(None) == "euclidean"
    assert resolve_hdbscan_metric("euclidean") == "euclidean"
    assert resolve_hdbscan_metric("cosine") == "euclidean"
    assert resolve_hdbscan_metric("correlation") == "euclidean"


def test_resolve_umap2d_params_mirrors_pass1():
    assert resolve_umap2d_params(10, 0.05, "cosine", None, None, None) == (
        10,
        0.05,
        "cosine",
    )
    assert resolve_umap2d_params(10, 0.05, "cosine", 20, None, None) == (
        20,
        0.05,
        "cosine",
    )
    assert resolve_umap2d_params(10, 0.05, "cosine", None, 0.2, None) == (
        10,
        0.2,
        "cosine",
    )
    assert resolve_umap2d_params(10, 0.05, "cosine", None, None, "euclidean") == (
        10,
        0.05,
        "euclidean",
    )


def test_user_cluster_columns():
    cols = {c.key for c in UserCluster.__table__.columns}
    assert "user_id" in cols
    assert "embedding_case" in cols
    assert "cluster_id" in cols
    assert "umap_x" in cols
    assert "umap_y" in cols
    assert "created_at" in cols
    assert "updated_at" in cols
    assert UserCluster.__tablename__ == "user_clusters"


# Tests for compute_clusters()

_RNG = np.random.default_rng(0)


def _two_cluster_matrix(n_per_cluster=40, dim=30):
    """Two tight Gaussian blobs far apart — should produce exactly 2 clusters."""
    a = _RNG.normal(loc=8.0, scale=0.2, size=(n_per_cluster, dim)).astype(np.float32)
    b = _RNG.normal(loc=-8.0, scale=0.2, size=(n_per_cluster, dim)).astype(np.float32)
    return np.vstack([a, b])


def test_compute_clusters_returns_cluster_result():
    matrix = _two_cluster_matrix()
    result = compute_clusters(
        matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0
    )
    assert isinstance(result, ClusterResult)


def test_compute_clusters_labels_shape():
    matrix = _two_cluster_matrix()
    result = compute_clusters(
        matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0
    )
    assert result.labels.shape == (80,)


def test_compute_clusters_coords_2d_shape():
    matrix = _two_cluster_matrix()
    result = compute_clusters(
        matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0
    )
    assert result.coords_2d.shape == (80, 2)


def test_compute_clusters_finds_two_clusters():
    matrix = _two_cluster_matrix()
    result = compute_clusters(
        matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0
    )
    assert result.n_clusters == 2


def test_compute_clusters_n_clusters_matches_unique_labels():
    matrix = _two_cluster_matrix()
    result = compute_clusters(
        matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0
    )
    unique_non_noise = set(result.labels[result.labels >= 0])
    assert result.n_clusters == len(unique_non_noise)


def test_compute_clusters_noise_ratio_valid_range():
    matrix = _two_cluster_matrix()
    result = compute_clusters(
        matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0
    )
    assert 0.0 <= result.noise_ratio <= 1.0


def test_compute_clusters_cluster_sizes_sorted_descending():
    matrix = _two_cluster_matrix()
    result = compute_clusters(
        matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0
    )
    assert result.cluster_sizes == sorted(result.cluster_sizes, reverse=True)


def test_compute_clusters_large_min_cluster_size_yields_noise():
    # 20 points total, min_cluster_size=30 → no cluster can form
    matrix = _two_cluster_matrix(n_per_cluster=10)
    result = compute_clusters(
        matrix, umap_n_components=5, hdbscan_min_cluster_size=30, random_state=0
    )
    assert result.n_clusters == 0
    assert result.noise_ratio == pytest.approx(1.0)
    assert result.cluster_sizes == []


def test_compute_clusters_matrix_nd_none_by_default():
    matrix = _two_cluster_matrix()
    result = compute_clusters(
        matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0
    )
    assert result.matrix_nd is None


def test_compute_clusters_matrix_nd_returned_when_requested():
    matrix = _two_cluster_matrix()
    result = compute_clusters(
        matrix,
        umap_n_components=5,
        hdbscan_min_cluster_size=10,
        random_state=0,
        return_nd_matrix=True,
    )
    assert result.matrix_nd is not None
    assert result.matrix_nd.shape == (80, 5)  # n_rows × umap_n_components


@patch("modules.clustering.hdbscan.HDBSCAN")
@patch("modules.clustering.UMAP")
def test_compute_clusters_passes_umap_n_jobs_to_both_umap_instances(
    mock_umap, mock_hdbscan
):
    """Both UMAP passes receive the same n_jobs as umap_n_jobs."""
    inst_nd = MagicMock()
    inst_nd.fit_transform.side_effect = lambda x: np.zeros(
        (x.shape[0], 5), dtype=np.float32
    )
    inst_2d = MagicMock()
    inst_2d.fit_transform.side_effect = lambda x: np.zeros(
        (x.shape[0], 2), dtype=np.float32
    )
    mock_umap.side_effect = [inst_nd, inst_2d]

    mock_clusterer = MagicMock()
    mock_clusterer.fit_predict.return_value = np.array([0] * 40 + [1] * 40)
    mock_hdbscan.return_value = mock_clusterer

    matrix = _two_cluster_matrix()
    compute_clusters(
        matrix,
        umap_n_components=5,
        hdbscan_min_cluster_size=10,
        random_state=0,
        umap_n_jobs=4,
    )

    assert mock_umap.call_count == 2
    assert mock_umap.call_args_list[0].kwargs["n_jobs"] == 4
    assert mock_umap.call_args_list[1].kwargs["n_jobs"] == 4
    assert mock_hdbscan.call_args.kwargs["metric"] == DEFAULT_HDBSCAN_METRIC
