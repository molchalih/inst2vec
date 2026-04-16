import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from modules.database import UserCluster
from modules.clustering import compute_clusters, ClusterResult


def test_user_cluster_columns():
    cols = {c.key for c in UserCluster.__table__.columns}
    assert "user_pk" in cols
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
    result = compute_clusters(matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0)
    assert isinstance(result, ClusterResult)


def test_compute_clusters_labels_shape():
    matrix = _two_cluster_matrix()
    result = compute_clusters(matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0)
    assert result.labels.shape == (80,)


def test_compute_clusters_coords_2d_shape():
    matrix = _two_cluster_matrix()
    result = compute_clusters(matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0)
    assert result.coords_2d.shape == (80, 2)


def test_compute_clusters_finds_two_clusters():
    matrix = _two_cluster_matrix()
    result = compute_clusters(matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0)
    assert result.n_clusters == 2


def test_compute_clusters_n_clusters_matches_unique_labels():
    matrix = _two_cluster_matrix()
    result = compute_clusters(matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0)
    unique_non_noise = set(result.labels[result.labels >= 0])
    assert result.n_clusters == len(unique_non_noise)


def test_compute_clusters_noise_ratio_valid_range():
    matrix = _two_cluster_matrix()
    result = compute_clusters(matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0)
    assert 0.0 <= result.noise_ratio <= 1.0


def test_compute_clusters_cluster_sizes_sorted_descending():
    matrix = _two_cluster_matrix()
    result = compute_clusters(matrix, umap_n_components=5, hdbscan_min_cluster_size=10, random_state=0)
    assert result.cluster_sizes == sorted(result.cluster_sizes, reverse=True)


def test_compute_clusters_large_min_cluster_size_yields_noise():
    # 20 points total, min_cluster_size=30 → no cluster can form
    matrix = _two_cluster_matrix(n_per_cluster=10)
    result = compute_clusters(matrix, umap_n_components=5, hdbscan_min_cluster_size=30, random_state=0)
    assert result.n_clusters == 0
    assert result.noise_ratio == pytest.approx(1.0)
    assert result.cluster_sizes == []
