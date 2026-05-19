from __future__ import annotations

import numpy as np
import pytest

from scripts.cluster_lab.runners import (
    get_runner,
    run,
    run_agglomerative,
    run_gmm,
    run_kmeans,
    run_pca_hdbscan,
    run_spectral,
    run_umap_hdbscan,
)


def _blobs(seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    parts = [rng.normal(c, 0.5, size=(80, 3)) for c in centers]
    return np.vstack(parts).astype(np.float32)


def test_kmeans_finds_three_blobs() -> None:
    mat = _blobs()
    out = run_kmeans(mat, {"k": 3, "random_state": 0, "normalized": 0})
    assert out["error"] is None
    assert out["n_clusters"] == 3
    assert out["silhouette"] > 0.5


def test_kmeans_bad_k_captures_error() -> None:
    mat = _blobs()
    out = run_kmeans(mat, {"k": -1, "random_state": 0, "normalized": 0})
    assert out["error"] is not None
    assert out["n_clusters"] is None


def test_gmm_three_blobs() -> None:
    mat = _blobs()
    out = run_gmm(
        mat, {"k": 3, "covariance_type": "full", "random_state": 0, "normalized": 0}
    )
    assert out["error"] is None
    assert out["n_clusters"] == 3


def test_agglomerative_ward() -> None:
    mat = _blobs()
    out = run_agglomerative(
        mat,
        {
            "k": 3,
            "linkage": "ward",
            "distance_metric": "euclidean",
            "normalized": 0,
        },
    )
    assert out["error"] is None
    assert out["n_clusters"] == 3


def test_spectral_three_blobs() -> None:
    mat = _blobs()
    out = run_spectral(
        mat,
        {
            "k": 3,
            "affinity": "nearest_neighbors",
            "n_neighbors": 10,
            "random_state": 0,
            "normalized": 0,
        },
    )
    assert out["error"] is None
    assert out["n_clusters"] == 3


def test_pca_hdbscan_three_blobs() -> None:
    mat = _blobs()
    out = run_pca_hdbscan(
        mat,
        {
            "pca_n_components": 2,
            "hdbscan_min_cluster_size": 10,
            "hdbscan_min_samples": None,
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_metric": "euclidean",
            "random_state": 0,
            "normalized": 0,
        },
    )
    assert out["error"] is None
    assert out["n_clusters"] >= 2


def test_umap_hdbscan_smoke() -> None:
    mat = _blobs()
    out = run_umap_hdbscan(
        mat,
        {
            "umap_n_components": 3,
            "umap_n_neighbors": 10,
            "umap_min_dist": 0.0,
            "umap_metric": "euclidean",
            "hdbscan_min_cluster_size": 10,
            "hdbscan_min_samples": None,
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_metric": "euclidean",
            "random_state": 42,
            "normalized": 0,
        },
    )
    assert out["error"] is None
    assert out["n_clusters"] >= 2


def test_umap_hdbscan_error_capture() -> None:
    mat = _blobs()
    out = run_umap_hdbscan(
        mat,
        {
            "umap_n_components": 3,
            "umap_n_neighbors": 10,
            "umap_min_dist": 0.0,
            "umap_metric": "definitely-not-a-metric",
            "hdbscan_min_cluster_size": 10,
            "hdbscan_min_samples": None,
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_metric": "euclidean",
            "random_state": 42,
            "normalized": 0,
        },
    )
    assert out["error"] is not None


def test_dispatch_table() -> None:
    assert get_runner("kmeans", "none") is run_kmeans
    assert get_runner("hdbscan", "umap") is run_umap_hdbscan
    with pytest.raises(KeyError):
        get_runner("xyz", "qqq")


def test_run_proxies_dispatch() -> None:
    mat = _blobs()
    out = run("kmeans", "none", mat, {"k": 3, "random_state": 0, "normalized": 0})
    assert out["algorithm"] == "kmeans"
    assert out["reducer"] == "none"
