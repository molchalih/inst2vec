from __future__ import annotations

import numpy as np
import pytest

from scripts.cluster_lab.metrics import (
    all_metrics,
    cluster_summary,
    safe_calinski_harabasz,
    safe_davies_bouldin,
    safe_silhouette,
)


def _three_blob_dataset(rng_seed: int = 0):
    rng = np.random.default_rng(rng_seed)
    centers = np.array([[0, 0], [10, 10], [-10, 10]], dtype=float)
    parts = [rng.normal(c, 0.5, size=(50, 2)) for c in centers]
    matrix = np.vstack(parts).astype(np.float32)
    labels = np.concatenate([np.full(50, i) for i in range(3)])
    return matrix, labels


def test_cluster_summary_basic() -> None:
    labels = np.array([0, 0, 0, 1, 1, -1])
    out = cluster_summary(labels)
    assert out["n_clusters"] == 2
    assert pytest.approx(out["noise_ratio"], abs=1e-6) == 1 / 6
    assert out["min_size"] == 2
    assert out["max_size"] == 3
    assert out["n_singletons"] == 0


def test_cluster_summary_all_noise() -> None:
    labels = np.array([-1, -1, -1])
    out = cluster_summary(labels)
    assert out["n_clusters"] == 0
    assert out["noise_ratio"] == 1.0


def test_metrics_on_three_blobs() -> None:
    matrix, labels = _three_blob_dataset()
    metrics = all_metrics(matrix, labels)
    assert metrics["n_clusters"] == 3
    assert metrics["silhouette"] is not None and metrics["silhouette"] > 0.5
    assert metrics["calinski_harabasz"] is not None and metrics["calinski_harabasz"] > 0
    assert metrics["davies_bouldin"] is not None and metrics["davies_bouldin"] > 0


def test_single_cluster_returns_none() -> None:
    matrix = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    labels = np.zeros(20, dtype=int)
    assert safe_silhouette(matrix, labels) is None
    assert safe_calinski_harabasz(matrix, labels) is None
    assert safe_davies_bouldin(matrix, labels) is None


def test_all_noise_returns_none() -> None:
    matrix = np.random.default_rng(0).normal(size=(20, 4)).astype(np.float32)
    labels = -np.ones(20, dtype=int)
    out = all_metrics(matrix, labels)
    assert out["silhouette"] is None
    assert out["calinski_harabasz"] is None
