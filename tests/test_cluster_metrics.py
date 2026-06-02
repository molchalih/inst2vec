"""Per-cluster quality metrics computed during the assign stage.

Covers ``modules.clustering.metrics.compute_cluster_metrics``: per-cluster
n_users, mean membership centrality, high-confidence fraction
(centrality > threshold), HDBSCAN cluster persistence, and per-cluster DBCV.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from modules.clustering.core import ClusterResult
from modules.clustering.metrics import (
    HIGH_CONFIDENCE_THRESHOLD,
    ClusterMetricRow,
    compute_cluster_metrics,
)

_RNG = np.random.default_rng(0)


def _three_blob_result(*, with_matrix_nd: bool = True) -> ClusterResult:
    """Three well-separated blobs in 4-D with hand-set centralities/persistence.

    Cluster 0: 3 users (centralities 0.9, 0.95, 0.4 → high-conf frac 2/3)
    Cluster 1: 2 users (centralities 1.0, 1.0 → high-conf frac 1.0)
    Cluster 2: 4 users (all 0.5 → high-conf frac 0.0)
    Plus two noise points (label -1) that must be ignored.
    """
    labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, -1, -1])
    centralities = np.array(
        [0.9, 0.95, 0.4, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0], dtype=np.float32
    )
    # cluster_persistence_ is indexed by cluster label 0..k-1.
    cluster_persistence = np.array([0.71, 0.62, 0.83], dtype=np.float32)
    centers = {0: 0.0, 1: 10.0, 2: 20.0, -1: 5.0}
    matrix_nd = np.array([centers[lbl] for lbl in labels], dtype=np.float64)[
        :, None
    ] + _RNG.normal(0, 0.05, size=(len(labels), 4))
    return ClusterResult(
        labels=labels,
        coords_2d=np.zeros((len(labels), 2), np.float32),
        n_clusters=3,
        noise_ratio=2 / len(labels),
        centralities=centralities,
        cluster_sizes=[4, 3, 2],
        cluster_persistence=cluster_persistence,
        matrix_nd=matrix_nd.astype(np.float32) if with_matrix_nd else None,
    )


def test_one_row_per_non_noise_cluster_sorted():
    rows = compute_cluster_metrics(_three_blob_result())
    assert [r.cluster_id for r in rows] == [0, 1, 2]
    assert all(isinstance(r, ClusterMetricRow) for r in rows)


def test_n_users_excludes_noise():
    rows = {r.cluster_id: r for r in compute_cluster_metrics(_three_blob_result())}
    assert rows[0].n_users == 3
    assert rows[1].n_users == 2
    assert rows[2].n_users == 4


def test_mean_centrality_per_cluster():
    rows = {r.cluster_id: r for r in compute_cluster_metrics(_three_blob_result())}
    assert rows[0].mean_centrality == np.float32([0.9, 0.95, 0.4]).mean()
    assert rows[1].mean_centrality == 1.0
    assert rows[2].mean_centrality == 0.5


def test_high_confidence_fraction_uses_threshold():
    rows = {r.cluster_id: r for r in compute_cluster_metrics(_three_blob_result())}
    # default threshold 0.8: cluster 0 → 2/3, cluster 1 → 1.0, cluster 2 → 0.0
    assert rows[0].high_conf_fraction == 2 / 3
    assert rows[1].high_conf_fraction == 1.0
    assert rows[2].high_conf_fraction == 0.0


def test_high_confidence_threshold_is_configurable():
    rows = {
        r.cluster_id: r
        for r in compute_cluster_metrics(_three_blob_result(), high_conf_threshold=0.3)
    }
    # threshold 0.3: cluster 0 → all three > 0.3 → 1.0; cluster 2 (all 0.5) → 1.0
    assert rows[0].high_conf_fraction == 1.0
    assert rows[2].high_conf_fraction == 1.0


def test_persistence_mapped_by_cluster_label():
    rows = {r.cluster_id: r for r in compute_cluster_metrics(_three_blob_result())}
    assert rows[0].persistence == np.float32(0.71)
    assert rows[1].persistence == np.float32(0.62)
    assert rows[2].persistence == np.float32(0.83)


def test_per_cluster_dbcv_present_for_separable_blobs():
    rows = compute_cluster_metrics(_three_blob_result())
    # well-separated → each cluster gets a finite DBCV score
    assert all(r.dbcv is not None for r in rows)
    assert all(np.isfinite(r.dbcv) for r in rows)


def test_dbcv_none_without_matrix_nd():
    rows = compute_cluster_metrics(_three_blob_result(with_matrix_nd=False))
    assert all(r.dbcv is None for r in rows)
    # other metrics still computed
    assert all(r.n_users > 0 for r in rows)


def test_default_threshold_constant():
    assert HIGH_CONFIDENCE_THRESHOLD == 0.8
