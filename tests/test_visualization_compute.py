"""Tests for modules.visualization.compute."""

from __future__ import annotations

import math

import numpy as np

from core.database import UserCluster, VisualizationCluster, VisualizationUser
from modules.visualization.compute import build_case_payload, fit_cluster_ellipse


def test_fit_cluster_ellipse_circular_cloud_has_equal_axes():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 2))
    xs, ys = pts[:, 0], pts[:, 1]
    e = fit_cluster_ellipse(xs, ys)
    # 2σ over unit-std cloud: rx,ry ≈ 2; angle is unstable for circles.
    assert abs(e.rx - e.ry) < 0.3
    assert 1.7 < e.rx < 2.3


def test_fit_cluster_ellipse_45deg_diagonal_recovers_angle():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(500) * 3.0
    noise = rng.standard_normal(500) * 0.3
    # Points stretched along the y=x line.
    xs = base + noise
    ys = base - noise
    e = fit_cluster_ellipse(xs, ys)
    # Angle wraps; principal axis is along ±45°.
    angle_norm = math.atan2(math.sin(2 * e.angle), math.cos(2 * e.angle)) / 2
    assert abs(abs(angle_norm) - math.pi / 4) < 0.1
    assert e.rx > 2 * e.ry  # clear major-axis dominance


def test_fit_cluster_ellipse_single_point_returns_degenerate_circle():
    xs = np.array([3.5])
    ys = np.array([-1.2])
    e = fit_cluster_ellipse(xs, ys)
    assert e.cx == 3.5
    assert e.cy == -1.2
    assert e.rx > 0 and e.ry > 0
    assert e.rx < 1e-3 and e.ry < 1e-3
    assert e.angle == 0.0


def _uc(uid: int, x: float, y: float, cid: int) -> UserCluster:
    return UserCluster(
        user_id=uid, embedding_case="video", cluster_id=cid, umap_x=x, umap_y=y
    )


def test_build_case_payload_groups_clusters_and_skips_noise():
    rows = [
        # Cluster 0
        _uc(0, 0.0, 0.0, 0),
        _uc(1, 1.0, 0.0, 0),
        _uc(2, 0.5, 1.0, 0),
        # Cluster 1
        _uc(3, 10.0, 10.0, 1),
        _uc(4, 11.0, 10.0, 1),
        _uc(5, 10.5, 11.0, 1),
        # Noise
        _uc(6, -5.0, -5.0, -1),
    ]
    payload = build_case_payload("video", "Visual", rows)
    assert payload.case == "video"
    assert payload.label == "Visual"
    # Every user (including noise) appears in users.
    assert len(payload.users) == 7
    assert all(isinstance(u, VisualizationUser) for u in payload.users)
    # Only the two real clusters get ellipses; noise is excluded.
    assert len(payload.clusters) == 2
    assert all(isinstance(c, VisualizationCluster) for c in payload.clusters)
    assert sorted(c.cluster_id for c in payload.clusters) == [0, 1]


def test_build_case_payload_labels_use_one_based_index():
    rows = [
        _uc(0, 0.0, 0.0, 0),
        _uc(1, 1.0, 1.0, 0),
        _uc(2, 5.0, 5.0, 2),
        _uc(3, 6.0, 6.0, 2),
    ]
    payload = build_case_payload("video", "Visual", rows)
    by_id = {c.cluster_id: c for c in payload.clusters}
    assert by_id[0].label == "Cluster 1"
    assert by_id[2].label == "Cluster 3"
    # Sizes track member count.
    assert by_id[0].size == 2
    assert by_id[2].size == 2


def test_build_case_payload_empty_input_returns_empty_lists():
    payload = build_case_payload("video", "Visual", [])
    assert payload.users == []
    assert payload.clusters == []
