import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.database import Base, ClusterRun


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _base_run_kwargs(**overrides):
    defaults = dict(
        embedding_case="video",
        umap_n_components=15, umap_n_neighbors=15, umap_min_dist=0.0, umap_metric="cosine",
        umap2d_n_neighbors=15, umap2d_min_dist=0.1, umap2d_metric="cosine",
        hdbscan_min_cluster_size=15, hdbscan_min_samples=None,
        hdbscan_cluster_selection_method="eom", hdbscan_metric="euclidean",
        random_state=42,
        n_clusters=5, noise_ratio=0.1, min_size=10, median_size=20, max_size=40,
    )
    defaults.update(overrides)
    return defaults


def _insert_run(session, **kwargs):
    row = ClusterRun(**_base_run_kwargs(**kwargs))
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


# --- Phase 1: filter ---

def test_filter_passes_run_within_bounds():
    eng = _make_engine()
    env = {
        "VALIDATION_MAX_NOISE_RATIO": "0.3",
        "VALIDATION_MIN_CLUSTERS": "3",
        "VALIDATION_MAX_CLUSTERS": "20",
    }
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=5)
        row_id = row.id

    from modules.cluster_validation import _phase_filter
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_filter(s, "video")
            assert s.get(ClusterRun, row_id).disqualified == 0


def test_filter_disqualifies_high_noise():
    eng = _make_engine()
    env = {
        "VALIDATION_MAX_NOISE_RATIO": "0.3",
        "VALIDATION_MIN_CLUSTERS": "3",
        "VALIDATION_MAX_CLUSTERS": "20",
    }
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.5, n_clusters=5)
        row_id = row.id

    from modules.cluster_validation import _phase_filter
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_filter(s, "video")
            assert s.get(ClusterRun, row_id).disqualified == 1


def test_filter_disqualifies_too_few_clusters():
    eng = _make_engine()
    env = {
        "VALIDATION_MAX_NOISE_RATIO": "0.3",
        "VALIDATION_MIN_CLUSTERS": "3",
        "VALIDATION_MAX_CLUSTERS": "20",
    }
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=1)
        row_id = row.id

    from modules.cluster_validation import _phase_filter
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_filter(s, "video")
            assert s.get(ClusterRun, row_id).disqualified == 1


def test_filter_disqualifies_too_many_clusters():
    eng = _make_engine()
    env = {
        "VALIDATION_MAX_NOISE_RATIO": "0.3",
        "VALIDATION_MIN_CLUSTERS": "3",
        "VALIDATION_MAX_CLUSTERS": "20",
    }
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=25)
        row_id = row.id

    from modules.cluster_validation import _phase_filter
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_filter(s, "video")
            assert s.get(ClusterRun, row_id).disqualified == 1


def test_filter_skips_already_set_rows():
    eng = _make_engine()
    env = {
        "VALIDATION_MAX_NOISE_RATIO": "0.3",
        "VALIDATION_MIN_CLUSTERS": "3",
        "VALIDATION_MAX_CLUSTERS": "20",
    }
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.5, n_clusters=5)
        row.disqualified = 0  # already set — should not be overwritten
        s.commit()
        row_id = row.id

    from modules.cluster_validation import _phase_filter
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_filter(s, "video")
            assert s.get(ClusterRun, row_id).disqualified == 0  # unchanged


# --- Phase 2: score helpers ---

def test_minmax_normalizes_range():
    from modules.cluster_validation import _minmax
    result = _minmax([0.0, 0.5, 1.0])
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_minmax_all_same_returns_zeros():
    from modules.cluster_validation import _minmax
    result = _minmax([0.4, 0.4, 0.4])
    assert result == [0.0, 0.0, 0.0]


def test_minmax_nan_treated_as_zero():
    from modules.cluster_validation import _minmax
    result = _minmax([float("nan"), 0.0, 1.0])
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(1.0)


def test_phase_score_populates_dbcv_and_silhouette():
    eng = _make_engine()
    rng = np.random.default_rng(0)
    matrix = np.vstack([
        rng.normal(8.0, 0.2, (40, 30)).astype(np.float32),
        rng.normal(-8.0, 0.2, (40, 30)).astype(np.float32),
    ])

    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.0, n_clusters=2, disqualified=0,
                          umap_n_components=5, hdbscan_min_cluster_size=10)
        row_id = row.id

    from modules.cluster_validation import _phase_score
    with Session(eng) as s:
        _phase_score(s, "video", matrix)
        updated = s.get(ClusterRun, row_id)
        assert updated.dbcv is not None
        assert updated.silhouette is not None


def test_phase_score_skips_already_scored_rows():
    eng = _make_engine()
    rng = np.random.default_rng(0)
    matrix = np.vstack([
        rng.normal(8.0, 0.2, (40, 30)).astype(np.float32),
        rng.normal(-8.0, 0.2, (40, 30)).astype(np.float32),
    ])

    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.0, n_clusters=2, disqualified=0,
                          umap_n_components=5, hdbscan_min_cluster_size=10)
        row.dbcv = 0.99
        row.silhouette = 0.88
        s.commit()
        row_id = row.id

    from modules.cluster_validation import _phase_score
    with Session(eng) as s:
        _phase_score(s, "video", matrix)
        updated = s.get(ClusterRun, row_id)
        assert updated.dbcv == pytest.approx(0.99)  # unchanged
        assert updated.silhouette == pytest.approx(0.88)  # unchanged
