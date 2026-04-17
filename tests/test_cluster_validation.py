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
