import sys
import os
from unittest.mock import patch

import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.database import ClusterRun


def test_cluster_run_tablename():
    assert ClusterRun.__tablename__ == "cluster_runs"


def test_cluster_run_columns():
    cols = {c.key for c in ClusterRun.__table__.columns}
    assert cols == {
        "id", "embedding_case",
        "umap_n_components", "umap_n_neighbors", "umap_min_dist", "umap_metric",
        "umap2d_n_neighbors", "umap2d_min_dist", "umap2d_metric",
        "hdbscan_min_cluster_size", "hdbscan_min_samples",
        "hdbscan_cluster_selection_method", "hdbscan_metric",
        "random_state",
        "n_clusters", "noise_ratio", "min_size", "median_size", "max_size",
        "created_at",
    }


def test_cluster_run_unique_constraint():
    constraint_names = {c.name for c in ClusterRun.__table__.constraints}
    assert "uq_cluster_runs_params" in constraint_names


def test_cluster_run_hdbscan_min_samples_nullable():
    col = ClusterRun.__table__.c["hdbscan_min_samples"]
    assert col.nullable is True


def test_parse_ints():
    from modules.cluster_search import _parse_ints
    assert _parse_ints("10 15 30") == [10, 15, 30]
    assert _parse_ints("42") == [42]


def test_parse_floats():
    from modules.cluster_search import _parse_floats
    assert _parse_floats("0.0 0.05") == [0.0, 0.05]
    assert _parse_floats("0.1") == [0.1]


def test_parse_strs():
    from modules.cluster_search import _parse_strs
    assert _parse_strs("cosine euclidean") == ["cosine", "euclidean"]
    assert _parse_strs("eom") == ["eom"]


def test_load_grid_combo_count():
    from modules.cluster_search import _load_grid
    env = {
        "CLUSTERING_UMAP_N_COMPONENTS": "10 15",
        "CLUSTERING_UMAP_N_NEIGHBORS": "10",
        "CLUSTERING_UMAP_MIN_DIST": "0.0",
        "CLUSTERING_UMAP_METRICS": "cosine",
        "CLUSTERING_UMAP2D_N_NEIGHBORS": "15",
        "CLUSTERING_UMAP2D_MIN_DIST": "0.1",
        "CLUSTERING_UMAP2D_METRICS": "cosine euclidean",
        "CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE": "10",
        "CLUSTERING_HDBSCAN_SELECTION": "eom",
        "CLUSTERING_HDBSCAN_METRICS": "euclidean",
        "CLUSTERING_RANDOM_STATE": "42",
    }
    with patch.dict(os.environ, env, clear=False):
        combos = _load_grid()
    # 3 cases × 2 umap_n_components × 1 nn × 1 md × 1 umap_metric
    # × 2 umap2d_metrics × 1 mcs × 1 selection × 1 hdbscan_metric = 12
    assert len(combos) == 12


def test_load_grid_combo_keys():
    from modules.cluster_search import _load_grid
    env = {
        "CLUSTERING_UMAP_N_COMPONENTS": "10",
        "CLUSTERING_UMAP_N_NEIGHBORS": "15",
        "CLUSTERING_UMAP_MIN_DIST": "0.0",
        "CLUSTERING_UMAP_METRICS": "cosine",
        "CLUSTERING_UMAP2D_N_NEIGHBORS": "15",
        "CLUSTERING_UMAP2D_MIN_DIST": "0.1",
        "CLUSTERING_UMAP2D_METRICS": "cosine",
        "CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE": "10",
        "CLUSTERING_HDBSCAN_SELECTION": "eom",
        "CLUSTERING_HDBSCAN_METRICS": "euclidean",
        "CLUSTERING_RANDOM_STATE": "42",
    }
    with patch.dict(os.environ, env, clear=False):
        combos = _load_grid()
    expected_keys = {
        "embedding_case", "umap_n_components", "umap_n_neighbors", "umap_min_dist",
        "umap_metric", "umap2d_n_neighbors", "umap2d_min_dist", "umap2d_metric",
        "hdbscan_min_cluster_size", "hdbscan_min_samples",
        "hdbscan_cluster_selection_method", "hdbscan_metric", "random_state",
    }
    assert set(combos[0].keys()) == expected_keys


def test_load_grid_umap2d_fixed_values():
    from modules.cluster_search import _load_grid
    env = {
        "CLUSTERING_UMAP_N_COMPONENTS": "10",
        "CLUSTERING_UMAP_N_NEIGHBORS": "15",
        "CLUSTERING_UMAP_MIN_DIST": "0.0",
        "CLUSTERING_UMAP_METRICS": "cosine",
        "CLUSTERING_UMAP2D_N_NEIGHBORS": "20",
        "CLUSTERING_UMAP2D_MIN_DIST": "0.2",
        "CLUSTERING_UMAP2D_METRICS": "cosine",
        "CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE": "10",
        "CLUSTERING_HDBSCAN_SELECTION": "eom",
        "CLUSTERING_HDBSCAN_METRICS": "euclidean",
        "CLUSTERING_RANDOM_STATE": "42",
    }
    with patch.dict(os.environ, env, clear=False):
        combos = _load_grid()
    for combo in combos:
        assert combo["umap2d_n_neighbors"] == 20
        assert combo["umap2d_min_dist"] == 0.2


from modules.database import Base, ClusterRun
from modules.clustering import ClusterResult


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _fake_matrix():
    rng = np.random.default_rng(0)
    return np.vstack([
        rng.normal(8.0, 0.2, (40, 30)).astype(np.float32),
        rng.normal(-8.0, 0.2, (40, 30)).astype(np.float32),
    ])


def _fake_result():
    labels = np.array([0] * 40 + [1] * 40)
    coords = np.zeros((80, 2), dtype=np.float32)
    return ClusterResult(labels=labels, coords_2d=coords, n_clusters=2,
                         noise_ratio=0.0, cluster_sizes=[40, 40])


@pytest.fixture()
def mem_engine(monkeypatch):
    eng = _make_engine()

    def _get_session():
        return Session(eng)

    monkeypatch.setattr("modules.cluster_search.engine", eng)
    monkeypatch.setattr("modules.cluster_search.get_session", _get_session)
    return eng


def test_run_cluster_search_inserts_rows(mem_engine, monkeypatch):
    env = {
        "CLUSTERING_UMAP_N_COMPONENTS": "5",
        "CLUSTERING_UMAP_N_NEIGHBORS": "5",
        "CLUSTERING_UMAP_MIN_DIST": "0.0",
        "CLUSTERING_UMAP_METRICS": "cosine",
        "CLUSTERING_UMAP2D_N_NEIGHBORS": "5",
        "CLUSTERING_UMAP2D_MIN_DIST": "0.1",
        "CLUSTERING_UMAP2D_METRICS": "cosine",
        "CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE": "10",
        "CLUSTERING_HDBSCAN_SELECTION": "eom",
        "CLUSTERING_HDBSCAN_METRICS": "euclidean",
        "CLUSTERING_RANDOM_STATE": "42",
        "CLUSTERING_JOBS": "1",
    }
    monkeypatch.setattr("modules.cluster_search.load_user_matrix",
                        lambda case: (_fake_matrix(), list(range(80))))
    monkeypatch.setattr("modules.cluster_search.compute_clusters",
                        lambda matrix, **kw: _fake_result())

    from modules.cluster_search import run_cluster_search
    with patch.dict(os.environ, env, clear=False):
        run_cluster_search()

    with Session(mem_engine) as s:
        count = s.query(ClusterRun).count()
    assert count == 3  # one row per embedding case (video, sandwich, audio)


def test_run_cluster_search_idempotent(mem_engine, monkeypatch):
    env = {
        "CLUSTERING_UMAP_N_COMPONENTS": "5",
        "CLUSTERING_UMAP_N_NEIGHBORS": "5",
        "CLUSTERING_UMAP_MIN_DIST": "0.0",
        "CLUSTERING_UMAP_METRICS": "cosine",
        "CLUSTERING_UMAP2D_N_NEIGHBORS": "5",
        "CLUSTERING_UMAP2D_MIN_DIST": "0.1",
        "CLUSTERING_UMAP2D_METRICS": "cosine",
        "CLUSTERING_HDBSCAN_MIN_CLUSTER_SIZE": "10",
        "CLUSTERING_HDBSCAN_SELECTION": "eom",
        "CLUSTERING_HDBSCAN_METRICS": "euclidean",
        "CLUSTERING_RANDOM_STATE": "42",
        "CLUSTERING_JOBS": "1",
    }
    monkeypatch.setattr("modules.cluster_search.load_user_matrix",
                        lambda case: (_fake_matrix(), list(range(80))))
    monkeypatch.setattr("modules.cluster_search.compute_clusters",
                        lambda matrix, **kw: _fake_result())

    from modules.cluster_search import run_cluster_search
    with patch.dict(os.environ, env, clear=False):
        run_cluster_search()
        run_cluster_search()  # second call — must not duplicate

    with Session(mem_engine) as s:
        count = s.query(ClusterRun).count()
    assert count == 3  # still 3, not 6


def test_validate_clustering_raises():
    from modules.cluster_search import validate_clustering
    with pytest.raises(NotImplementedError):
        validate_clustering()
