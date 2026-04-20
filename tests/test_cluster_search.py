import sys
import os
from unittest.mock import patch

import pytest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.database import Base, ClusterRun


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
        "disqualified", "dbcv", "silhouette", "composite_score",
        "bootstrap_stability", "bootstrap_n_runs", "param_plateau_score",
        "in_current_grid", "dataset_fingerprint",
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


def test_compute_dataset_fingerprint_deterministic():
    from modules.cluster_search import _compute_dataset_fingerprint
    fp1 = _compute_dataset_fingerprint([3, 1, 2])
    fp2 = _compute_dataset_fingerprint([1, 2, 3])
    assert fp1 == fp2  # order-independent
    assert len(fp1) == 64  # SHA-256 hex


def test_compute_dataset_fingerprint_differs_on_different_users():
    from modules.cluster_search import _compute_dataset_fingerprint
    assert _compute_dataset_fingerprint([1, 2]) != _compute_dataset_fingerprint([1, 3])


def test_combo_key_excludes_embedding_case():
    from modules.cluster_search import _combo_key
    combo = dict(embedding_case="video", umap_n_components=15, umap_n_neighbors=15,
                 umap_min_dist=0.0, umap_metric="cosine", umap2d_n_neighbors=15,
                 umap2d_min_dist=0.1, umap2d_metric="cosine",
                 hdbscan_min_cluster_size=15, hdbscan_min_samples=None,
                 hdbscan_cluster_selection_method="eom", hdbscan_metric="euclidean",
                 random_state=42)
    key = _combo_key(combo)
    assert ("embedding_case", "video") not in key
    assert ("umap_n_components", 15) in key


def test_run_cluster_search_marks_stale_rows_when_grid_shrinks(mem_engine, monkeypatch):
    """A row inserted under old grid becomes in_current_grid=0 when grid changes."""
    env_old = {
        "CLUSTERING_UMAP_N_COMPONENTS": "5 10",  # two combos per case
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
    }
    env_new = {**env_old, "CLUSTERING_UMAP_N_COMPONENTS": "5"}  # only one combo per case

    monkeypatch.setattr("modules.cluster_search.load_user_matrix",
                        lambda case: (_fake_matrix(), list(range(80))))
    monkeypatch.setattr("modules.cluster_search.compute_clusters",
                        lambda matrix, **kw: _fake_result())

    from modules.cluster_search import run_cluster_search

    # First run with wide grid
    with patch.dict(os.environ, env_old, clear=False):
        run_cluster_search()

    with Session(mem_engine) as s:
        assert s.query(ClusterRun).count() == 6  # 3 cases × 2 nc combos

    # Second run with narrow grid — nc=10 rows become stale
    with patch.dict(os.environ, env_new, clear=False):
        run_cluster_search()

    with Session(mem_engine) as s:
        stale = s.query(ClusterRun).filter(ClusterRun.in_current_grid == 0).count()
        current = s.query(ClusterRun).filter(ClusterRun.in_current_grid == 1).count()
        assert stale == 3   # nc=10 rows for each of 3 cases
        assert current == 3  # nc=5 rows for each of 3 cases


def test_run_cluster_search_invalidates_rows_on_dataset_change(mem_engine, monkeypatch):
    """Rows computed on a different dataset fingerprint get in_current_grid=0."""
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
    }

    call_count = {"n": 0}

    def fake_matrix(case):
        call_count["n"] += 1
        if call_count["n"] <= 3:
            pks = list(range(80))
            return (_fake_matrix(), pks)
        else:
            rng = np.random.default_rng(1)
            matrix = np.vstack([
                rng.normal(8.0, 0.2, (50, 30)).astype(np.float32),
                rng.normal(-8.0, 0.2, (50, 30)).astype(np.float32),
            ])
            return (matrix, list(range(100)))

    monkeypatch.setattr("modules.cluster_search.load_user_matrix", fake_matrix)
    monkeypatch.setattr("modules.cluster_search.compute_clusters",
                        lambda matrix, **kw: _fake_result())

    from modules.cluster_search import run_cluster_search

    with patch.dict(os.environ, env, clear=False):
        run_cluster_search()  # uses pks 0..79

    with Session(mem_engine) as s:
        fp_before = s.query(ClusterRun).first().dataset_fingerprint
        assert fp_before is not None

    with patch.dict(os.environ, env, clear=False):
        run_cluster_search()  # uses pks 0..99 — fingerprint changes

    with Session(mem_engine) as s:
        stale = s.query(ClusterRun).filter(ClusterRun.in_current_grid == 0).all()
        assert len(stale) == 3  # old rows invalidated
        current = s.query(ClusterRun).filter(ClusterRun.in_current_grid == 1).all()
        assert len(current) == 3  # new rows inserted


def test_run_cluster_search_new_rows_have_fingerprint_and_in_current_grid(mem_engine, monkeypatch):
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
    }
    monkeypatch.setattr("modules.cluster_search.load_user_matrix",
                        lambda case: (_fake_matrix(), list(range(80))))
    monkeypatch.setattr("modules.cluster_search.compute_clusters",
                        lambda matrix, **kw: _fake_result())

    from modules.cluster_search import run_cluster_search
    with patch.dict(os.environ, env, clear=False):
        run_cluster_search()

    with Session(mem_engine) as s:
        rows = s.query(ClusterRun).all()
        for row in rows:
            assert row.in_current_grid == 1
            assert row.dataset_fingerprint is not None
            assert len(row.dataset_fingerprint) == 64


