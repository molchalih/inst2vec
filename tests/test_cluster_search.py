import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, __file__[: __file__.rfind("/")] + "/..")

from modules.clustering import ClusterResult
from modules.database import Base, ClusterRun


def _make_search_settings(**overrides):
    """Create a SimpleNamespace for search settings with defaults from config.toml."""
    defaults = {
        "umap_n_components": [5],
        "umap_n_neighbors": [5],
        "umap_min_dist": [0.0],
        "umap_metrics": ["cosine"],
        "umap2d_n_neighbors": 5,
        "umap2d_min_dist": 0.1,
        "umap2d_metrics": ["cosine"],
        "hdbscan_min_cluster_size": [10],
        "hdbscan_selection": ["eom"],
        "random_state": 42,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_cluster_run_tablename():
    assert ClusterRun.__tablename__ == "cluster_runs"


def test_cluster_run_columns():
    cols = {c.key for c in ClusterRun.__table__.columns}
    assert cols == {
        "id",
        "embedding_case",
        "umap_n_components",
        "umap_n_neighbors",
        "umap_min_dist",
        "umap_metric",
        "umap2d_n_neighbors",
        "umap2d_min_dist",
        "umap2d_metric",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "hdbscan_cluster_selection_method",
        "hdbscan_metric",
        "random_state",
        "n_clusters",
        "noise_ratio",
        "min_size",
        "median_size",
        "max_size",
        "created_at",
        "passes_validation",
        "dbcv",
        "silhouette",
        "param_plateau_score",
    }


def test_cluster_run_unique_constraint():
    constraint_names = {c.name for c in ClusterRun.__table__.constraints}
    assert "uq_cluster_runs_params" in constraint_names


def test_cluster_run_hdbscan_min_samples_nullable():
    col = ClusterRun.__table__.c["hdbscan_min_samples"]
    assert col.nullable is True


def test_cluster_run_passes_validation_nullable():
    col = ClusterRun.__table__.c["passes_validation"]
    assert col.nullable is True


def test_load_grid_combo_count():
    from types import SimpleNamespace

    from modules.clustering.search import _load_grid

    settings = SimpleNamespace(
        umap_n_components=[10, 15],
        umap_n_neighbors=[10],
        umap_min_dist=[0.0],
        umap_metrics=["cosine"],
        umap2d_n_neighbors=15,
        umap2d_min_dist=0.1,
        umap2d_metrics=["cosine", "euclidean"],
        hdbscan_min_cluster_size=[10],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    combos = _load_grid(settings)
    # 3 cases × 2 umap_n_components × 1 nn × 1 md × 1 umap_metric
    # × 2 umap2d_metrics × 1 mcs × 1 selection = 12 (HDBSCAN metric fixed, not swept)
    assert len(combos) == 12


def test_load_grid_ignores_hdbscan_metric_env_dimension():
    from types import SimpleNamespace

    from modules.clustering.search import _load_grid

    settings = SimpleNamespace(
        umap_n_components=[10, 15],
        umap_n_neighbors=[10],
        umap_min_dist=[0.0],
        umap_metrics=["cosine"],
        umap2d_n_neighbors=15,
        umap2d_min_dist=0.1,
        umap2d_metrics=["cosine", "euclidean"],
        hdbscan_min_cluster_size=[10],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    combos = _load_grid(settings)
    assert len(combos) == 12
    assert {combo["hdbscan_metric"] for combo in combos} == {"euclidean"}


def test_load_grid_combo_keys():
    from types import SimpleNamespace

    from modules.clustering.search import _load_grid

    settings = SimpleNamespace(
        umap_n_components=[10],
        umap_n_neighbors=[15],
        umap_min_dist=[0.0],
        umap_metrics=["cosine"],
        umap2d_n_neighbors=15,
        umap2d_min_dist=0.1,
        umap2d_metrics=["cosine"],
        hdbscan_min_cluster_size=[10],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    combos = _load_grid(settings)
    expected_keys = {
        "embedding_case",
        "umap_n_components",
        "umap_n_neighbors",
        "umap_min_dist",
        "umap_metric",
        "umap2d_n_neighbors",
        "umap2d_min_dist",
        "umap2d_metric",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "hdbscan_cluster_selection_method",
        "hdbscan_metric",
        "random_state",
    }
    assert set(combos[0].keys()) == expected_keys


def test_load_grid_umap2d_fixed_values():
    from types import SimpleNamespace

    from modules.clustering.search import _load_grid

    settings = SimpleNamespace(
        umap_n_components=[10],
        umap_n_neighbors=[15],
        umap_min_dist=[0.0],
        umap_metrics=["cosine"],
        umap2d_n_neighbors=20,
        umap2d_min_dist=0.2,
        umap2d_metrics=["cosine"],
        hdbscan_min_cluster_size=[10],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    combos = _load_grid(settings)
    for combo in combos:
        assert combo["umap2d_n_neighbors"] == 20
        assert combo["umap2d_min_dist"] == 0.2


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _fake_matrix():
    rng = np.random.default_rng(0)
    return np.vstack(
        [
            rng.normal(8.0, 0.2, (40, 30)).astype(np.float32),
            rng.normal(-8.0, 0.2, (40, 30)).astype(np.float32),
        ]
    )


def _fake_result():
    labels = np.array([0] * 40 + [1] * 40)
    coords = np.zeros((80, 2), dtype=np.float32)
    return ClusterResult(
        labels=labels,
        coords_2d=coords,
        n_clusters=2,
        noise_ratio=0.0,
        cluster_sizes=[40, 40],
    )


@pytest.fixture()
def mem_engine(monkeypatch):
    eng = _make_engine()

    def _get_session():
        return Session(eng)

    monkeypatch.setattr("modules.database.engine._main_engine", eng)
    monkeypatch.setattr("modules.clustering.search.get_session", _get_session)
    return eng


def test_run_cluster_search_inserts_rows(mem_engine, monkeypatch):
    monkeypatch.setattr(
        "modules.clustering.search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr(
        "modules.clustering.search.compute_clusters",
        lambda matrix, **kw: _fake_result(),
    )

    from modules.clustering.search import run_cluster_search

    settings = _make_search_settings()
    run_cluster_search(settings)

    with Session(mem_engine) as s:
        count = s.query(ClusterRun).count()
    assert count == 3  # one row per embedding case (video, sandwich, audio)


def test_run_cluster_search_new_rows_have_passes_validation_none(
    mem_engine, monkeypatch
):
    """New rows inserted by run_cluster_search have passes_validation=None (pending)."""
    monkeypatch.setattr(
        "modules.clustering.search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr(
        "modules.clustering.search.compute_clusters",
        lambda matrix, **kw: _fake_result(),
    )

    from modules.clustering.search import run_cluster_search

    settings = _make_search_settings()
    run_cluster_search(settings)

    with Session(mem_engine) as s:
        rows = s.query(ClusterRun).all()
        for row in rows:
            assert row.passes_validation is None


def test_run_cluster_search_skips_no_embeddings_case(mem_engine, monkeypatch):
    """If a case has zero embeddings, search skips it (no rows inserted or mutated)."""

    def fake_load(case):
        if case == "video":
            return (np.zeros((0, 30), dtype=np.float32), [])
        return (_fake_matrix(), list(range(80)))

    monkeypatch.setattr("modules.clustering.search.load_user_matrix", fake_load)
    monkeypatch.setattr(
        "modules.clustering.search.compute_clusters",
        lambda matrix, **kw: _fake_result(),
    )

    from modules.clustering.search import run_cluster_search

    settings = _make_search_settings()
    run_cluster_search(settings)

    with Session(mem_engine) as s:
        video_count = (
            s.query(ClusterRun).filter(ClusterRun.embedding_case == "video").count()
        )
        assert video_count == 0  # nothing inserted for video


def test_run_cluster_search_idempotent(mem_engine, monkeypatch):
    monkeypatch.setattr(
        "modules.clustering.search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr(
        "modules.clustering.search.compute_clusters",
        lambda matrix, **kw: _fake_result(),
    )

    from modules.clustering.search import run_cluster_search

    settings = _make_search_settings()
    run_cluster_search(settings)
    run_cluster_search(settings)  # second call — must not duplicate

    with Session(mem_engine) as s:
        count = s.query(ClusterRun).count()
    assert count == 3  # still 3, not 6


def test_run_cluster_search_uses_single_thread_umap_per_combo(mem_engine, monkeypatch):
    """Grid search does not enable UMAP internal threading; parallelism is only via grid workers."""
    received: list[int | None] = []

    def capture_compute(matrix, **kw):
        received.append(kw.get("umap_n_jobs"))
        return _fake_result()

    monkeypatch.setenv("CLUSTERING_GRID_WORKERS", "1")
    monkeypatch.setenv("CLUSTERING_JOBS", "99")
    monkeypatch.setattr(
        "modules.clustering.search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr("modules.clustering.search.compute_clusters", capture_compute)

    from modules.clustering.search import run_cluster_search

    settings = _make_search_settings()
    run_cluster_search(settings)

    assert all(j in (None, 1) for j in received), received
    assert received, "compute_clusters should have been called"


def test_run_cluster_search_parallel_workers_uses_thread_pool(mem_engine, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    max_workers_seen: list[int | None] = []

    class RecordingPool(ThreadPoolExecutor):
        def __init__(self, *args, max_workers=None, **kwargs):
            max_workers_seen.append(max_workers)
            super().__init__(*args, max_workers=max_workers, **kwargs)

    def tracking_compute(matrix, **kw):
        assert kw.get("umap_n_jobs") in (None, 1)
        return _fake_result()

    monkeypatch.setattr("modules.clustering.search.ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(
        "modules.clustering.search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr("modules.clustering.search.compute_clusters", tracking_compute)

    from modules.clustering.search import run_cluster_search

    settings = _make_search_settings(umap_n_components=[5, 6])
    run_cluster_search(settings, clustering_grid_workers=3)

    with Session(mem_engine) as s:
        assert s.query(ClusterRun).count() == 6
    assert max_workers_seen == [3, 3, 3]
