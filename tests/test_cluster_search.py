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
        "eligibility",
        "dbcv",
        "silhouette",
        "param_plateau_score",
        "in_current_grid",
        "dataset_hash",
        "validation_config_hash",
    }


def test_cluster_run_unique_constraint():
    constraint_names = {c.name for c in ClusterRun.__table__.constraints}
    assert "uq_cluster_runs_params" in constraint_names


def test_cluster_run_hdbscan_min_samples_nullable():
    col = ClusterRun.__table__.c["hdbscan_min_samples"]
    assert col.nullable is True


def test_load_grid_combo_count():
    from types import SimpleNamespace

    from modules.cluster_search import _load_grid

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

    from modules.cluster_search import _load_grid

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

    from modules.cluster_search import _load_grid

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

    from modules.cluster_search import _load_grid

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
    monkeypatch.setattr("modules.cluster_search.get_session", _get_session)
    return eng


def test_run_cluster_search_inserts_rows(mem_engine, monkeypatch):
    monkeypatch.setattr(
        "modules.cluster_search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr(
        "modules.cluster_search.compute_clusters", lambda matrix, **kw: _fake_result()
    )

    from modules.cluster_search import run_cluster_search

    settings = _make_search_settings()
    run_cluster_search(settings)

    with Session(mem_engine) as s:
        count = s.query(ClusterRun).count()
    assert count == 3  # one row per embedding case (video, sandwich, audio)


def test_run_cluster_search_invalidates_rows_when_no_embeddings_for_case(
    mem_engine, monkeypatch
):
    """If a case has zero embeddings, existing ClusterRun rows for that case must not stay current."""
    with Session(mem_engine) as s:
        row = ClusterRun(
            embedding_case="video",
            umap_n_components=5,
            umap_n_neighbors=5,
            umap_min_dist=0.0,
            umap_metric="cosine",
            umap2d_n_neighbors=5,
            umap2d_min_dist=0.1,
            umap2d_metric="cosine",
            hdbscan_min_cluster_size=10,
            hdbscan_min_samples=None,
            hdbscan_cluster_selection_method="eom",
            hdbscan_metric="euclidean",
            random_state=42,
            n_clusters=3,
            noise_ratio=0.1,
            min_size=1,
            median_size=2,
            max_size=5,
            in_current_grid=1,
            eligibility=1,
            dataset_hash="old",
        )
        s.add(row)
        s.commit()
        video_row_id = row.id

    def fake_load(case):
        if case == "video":
            return (np.zeros((0, 30), dtype=np.float32), [])
        return (_fake_matrix(), list(range(80)))

    monkeypatch.setattr("modules.cluster_search.load_user_matrix", fake_load)
    monkeypatch.setattr(
        "modules.cluster_search.compute_clusters", lambda matrix, **kw: _fake_result()
    )

    from modules.cluster_search import run_cluster_search

    settings = _make_search_settings()
    run_cluster_search(settings)

    with Session(mem_engine) as s:
        video_row = s.get(ClusterRun, video_row_id)
        assert video_row is not None
        assert video_row.in_current_grid == 0
        assert video_row.eligibility == 2


def test_run_cluster_search_idempotent(mem_engine, monkeypatch):
    monkeypatch.setattr(
        "modules.cluster_search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr(
        "modules.cluster_search.compute_clusters", lambda matrix, **kw: _fake_result()
    )

    from modules.cluster_search import run_cluster_search

    settings = _make_search_settings()
    run_cluster_search(settings)
    run_cluster_search(settings)  # second call — must not duplicate

    with Session(mem_engine) as s:
        count = s.query(ClusterRun).count()
    assert count == 3  # still 3, not 6


def test_compute_dataset_hash_deterministic():
    from modules.cluster_search import _compute_dataset_hash

    fp1 = _compute_dataset_hash([3, 1, 2])
    fp2 = _compute_dataset_hash([1, 2, 3])
    assert fp1 == fp2  # order-independent
    assert len(fp1) == 64  # SHA-256 hex


def test_compute_dataset_hash_differs_on_different_users():
    from modules.cluster_search import _compute_dataset_hash

    assert _compute_dataset_hash([1, 2]) != _compute_dataset_hash([1, 3])


def test_combo_key_excludes_embedding_case():
    from modules.cluster_search import _combo_key

    combo = dict(
        embedding_case="video",
        umap_n_components=15,
        umap_n_neighbors=15,
        umap_min_dist=0.0,
        umap_metric="cosine",
        umap2d_n_neighbors=15,
        umap2d_min_dist=0.1,
        umap2d_metric="cosine",
        hdbscan_min_cluster_size=15,
        hdbscan_min_samples=None,
        hdbscan_cluster_selection_method="eom",
        hdbscan_metric="euclidean",
        random_state=42,
    )
    key = _combo_key(combo)
    assert ("embedding_case", "video") not in key
    assert ("umap_n_components", 15) in key


def test_run_cluster_search_marks_stale_rows_when_grid_shrinks(mem_engine, monkeypatch):
    """A row inserted under old grid becomes in_current_grid=0 when grid changes."""
    monkeypatch.setattr(
        "modules.cluster_search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr(
        "modules.cluster_search.compute_clusters", lambda matrix, **kw: _fake_result()
    )

    from modules.cluster_search import run_cluster_search

    # First run with wide grid
    settings_old = _make_search_settings(umap_n_components=[5, 10])
    run_cluster_search(settings_old)

    with Session(mem_engine) as s:
        assert s.query(ClusterRun).count() == 6  # 3 cases × 2 nc combos

    # Second run with narrow grid — nc=10 rows become stale
    settings_new = _make_search_settings(umap_n_components=[5])
    run_cluster_search(settings_new)

    with Session(mem_engine) as s:
        stale = s.query(ClusterRun).filter(ClusterRun.in_current_grid == 0).count()
        current = s.query(ClusterRun).filter(ClusterRun.in_current_grid == 1).count()
        assert stale == 3  # nc=10 rows for each of 3 cases
        assert current == 3  # nc=5 rows for each of 3 cases


def test_run_cluster_search_invalidates_rows_on_dataset_change(mem_engine, monkeypatch):
    """Rows computed on a different dataset fingerprint get in_current_grid=0."""
    call_count = {"n": 0}

    def fake_matrix(case):
        call_count["n"] += 1
        if call_count["n"] <= 3:
            pks = list(range(80))
            return (_fake_matrix(), pks)
        else:
            rng = np.random.default_rng(1)
            matrix = np.vstack(
                [
                    rng.normal(8.0, 0.2, (50, 30)).astype(np.float32),
                    rng.normal(-8.0, 0.2, (50, 30)).astype(np.float32),
                ]
            )
            return (matrix, list(range(100)))

    monkeypatch.setattr("modules.cluster_search.load_user_matrix", fake_matrix)
    monkeypatch.setattr(
        "modules.cluster_search.compute_clusters", lambda matrix, **kw: _fake_result()
    )

    from modules.cluster_search import run_cluster_search

    settings = _make_search_settings()
    run_cluster_search(settings)  # uses pks 0..79

    with Session(mem_engine) as s:
        result = s.query(ClusterRun).first()
        assert result is not None
        fp_before = result.dataset_hash
        assert fp_before is not None

    run_cluster_search(settings)  # uses pks 0..99 — fingerprint changes

    with Session(mem_engine) as s:
        stale = s.query(ClusterRun).filter(ClusterRun.in_current_grid == 0).all()
        assert len(stale) == 3  # old rows invalidated
        current = s.query(ClusterRun).filter(ClusterRun.in_current_grid == 1).all()
        assert len(current) == 3  # new rows inserted


def test_run_cluster_search_uses_single_thread_umap_per_combo(mem_engine, monkeypatch):
    """Grid search does not enable UMAP internal threading; parallelism is only via grid workers."""
    received: list[int | None] = []

    def capture_compute(matrix, **kw):
        received.append(kw.get("umap_n_jobs"))
        return _fake_result()

    monkeypatch.setenv("CLUSTERING_GRID_WORKERS", "1")
    monkeypatch.setenv("CLUSTERING_JOBS", "99")
    monkeypatch.setattr(
        "modules.cluster_search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr("modules.cluster_search.compute_clusters", capture_compute)

    from modules.cluster_search import run_cluster_search

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

    monkeypatch.setattr("modules.cluster_search.ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(
        "modules.cluster_search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr("modules.cluster_search.compute_clusters", tracking_compute)

    from modules.cluster_search import run_cluster_search

    settings = _make_search_settings(umap_n_components=[5, 6])
    run_cluster_search(settings, clustering_grid_workers=3)

    with Session(mem_engine) as s:
        assert s.query(ClusterRun).count() == 6
    assert max_workers_seen == [3, 3, 3]


def test_run_cluster_search_new_rows_have_dataset_hash_and_in_current_grid(
    mem_engine, monkeypatch
):
    monkeypatch.setattr(
        "modules.cluster_search.load_user_matrix",
        lambda case: (_fake_matrix(), list(range(80))),
    )
    monkeypatch.setattr(
        "modules.cluster_search.compute_clusters", lambda matrix, **kw: _fake_result()
    )

    from modules.cluster_search import run_cluster_search

    settings = _make_search_settings()
    run_cluster_search(settings)

    with Session(mem_engine) as s:
        rows = s.query(ClusterRun).all()
        for row in rows:
            assert row.in_current_grid == 1
            assert row.dataset_hash is not None
            assert len(row.dataset_hash) == 64
