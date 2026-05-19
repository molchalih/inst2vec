import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, __file__[: __file__.rfind("/")] + "/..")

from core.database import Base, ClusterRun
from modules.clustering.core import ClusterResult
from tests._clustering_helpers import (
    _make_minimal_search_settings,
    _mutate_one_embedding,
    _seed_search_dataset,
)


def _make_search_settings(*, gemini_enabled: bool = False, **overrides):
    """Create a full-shaped settings namespace for run_cluster_search tests.

    Returns a namespace with `.search` (grid hyperparameters) and
    `.embeddings.gemini_enabled` so `default_cases(settings)` works.
    """
    defaults = {
        "umap_n_components": [5],
        "umap_n_neighbors": [5],
        "umap_min_dist": [0.0],
        "umap_metrics": ["cosine"],
        "umap2d_n_neighbors": 5,
        "umap2d_min_dist": 0.1,
        "umap2d_metrics": ["cosine"],
        "hdbscan_min_cluster_size": [10],
        "hdbscan_min_samples": [],
        "hdbscan_selection": ["eom"],
        "random_state": 42,
    }
    defaults.update(overrides)
    return SimpleNamespace(
        search=SimpleNamespace(**defaults),
        embeddings=SimpleNamespace(gemini_enabled=gemini_enabled),
    )


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
        hdbscan_min_samples=[],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    combos = _load_grid(settings, cases=("video", "sandwich", "audio"))
    # 3 cases × 2 umap_n_components × 1 nn × 1 md × 1 umap_metric
    # × 2 umap2d_metrics × 1 mcs × 1 selection = 12 (HDBSCAN metric fixed, not swept)
    assert len(combos) == 12


def test_load_grid_iterates_gemini_when_enabled():
    """Regression: when gemini is included in cases, _load_grid emits
    combos for it. Previously the clustering loops hardcoded the three-case
    literal so gemini embeddings were produced but never clustered."""
    from types import SimpleNamespace

    from modules.clustering.search import _load_grid

    settings = SimpleNamespace(
        umap_n_components=[10],
        umap_n_neighbors=[10],
        umap_min_dist=[0.0],
        umap_metrics=["cosine"],
        umap2d_n_neighbors=15,
        umap2d_min_dist=0.1,
        umap2d_metrics=["cosine"],
        hdbscan_min_cluster_size=[10],
        hdbscan_min_samples=[],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    combos = _load_grid(settings, cases=("video", "sandwich", "audio", "gemini"))
    case_set = {c["embedding_case"] for c in combos}
    assert case_set == {"video", "sandwich", "audio", "gemini"}


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
        hdbscan_min_samples=[],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    combos = _load_grid(settings, cases=("video", "sandwich", "audio"))
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
        hdbscan_min_samples=[],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    combos = _load_grid(settings, cases=("video", "sandwich", "audio"))
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
        hdbscan_min_samples=[],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    combos = _load_grid(settings, cases=("video", "sandwich", "audio"))
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

    monkeypatch.setattr("core.database.engine._main_engine", eng)
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


# ── fingerprint integration tests ────────────────────────────────────────────
# These tests use the conftest-initialised in-memory DB (not mem_engine) so
# fingerprint StageState rows land in the same engine as ClusterRun rows.


def test_unchanged_fingerprint_skips_recomputation(monkeypatch):
    """Second call to run_cluster_search with unchanged inputs is a no-op."""
    from core.database import ClusterRun, get_session
    from modules.clustering import run_cluster_search

    monkeypatch.setattr(
        "modules.clustering.search.compute_clusters",
        lambda matrix, **kw: _fake_result(),
    )

    _seed_search_dataset()
    settings = _make_minimal_search_settings()
    run_cluster_search(settings)
    session = get_session()
    try:
        first_ids = {r.id for r in session.query(ClusterRun.id).all()}
    finally:
        session.close()
    assert first_ids  # something was inserted

    run_cluster_search(settings)
    session = get_session()
    try:
        second_ids = {r.id for r in session.query(ClusterRun.id).all()}
    finally:
        session.close()
    assert second_ids == first_ids  # no rewrites, same row ids


def test_changed_embeddings_wipes_and_recomputes(monkeypatch):
    """Mutating a UserEmbedding blob invalidates fingerprint and rewrites.

    Asserts that the ClusterRun row was deleted and reinserted (not just
    skipped) by making the second compute return a different n_clusters value
    and checking that the stored row reflects the new value.  If the delete
    path were broken the old row would remain and n_clusters would not change.
    """
    from core.database import ClusterRun, StageState, get_session
    from modules.clustering import run_cluster_search

    call_count = [0]

    def counting_compute(matrix, **kw):
        # Return n_clusters=2 on the first call, n_clusters=3 on subsequent calls.
        n = 2 if call_count[0] == 0 else 3
        call_count[0] += 1
        labels = np.array([i % n for i in range(80)])
        coords = np.zeros((80, 2), dtype=np.float32)
        sizes = [80 // n] * n
        return ClusterResult(
            labels=labels,
            coords_2d=coords,
            n_clusters=n,
            noise_ratio=0.0,
            cluster_sizes=sizes,
        )

    monkeypatch.setattr("modules.clustering.search.compute_clusters", counting_compute)

    _seed_search_dataset()
    settings = _make_minimal_search_settings()
    run_cluster_search(settings)

    session = get_session()
    try:
        assert session.query(ClusterRun).count() == 1  # video only
        first_hash = session.get(StageState, ("cluster_search", "video")).data_hash
        first_n_clusters = session.query(ClusterRun).first().n_clusters
    finally:
        session.close()

    calls_after_first = call_count[0]
    assert calls_after_first >= 1
    assert first_n_clusters == 2

    _mutate_one_embedding()
    run_cluster_search(settings)

    session = get_session()
    try:
        assert session.query(ClusterRun).count() == 1  # still video only
        second_hash = session.get(StageState, ("cluster_search", "video")).data_hash
        second_n_clusters = session.query(ClusterRun).first().n_clusters
    finally:
        session.close()

    # fingerprint changed → compute ran again → new StageState data_hash
    assert call_count[0] > calls_after_first
    assert first_hash != second_hash  # fingerprint was updated
    # delete-then-reinsert path stored the new result from the second compute
    assert second_n_clusters == 3


def test_changed_grid_config_wipes_and_recomputes(monkeypatch):
    """Changing the grid (different umap_n_components) wipes old rows."""
    from core.database import ClusterRun, get_session
    from modules.clustering import run_cluster_search

    monkeypatch.setattr(
        "modules.clustering.search.compute_clusters",
        lambda matrix, **kw: _fake_result(),
    )

    _seed_search_dataset()
    run_cluster_search(_make_minimal_search_settings(umap_n_components=[3]))
    session = get_session()
    try:
        first_components = {
            r.umap_n_components
            for r in session.query(ClusterRun.umap_n_components).all()
        }
    finally:
        session.close()
    assert first_components == {3}

    run_cluster_search(_make_minimal_search_settings(umap_n_components=[4]))
    session = get_session()
    try:
        second_components = {
            r.umap_n_components
            for r in session.query(ClusterRun.umap_n_components).all()
        }
    finally:
        session.close()
    assert second_components == {4}  # old 3-component rows wiped


def test_no_user_embeddings_seals_empty_state():
    """Empty matrix: no rows inserted but StageState seal exists for the case."""
    from core.database import (
        Base,
        Clip,
        ClusterRun,
        StageState,
        User,
        UserEmbedding,
        get_engine,
        get_session,
    )
    from modules.clustering import run_cluster_search

    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for m in (ClusterRun, StageState, UserEmbedding, Clip, User):
            session.query(m).delete()
        session.commit()
    finally:
        session.close()

    run_cluster_search(_make_minimal_search_settings())
    session = get_session()
    try:
        # No analysis users -> no ClusterRun rows for any case.
        assert session.query(ClusterRun).count() == 0
        # The empty-matrix path still seals StageState so a subsequent
        # run with the same empty inputs short-circuits via fingerprint
        # match (per spec: "empty matrix seals empty state").
        for case in ("video", "sandwich", "audio"):
            assert session.get(StageState, ("cluster_search", case)) is not None
    finally:
        session.close()
