import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import Base, ClusterRun


def _make_settings(**overrides):
    """Create a settings object for validation tests."""
    defaults = {
        "plateau_drop_threshold": 0.05,
        "max_noise_ratio": 0.3,
        "min_clusters": 3,
        "max_clusters": 20,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _base_run_kwargs(**overrides):
    defaults = dict(
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
        n_clusters=5,
        noise_ratio=0.1,
        min_size=10,
        median_size=20,
        max_size=40,
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
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=5)
        row_id = row.id

    from modules.clustering.validation import _phase_filter

    with Session(eng) as s:
        _phase_filter(s, "video", _make_settings())
        result = s.get(ClusterRun, row_id)
        assert result is not None
        assert result.passes_validation is True


def test_filter_disqualifies_high_noise():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.5, n_clusters=5)
        row_id = row.id

    from modules.clustering.validation import _phase_filter

    with Session(eng) as s:
        _phase_filter(s, "video", _make_settings())
        result = s.get(ClusterRun, row_id)
        assert result is not None
        assert result.passes_validation is False


def test_filter_disqualifies_too_few_clusters():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=1)
        row_id = row.id

    from modules.clustering.validation import _phase_filter

    with Session(eng) as s:
        _phase_filter(s, "video", _make_settings())
        result = s.get(ClusterRun, row_id)
        assert result is not None
        assert result.passes_validation is False


def test_filter_disqualifies_too_many_clusters():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=25)
        row_id = row.id

    from modules.clustering.validation import _phase_filter

    with Session(eng) as s:
        _phase_filter(s, "video", _make_settings())
        result = s.get(ClusterRun, row_id)
        assert result is not None
        assert result.passes_validation is False


def test_filter_processes_all_rows():
    """_phase_filter processes all rows for the case."""
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=5)
        # passes_validation starts as None
        assert row.passes_validation is None
        row_id = row.id

    from modules.clustering.validation import _phase_filter

    with Session(eng) as s:
        _phase_filter(s, "video", _make_settings())
        result = s.get(ClusterRun, row_id)
        assert result is not None
        assert result.passes_validation is not None


# --- Phase 2: score ---


def test_phase_score_populates_dbcv_and_silhouette(monkeypatch):
    eng = _make_engine()

    def _get_session():
        return Session(eng)

    monkeypatch.setattr("modules.clustering.validation.get_session", _get_session)

    rng = np.random.default_rng(0)
    matrix = np.vstack(
        [
            rng.normal(8.0, 0.2, (40, 30)).astype(np.float32),
            rng.normal(-8.0, 0.2, (40, 30)).astype(np.float32),
        ]
    )

    with Session(eng) as s:
        row = _insert_run(
            s,
            noise_ratio=0.0,
            n_clusters=2,
            passes_validation=True,
            umap_n_components=5,
            hdbscan_min_cluster_size=10,
        )
        row_id = row.id

    from modules.clustering.validation import _phase_score

    with Session(eng) as s:
        _phase_score(s, "video", matrix)
        updated = s.get(ClusterRun, row_id)
        assert updated is not None
        assert updated.dbcv is not None
        assert updated.silhouette is not None


def test_phase_score_skips_already_scored_rows(monkeypatch):
    eng = _make_engine()

    def _get_session():
        return Session(eng)

    monkeypatch.setattr("modules.clustering.validation.get_session", _get_session)

    rng = np.random.default_rng(0)
    matrix = np.vstack(
        [
            rng.normal(8.0, 0.2, (40, 30)).astype(np.float32),
            rng.normal(-8.0, 0.2, (40, 30)).astype(np.float32),
        ]
    )

    with Session(eng) as s:
        row = _insert_run(
            s,
            noise_ratio=0.0,
            n_clusters=2,
            passes_validation=True,
            umap_n_components=5,
            hdbscan_min_cluster_size=10,
        )
        row.dbcv = 0.99
        row.silhouette = 0.88
        s.commit()
        row_id = row.id

    from modules.clustering.validation import _phase_score

    with Session(eng) as s:
        _phase_score(s, "video", matrix)
        updated = s.get(ClusterRun, row_id)
        assert updated is not None
        assert updated.dbcv == pytest.approx(0.99)
        assert updated.silhouette == pytest.approx(0.88)


def test_compute_row_scores_uses_explicit_euclidean_metrics(monkeypatch):
    captured = {}

    class DummyResult:
        def __init__(self):
            self.matrix_nd = np.array(
                [[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]],
                dtype=np.float32,
            )
            self.labels = np.array([0, 0, 1, 1], dtype=int)

    monkeypatch.setattr(
        "modules.clustering.validation.compute_clusters",
        lambda matrix, return_nd_matrix, **params: DummyResult(),
    )

    def fake_validity_index(X, labels, metric):
        captured["dbcv_metric"] = metric
        return 0.42

    def fake_silhouette_score(X, labels, metric):
        captured["silhouette_metric"] = metric
        return 0.51

    monkeypatch.setattr(
        "modules.clustering.validation.hdbscan.validity.validity_index",
        fake_validity_index,
    )
    monkeypatch.setattr(
        "modules.clustering.validation.silhouette_score",
        fake_silhouette_score,
    )

    from modules.clustering.validation import _compute_row_scores

    outcome = _compute_row_scores(
        np.zeros((4, 2), dtype=np.float32),
        {
            "umap_n_components": 2,
            "umap_n_neighbors": 2,
            "umap_min_dist": 0.0,
            "umap_metric": "cosine",
            "umap2d_n_neighbors": 2,
            "umap2d_min_dist": 0.1,
            "umap2d_metric": "cosine",
            "hdbscan_min_cluster_size": 2,
            "hdbscan_min_samples": None,
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_metric": "cosine",
            "random_state": 42,
        },
    )

    assert outcome == (0.42, 0.51)
    assert captured == {
        "dbcv_metric": "euclidean",
        "silhouette_metric": "euclidean",
    }


# --- Plateau neighbor logic ---


def test_find_param_neighbors_one_step_difference():
    from modules.clustering.validation import _find_param_neighbors

    eng = _make_engine()

    with Session(eng) as s:
        target = _insert_run(
            s, umap_n_components=10, umap_n_neighbors=15, random_state=1
        )
        neighbor = _insert_run(
            s, umap_n_components=15, umap_n_neighbors=15, random_state=1
        )
        non_neighbor = _insert_run(
            s, umap_n_components=20, umap_n_neighbors=10, random_state=1
        )
        two_away = _insert_run(
            s, umap_n_components=20, umap_n_neighbors=15, random_state=1
        )

        result = _find_param_neighbors(target, [neighbor, non_neighbor, two_away])
        result_ids = {r.id for r in result}
        assert neighbor.id in result_ids
        assert non_neighbor.id not in result_ids
        assert two_away.id not in result_ids


def test_find_param_neighbors_categorical_any_other_value():
    from modules.clustering.validation import _find_param_neighbors

    eng = _make_engine()

    with Session(eng) as s:
        target = _insert_run(s, umap_metric="cosine", random_state=1)
        neighbor = _insert_run(s, umap_metric="euclidean", random_state=1)
        non_neighbor = _insert_run(
            s, umap_metric="euclidean", umap_n_components=10, random_state=1
        )

        result = _find_param_neighbors(target, [neighbor, non_neighbor])
        result_ids = {r.id for r in result}
        assert neighbor.id in result_ids
        assert non_neighbor.id not in result_ids


# --- Phase 3: plateau ---


def test_phase_plateau_uses_dbcv_of_neighbors():
    """param_plateau_score is the mean DBCV of grid-adjacent neighbors."""
    eng = _make_engine()
    with Session(eng) as s:
        r1 = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=10,
            random_state=1,
        )
        r1.dbcv = 0.8
        r2 = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=15,
            random_state=1,
        )
        r2.dbcv = 0.6
        s.commit()
        id1 = r1.id

    from modules.clustering.validation import _phase_plateau

    with Session(eng) as s:
        _phase_plateau(s, "video", _make_settings())
        r1_updated = s.get(ClusterRun, id1)
        assert r1_updated is not None
        # r1's only neighbor is r2 (dbcv=0.6), so plateau = 0.6
        assert r1_updated.param_plateau_score == pytest.approx(0.6, abs=1e-5)


def test_phase_plateau_covers_all_scored_rows():
    """Plateau is computed for every qualifying row, not just a top-N subset."""
    eng = _make_engine()
    with Session(eng) as s:
        rows = []
        for i, nc in enumerate([10, 15, 20]):
            r = _insert_run(
                s,
                passes_validation=True,
                noise_ratio=0.1,
                n_clusters=5,
                umap_n_components=nc,
                random_state=1,
            )
            r.dbcv = 0.5 + i * 0.1
            rows.append(r)
        s.commit()
        ids = [r.id for r in rows]

    from modules.clustering.validation import _phase_plateau

    with Session(eng) as s:
        _phase_plateau(s, "video", _make_settings())
        for rid in ids:
            result = s.get(ClusterRun, rid)
            assert result is not None
            assert result.param_plateau_score is not None


def test_phase_plateau_no_neighbors_falls_back_to_own_dbcv():
    """An isolated config with no grid-adjacent neighbors gets plateau = own dbcv."""
    eng = _make_engine()
    with Session(eng) as s:
        r = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=10,
            random_state=1,
        )
        r.dbcv = 0.75
        s.commit()
        rid = r.id

    from modules.clustering.validation import _phase_plateau

    with Session(eng) as s:
        _phase_plateau(s, "video", _make_settings())
        updated = s.get(ClusterRun, rid)
        assert updated is not None
        # No neighbors → fallback to own dbcv → drop = 0 → not rejected by filter
        assert updated.param_plateau_score == pytest.approx(0.75, abs=1e-5)


def test_phase_plateau_skips_already_set():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=10,
            random_state=1,
        )
        row.dbcv = 0.9
        row.param_plateau_score = 0.5  # already set
        s.commit()
        row_id = row.id

    from modules.clustering.validation import _phase_plateau

    with Session(eng) as s:
        _phase_plateau(s, "video", _make_settings())
        result = s.get(ClusterRun, row_id)
        assert result is not None
        assert result.param_plateau_score == pytest.approx(0.5)


# --- _select_best ---


def test_select_best_picks_highest_dbcv():
    """_select_best selects the run with highest DBCV among plateau survivors."""
    eng = _make_engine()
    with Session(eng) as s:
        r1 = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=10,
            random_state=1,
        )
        r1.dbcv = 0.9
        r1.param_plateau_score = 0.88  # drop=0.02, within threshold
        r2 = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=15,
            random_state=1,
        )
        r2.dbcv = 0.6
        r2.param_plateau_score = 0.59  # drop=0.01, within threshold
        s.commit()
        id1 = r1.id

    from modules.clustering.validation import _select_best

    with Session(eng) as s:
        result = _select_best(s, "video", _make_settings())
        assert result is not None
        assert result.id == id1


def test_select_best_rejects_sharp_peak_by_plateau_filter():
    """A run whose DBCV far exceeds its neighborhood mean is rejected as a sharp peak."""
    eng = _make_engine()
    with Session(eng) as s:
        # r1 has higher DBCV but is a sharp peak
        r1 = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=10,
            random_state=1,
        )
        r1.dbcv = 0.9
        r1.param_plateau_score = 0.3  # drop=0.6, far exceeds threshold → rejected
        # r2 has lower DBCV but is on a stable plateau
        r2 = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=15,
            random_state=1,
        )
        r2.dbcv = 0.7
        r2.param_plateau_score = 0.68  # drop=0.02, within threshold → survives
        s.commit()
        id2 = r2.id

    from modules.clustering.validation import _select_best

    with Session(eng) as s:
        result = _select_best(s, "video", _make_settings())
        assert result is not None
        assert result.id == id2


def test_select_best_falls_back_when_all_rejected():
    """When all runs fail the plateau filter, fall back to highest DBCV."""
    eng = _make_engine()
    with Session(eng) as s:
        r1 = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=10,
            random_state=1,
        )
        r1.dbcv = 0.9
        r1.param_plateau_score = 0.1  # sharp peak — would be rejected
        r2 = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=15,
            random_state=1,
        )
        r2.dbcv = 0.7
        r2.param_plateau_score = 0.2  # also sharp — would be rejected
        s.commit()
        id1 = r1.id

    from modules.clustering.validation import _select_best

    with Session(eng) as s:
        result = _select_best(s, "video", _make_settings())
        assert result is not None
        assert result.id == id1  # fallback: highest DBCV wins


def test_select_best_returns_none_when_no_eligible_runs():
    eng = _make_engine()
    with Session(eng) as s:
        _insert_run(s, passes_validation=True, noise_ratio=0.1, n_clusters=5)
        # no dbcv set → not included in list_best_candidate_rows
        s.commit()

    from modules.clustering.validation import _select_best

    with Session(eng) as s:
        assert _select_best(s, "video", _make_settings()) is None


def test_select_best_ignores_disqualified():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(
            s,
            passes_validation=False,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=10,
            random_state=1,
        )
        row.dbcv = 0.9
        row.param_plateau_score = 0.88
        s.commit()

    from modules.clustering.validation import _select_best

    with Session(eng) as s:
        assert _select_best(s, "video", _make_settings()) is None


def test_select_best_ignores_cluster_override_env(monkeypatch):
    """CLUSTER_OVERRIDE_* is unsupported; selection uses plateau + DBCV only."""
    eng = _make_engine()
    with Session(eng) as s:
        r1 = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=10,
            random_state=1,
        )
        r1.dbcv = 0.9
        r1.param_plateau_score = 0.88
        r2 = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=15,
            random_state=1,
        )
        r2.dbcv = 0.1
        r2.param_plateau_score = 0.09
        s.commit()
        low_id = r2.id

    monkeypatch.setenv("CLUSTER_OVERRIDE_VIDEO", str(low_id))
    from modules.clustering.validation import _select_best

    with Session(eng) as s:
        result = _select_best(s, "video", _make_settings())
        assert result is not None
        assert result.id != low_id
        assert result.dbcv == 0.9


def test_select_best_delegates_to_shared_selector(monkeypatch):
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(
            s,
            passes_validation=True,
            noise_ratio=0.1,
            n_clusters=5,
            umap_n_components=10,
            random_state=1,
        )
        row.dbcv = 0.9
        row.param_plateau_score = 0.88
        s.commit()
        row_id = row.id

    calls = {}

    def fake_pick(rows, threshold=None):
        calls["count"] = len(rows)
        return rows[0]

    monkeypatch.setattr(
        "modules.clustering.validation.pick_best_cluster_run", fake_pick
    )

    from modules.clustering.validation import _select_best

    with Session(eng) as s:
        result = _select_best(s, "video", _make_settings())

    assert result is not None
    assert result.id == row_id
    assert calls["count"] == 1


# --- Orchestration ---


def test_validate_clustering_phase_order(monkeypatch):
    """validate_clustering: fingerprint check → load → _compute_updates → write.

    The new orchestration no longer calls _phase_filter/_phase_score/_phase_plateau
    as top-level callables (they are still available for direct use in other tests).
    Instead, filter→score→plateau logic runs inside _compute_updates. We verify:
    - _compute_updates is called for the video case (non-empty matrix)
    - no forbidden operations (invalidate, bootstrap, composite) appear
    - filter logic (passes_validation) runs before score logic (dbcv set)
      by inspecting what _compute_updates returns
    """
    # Seed one ClusterRun row so _compute_updates is guaranteed to be invoked
    # (fingerprint is stale on a fresh DB, and the matrix is non-empty).
    from core.database import Base, ClusterRun, get_engine, get_session

    Base.metadata.create_all(get_engine())
    seed_session = get_session()
    try:
        seed_session.query(ClusterRun).delete()
        seed_session.commit()
        seed_session.add(ClusterRun(**_base_run_kwargs(embedding_case="video")))
        seed_session.commit()
    finally:
        seed_session.close()

    sequence = []

    def fake_load_matrix(case):
        if case == "video":
            return (np.ones((5, 10), dtype=np.float32), list(range(5)))
        return (np.zeros((0, 10), dtype=np.float32), [])

    from modules.clustering import validation as validation_mod

    original_compute_updates = validation_mod._compute_updates

    def spy_compute_updates(case, matrix, settings, workers=1):
        sequence.append(("compute_updates", case))
        return original_compute_updates(case, matrix, settings, workers)

    monkeypatch.setattr(
        "modules.clustering.validation.load_user_matrix", fake_load_matrix
    )
    monkeypatch.setattr(
        "modules.clustering.validation._compute_updates", spy_compute_updates
    )

    from modules.clustering.validation import validate_clustering

    validate_clustering(_make_settings())

    video_seq = [(op, c) for op, c in sequence if c == "video"]

    ops = [op for op, _c in sequence]
    assert "invalidate" not in ops, "invalidate must not be called"
    assert "bootstrap" not in ops, "bootstrap must not be called"
    assert "composite" not in ops, "composite must not be called"

    # _compute_updates must be called for the video case since the matrix is
    # non-empty and fingerprint is stale (fresh DB with one seeded ClusterRun).
    assert ("compute_updates", "video") in video_seq, (
        f"_compute_updates was not called for video; sequence={video_seq}"
    )


def test_phase_score_uses_thread_pool_when_workers_gt_one(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    eng = _make_engine()

    def _get_session():
        return Session(eng)

    monkeypatch.setattr("modules.clustering.validation.get_session", _get_session)

    max_workers_seen: list[int | None] = []

    class RecordingPool(ThreadPoolExecutor):
        def __init__(self, *args, max_workers=None, **kwargs):
            max_workers_seen.append(max_workers)
            super().__init__(*args, max_workers=max_workers, **kwargs)

    calls = {"n": 0}

    def counting_compute(matrix, **kw):
        calls["n"] += 1
        assert kw.get("umap_n_jobs") in (None, 1)
        from modules.clustering import ClusterResult

        labels = np.zeros(matrix.shape[0], dtype=np.int32)
        return ClusterResult(
            labels=labels,
            coords_2d=np.zeros((matrix.shape[0], 2), dtype=np.float32),
            n_clusters=1,
            noise_ratio=0.0,
            cluster_sizes=[matrix.shape[0]],
            matrix_nd=np.zeros((matrix.shape[0], 2), dtype=np.float32),
        )

    with Session(eng) as s:
        for nc in (15, 16):
            row = _insert_run(s, umap_n_components=nc, passes_validation=True)
            row.dbcv = None
            s.commit()

    monkeypatch.setattr(
        "modules.clustering.validation.ThreadPoolExecutor", RecordingPool
    )
    monkeypatch.setattr(
        "modules.clustering.validation.compute_clusters", counting_compute
    )

    from modules.clustering.validation import _phase_score

    matrix = np.ones((20, 8), dtype=np.float32)

    with Session(eng) as session:
        _phase_score(session, "video", matrix, clustering_grid_workers=4)

    assert calls["n"] == 2
    assert max_workers_seen == [4]


# --- New tests for Task 3 semantics ---


def test_passes_validation_none_means_pending():
    from core.database import ClusterRun, get_session

    session = get_session()
    try:
        session.query(ClusterRun).delete()
        session.commit()
        row = ClusterRun(
            embedding_case="video",
            umap_n_components=5,
            umap_n_neighbors=15,
            umap_min_dist=0.1,
            umap_metric="cosine",
            umap2d_n_neighbors=15,
            umap2d_min_dist=0.1,
            umap2d_metric="cosine",
            hdbscan_min_cluster_size=15,
            hdbscan_min_samples=None,
            hdbscan_cluster_selection_method="eom",
            hdbscan_metric="euclidean",
            random_state=42,
            n_clusters=5,
            noise_ratio=0.1,
            min_size=10,
            median_size=20,
            max_size=30,
        )
        session.add(row)
        session.commit()
        assert row.passes_validation is None
    finally:
        session.close()


def test_list_best_candidate_rows_filters_by_passes_validation():
    from core.database import ClusterRun, get_session
    from modules.clustering import list_best_candidate_rows

    session = get_session()
    try:
        session.query(ClusterRun).delete()
        session.commit()
        base = dict(
            embedding_case="video",
            umap_n_components=5,
            umap_n_neighbors=15,
            umap_min_dist=0.1,
            umap_metric="cosine",
            umap2d_n_neighbors=15,
            umap2d_min_dist=0.1,
            umap2d_metric="cosine",
            hdbscan_min_samples=None,
            hdbscan_cluster_selection_method="eom",
            hdbscan_metric="euclidean",
            random_state=42,
            n_clusters=5,
            noise_ratio=0.1,
            min_size=10,
            median_size=20,
            max_size=30,
            dbcv=0.5,
            silhouette=0.4,
            param_plateau_score=0.45,
        )
        passed = ClusterRun(**base, hdbscan_min_cluster_size=15, passes_validation=True)
        failed = ClusterRun(
            **base, hdbscan_min_cluster_size=16, passes_validation=False
        )
        pending = ClusterRun(
            **{**base, "dbcv": None, "param_plateau_score": None},
            hdbscan_min_cluster_size=17,
            passes_validation=None,
        )
        session.add_all([passed, failed, pending])
        session.commit()

        rows = list_best_candidate_rows(session, "video")
        ids = sorted(r.hdbscan_min_cluster_size for r in rows)
        assert ids == [15]
    finally:
        session.close()


# ── fingerprint integration tests ────────────────────────────────────────────
# These tests use the conftest-initialised in-memory DB so fingerprint
# StageState rows land in the same engine as ClusterRun rows.


def _seed_validate_dataset() -> object:
    """Seed Users/Clips/UserEmbeddings, run cluster_search, return validation settings."""
    from modules.clustering import run_cluster_search
    from tests._clustering_helpers import (
        _make_minimal_search_settings,
        _seed_search_dataset,
    )

    _seed_search_dataset()
    search_settings = _make_minimal_search_settings()
    run_cluster_search(search_settings)
    return SimpleNamespace(
        max_noise_ratio=0.9,
        min_clusters=1,
        max_clusters=20,
        plateau_drop_threshold=0.05,
    )


def test_validate_unchanged_fingerprint_skips_recomputation(monkeypatch):
    """Second validate_clustering call is a no-op when inputs unchanged."""
    from modules.clustering import validate_clustering
    from modules.clustering import validation as validation_mod

    settings = _seed_validate_dataset()
    validate_clustering(settings)

    calls = []
    original = validation_mod._compute_row_scores

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(validation_mod, "_compute_row_scores", spy)
    validate_clustering(settings)
    assert calls == []  # no rescoring happened


def test_validate_changed_config_invalidates_and_rewrites_fields():
    """Changing plateau_drop_threshold triggers a recompute on next call."""
    from core.database import ClusterRun, get_session
    from modules.clustering import validate_clustering

    settings = _seed_validate_dataset()
    validate_clustering(settings)

    # Mark a row's fields as sentinels so we can detect overwrites.
    session = get_session()
    try:
        row = session.query(ClusterRun).first()
        target_id = row.id
        row.passes_validation = False
        row.dbcv = -999.0
        row.silhouette = -999.0
        row.param_plateau_score = -999.0
        session.commit()
    finally:
        session.close()

    changed = SimpleNamespace(
        max_noise_ratio=settings.max_noise_ratio,
        min_clusters=settings.min_clusters,
        max_clusters=settings.max_clusters,
        plateau_drop_threshold=0.20,
    )
    validate_clustering(changed)

    session = get_session()
    try:
        row = session.get(ClusterRun, target_id)
        assert row.dbcv != -999.0
        assert row.n_clusters is not None
        assert row.noise_ratio is not None
    finally:
        session.close()


def test_validate_passes_validation_semantics_pending_pass_fail():
    """After validate runs, all rows for the case have passes_validation set."""
    from core.database import ClusterRun, get_session
    from modules.clustering import validate_clustering

    settings = _seed_validate_dataset()
    validate_clustering(settings)

    session = get_session()
    try:
        rows = (
            session.query(ClusterRun).filter(ClusterRun.embedding_case == "video").all()
        )
        assert all(r.passes_validation is not None for r in rows)
        for r in rows:
            assert isinstance(r.passes_validation, bool)
    finally:
        session.close()


def test_validate_score_value_error_marks_false(monkeypatch):
    """compute_clusters ValueError inside scoring sets passes_validation=False."""
    from core.database import ClusterRun, get_session
    from modules.clustering import validate_clustering
    from modules.clustering import validation as validation_mod

    settings = _seed_validate_dataset()

    def boom(matrix, params):
        return "value_error"

    monkeypatch.setattr(validation_mod, "_compute_row_scores", boom)
    validate_clustering(settings)

    session = get_session()
    try:
        rows = (
            session.query(ClusterRun).filter(ClusterRun.embedding_case == "video").all()
        )
        assert all(r.passes_validation is not None for r in rows)
        assert any(r.passes_validation is False for r in rows)
    finally:
        session.close()
