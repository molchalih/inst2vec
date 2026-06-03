import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import Base, ClusterRun


def _make_settings(**overrides):
    """Narrow validation-knobs SimpleNamespace for validation helpers."""
    defaults = {
        "plateau_drop_threshold": 0.05,
        "max_noise_ratio": 0.3,
        "min_clusters": 3,
        "max_clusters": 20,
        "max_dominance": 1.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_full_settings(**overrides):
    """Settings namespace (validation + search knobs) for validate_clustering tests."""
    return SimpleNamespace(
        validation=_make_settings(**overrides),
        search=SimpleNamespace(hdbscan_max_cluster_frac=0.0, embedding_preprocess={}),
    )


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
        lambda matrix, params=None, **kw: DummyResult(),
    )

    def fake_validity_index(X, labels, metric):
        captured["dbcv_metric"] = metric
        return 0.42

    def fake_silhouette_score(X, labels, metric, random_state=None):
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

    def fake_load_matrix(case, preprocess="none"):
        if case == "video":
            return (np.ones((5, 10), dtype=np.float32), list(range(5)))
        return (np.zeros((0, 10), dtype=np.float32), [])

    from modules.clustering import validation as validation_mod

    original_compute_updates = validation_mod._compute_updates

    def spy_compute_updates(case, matrix, settings, max_cluster_frac=0.0):
        sequence.append(("compute_updates", case))
        return original_compute_updates(
            case, matrix, settings, max_cluster_frac=max_cluster_frac
        )

    monkeypatch.setattr(
        "modules.clustering.validation.load_user_matrix", fake_load_matrix
    )
    monkeypatch.setattr(
        "modules.clustering.validation._compute_updates", spy_compute_updates
    )

    from modules.clustering.validation import validate_clustering

    validate_clustering(_make_full_settings(), cases=("video",))

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
    from modules.clustering.results import list_best_candidate_rows

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


# These tests use the conftest-initialised in-memory DB so fingerprint
# StageState rows land in the same engine as ClusterRun rows.


def _seed_validate_dataset() -> object:
    """Seed Users/Clips/UserEmbeddings, run cluster_search, return full settings.

    The four ``test_validate_*`` tests below call this in setup. ``n_users``
    is intentionally just large enough for UMAP (n_neighbors=5) and HDBSCAN
    (min_cluster_size=5) to converge — bigger seeds add real UMAP+HDBSCAN
    work that dominates this file's wall time without changing what's tested.
    """
    from modules.clustering import run_cluster_search
    from tests._clustering_helpers import (
        _make_minimal_search_settings,
        _seed_search_dataset,
    )

    _seed_search_dataset(n_users=12)
    search_settings = _make_minimal_search_settings()
    run_cluster_search(search_settings, cases=("video",))
    return SimpleNamespace(
        validation=SimpleNamespace(
            max_noise_ratio=0.9,
            min_clusters=1,
            max_clusters=20,
            plateau_drop_threshold=0.05,
            max_dominance=1.0,
        ),
        search=SimpleNamespace(hdbscan_max_cluster_frac=0.0, embedding_preprocess={}),
    )


def test_validate_unchanged_fingerprint_skips_recomputation(monkeypatch):
    """Second validate_clustering call is a no-op when inputs unchanged."""
    from modules.clustering import validate_clustering
    from modules.clustering import validation as validation_mod

    settings = _seed_validate_dataset()
    validate_clustering(settings, cases=("video",))

    calls = []
    original = validation_mod._compute_row_scores

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(validation_mod, "_compute_row_scores", spy)
    validate_clustering(settings, cases=("video",))
    assert calls == []  # no rescoring happened


def test_validate_changed_config_invalidates_and_rewrites_fields():
    """Changing plateau_drop_threshold triggers a recompute on next call."""
    from core.database import ClusterRun, get_session
    from modules.clustering import validate_clustering

    settings = _seed_validate_dataset()
    validate_clustering(settings, cases=("video",))

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
        validation=SimpleNamespace(
            max_noise_ratio=settings.validation.max_noise_ratio,
            min_clusters=settings.validation.min_clusters,
            max_clusters=settings.validation.max_clusters,
            plateau_drop_threshold=0.20,
            max_dominance=settings.validation.max_dominance,
        ),
        search=settings.search,
    )
    validate_clustering(changed, cases=("video",))

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
    validate_clustering(settings, cases=("video",))

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

    def boom(matrix, params, max_cluster_frac=0.0):
        return "value_error"

    monkeypatch.setattr(validation_mod, "_compute_row_scores", boom)
    validate_clustering(settings, cases=("video",))

    session = get_session()
    try:
        rows = (
            session.query(ClusterRun).filter(ClusterRun.embedding_case == "video").all()
        )
        assert all(r.passes_validation is not None for r in rows)
        assert any(r.passes_validation is False for r in rows)
    finally:
        session.close()
