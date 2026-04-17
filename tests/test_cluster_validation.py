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


# --- Phase 3: composite ---

def test_phase_composite_weights():
    eng = _make_engine()

    with Session(eng) as s:
        r1 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=10, random_state=1)
        r1.dbcv = 1.0
        r1.silhouette = 1.0
        r2 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=15, random_state=1)
        r2.dbcv = 0.0
        r2.silhouette = 0.0
        s.commit()
        id1, id2 = r1.id, r2.id

    from modules.cluster_validation import _phase_composite
    with Session(eng) as s:
        _phase_composite(s, "video")
        top = s.get(ClusterRun, id1)
        bottom = s.get(ClusterRun, id2)
        # No bootstrap yet → stability=0 for both; top should score 0.5*1 + 0.2*1 + 0.3*0 = 0.7
        assert top.composite_score == pytest.approx(0.7, abs=1e-5)
        assert bottom.composite_score == pytest.approx(0.0, abs=1e-5)


def test_phase_composite_incorporates_bootstrap_stability():
    eng = _make_engine()

    with Session(eng) as s:
        r1 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=10, random_state=1)
        r1.dbcv = 0.8
        r1.silhouette = 0.8
        r1.bootstrap_stability = 1.0  # high stability
        r2 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=15, random_state=1)
        r2.dbcv = 0.8
        r2.silhouette = 0.8
        r2.bootstrap_stability = 0.0  # no stability
        s.commit()
        id1, id2 = r1.id, r2.id

    from modules.cluster_validation import _phase_composite
    with Session(eng) as s:
        _phase_composite(s, "video")
        r1_updated = s.get(ClusterRun, id1)
        r2_updated = s.get(ClusterRun, id2)
        # dbcv and silhouette are equal → norm=0 for both; only stability differs
        # stab_norm: r1=1.0, r2=0.0 → composite: r1=0.3, r2=0.0
        assert r1_updated.composite_score > r2_updated.composite_score
        assert r1_updated.composite_score == pytest.approx(0.3, abs=1e-5)
        assert r2_updated.composite_score == pytest.approx(0.0, abs=1e-5)


# --- Phase 4: bootstrap ---

def test_ari_non_noise_excludes_noise_from_both():
    from modules.cluster_validation import _ari_non_noise
    # Points 0,1,2 non-noise in both; point 3 noise in a; point 4 noise in b
    a = np.array([0, 0, 1, -1,  1])
    b = np.array([0, 0, 1,  1, -1])
    # Only indices 0,1,2 count (non-noise in BOTH)
    ari = _ari_non_noise(a, b)
    assert ari == pytest.approx(1.0)  # perfect agreement on non-noise subset


def test_ari_non_noise_returns_zero_when_no_non_noise_overlap():
    from modules.cluster_validation import _ari_non_noise
    a = np.array([-1, -1, -1])
    b = np.array([0, 1, 2])
    assert _ari_non_noise(a, b) == pytest.approx(0.0)


def test_phase_bootstrap_populates_stability():
    eng = _make_engine()
    rng = np.random.default_rng(0)
    matrix = np.vstack([
        rng.normal(8.0, 0.2, (40, 30)).astype(np.float32),
        rng.normal(-8.0, 0.2, (40, 30)).astype(np.float32),
    ])

    with Session(eng) as s:
        row = _insert_run(s, disqualified=0, noise_ratio=0.0, n_clusters=2,
                          umap_n_components=5, hdbscan_min_cluster_size=10)
        row.dbcv = 0.9
        row.composite_score = 0.8
        s.commit()
        row_id = row.id

    env = {"VALIDATION_TOP_N_BOOTSTRAP": "5", "VALIDATION_BOOTSTRAP_N": "3"}
    from modules.cluster_validation import _phase_bootstrap
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_bootstrap(s, "video", matrix)
            updated = s.get(ClusterRun, row_id)
            assert updated.bootstrap_stability is not None
            assert updated.bootstrap_n_runs == 3


def test_phase_bootstrap_skips_already_set():
    eng = _make_engine()
    matrix = np.ones((80, 30), dtype=np.float32)

    with Session(eng) as s:
        row = _insert_run(s, disqualified=0, noise_ratio=0.0, n_clusters=2,
                          umap_n_components=5, hdbscan_min_cluster_size=10)
        row.dbcv = 0.9
        row.bootstrap_stability = 0.75
        row.bootstrap_n_runs = 10
        s.commit()
        row_id = row.id

    env = {"VALIDATION_TOP_N_BOOTSTRAP": "5", "VALIDATION_BOOTSTRAP_N": "3"}
    from modules.cluster_validation import _phase_bootstrap
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_bootstrap(s, "video", matrix)
            updated = s.get(ClusterRun, row_id)
            assert updated.bootstrap_stability == pytest.approx(0.75)  # unchanged
            assert updated.bootstrap_n_runs == 10  # unchanged


# --- Phase 5: plateau ---

def test_find_param_neighbors_one_step_difference():
    from modules.cluster_validation import _find_param_neighbors
    eng = _make_engine()

    with Session(eng) as s:
        target = _insert_run(s, umap_n_components=10, umap_n_neighbors=15, random_state=1)
        neighbor = _insert_run(s, umap_n_components=15, umap_n_neighbors=15, random_state=1)
        non_neighbor = _insert_run(s, umap_n_components=20, umap_n_neighbors=10, random_state=1)
        two_away = _insert_run(s, umap_n_components=20, umap_n_neighbors=15, random_state=1)

        result = _find_param_neighbors(target, [neighbor, non_neighbor, two_away])
        result_ids = {r.id for r in result}
        # neighbor differs only in umap_n_components by one step (10→15 in [10,15,20])
        assert neighbor.id in result_ids
        # non_neighbor differs in two params
        assert non_neighbor.id not in result_ids
        # two_away differs only in umap_n_components but by two steps (10→20)
        assert two_away.id not in result_ids


def test_find_param_neighbors_categorical_any_other_value():
    from modules.cluster_validation import _find_param_neighbors
    eng = _make_engine()

    with Session(eng) as s:
        target = _insert_run(s, umap_metric="cosine", random_state=1)
        neighbor = _insert_run(s, umap_metric="euclidean", random_state=1)
        # differs in umap_metric (categorical) AND umap_n_components → not a neighbor
        non_neighbor = _insert_run(s, umap_metric="euclidean", umap_n_components=10, random_state=1)

        result = _find_param_neighbors(target, [neighbor, non_neighbor])
        result_ids = {r.id for r in result}
        assert neighbor.id in result_ids
        assert non_neighbor.id not in result_ids


def test_phase_plateau_populates_top_rows():
    eng = _make_engine()

    with Session(eng) as s:
        # Two rows that are neighbors (differ only in umap_n_components)
        r1 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=10, random_state=1)
        r1.composite_score = 0.9
        r2 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=15, random_state=1)
        r2.composite_score = 0.7
        s.commit()
        id1, id2 = r1.id, r2.id

    env = {"VALIDATION_TOP_N_PLATEAU": "5"}
    from modules.cluster_validation import _phase_plateau
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_plateau(s, "video")
            updated1 = s.get(ClusterRun, id1)
            updated2 = s.get(ClusterRun, id2)
            # r1's only neighbor is r2 (score=0.7)
            assert updated1.param_plateau_score == pytest.approx(0.7, abs=1e-5)
            # r2's only neighbor is r1 (score=0.9)
            assert updated2.param_plateau_score == pytest.approx(0.9, abs=1e-5)


def test_phase_plateau_skips_already_set():
    eng = _make_engine()

    with Session(eng) as s:
        row = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                          umap_n_components=10, random_state=1)
        row.composite_score = 0.9
        row.param_plateau_score = 0.5  # already set — should not be overwritten
        s.commit()
        row_id = row.id

    env = {"VALIDATION_TOP_N_PLATEAU": "5"}
    from modules.cluster_validation import _phase_plateau
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_plateau(s, "video")
            assert s.get(ClusterRun, row_id).param_plateau_score == pytest.approx(0.5)  # unchanged


# --- _select_best ---

def test_select_best_picks_highest_final_score():
    eng = _make_engine()
    with Session(eng) as s:
        r1 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=10, random_state=1)
        r1.composite_score = 0.8
        r1.param_plateau_score = 0.6  # final = 0.7*0.8 + 0.3*0.6 = 0.74
        r2 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=15, random_state=1)
        r2.composite_score = 0.6
        r2.param_plateau_score = 0.9  # final = 0.7*0.6 + 0.3*0.9 = 0.69
        s.commit()
        id1 = r1.id

    from modules.cluster_validation import _select_best
    with Session(eng) as s:
        result = _select_best(s, "video")
        assert result is not None
        assert result.id == id1


def test_select_best_returns_none_when_no_eligible_runs():
    eng = _make_engine()
    with Session(eng) as s:
        _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5)
        # no composite_score set → not eligible

    from modules.cluster_validation import _select_best
    with Session(eng) as s:
        assert _select_best(s, "video") is None


def test_select_best_ignores_disqualified():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, disqualified=1, noise_ratio=0.1, n_clusters=5,
                          umap_n_components=10, random_state=1)
        row.composite_score = 0.9
        row.param_plateau_score = 0.9
        s.commit()

    from modules.cluster_validation import _select_best
    with Session(eng) as s:
        assert _select_best(s, "video") is None


def test_select_best_env_override(monkeypatch):
    eng = _make_engine()
    with Session(eng) as s:
        r1 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=10, random_state=1)
        r1.composite_score = 0.9
        r1.param_plateau_score = 0.9  # best by score
        r2 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=15, random_state=1)
        r2.composite_score = 0.1
        r2.param_plateau_score = 0.1  # worst by score
        s.commit()
        id2 = r2.id

    monkeypatch.setenv("CLUSTER_OVERRIDE_VIDEO", str(id2))
    from modules.cluster_validation import _select_best
    with Session(eng) as s:
        result = _select_best(s, "video")
        assert result is not None
        assert result.id == id2  # forced, not the best-scoring


def test_select_best_env_override_missing_id_raises(monkeypatch):
    eng = _make_engine()
    monkeypatch.setenv("CLUSTER_OVERRIDE_VIDEO", "99999")
    from modules.cluster_validation import _select_best
    with Session(eng) as s:
        with pytest.raises(ValueError, match="CLUSTER_OVERRIDE_VIDEO"):
            _select_best(s, "video")
