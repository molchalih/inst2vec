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
        row.in_current_grid = 1
        s.commit()
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
        row.in_current_grid = 1
        s.commit()
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
        row.in_current_grid = 1
        s.commit()
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
        row.in_current_grid = 1
        s.commit()
        row_id = row.id

    from modules.cluster_validation import _phase_filter
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_filter(s, "video")
            assert s.get(ClusterRun, row_id).disqualified == 1


def test_filter_ignores_stale_rows():
    """_phase_filter must not touch rows with in_current_grid=0."""
    eng = _make_engine()
    env = {
        "VALIDATION_MAX_NOISE_RATIO": "0.3",
        "VALIDATION_MIN_CLUSTERS": "3",
        "VALIDATION_MAX_CLUSTERS": "20",
    }
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=5)
        row.in_current_grid = 0
        row.disqualified = 1
        s.commit()
        row_id = row.id

    from modules.cluster_validation import _phase_filter
    with patch.dict(os.environ, env):
        with Session(eng) as s:
            _phase_filter(s, "video")
            assert s.get(ClusterRun, row_id).disqualified == 1


# --- Phase 2: score ---

def test_phase_score_populates_dbcv_and_silhouette(monkeypatch):
    eng = _make_engine()

    def _get_session():
        return Session(eng)

    monkeypatch.setattr("modules.cluster_validation.get_session", _get_session)

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


def test_phase_score_skips_already_scored_rows(monkeypatch):
    eng = _make_engine()

    def _get_session():
        return Session(eng)

    monkeypatch.setattr("modules.cluster_validation.get_session", _get_session)

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
        assert updated.dbcv == pytest.approx(0.99)
        assert updated.silhouette == pytest.approx(0.88)


# --- Plateau neighbor logic ---

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
        assert neighbor.id in result_ids
        assert non_neighbor.id not in result_ids
        assert two_away.id not in result_ids


def test_find_param_neighbors_categorical_any_other_value():
    from modules.cluster_validation import _find_param_neighbors
    eng = _make_engine()

    with Session(eng) as s:
        target = _insert_run(s, umap_metric="cosine", random_state=1)
        neighbor = _insert_run(s, umap_metric="euclidean", random_state=1)
        non_neighbor = _insert_run(s, umap_metric="euclidean", umap_n_components=10, random_state=1)

        result = _find_param_neighbors(target, [neighbor, non_neighbor])
        result_ids = {r.id for r in result}
        assert neighbor.id in result_ids
        assert non_neighbor.id not in result_ids


# --- Phase 3: plateau ---

def test_phase_plateau_uses_dbcv_of_neighbors():
    """param_plateau_score is the mean DBCV of grid-adjacent neighbors."""
    eng = _make_engine()
    with Session(eng) as s:
        r1 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=10, random_state=1)
        r1.in_current_grid = 1
        r1.dbcv = 0.8
        r2 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=15, random_state=1)
        r2.in_current_grid = 1
        r2.dbcv = 0.6
        s.commit()
        id1 = r1.id

    from modules.cluster_validation import _phase_plateau
    with Session(eng) as s:
        _phase_plateau(s, "video")
        r1_updated = s.get(ClusterRun, id1)
        # r1's only neighbor is r2 (dbcv=0.6), so plateau = 0.6
        assert r1_updated.param_plateau_score == pytest.approx(0.6, abs=1e-5)


def test_phase_plateau_covers_all_scored_rows():
    """Plateau is computed for every qualifying row, not just a top-N subset."""
    eng = _make_engine()
    with Session(eng) as s:
        rows = []
        for i, nc in enumerate([10, 15, 20]):
            r = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                            umap_n_components=nc, random_state=1)
            r.in_current_grid = 1
            r.dbcv = 0.5 + i * 0.1
            rows.append(r)
        s.commit()
        ids = [r.id for r in rows]

    from modules.cluster_validation import _phase_plateau
    with Session(eng) as s:
        _phase_plateau(s, "video")
        for rid in ids:
            assert s.get(ClusterRun, rid).param_plateau_score is not None


def test_phase_plateau_no_neighbors_falls_back_to_own_dbcv():
    """An isolated config with no grid-adjacent neighbors gets plateau = own dbcv."""
    eng = _make_engine()
    with Session(eng) as s:
        r = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                        umap_n_components=10, random_state=1)
        r.in_current_grid = 1
        r.dbcv = 0.75
        s.commit()
        rid = r.id

    from modules.cluster_validation import _phase_plateau
    with Session(eng) as s:
        _phase_plateau(s, "video")
        updated = s.get(ClusterRun, rid)
        # No neighbors → fallback to own dbcv → drop = 0 → not rejected by filter
        assert updated.param_plateau_score == pytest.approx(0.75, abs=1e-5)


def test_phase_plateau_skips_already_set():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                          umap_n_components=10, random_state=1)
        row.in_current_grid = 1
        row.dbcv = 0.9
        row.param_plateau_score = 0.5  # already set
        s.commit()
        row_id = row.id

    from modules.cluster_validation import _phase_plateau
    with Session(eng) as s:
        _phase_plateau(s, "video")
        assert s.get(ClusterRun, row_id).param_plateau_score == pytest.approx(0.5)


# --- _select_best ---

def test_select_best_picks_highest_dbcv():
    """_select_best selects the run with highest DBCV among plateau survivors."""
    eng = _make_engine()
    with Session(eng) as s:
        r1 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=10, random_state=1)
        r1.in_current_grid = 1
        r1.dbcv = 0.9
        r1.param_plateau_score = 0.88  # drop=0.02, within threshold
        r2 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=15, random_state=1)
        r2.in_current_grid = 1
        r2.dbcv = 0.6
        r2.param_plateau_score = 0.59  # drop=0.01, within threshold
        s.commit()
        id1 = r1.id

    from modules.cluster_validation import _select_best
    with patch.dict(os.environ, {"VALIDATION_PLATEAU_DROP_THRESHOLD": "0.05"}):
        with Session(eng) as s:
            result = _select_best(s, "video")
            assert result is not None
            assert result.id == id1


def test_select_best_rejects_sharp_peak_by_plateau_filter():
    """A run whose DBCV far exceeds its neighborhood mean is rejected as a sharp peak."""
    eng = _make_engine()
    with Session(eng) as s:
        # r1 has higher DBCV but is a sharp peak
        r1 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=10, random_state=1)
        r1.in_current_grid = 1
        r1.dbcv = 0.9
        r1.param_plateau_score = 0.3   # drop=0.6, far exceeds threshold → rejected
        # r2 has lower DBCV but is on a stable plateau
        r2 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=15, random_state=1)
        r2.in_current_grid = 1
        r2.dbcv = 0.7
        r2.param_plateau_score = 0.68  # drop=0.02, within threshold → survives
        s.commit()
        id2 = r2.id

    from modules.cluster_validation import _select_best
    with patch.dict(os.environ, {"VALIDATION_PLATEAU_DROP_THRESHOLD": "0.05"}):
        with Session(eng) as s:
            result = _select_best(s, "video")
            assert result is not None
            assert result.id == id2


def test_select_best_falls_back_when_all_rejected():
    """When all runs fail the plateau filter, fall back to highest DBCV."""
    eng = _make_engine()
    with Session(eng) as s:
        r1 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=10, random_state=1)
        r1.in_current_grid = 1
        r1.dbcv = 0.9
        r1.param_plateau_score = 0.1  # sharp peak — would be rejected
        r2 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=15, random_state=1)
        r2.in_current_grid = 1
        r2.dbcv = 0.7
        r2.param_plateau_score = 0.2  # also sharp — would be rejected
        s.commit()
        id1 = r1.id

    from modules.cluster_validation import _select_best
    with patch.dict(os.environ, {"VALIDATION_PLATEAU_DROP_THRESHOLD": "0.05"}):
        with Session(eng) as s:
            result = _select_best(s, "video")
            assert result is not None
            assert result.id == id1  # fallback: highest DBCV wins


def test_select_best_returns_none_when_no_eligible_runs():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5)
        row.in_current_grid = 1
        # no dbcv set → not eligible
        s.commit()

    from modules.cluster_validation import _select_best
    with Session(eng) as s:
        assert _select_best(s, "video") is None


def test_select_best_ignores_disqualified():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, disqualified=1, noise_ratio=0.1, n_clusters=5,
                          umap_n_components=10, random_state=1)
        row.in_current_grid = 1
        row.dbcv = 0.9
        row.param_plateau_score = 0.88
        s.commit()

    from modules.cluster_validation import _select_best
    with Session(eng) as s:
        assert _select_best(s, "video") is None


def test_select_best_ignores_cluster_override_env(monkeypatch):
    """CLUSTER_OVERRIDE_* is unsupported; selection uses plateau + DBCV only."""
    eng = _make_engine()
    with Session(eng) as s:
        r1 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=10, random_state=1)
        r1.in_current_grid = 1
        r1.dbcv = 0.9
        r1.param_plateau_score = 0.88
        r2 = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                         umap_n_components=15, random_state=1)
        r2.in_current_grid = 1
        r2.dbcv = 0.1
        r2.param_plateau_score = 0.09
        s.commit()
        low_id = r2.id

    monkeypatch.setenv("CLUSTER_OVERRIDE_VIDEO", str(low_id))
    from modules.cluster_validation import _select_best
    with Session(eng) as s:
        result = _select_best(s, "video")
        assert result is not None
        assert result.id != low_id
        assert result.dbcv == 0.9


def test_select_best_delegates_to_shared_selector(monkeypatch):
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, disqualified=0, noise_ratio=0.1, n_clusters=5,
                          umap_n_components=10, random_state=1)
        row.in_current_grid = 1
        row.dbcv = 0.9
        row.param_plateau_score = 0.88
        s.commit()
        row_id = row.id

    calls = {}

    def fake_pick(rows, threshold=None):
        calls["count"] = len(rows)
        return rows[0]

    monkeypatch.setattr("modules.cluster_validation.pick_best_cluster_run", fake_pick)

    from modules.cluster_validation import _select_best
    with Session(eng) as s:
        result = _select_best(s, "video")

    assert result is not None
    assert result.id == row_id
    assert calls["count"] == 1


# --- Config hash ---

def test_compute_validation_config_hash_is_deterministic():
    from modules.cluster_validation import _compute_validation_config_hash
    env = {
        "VALIDATION_MAX_NOISE_RATIO": "0.3",
        "VALIDATION_MIN_CLUSTERS": "3",
        "VALIDATION_MAX_CLUSTERS": "20",
        "VALIDATION_PLATEAU_DROP_THRESHOLD": "0.05",
    }
    with patch.dict(os.environ, env, clear=False):
        h1 = _compute_validation_config_hash()
        h2 = _compute_validation_config_hash()
    assert h1 == h2
    assert len(h1) == 16


def test_compute_validation_config_hash_changes_with_env():
    from modules.cluster_validation import _compute_validation_config_hash
    base_env = {
        "VALIDATION_MAX_NOISE_RATIO": "0.3",
        "VALIDATION_MIN_CLUSTERS": "3",
        "VALIDATION_MAX_CLUSTERS": "20",
        "VALIDATION_PLATEAU_DROP_THRESHOLD": "0.05",
    }
    changed_env = {**base_env, "VALIDATION_PLATEAU_DROP_THRESHOLD": "0.10"}
    with patch.dict(os.environ, base_env, clear=False):
        h_base = _compute_validation_config_hash()
    with patch.dict(os.environ, changed_env, clear=False):
        h_changed = _compute_validation_config_hash()
    assert h_base != h_changed


def test_compute_validation_config_hash_uses_defaults_when_env_absent():
    from modules.cluster_validation import _compute_validation_config_hash
    keys = [
        "VALIDATION_MAX_NOISE_RATIO", "VALIDATION_MIN_CLUSTERS", "VALIDATION_MAX_CLUSTERS",
        "VALIDATION_PLATEAU_DROP_THRESHOLD",
    ]
    original = {k: os.environ.pop(k) for k in keys if k in os.environ}
    try:
        h1 = _compute_validation_config_hash()
        h2 = _compute_validation_config_hash()
        assert h1 == h2
    finally:
        os.environ.update(original)


# --- Stale row invalidation ---

def test_cluster_run_has_validation_config_hash_field():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s)
        row.validation_config_hash = "abc123def456abcd"
        s.commit()
        s.refresh(row)
        assert row.validation_config_hash == "abc123def456abcd"


def test_invalidate_stale_rows_nulls_plateau_when_hash_differs():
    """Invalidation nulls param_plateau_score but preserves dbcv and silhouette."""
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=5)
        row.in_current_grid = 1
        row.validation_config_hash = "oldhash00000000"
        row.param_plateau_score = 0.7
        row.dbcv = 0.9
        row.silhouette = 0.85
        s.commit()
        row_id = row.id

    from modules.cluster_validation import _invalidate_stale_rows
    with Session(eng) as s:
        _invalidate_stale_rows(s, "video", "newhash00000000")
        updated = s.get(ClusterRun, row_id)
        assert updated.param_plateau_score is None
        assert updated.dbcv == pytest.approx(0.9)
        assert updated.silhouette == pytest.approx(0.85)
        assert updated.validation_config_hash == "newhash00000000"


def test_invalidate_stale_rows_treats_null_hash_as_stale():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=5)
        row.in_current_grid = 1
        row.validation_config_hash = None
        row.param_plateau_score = 0.4
        s.commit()
        row_id = row.id

    from modules.cluster_validation import _invalidate_stale_rows
    with Session(eng) as s:
        _invalidate_stale_rows(s, "video", "currenthash0000")
        updated = s.get(ClusterRun, row_id)
        assert updated.param_plateau_score is None
        assert updated.validation_config_hash == "currenthash0000"


def test_invalidate_stale_rows_skips_matching_hash():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=5)
        row.in_current_grid = 1
        row.validation_config_hash = "currenthash0000"
        row.param_plateau_score = 0.7
        s.commit()
        row_id = row.id

    from modules.cluster_validation import _invalidate_stale_rows
    with Session(eng) as s:
        _invalidate_stale_rows(s, "video", "currenthash0000")
        updated = s.get(ClusterRun, row_id)
        assert updated.param_plateau_score == pytest.approx(0.7)


def test_invalidate_stale_rows_ignores_non_current_grid_rows():
    eng = _make_engine()
    with Session(eng) as s:
        row = _insert_run(s, noise_ratio=0.1, n_clusters=5)
        row.in_current_grid = 0
        row.validation_config_hash = "oldhash00000000"
        row.param_plateau_score = 0.7
        s.commit()
        row_id = row.id

    from modules.cluster_validation import _invalidate_stale_rows
    with Session(eng) as s:
        _invalidate_stale_rows(s, "video", "newhash00000000")
        updated = s.get(ClusterRun, row_id)
        assert updated.param_plateau_score == pytest.approx(0.7)
        assert updated.validation_config_hash == "oldhash00000000"


def test_invalidate_stale_rows_only_affects_matching_case():
    eng = _make_engine()
    with Session(eng) as s:
        video_row = _insert_run(s, embedding_case="video", noise_ratio=0.1, n_clusters=5,
                                umap_n_components=10, random_state=1)
        video_row.in_current_grid = 1
        video_row.validation_config_hash = "oldhash00000000"
        video_row.param_plateau_score = 0.8

        audio_row = _insert_run(s, embedding_case="audio", noise_ratio=0.1, n_clusters=5,
                                umap_n_components=10, random_state=1)
        audio_row.in_current_grid = 1
        audio_row.validation_config_hash = "oldhash00000000"
        audio_row.param_plateau_score = 0.9
        s.commit()
        vid_id, aud_id = video_row.id, audio_row.id

    from modules.cluster_validation import _invalidate_stale_rows
    with Session(eng) as s:
        _invalidate_stale_rows(s, "video", "newhash00000000")
        vid = s.get(ClusterRun, vid_id)
        aud = s.get(ClusterRun, aud_id)
        assert vid.param_plateau_score is None
        assert aud.param_plateau_score == pytest.approx(0.9)


# --- Orchestration ---

def test_validate_clustering_phase_order(monkeypatch):
    """invalidate → filter → score → plateau → select, no bootstrap or composite."""
    from unittest.mock import MagicMock
    sequence = []

    def fake_invalidate(session, case, current_hash):
        sequence.append(("invalidate", case, current_hash))

    def fake_load_matrix(case):
        if case == "video":
            return (np.ones((5, 10), dtype=np.float32), list(range(5)))
        return (np.zeros((0, 10), dtype=np.float32), [])

    def fake_phase_filter(session, case):
        sequence.append(("filter", case))

    def fake_phase_score(session, case, matrix):
        sequence.append(("score", case))

    def fake_phase_plateau(session, case):
        sequence.append(("plateau", case))

    def fake_select_best(session, case):
        return None

    monkeypatch.setattr("modules.cluster_validation._invalidate_stale_rows", fake_invalidate)
    monkeypatch.setattr("modules.cluster_validation.load_user_matrix", fake_load_matrix)
    monkeypatch.setattr("modules.cluster_validation._phase_filter", fake_phase_filter)
    monkeypatch.setattr("modules.cluster_validation._phase_score", fake_phase_score)
    monkeypatch.setattr("modules.cluster_validation._phase_plateau", fake_phase_plateau)
    monkeypatch.setattr("modules.cluster_validation._select_best", fake_select_best)
    monkeypatch.setattr("modules.cluster_validation.get_session", lambda: MagicMock())

    from modules.cluster_validation import validate_clustering
    validate_clustering()

    video_seq = [(op, c) for op, c, *_ in sequence if c == "video"]
    assert video_seq, "no calls recorded for video case"

    ops = [op for op, _c in video_seq]
    assert "bootstrap" not in ops, "bootstrap must not be called"
    assert "composite" not in ops, "composite must not be called"

    assert ("invalidate", "video") in video_seq
    assert ("filter", "video") in video_seq
    assert ("score", "video") in video_seq
    assert ("plateau", "video") in video_seq

    invalidate_idx = video_seq.index(("invalidate", "video"))
    filter_idx = video_seq.index(("filter", "video"))
    assert invalidate_idx < filter_idx, "invalidation must come before filter"

    video_hash = next(h for op, c, h in sequence if c == "video" and op == "invalidate")
    assert len(video_hash) == 16
    assert all(ch in "0123456789abcdef" for ch in video_hash)


def test_phase_score_uses_thread_pool_when_workers_gt_one(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    eng = _make_engine()

    def _get_session():
        return Session(eng)

    monkeypatch.setattr("modules.cluster_validation.get_session", _get_session)

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
            row = _insert_run(s, umap_n_components=nc, disqualified=0)
            row.in_current_grid = 1
            row.dbcv = None
            s.commit()

    monkeypatch.setattr("modules.cluster_validation.ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr("modules.cluster_validation.compute_clusters", counting_compute)

    from modules.cluster_validation import _phase_score

    matrix = np.ones((20, 8), dtype=np.float32)
    env = {"CLUSTERING_GRID_WORKERS": "4"}

    with patch.dict(os.environ, env, clear=False):
        with Session(eng) as session:
            _phase_score(session, "video", matrix)

    assert calls["n"] == 2
    assert max_workers_seen == [4]
