"""Tests for the mega-cluster fixes: embedding preprocessing, the eom
max_cluster_size cap, and the validation dominance guard."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.database import Base, ClusterRun, get_engine, get_session

# ── preprocess_matrix ─────────────────────────────────────────────────────────


def test_preprocess_none_is_identity():
    from modules.clustering.core import preprocess_matrix

    m = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert np.array_equal(preprocess_matrix(m, "none"), m)


def test_preprocess_center_zeros_column_mean():
    from modules.clustering.core import preprocess_matrix

    m = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]], dtype=np.float64)
    out = preprocess_matrix(m, "center")
    assert np.allclose(out.mean(axis=0), 0.0)
    # spread (relative geometry) is preserved by centering
    assert np.allclose(out, m - m.mean(axis=0))


def test_preprocess_standardize_unit_variance():
    from modules.clustering.core import preprocess_matrix

    rng = np.random.default_rng(0)
    m = rng.standard_normal((50, 4)) * np.array([1.0, 5.0, 0.1, 20.0])
    out = preprocess_matrix(m, "standardize")
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-9)


def test_preprocess_standardize_handles_zero_variance_column():
    from modules.clustering.core import preprocess_matrix

    m = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]], dtype=np.float64)
    out = preprocess_matrix(m, "standardize")
    assert np.all(np.isfinite(out))  # no divide-by-zero blowup
    assert np.allclose(out[:, 1], 0.0)


# ── compute_clusters max_cluster_frac → HDBSCAN max_cluster_size ────────────────


def test_compute_clusters_translates_frac_to_max_cluster_size(monkeypatch):
    import modules.clustering.core as core

    captured: dict = {}

    class FakeUMAP:
        def __init__(self, **kw):
            self._nc = kw.get("n_components")

        def fit_transform(self, m):
            nc = self._nc or m.shape[1]
            return m[:, :nc] if nc <= m.shape[1] else m

    class FakeHDBSCAN:
        def __init__(self, **kw):
            captured.update(kw)

        def fit_predict(self, m):
            self.probabilities_ = np.ones(m.shape[0], dtype=np.float32)
            return np.zeros(m.shape[0], dtype=int)

    monkeypatch.setattr(core, "UMAP", FakeUMAP)
    monkeypatch.setattr(core.hdbscan, "HDBSCAN", FakeHDBSCAN)

    m = np.random.default_rng(1).standard_normal((40, 6)).astype(np.float32)
    core.compute_clusters(
        m, core.ClusterParams(umap_n_components=3, hdbscan_max_cluster_frac=0.25)
    )
    assert captured["max_cluster_size"] == 10  # round(0.25 * 40)

    captured.clear()
    core.compute_clusters(
        m, core.ClusterParams(umap_n_components=3, hdbscan_max_cluster_frac=0.0)
    )
    assert captured["max_cluster_size"] == 0  # disabled


# ── validation dominance guard ─────────────────────────────────────────────────


def _seed_run(session, case, **over):
    base = dict(
        embedding_case=case,
        umap_n_components=3,
        umap_n_neighbors=5,
        umap_min_dist=0.0,
        umap_metric="cosine",
        umap2d_n_neighbors=5,
        umap2d_min_dist=0.1,
        umap2d_metric="cosine",
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=None,
        hdbscan_cluster_selection_method="eom",
        hdbscan_metric="euclidean",
        random_state=42,
        n_clusters=4,
        noise_ratio=0.0,
        min_size=5,
        median_size=7,
        max_size=8,
    )
    base.update(over)
    row = ClusterRun(**base)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row.id


def test_dominance_guard_rejects_dominant_run(monkeypatch):
    from modules.clustering import validation as vmod

    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        session.query(ClusterRun).delete()
        session.commit()
        # n_total=30: dominant max_size=20 → 0.667 > 0.4; balanced max_size=8 → 0.267
        dominant_id = _seed_run(
            session, "video", max_size=20, hdbscan_min_cluster_size=5
        )
        balanced_id = _seed_run(
            session, "video", max_size=8, hdbscan_min_cluster_size=6
        )
    finally:
        session.close()

    # avoid real UMAP/HDBSCAN scoring for the passing row
    monkeypatch.setattr(vmod, "_compute_row_scores", lambda *a, **k: (0.5, 0.4))

    settings = SimpleNamespace(
        max_noise_ratio=0.9,
        min_clusters=1,
        max_clusters=20,
        max_dominance=0.4,
    )
    matrix = np.ones((30, 6), dtype=np.float32)
    updates = vmod._compute_updates("video", matrix, settings)

    assert updates[dominant_id]["passes_validation"] is False
    assert updates[balanced_id]["passes_validation"] is True


def test_dominance_guard_disabled_keeps_rounded_lone_cluster(monkeypatch):
    """max_dominance=1.0 disables the guard: rounding in the stored noise_ratio
    must not let a lone cluster's reconstructed dominance tip past 1.0."""
    from modules.clustering import validation as vmod

    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        session.query(ClusterRun).delete()
        session.commit()
        # 20 noisy of 30 → stored noise_ratio 0.6667; one 10-user cluster.
        # Naive 10 / (30 * (1 - 0.6667)) = 1.0001 > 1.0 would wrongly reject.
        run_id = _seed_run(
            session,
            "video",
            n_clusters=1,
            noise_ratio=round(20 / 30, 4),
            max_size=10,
        )
    finally:
        session.close()

    monkeypatch.setattr(vmod, "_compute_row_scores", lambda *a, **k: (0.5, 0.4))
    settings = SimpleNamespace(
        max_noise_ratio=0.9,
        min_clusters=1,
        max_clusters=20,
        max_dominance=1.0,
    )
    matrix = np.ones((30, 6), dtype=np.float32)
    updates = vmod._compute_updates("video", matrix, settings)

    assert updates[run_id]["passes_validation"] is True
