import inspect
from types import SimpleNamespace

from modules.clustering import search as cs_mod
from modules.clustering.search import _load_grid


def test_run_cluster_search_accepts_settings():
    sig = inspect.signature(cs_mod.run_cluster_search)
    assert "settings" in sig.parameters


def _full_settings(**overrides):
    """Build a SearchSettings-shaped namespace for _load_grid."""
    base = dict(
        umap_n_components=[20],
        umap_n_neighbors=[10],
        umap_min_dist=[0.0],
        umap_metrics=["cosine"],
        umap2d_n_neighbors=15,
        umap2d_min_dist=0.1,
        umap2d_metrics=["cosine"],
        hdbscan_min_cluster_size=[15],
        hdbscan_min_samples=[],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_load_grid_empty_min_samples_yields_none():
    """Empty hdbscan_min_samples preserves current behavior (single combo, min_samples=None)."""
    combos = _load_grid(_full_settings(), cases=["video"])
    assert len(combos) == 1
    assert combos[0]["hdbscan_min_samples"] is None


def test_load_grid_enumerates_min_samples():
    """Non-empty hdbscan_min_samples expands the cartesian product."""
    settings = _full_settings(hdbscan_min_samples=[5, 10, 15])
    combos = _load_grid(settings, cases=["video"])
    assert len(combos) == 3
    assert sorted(c["hdbscan_min_samples"] for c in combos) == [5, 10, 15]


def test_load_grid_full_cartesian_includes_min_samples():
    """All axes multiply together including hdbscan_min_samples."""
    settings = _full_settings(
        umap_n_components=[15, 20],
        hdbscan_min_cluster_size=[13, 15],
        hdbscan_min_samples=[5, 10],
    )
    combos = _load_grid(settings, cases=["video"])
    assert len(combos) == 2 * 2 * 2  # n_components × min_cluster_size × min_samples
