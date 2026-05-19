"""Smoke test: production config.toml [search] grid loads and expands within budget.

Pins the post-tuning grid against accidental config drift. Budget is set to
500 to absorb future single-axis additions; alarm only if it explodes.
"""

import tomllib
from pathlib import Path

import pytest

from core.config import SearchSettings
from modules.clustering.search import _load_grid

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def search_settings() -> SearchSettings:
    with open(PROJECT_ROOT / "config.toml", "rb") as f:
        raw = tomllib.load(f)
    return SearchSettings(**raw["search"])


def test_production_search_grid_loads_and_expands(search_settings: SearchSettings):
    # Non-empty field assertions give a precise failure signal if a key
    # is dropped from config.toml; the grid-size band would also fire,
    # but with a less useful "grid size 0" error.
    assert search_settings.umap_n_components, "umap_n_components must be non-empty"
    assert search_settings.umap_n_neighbors, "umap_n_neighbors must be non-empty"
    assert search_settings.umap_min_dist, "umap_min_dist must be non-empty"
    assert search_settings.umap_metrics, "umap_metrics must be non-empty"
    assert search_settings.hdbscan_min_cluster_size, (
        "hdbscan_min_cluster_size must be non-empty"
    )
    assert search_settings.hdbscan_selection, "hdbscan_selection must be non-empty"

    combos = _load_grid(search_settings, cases=["sandwich"])
    assert 100 <= len(combos) <= 500, (
        f"grid size {len(combos)} outside expected production band [100, 500]"
    )


def test_production_grid_drops_known_bad_values(search_settings: SearchSettings):
    """Values empirically shown to fail in cluster_param_analysis must be excluded.

    Empirical failure rates live in config.toml's [search] comment; the
    assertion messages here intentionally point readers there rather than
    duplicate the numbers in two places.
    """
    assert "euclidean" not in search_settings.umap_metrics, (
        "umap_metric=euclidean empirically excluded — see config.toml [search]"
    )
    assert "leaf" not in search_settings.hdbscan_selection, (
        "leaf produces 25+ clusters, outside 8-14 target band"
    )
    assert 0.05 not in [float(x) for x in search_settings.umap_min_dist], (
        "umap_min_dist=0.05 empirically excluded — see config.toml [search]"
    )
    assert 15 not in search_settings.umap_n_neighbors, (
        "umap_n_neighbors=15 dropped — lower quality than 10, shifts cluster count"
    )
    assert 30 not in search_settings.umap_n_neighbors, (
        "umap_n_neighbors=30 empirically excluded — see config.toml [search]"
    )
    assert 25 not in search_settings.hdbscan_min_cluster_size, (
        "hdbscan_min_cluster_size=25 empirically excluded — see config.toml [search]"
    )
