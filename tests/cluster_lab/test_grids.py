from __future__ import annotations

from scripts.cluster_lab import db as cdb
from scripts.cluster_lab.grids import ALL_GRIDS, GRID_REGISTRY


def test_kmeans_smoke_one_config() -> None:
    items = list(GRID_REGISTRY["kmeans_smoke"]())
    assert len(items) == 1
    algo, reducer, cfg = items[0]
    assert algo == "kmeans" and reducer == "none"
    assert cfg["k"] == 10


def test_each_grid_has_unique_hashes() -> None:
    for name, fn in GRID_REGISTRY.items():
        hashes = set()
        n = 0
        for algo, reducer, cfg in fn():
            h = cdb.config_hash({**cfg, "algorithm": algo, "reducer": reducer})
            assert h not in hashes, f"collision in {name}"
            hashes.add(h)
            n += 1
        assert n > 0, f"empty grid {name}"


def test_all_grids_in_registry() -> None:
    for g in ALL_GRIDS:
        assert g in GRID_REGISTRY


def test_grid_counts_summary() -> None:
    """Print grid sizes for visibility; assert reasonable ranges."""
    sizes = {name: sum(1 for _ in fn()) for name, fn in GRID_REGISTRY.items()}
    # umap_extension is the big one (3 nn × 5 nc × 4 md × 3 met × 3 mcs × 2 csm × 2 hdm = 2160)
    assert sizes["umap_extension"] == 3 * 5 * 4 * 3 * 3 * 2 * 2
    assert sizes["seed_stability"] == 8 * 10
    assert sizes["min_samples_sweep"] == 5 * 10
    assert sizes["min_cluster_size_extension"] == 4 * 2 * 3 * 2 * 2 * 2
    assert sizes["kmeans"] == 13 * 2
    assert sizes["gmm"] == 6 * 4
    assert sizes["agglomerative"] == 6 * 3 + 6
    assert sizes["spectral"] == 6 * 2
    assert sizes["pca_hdbscan"] == 4 * 3 * 2
    assert sizes["hdbscan_direct_normalized"] == 3 * 2 * 3
