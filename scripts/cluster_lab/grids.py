"""Experiment grid generators.

Each grid yields ``(algorithm, reducer, config_dict)`` tuples. Config dict
keys feed straight into runners.run() and into config_hash().
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

# Top viable corner from analyzer report.md (sandwich case, May 19 run).
# These are the (umap_n_neighbors, umap_n_components, umap_min_dist, umap_metric,
# hdbscan_min_cluster_size, hdbscan_cluster_selection_method, hdbscan_metric)
# combinations that consistently scored quality_score > 0.5.
TOP_VIABLE_CORNER: list[dict[str, Any]] = [
    {
        "umap_n_components": 20,
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.0,
        "umap_metric": "cosine",
        "hdbscan_min_cluster_size": 15,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_metric": "cosine",
    },
    {
        "umap_n_components": 10,
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.0,
        "umap_metric": "correlation",
        "hdbscan_min_cluster_size": 10,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_metric": "cosine",
    },
    {
        "umap_n_components": 20,
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.0,
        "umap_metric": "cosine",
        "hdbscan_min_cluster_size": 10,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_metric": "cosine",
    },
    {
        "umap_n_components": 35,
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.0,
        "umap_metric": "cosine",
        "hdbscan_min_cluster_size": 15,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_metric": "cosine",
    },
    {
        "umap_n_components": 40,
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.0,
        "umap_metric": "correlation",
        "hdbscan_min_cluster_size": 10,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_metric": "cosine",
    },
    {
        "umap_n_components": 40,
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.05,
        "umap_metric": "correlation",
        "hdbscan_min_cluster_size": 10,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_metric": "cosine",
    },
    {
        "umap_n_components": 30,
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.0,
        "umap_metric": "correlation",
        "hdbscan_min_cluster_size": 10,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_metric": "cosine",
    },
    {
        "umap_n_components": 30,
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.0,
        "umap_metric": "cosine",
        "hdbscan_min_cluster_size": 10,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_metric": "cosine",
    },
    {
        "umap_n_components": 35,
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.0,
        "umap_metric": "cosine",
        "hdbscan_min_cluster_size": 10,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_metric": "cosine",
    },
    {
        "umap_n_components": 10,
        "umap_n_neighbors": 10,
        "umap_min_dist": 0.05,
        "umap_metric": "cosine",
        "hdbscan_min_cluster_size": 10,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_metric": "cosine",
    },
]


def grid_kmeans_smoke() -> Iterator[tuple[str, str, dict]]:
    """Single config — KMeans k=10 normalized. Used as smoke test."""
    yield "kmeans", "none", {"k": 10, "random_state": 42, "normalized": 1}


def grid_umap_extension() -> Iterator[tuple[str, str, dict]]:
    """Fill the UMAP gaps the legacy data didn't hit."""
    new_neighbors = [5, 7, 50]
    new_n_components = [3, 5, 7, 50, 75]
    new_min_dist = [0.1, 0.25]
    for nn, nc, md, met, mcs, csm, hdm in product(
        new_neighbors,
        new_n_components,
        [0.0, 0.05, *new_min_dist],
        ["cosine", "correlation", "euclidean"],
        [10, 15, 25],
        ["eom", "leaf"],
        ["cosine", "euclidean"],
    ):
        yield (
            "hdbscan",
            "umap",
            {
                "umap_n_components": nc,
                "umap_n_neighbors": nn,
                "umap_min_dist": md,
                "umap_metric": met,
                "hdbscan_min_cluster_size": mcs,
                "hdbscan_min_samples": None,
                "hdbscan_cluster_selection_method": csm,
                "hdbscan_metric": hdm,
                "random_state": 42,
                "normalized": 0,
            },
        )


def grid_min_cluster_size_extension() -> Iterator[tuple[str, str, dict]]:
    """Extend hdbscan_min_cluster_size into 35..100 over the viable corner."""
    big_sizes = [35, 50, 75, 100]
    for mcs, csm, nc, nn, md, met in product(
        big_sizes,
        ["eom", "leaf"],
        [10, 15, 20],
        [10, 15],
        [0.0, 0.05],
        ["cosine", "correlation"],
    ):
        yield (
            "hdbscan",
            "umap",
            {
                "umap_n_components": nc,
                "umap_n_neighbors": nn,
                "umap_min_dist": md,
                "umap_metric": met,
                "hdbscan_min_cluster_size": mcs,
                "hdbscan_min_samples": None,
                "hdbscan_cluster_selection_method": csm,
                "hdbscan_metric": "cosine",
                "random_state": 42,
                "normalized": 0,
            },
        )


def grid_min_samples_sweep() -> Iterator[tuple[str, str, dict]]:
    """First-time variation of hdbscan_min_samples on the top viable corner."""
    samples = [1, 5, 10, 15, 25]
    for ms, base in product(samples, TOP_VIABLE_CORNER):
        cfg = {
            **base,
            "hdbscan_min_samples": ms,
            "random_state": 42,
            "normalized": 0,
        }
        yield "hdbscan", "umap", cfg


def grid_seed_stability() -> Iterator[tuple[str, str, dict]]:
    """Replay top-10 viable corner configs across 8 seeds."""
    seeds = [0, 1, 2, 7, 13, 42, 101, 202]
    for s, base in product(seeds, TOP_VIABLE_CORNER):
        cfg = {
            **base,
            "hdbscan_min_samples": None,
            "random_state": s,
            "normalized": 0,
        }
        yield "hdbscan", "umap", cfg


def grid_kmeans() -> Iterator[tuple[str, str, dict]]:
    ks = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 80]
    for k, norm in product(ks, [0, 1]):
        yield "kmeans", "none", {"k": k, "random_state": 42, "normalized": norm}


def grid_gmm() -> Iterator[tuple[str, str, dict]]:
    for k, cov in product(
        [5, 10, 15, 20, 25, 30], ["full", "diag", "tied", "spherical"]
    ):
        yield (
            "gmm",
            "none",
            {
                "k": k,
                "covariance_type": cov,
                "random_state": 42,
                "normalized": 1,
            },
        )


def grid_agglomerative() -> Iterator[tuple[str, str, dict]]:
    for k, linkage in product([5, 10, 15, 20, 25, 30], ["ward", "average", "complete"]):
        yield (
            "agglomerative",
            "none",
            {
                "k": k,
                "linkage": linkage,
                "distance_metric": "euclidean",
                "normalized": 1,
            },
        )
    for k in [5, 10, 15, 20, 25, 30]:
        yield (
            "agglomerative",
            "none",
            {
                "k": k,
                "linkage": "average",
                "distance_metric": "cosine",
                "normalized": 1,
            },
        )


def grid_spectral() -> Iterator[tuple[str, str, dict]]:
    for k, nn in product([5, 10, 15, 20, 25, 30], [10, 20]):
        yield (
            "spectral",
            "none",
            {
                "k": k,
                "affinity": "nearest_neighbors",
                "n_neighbors": nn,
                "random_state": 42,
                "normalized": 1,
            },
        )


def grid_pca_hdbscan() -> Iterator[tuple[str, str, dict]]:
    for pc, mcs, csm in product([10, 25, 50, 100], [10, 15, 25], ["eom", "leaf"]):
        yield (
            "hdbscan",
            "pca",
            {
                "pca_n_components": pc,
                "hdbscan_min_cluster_size": mcs,
                "hdbscan_min_samples": None,
                "hdbscan_cluster_selection_method": csm,
                "hdbscan_metric": "euclidean",
                "random_state": 42,
                "normalized": 0,
            },
        )


def grid_hdbscan_direct_normalized() -> Iterator[tuple[str, str, dict]]:
    for mcs, csm, ms in product([10, 15, 25], ["eom", "leaf"], [None, 5, 10]):
        yield (
            "hdbscan",
            "none",
            {
                "hdbscan_min_cluster_size": mcs,
                "hdbscan_min_samples": ms,
                "hdbscan_cluster_selection_method": csm,
                "hdbscan_metric": "euclidean",
                "random_state": 42,
                "normalized": 1,
            },
        )


GRID_REGISTRY = {
    "kmeans_smoke": grid_kmeans_smoke,
    "umap_extension": grid_umap_extension,
    "min_cluster_size_extension": grid_min_cluster_size_extension,
    "min_samples_sweep": grid_min_samples_sweep,
    "seed_stability": grid_seed_stability,
    "kmeans": grid_kmeans,
    "gmm": grid_gmm,
    "agglomerative": grid_agglomerative,
    "spectral": grid_spectral,
    "pca_hdbscan": grid_pca_hdbscan,
    "hdbscan_direct_normalized": grid_hdbscan_direct_normalized,
}


ALL_GRIDS = (
    "umap_extension",
    "min_cluster_size_extension",
    "min_samples_sweep",
    "seed_stability",
    "kmeans",
    "gmm",
    "agglomerative",
    "spectral",
    "pca_hdbscan",
    "hdbscan_direct_normalized",
)
