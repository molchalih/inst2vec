"""
Parameter search tool for clustering. Runs compute_clusters with given params,
prints stats, and appends one row to scripts/clustering_results.csv.

Usage:
    python scripts/tune_clustering.py --embedding-case video --hdbscan-min-cluster-size 20
"""
import sys
import os
import argparse
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from modules.clustering import compute_clusters, load_user_matrix

CSV_PATH = os.path.join(os.path.dirname(__file__), "clustering_results.csv")
CSV_FIELDS = [
    "timestamp", "embedding_case",
    "umap_n_components", "umap_n_neighbors", "umap_min_dist", "umap_metric",
    "umap2d_n_neighbors", "umap2d_min_dist", "umap2d_metric",
    "hdbscan_min_cluster_size", "hdbscan_min_samples",
    "hdbscan_cluster_selection_method", "hdbscan_metric",
    "random_state",
    "n_clusters", "noise_ratio", "min_size", "median_size", "max_size",
]


def main():
    parser = argparse.ArgumentParser(description="Tune clustering parameters")
    parser.add_argument("--embedding-case", required=True, choices=["video", "sandwich", "audio"])
    parser.add_argument("--umap-n-components", type=int, default=15)
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--umap-metric", type=str, default="cosine")
    parser.add_argument("--umap2d-n-neighbors", type=int, default=15)
    parser.add_argument("--umap2d-min-dist", type=float, default=0.1)
    parser.add_argument("--umap2d-metric", type=str, default="cosine")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=15)
    parser.add_argument("--hdbscan-min-samples", type=int, default=None)
    parser.add_argument("--hdbscan-cluster-selection-method", type=str, default="eom", choices=["eom", "leaf"])
    parser.add_argument("--hdbscan-metric", type=str, default="euclidean")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    params = dict(
        umap_n_components=args.umap_n_components,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        umap2d_n_neighbors=args.umap2d_n_neighbors,
        umap2d_min_dist=args.umap2d_min_dist,
        umap2d_metric=args.umap2d_metric,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
        hdbscan_metric=args.hdbscan_metric,
        random_state=args.seed,
    )

    print(
        f"[tune] {args.embedding_case} | "
        f"umap_n_components={args.umap_n_components} n_neighbors={args.umap_n_neighbors} "
        f"min_dist={args.umap_min_dist} metric={args.umap_metric} | "
        f"hdbscan_min_cluster_size={args.hdbscan_min_cluster_size} "
        f"method={args.hdbscan_cluster_selection_method}"
    )

    matrix, user_pks = load_user_matrix(args.embedding_case)
    if matrix.shape[0] == 0:
        print(f"[tune] no embeddings found for case '{args.embedding_case}'")
        sys.exit(1)

    result = compute_clusters(matrix, **params)

    if result.cluster_sizes:
        sizes_str = (
            f"min={min(result.cluster_sizes)} "
            f"median={int(np.median(result.cluster_sizes))} "
            f"max={max(result.cluster_sizes)}"
        )
        min_size = min(result.cluster_sizes)
        median_size = int(np.median(result.cluster_sizes))
        max_size = max(result.cluster_sizes)
    else:
        sizes_str = "n/a"
        min_size = median_size = max_size = 0

    print(
        f"[tune] → {result.n_clusters} clusters, "
        f"{result.noise_ratio:.1%} noise, sizes: {sizes_str}"
    )

    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "embedding_case": args.embedding_case,
            "umap_n_components": args.umap_n_components,
            "umap_n_neighbors": args.umap_n_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "umap_metric": args.umap_metric,
            "umap2d_n_neighbors": args.umap2d_n_neighbors,
            "umap2d_min_dist": args.umap2d_min_dist,
            "umap2d_metric": args.umap2d_metric,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "hdbscan_cluster_selection_method": args.hdbscan_cluster_selection_method,
            "hdbscan_metric": args.hdbscan_metric,
            "random_state": args.seed,
            "n_clusters": result.n_clusters,
            "noise_ratio": round(result.noise_ratio, 4),
            "min_size": min_size,
            "median_size": median_size,
            "max_size": max_size,
        })

    print(f"[tune] appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
