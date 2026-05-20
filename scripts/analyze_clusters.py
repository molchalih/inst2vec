"""Driver — one entry point for the cluster-analysis package.

Reads cluster_runs from data/inst2vec.db (read-only), scores with the
DBCV-aligned quality formula, optionally runs in-memory seed stability
on the top-K configs, and writes report.md + companions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cluster_analysis import (  # noqa: E402
    DB_PATH,
    DEFAULT_SEEDS,
    DEFAULT_TOP_K,
    OUT_DIR_DEFAULT,
    diagnostics,
    loader,
    quality,
    report,
    stability,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case",
        default="sandwich",
        choices=("audio", "video", "sandwich"),
        help="embedding_case to analyze (default: sandwich)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="number of configs to feed stability re-runs",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=len(DEFAULT_SEEDS),
        help="seeds per config for stability (uses DEFAULT_SEEDS[:n_seeds])",
    )
    p.add_argument(
        "--no-stability",
        action="store_true",
        help="skip the in-memory seed re-runs",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(OUT_DIR_DEFAULT),
        help="output directory for report.md + data/ + plots/",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    print(f"Loading cluster_runs from {DB_PATH} (case={args.case})…")
    runs = loader.load_cluster_runs(DB_PATH, case=args.case)
    print(f"  {len(runs)} rows; {int(runs['dbcv'].notna().sum())} viable")

    scored = quality.quality_score(runs)

    stab_df = None
    if not args.no_stability:
        print(f"Loading embedding matrix (case={args.case})…")
        try:
            matrix, _user_ids = loader.load_sandwich_matrix(DB_PATH, case=args.case)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  skipping stability — {exc}")
            matrix = None

        if matrix is not None:
            top = diagnostics.top_n(quality.ridge_filter(scored), n=args.top_k)
            seeds = list(DEFAULT_SEEDS[: args.n_seeds])
            grid_axis_cols = [
                "umap_n_components",
                "umap_n_neighbors",
                "umap_min_dist",
                "umap_metric",
                "hdbscan_min_cluster_size",
                "hdbscan_min_samples",
                "hdbscan_cluster_selection_method",
                "hdbscan_metric",
            ]
            configs = top[grid_axis_cols].to_dict("records")
            print(f"Running stability: {len(configs)} configs × {len(seeds)} seeds…")
            stab_df = stability.stability_for(
                matrix,
                configs,
                seeds=seeds,
                label_producer=stability.umap_hdbscan_labels,
            )

    print(f"Writing report to {args.out_dir}/…")
    path = report.write_report(
        runs=scored, stability=stab_df, out_dir=args.out_dir, case=args.case
    )
    print(f"  done: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
