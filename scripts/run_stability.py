"""Compute seed stability + cross-method ARI for the cluster lab.

Reads `data/cluster_testing.db`, picks the top configs by silhouette per
(algorithm, reducer) bucket, re-runs each config across multiple seeds to
compute pairwise ARI/NMI, and emits a cross-method ARI matrix.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cluster_lab import db as cdb  # noqa: E402
from scripts.cluster_lab.loader import load_sandwich_matrix  # noqa: E402
from scripts.cluster_lab.stability import (  # noqa: E402
    compute_cross_method_ari,
    compute_seed_stability,
)

CONFIG_COL_BY_ALGO = {
    "hdbscan_umap": [
        "umap_n_components",
        "umap_n_neighbors",
        "umap_min_dist",
        "umap_metric",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "hdbscan_cluster_selection_method",
        "hdbscan_metric",
    ],
    "hdbscan_pca": [
        "pca_n_components",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "hdbscan_cluster_selection_method",
        "hdbscan_metric",
    ],
    "hdbscan_none": [
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "hdbscan_cluster_selection_method",
        "hdbscan_metric",
    ],
    "kmeans_none": ["k"],
    "gmm_none": ["k", "covariance_type"],
    "agglomerative_none": ["k", "linkage", "distance_metric"],
    "spectral_none": ["k", "affinity", "n_neighbors"],
}


def _row_to_cfg(row: sqlite3.Row, cols: list[str]) -> dict:
    return {c: row[c] for c in cols if row[c] is not None}


def top_configs_for_seed_stability(
    db_path: str, k_top: int = 5
) -> list[tuple[str, str, dict]]:
    """Top viable hdbscan+umap configs by silhouette, distinct by their
    non-seed config_key (so we don't re-pick the same shape under different
    seeds the legacy data may have)."""
    conn = cdb.connect(db_path)
    # Filter out degenerate 2-cluster solutions (1 big blob + noise inflates
    # silhouette but carries no information).
    rows = conn.execute(
        "SELECT * FROM cluster_runs "
        "WHERE algorithm='hdbscan' AND reducer='umap' "
        "AND silhouette IS NOT NULL AND dbcv IS NOT NULL "
        "AND n_clusters >= 5 AND noise_ratio < 0.5 "
        "ORDER BY silhouette DESC LIMIT 200"
    ).fetchall()
    seen: set[str] = set()
    picks: list[tuple[str, str, dict]] = []
    cols = CONFIG_COL_BY_ALGO["hdbscan_umap"] + ["normalized"]
    for r in rows:
        cfg = _row_to_cfg(r, cols)
        cfg["normalized"] = int(cfg.get("normalized", 0))
        key = cdb.config_hash({**cfg, "algorithm": "hdbscan", "reducer": "umap"})
        if key in seen:
            continue
        seen.add(key)
        picks.append(("hdbscan", "umap", cfg))
        if len(picks) >= k_top:
            break
    conn.close()
    return picks


def top_configs_per_algorithm(
    db_path: str, k_per_algo: int = 2
) -> list[tuple[str, str, str, dict]]:
    """Returns (label, algorithm, reducer, cfg) for cross-method comparison.

    Pulls top-K configs by silhouette per (algorithm, reducer) bucket.
    """
    conn = cdb.connect(db_path)
    buckets = list(cdb.count_by_algo(conn))
    out: list[tuple[str, str, str, dict]] = []
    for algo, reducer, _ in buckets:
        key = f"{algo}_{reducer}"
        cols = CONFIG_COL_BY_ALGO.get(key)
        if cols is None:
            continue
        rows = conn.execute(
            "SELECT * FROM cluster_runs "
            "WHERE algorithm = ? AND reducer = ? "
            "AND silhouette IS NOT NULL "
            "AND n_clusters >= 5 "
            "ORDER BY silhouette DESC LIMIT ?",
            (algo, reducer, k_per_algo),
        ).fetchall()
        for i, r in enumerate(rows):
            cfg = _row_to_cfg(r, [*cols, "normalized", "random_state"])
            cfg["normalized"] = int(cfg.get("normalized", 0))
            label = f"{algo}/{reducer}#{i + 1}"
            out.append((label, algo, reducer, cfg))
    conn.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/cluster_testing.db")
    ap.add_argument("--src-db", default="data/old/inst2vec.db")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2,42,101")
    ap.add_argument("--top-stability", type=int, default=8)
    ap.add_argument("--top-cross", type=int, default=2)
    ap.add_argument("--output-dir", default="scripts/output/cluster_lab")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    matrix, _ = load_sandwich_matrix(args.src_db)
    print(f"matrix {matrix.shape}")

    print(f"=== seed stability (top {args.top_stability} hdbscan/umap configs) ===")
    top_stab = top_configs_for_seed_stability(args.db, k_top=args.top_stability)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"  seeds: {seeds}")
    stab_rows = compute_seed_stability(
        matrix, top_stab, seeds=seeds, db_path=args.db, max_workers=args.workers
    )
    df = pd.DataFrame(stab_rows)
    df.to_csv(out_dir / "data" / "stability.csv", index=False)
    print(
        df[
            [
                "algorithm",
                "config_key",
                "mean_ari",
                "std_ari",
                "mean_nmi",
                "median_n_clusters",
            ]
        ]
    )

    print(f"=== cross-method ARI (top {args.top_cross} per algorithm) ===")
    labelled = top_configs_per_algorithm(args.db, k_per_algo=args.top_cross)
    print(f"  {len(labelled)} configs in comparison")
    ari, nmi, _ = compute_cross_method_ari(
        matrix, labelled, db_path=args.db, max_workers=args.workers
    )
    labels = [lbl[0] for lbl in labelled]
    pd.DataFrame(ari, index=labels, columns=labels).to_csv(
        out_dir / "data" / "cross_method_ari.csv"
    )
    pd.DataFrame(nmi, index=labels, columns=labels).to_csv(
        out_dir / "data" / "cross_method_nmi.csv"
    )

    # Persist the labelled list so the analyzer can render headers cleanly.
    (out_dir / "data" / "cross_method_labels.json").write_text(
        json.dumps(
            [
                {
                    "label": label,
                    "algorithm": algo,
                    "reducer": reducer,
                    "config": _cfg_to_jsonable(cfg),
                }
                for label, algo, reducer, cfg in labelled
            ],
            indent=2,
        )
    )
    print("done.")


def _cfg_to_jsonable(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
