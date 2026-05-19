"""One-shot migrator: copy sandwich cluster_runs rows from the legacy DB
into data/cluster_testing.db with the cluster-lab schema and config-hash.

Idempotent: re-running skips rows already present in the destination.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Allow running as a script (no package install).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cluster_lab import db as cdb  # noqa: E402

SRC_DEFAULT = "data/old/inst2vec.db"
DST_DEFAULT = "data/cluster_testing.db"


LEGACY_COLS = (
    "embedding_case",
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "hdbscan_cluster_selection_method",
    "hdbscan_metric",
    "random_state",
    "n_clusters",
    "noise_ratio",
    "min_size",
    "median_size",
    "max_size",
    "dbcv",
    "silhouette",
)


def build_row(legacy: sqlite3.Row) -> dict:
    """Convert a legacy cluster_runs row into a cluster-lab row dict."""
    cfg = {
        "algorithm": "hdbscan",
        "reducer": "umap",
        "normalized": 0,
        "random_state": int(legacy["random_state"]),
        "umap_n_components": int(legacy["umap_n_components"]),
        "umap_n_neighbors": int(legacy["umap_n_neighbors"]),
        "umap_min_dist": float(legacy["umap_min_dist"]),
        "umap_metric": legacy["umap_metric"],
        "hdbscan_min_cluster_size": int(legacy["hdbscan_min_cluster_size"]),
        "hdbscan_min_samples": (
            int(legacy["hdbscan_min_samples"])
            if legacy["hdbscan_min_samples"] is not None
            else None
        ),
        "hdbscan_cluster_selection_method": legacy["hdbscan_cluster_selection_method"],
        "hdbscan_metric": legacy["hdbscan_metric"],
    }
    row = {
        "config_hash": cdb.config_hash(cfg),
        "algorithm": cfg["algorithm"],
        "reducer": cfg["reducer"],
        "normalized": cfg["normalized"],
        "embedding_case": legacy["embedding_case"],
        "random_state": cfg["random_state"],
        "umap_n_components": cfg["umap_n_components"],
        "umap_n_neighbors": cfg["umap_n_neighbors"],
        "umap_min_dist": cfg["umap_min_dist"],
        "umap_metric": cfg["umap_metric"],
        "hdbscan_min_cluster_size": cfg["hdbscan_min_cluster_size"],
        "hdbscan_min_samples": cfg["hdbscan_min_samples"],
        "hdbscan_cluster_selection_method": cfg["hdbscan_cluster_selection_method"],
        "hdbscan_metric": cfg["hdbscan_metric"],
        "n_clusters": int(legacy["n_clusters"])
        if legacy["n_clusters"] is not None
        else None,
        "noise_ratio": float(legacy["noise_ratio"])
        if legacy["noise_ratio"] is not None
        else None,
        "min_size": int(legacy["min_size"]) if legacy["min_size"] is not None else None,
        "median_size": int(legacy["median_size"])
        if legacy["median_size"] is not None
        else None,
        "max_size": int(legacy["max_size"]) if legacy["max_size"] is not None else None,
        "dbcv": float(legacy["dbcv"]) if legacy["dbcv"] is not None else None,
        "silhouette": float(legacy["silhouette"])
        if legacy["silhouette"] is not None
        else None,
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dst", default=DST_DEFAULT)
    ap.add_argument("--case", default="sandwich")
    args = ap.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        raise SystemExit(f"source DB not found: {src_path}")

    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    dst = cdb.connect(args.dst)
    cdb.init_schema(dst)

    existing = cdb.bulk_existing_hashes(dst)
    total = 0
    inserted = 0
    skipped = 0
    seen_hashes: set[str] = set()

    cur = src.execute(
        f"SELECT {','.join(LEGACY_COLS)} FROM cluster_runs WHERE embedding_case = ?",
        (args.case,),
    )
    dst.execute("BEGIN")
    for row in cur:
        total += 1
        new_row = build_row(row)
        h = new_row["config_hash"]
        if h in existing or h in seen_hashes:
            skipped += 1
            continue
        cdb.insert_row(dst, new_row)
        seen_hashes.add(h)
        inserted += 1
    dst.execute("COMMIT")
    src.close()
    dst.close()

    print(f"migrated {inserted} / {total}, skipped {skipped}")


if __name__ == "__main__":
    main()
