#!/usr/bin/env python
"""One-shot: import scripts/clustering_results.csv into the ClusterRun DB table.

Usage:
    python scripts/import_cluster_csv.py
    python scripts/import_cluster_csv.py --csv /path/to/other.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.exc import IntegrityError
from modules.database import Base, engine, get_session, ClusterRun

DEFAULT_CSV = Path(__file__).parent / "clustering_results.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import clustering_results.csv into DB")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), metavar="PATH",
                        help=f"CSV to import (default: {DEFAULT_CSV})")
    args = parser.parse_args()

    Base.metadata.create_all(engine)

    inserted = 0
    skipped = 0

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ms = row.get("hdbscan_min_samples", "").strip()
            run = ClusterRun(
                embedding_case=row["embedding_case"],
                umap_n_components=int(row["umap_n_components"]),
                umap_n_neighbors=int(row["umap_n_neighbors"]),
                umap_min_dist=float(row["umap_min_dist"]),
                umap_metric=row["umap_metric"],
                umap2d_n_neighbors=int(row["umap2d_n_neighbors"]),
                umap2d_min_dist=float(row["umap2d_min_dist"]),
                umap2d_metric=row["umap2d_metric"],
                hdbscan_min_cluster_size=int(row["hdbscan_min_cluster_size"]),
                hdbscan_min_samples=int(ms) if ms else None,
                hdbscan_cluster_selection_method=row["hdbscan_cluster_selection_method"],
                hdbscan_metric=row["hdbscan_metric"],
                random_state=int(row["random_state"]),
                n_clusters=int(row["n_clusters"]),
                noise_ratio=float(row["noise_ratio"]),
                min_size=int(row["min_size"]),
                median_size=int(row["median_size"]),
                max_size=int(row["max_size"]),
            )
            session = get_session()
            try:
                session.add(run)
                session.commit()
                inserted += 1
            except IntegrityError:
                session.rollback()
                skipped += 1
            finally:
                session.close()

    print(f"Done — {inserted} inserted, {skipped} skipped (already in DB)")


if __name__ == "__main__":
    main()
