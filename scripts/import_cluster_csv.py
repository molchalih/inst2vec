#!/usr/bin/env python
"""One-shot: import clustering_results.csv into the cluster_runs table.

Default CSV: data/clustering_results.csv (repo root relative).

Usage:
    python scripts/import_cluster_csv.py
    python scripts/import_cluster_csv.py --csv /path/to/other.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.exc import IntegrityError
from modules.database import Base, engine, get_session, ClusterRun

DEFAULT_CSV = Path(__file__).parent.parent / "data" / "clustering_results.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import clustering_results.csv into DB")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), metavar="PATH",
                        help=f"CSV to import (default: {DEFAULT_CSV})")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"Error: CSV not found: {csv_path}")

    Base.metadata.create_all(engine)

    inserted = 0
    skipped = 0
    errors = 0

    session = get_session()
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                ms = row.get("hdbscan_min_samples", "").strip()
                try:
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
                except (ValueError, KeyError) as exc:
                    print(f"  row {row_num}: skipping malformed row — {exc}")
                    errors += 1
                    continue

                try:
                    # SQLite treats NULL as distinct in UNIQUE constraints, so IntegrityError
                    # is never raised for rows where hdbscan_min_samples=None. Pre-check first.
                    exists = session.query(ClusterRun).filter_by(
                        embedding_case=run.embedding_case,
                        umap_n_components=run.umap_n_components,
                        umap_n_neighbors=run.umap_n_neighbors,
                        umap_min_dist=run.umap_min_dist,
                        umap_metric=run.umap_metric,
                        umap2d_n_neighbors=run.umap2d_n_neighbors,
                        umap2d_min_dist=run.umap2d_min_dist,
                        umap2d_metric=run.umap2d_metric,
                        hdbscan_min_cluster_size=run.hdbscan_min_cluster_size,
                        hdbscan_min_samples=run.hdbscan_min_samples,
                        hdbscan_cluster_selection_method=run.hdbscan_cluster_selection_method,
                        hdbscan_metric=run.hdbscan_metric,
                        random_state=run.random_state,
                    ).first()
                    if exists:
                        skipped += 1
                        continue
                    session.add(run)
                    session.flush()
                    inserted += 1
                    if inserted % 500 == 0:
                        session.commit()
                except IntegrityError:
                    session.rollback()
                    skipped += 1
        session.commit()
    finally:
        session.close()

    print(f"Done — {inserted} inserted, {skipped} skipped (already in DB), {errors} malformed")


if __name__ == "__main__":
    main()
