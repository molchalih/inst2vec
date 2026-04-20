#!/usr/bin/env python
"""Null out bootstrap and stale composite/plateau columns in cluster_runs.

Run once after upgrading to the DBCV-only cluster validation pipeline.
Safe to run multiple times (idempotent — only touches non-NULL rows).

Usage:
    source .venv/bin/activate
    python scripts/migrate_drop_bootstrap.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import create_engine, inspect, text

engine = create_engine(os.environ["DATABASE_URL"])

COLS_TO_NULL = [
    "bootstrap_stability",
    "bootstrap_n_runs",
    "composite_score",
    "param_plateau_score",
]

inspector = inspect(engine)
if "cluster_runs" not in inspector.get_table_names():
    print("cluster_runs table not found — nothing to do.")
    sys.exit(0)

existing_cols = {c["name"] for c in inspector.get_columns("cluster_runs")}

with engine.begin() as conn:
    for col in COLS_TO_NULL:
        if col not in existing_cols:
            print(f"  {col}: column does not exist — skipped")
            continue
        result = conn.execute(
            text(f"UPDATE cluster_runs SET {col} = NULL WHERE {col} IS NOT NULL")
        )
        print(f"  {col}: nulled {result.rowcount} rows")

print("\nDone. Re-run the pipeline to recompute scores with the new DBCV-based logic.")
