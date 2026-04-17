"""One-shot: apply cluster_runs schema migration (adds Phase 6b validation columns).

Usage:
    python scripts/migrate_cluster_runs.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from modules.database import _migrate_cluster_runs_table

if __name__ == "__main__":
    _migrate_cluster_runs_table()
    print("cluster_runs migration complete.")
