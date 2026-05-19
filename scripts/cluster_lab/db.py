"""SQLite schema for cluster_testing.db + idempotent insert helpers.

Schema mirrors the legacy cluster_runs table but:
  - adds (algorithm, reducer, normalized, extra_params, calinski_harabasz,
    davies_bouldin, n_singletons, error) columns,
  - replaces the legacy multi-column UNIQUE with a single config_hash UNIQUE.

Every row we write is fingerprinted via a stable hash of its (algorithm,
reducer, normalized, random_state, hyperparams, extra_params) so the
orchestrator can skip re-runs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cluster_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_hash TEXT UNIQUE NOT NULL,
    algorithm TEXT NOT NULL,
    reducer TEXT NOT NULL,
    normalized INTEGER NOT NULL DEFAULT 0,
    embedding_case TEXT NOT NULL DEFAULT 'sandwich',
    random_state INTEGER,
    extra_params TEXT,

    umap_n_components INTEGER,
    umap_n_neighbors INTEGER,
    umap_min_dist REAL,
    umap_metric TEXT,
    pca_n_components INTEGER,

    hdbscan_min_cluster_size INTEGER,
    hdbscan_min_samples INTEGER,
    hdbscan_cluster_selection_method TEXT,
    hdbscan_metric TEXT,

    k INTEGER,
    covariance_type TEXT,
    linkage TEXT,
    distance_metric TEXT,
    affinity TEXT,
    n_neighbors INTEGER,

    n_clusters INTEGER,
    noise_ratio REAL,
    n_singletons INTEGER,
    min_size INTEGER,
    median_size INTEGER,
    max_size INTEGER,

    dbcv REAL,
    silhouette REAL,
    calinski_harabasz REAL,
    davies_bouldin REAL,

    error TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_algo_reducer ON cluster_runs(algorithm, reducer);
CREATE INDEX IF NOT EXISTS idx_silhouette ON cluster_runs(silhouette);
"""


# Columns that exist on the row dict and map directly into a SQL column.
ROW_COLUMNS = (
    "config_hash",
    "algorithm",
    "reducer",
    "normalized",
    "embedding_case",
    "random_state",
    "extra_params",
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "pca_n_components",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "hdbscan_cluster_selection_method",
    "hdbscan_metric",
    "k",
    "covariance_type",
    "linkage",
    "distance_metric",
    "affinity",
    "n_neighbors",
    "n_clusters",
    "noise_ratio",
    "n_singletons",
    "min_size",
    "median_size",
    "max_size",
    "dbcv",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "error",
)


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with sane defaults (FOREIGN KEYS off, WAL on)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)


def _canonicalize(value: Any) -> Any:
    """Coerce numpy / pandas / dict values into JSON-stable shapes."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        # Treat NaN as None so the hash is stable across (Nones / NaN) producers.
        if value != value:
            return None
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    return str(value)


def config_hash(config: dict[str, Any]) -> str:
    """Stable sha1 over the canonical config dict.

    Identical configs produce identical hashes regardless of dict ordering;
    near-identical configs (differing in any single param including
    `random_state`) produce different hashes.
    """
    canonical = _canonicalize(config)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def row_exists(conn: sqlite3.Connection, h: str) -> bool:
    cur = conn.execute("SELECT 1 FROM cluster_runs WHERE config_hash = ? LIMIT 1", (h,))
    return cur.fetchone() is not None


def insert_row(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Insert a row, ignoring missing columns (fill with NULL)."""
    cols = [c for c in ROW_COLUMNS if c in row]
    placeholders = ",".join("?" for _ in cols)
    sql = (
        f"INSERT OR IGNORE INTO cluster_runs ({','.join(cols)}) VALUES ({placeholders})"
    )
    values: list[Any] = []
    for c in cols:
        v = row[c]
        if isinstance(v, (dict, list)):
            values.append(json.dumps(v, sort_keys=True))
        else:
            values.append(v)
    conn.execute(sql, values)


def bulk_existing_hashes(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT config_hash FROM cluster_runs")}


def count_by_algo(conn: sqlite3.Connection) -> Iterable[tuple[str, str, int]]:
    cur = conn.execute(
        "SELECT algorithm, reducer, COUNT(*) FROM cluster_runs GROUP BY algorithm, reducer"
    )
    return cur.fetchall()
