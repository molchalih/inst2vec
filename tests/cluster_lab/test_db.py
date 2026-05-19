"""Tests for the cluster_testing.db schema + idempotence helpers."""

from __future__ import annotations

from pathlib import Path

from scripts.cluster_lab import db


def test_init_schema_creates_table(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.db")
    db.init_schema(conn)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = {r[0] for r in cur.fetchall()}
    assert "cluster_runs" in names


def test_config_hash_is_order_independent() -> None:
    a = {"algorithm": "kmeans", "k": 10, "random_state": 42}
    b = {"random_state": 42, "k": 10, "algorithm": "kmeans"}
    assert db.config_hash(a) == db.config_hash(b)


def test_config_hash_differs_on_seed() -> None:
    a = {"algorithm": "kmeans", "k": 10, "random_state": 42}
    b = {"algorithm": "kmeans", "k": 10, "random_state": 7}
    assert db.config_hash(a) != db.config_hash(b)


def test_insert_then_row_exists(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.db")
    db.init_schema(conn)
    cfg = {"algorithm": "kmeans", "reducer": "none", "k": 8, "random_state": 42}
    h = db.config_hash(cfg)
    assert not db.row_exists(conn, h)
    db.insert_row(
        conn,
        {
            "config_hash": h,
            "algorithm": "kmeans",
            "reducer": "none",
            "normalized": 0,
            "embedding_case": "sandwich",
            "random_state": 42,
            "k": 8,
            "silhouette": 0.42,
        },
    )
    assert db.row_exists(conn, h)


def test_nan_normalized_to_none() -> None:
    a = {"algorithm": "x", "param": float("nan")}
    b = {"algorithm": "x", "param": None}
    assert db.config_hash(a) == db.config_hash(b)
