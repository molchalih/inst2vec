from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.cluster_lab import db as cdb
from scripts.cluster_lab.orchestrator import run_grid


def _blobs():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
    parts = [rng.normal(c, 0.5, size=(60, 3)) for c in centers]
    return np.vstack(parts).astype(np.float32)


def _kmeans_grid(ks):
    for k in ks:
        yield "kmeans", "none", {"k": k, "random_state": 42, "normalized": 0}


def test_run_grid_populates_db(tmp_path: Path) -> None:
    mat = _blobs()
    db_path = str(tmp_path / "x.db")
    summary = run_grid(
        mat,
        _kmeans_grid([2, 3, 4, 5]),
        db_path=db_path,
        name="t",
        max_workers=2,
        verbose=False,
    )
    assert summary["submitted"] == 4
    assert summary["errors"] == 0
    conn = cdb.connect(db_path)
    n = conn.execute("SELECT COUNT(*) FROM cluster_runs").fetchone()[0]
    assert n == 4


def test_run_grid_skip_existing(tmp_path: Path) -> None:
    mat = _blobs()
    db_path = str(tmp_path / "x.db")
    run_grid(
        mat,
        _kmeans_grid([2, 3]),
        db_path=db_path,
        name="t",
        max_workers=2,
        verbose=False,
    )
    summary = run_grid(
        mat,
        _kmeans_grid([2, 3, 4]),
        db_path=db_path,
        name="t",
        max_workers=2,
        verbose=False,
    )
    assert summary["submitted"] == 1
    assert summary["skipped"] == 2


def test_run_grid_captures_errors(tmp_path: Path) -> None:
    mat = _blobs()
    db_path = str(tmp_path / "x.db")

    def bad_grid():
        yield "kmeans", "none", {"k": -1, "random_state": 0, "normalized": 0}
        yield "kmeans", "none", {"k": 5, "random_state": 0, "normalized": 0}

    summary = run_grid(
        mat, bad_grid(), db_path=db_path, name="t", max_workers=2, verbose=False
    )
    assert summary["errors"] == 1
    assert summary["completed"] == 2
