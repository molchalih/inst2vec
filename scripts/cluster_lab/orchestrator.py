"""Parallel orchestrator: dispatch (algorithm, reducer, config) tuples to a
ProcessPoolExecutor, skip configs already present in the DB by config_hash,
and stream results into SQLite as they complete.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from scripts.cluster_lab import db as cdb
from scripts.cluster_lab.runners import run

# Sharing a 800×4096 float32 matrix per worker over fork is fine on macOS/Linux;
# we just keep the matrix in a module-level slot so workers don't have to
# pickle it on every call.
_MATRIX_CACHE: dict[str, np.ndarray] = {}


def _set_matrix(key: str, matrix: np.ndarray) -> None:
    _MATRIX_CACHE[key] = matrix


def _worker_init(key: str, matrix_bytes: bytes, shape: tuple, dtype: str) -> None:
    arr = np.frombuffer(matrix_bytes, dtype=np.dtype(dtype)).reshape(shape)
    _MATRIX_CACHE[key] = arr


def _worker_run(
    matrix_key: str,
    algorithm: str,
    reducer: str,
    config: dict,
    config_hash: str,
) -> dict:
    """Top-level worker function — must be picklable."""
    matrix = _MATRIX_CACHE[matrix_key]
    result = run(algorithm, reducer, matrix, config)
    # The runner already computes its own hash; we override to the canonical
    # hash the orchestrator used for skip-check so the DB key is stable.
    result["config_hash"] = config_hash
    return result


def _summarize(rows: list[dict]) -> str:
    n = len(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    n_ok = n - n_err
    return f"{n_ok} ok, {n_err} errored"


def run_grid(
    matrix: np.ndarray,
    grid_iter: Iterable[tuple[str, str, dict]],
    db_path: str,
    name: str = "grid",
    max_workers: int = 5,
    verbose: bool = True,
    embedding_case: str = "sandwich",
    progress_every: int = 25,
) -> dict:
    """Dispatch a grid through a worker pool, stream rows into the DB.

    Returns summary dict: {submitted, skipped, completed, errors, elapsed}.
    """
    conn = cdb.connect(db_path)
    cdb.init_schema(conn)
    existing = cdb.bulk_existing_hashes(conn)

    matrix = np.ascontiguousarray(matrix)
    matrix_bytes = matrix.tobytes()
    matrix_shape = matrix.shape
    matrix_dtype = str(matrix.dtype)
    matrix_key = "M"

    pending = []
    submitted = 0
    skipped = 0
    duplicate_in_grid = 0
    seen: set[str] = set()
    for algorithm, reducer, cfg in grid_iter:
        cfg_full = {
            **cfg,
            "algorithm": algorithm,
            "reducer": reducer,
            "embedding_case": embedding_case,
        }
        h = cdb.config_hash(cfg_full)
        if h in existing or h in seen:
            if h in seen:
                duplicate_in_grid += 1
            else:
                skipped += 1
            continue
        seen.add(h)
        cfg_with_case = {**cfg, "embedding_case": embedding_case}
        pending.append((algorithm, reducer, cfg_with_case, h))

    if verbose:
        print(
            f"[{name}] {len(pending)} configs to run, "
            f"{skipped} already in DB, {duplicate_in_grid} dupes inside grid"
        )

    if not pending:
        conn.close()
        return {
            "name": name,
            "submitted": 0,
            "skipped": skipped,
            "completed": 0,
            "errors": 0,
            "elapsed": 0.0,
        }

    completed_rows: list[dict] = []
    errors = 0
    t0 = time.time()
    ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(matrix_key, matrix_bytes, matrix_shape, matrix_dtype),
    ) as pool:
        futs = {
            pool.submit(_worker_run, matrix_key, algo, red, cfg, h): (algo, red, cfg, h)
            for algo, red, cfg, h in pending
        }
        submitted = len(futs)
        for n_done, fut in enumerate(as_completed(futs), start=1):
            algo, red, cfg, h = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:  # pragma: no cover — worker death
                row = {
                    "config_hash": h,
                    "algorithm": algo,
                    "reducer": red,
                    "normalized": int(cfg.get("normalized", 0)),
                    "embedding_case": embedding_case,
                    "random_state": cfg.get("random_state"),
                    "error": f"worker death: {exc!r}",
                }
                for k in (
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
                ):
                    if k in cfg:
                        row[k] = cfg[k]
            if row.get("error"):
                errors += 1
            cdb.insert_row(conn, row)
            completed_rows.append(row)
            if verbose and (n_done % progress_every == 0 or n_done == submitted):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (submitted - n_done) / rate if rate else 0
                print(
                    f"[{name}] {n_done}/{submitted} done "
                    f"({_summarize(completed_rows)}), "
                    f"{elapsed:.1f}s elapsed, eta {eta:.1f}s"
                )
    conn.close()
    return {
        "name": name,
        "submitted": submitted,
        "skipped": skipped,
        "completed": len(completed_rows),
        "errors": errors,
        "elapsed": time.time() - t0,
    }
