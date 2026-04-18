"""Explore PCA whitening strategies for audio embeddings.

Runs UMAP+HDBSCAN for 13 whitening variants × 64 param combos (832 total)
using ProcessPoolExecutor(max_workers=8). Appends results to
scripts/audio_whitening_results.csv. Safe to interrupt and resume.

Usage:
    python scripts/explore_audio_whitening.py
"""
from __future__ import annotations

import csv
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_CSV = Path(__file__).parent / "audio_whitening_results.csv"
PARAMS_CSV = Path(__file__).parent / "audio_best_params.csv"
MAX_WORKERS = 8

PARAM_COLS = [
    "umap_n_components",
    "umap_n_neighbors",
    "umap_min_dist",
    "umap_metric",
    "hdbscan_min_cluster_size",
    "hdbscan_cluster_selection_method",
    "hdbscan_metric",
]

FIELDNAMES = PARAM_COLS + [
    "whitening",
    "n_clusters",
    "noise_ratio",
    "dbcv",
    "silhouette",
]

# (id, n_components, use_scaler) — None n_components means no-op (baseline)
WHITENING_SPECS: list[tuple[str, int | None, bool]] = [
    ("none", None, False),
    ("whiten_32", 32, False),
    ("whiten_64", 64, False),
    ("whiten_128", 128, False),
    ("whiten_256", 256, False),
    ("scale_whiten_32", 32, True),
    ("scale_whiten_64", 64, True),
    ("scale_whiten_128", 128, True),
    ("scale_whiten_256", 256, True),
]


def _build_whitened_matrix(
    matrix: np.ndarray,
    n_components: int | None,
    use_scaler: bool,
) -> np.ndarray:
    """Apply whitening transform to matrix. Returns float32 array."""
    if n_components is None:
        return matrix
    X = matrix.astype(np.float64)
    if use_scaler:
        X = StandardScaler().fit_transform(X)
    X = PCA(n_components=n_components, whiten=True).fit_transform(X)
    return X.astype(np.float32)


def _load_done(csv_path: str, param_cols: list[str]) -> frozenset[tuple]:
    """Return set of (whitening, *param_values) tuples already in the CSV."""
    p = Path(csv_path)
    if not p.exists():
        return frozenset()
    done = set()
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["whitening"],) + tuple(row[c] for c in param_cols)
            done.add(key)
    return frozenset(done)


def _append_row(csv_path: str, row: dict) -> None:
    """Append one row to CSV; writes header if file is new or empty."""
    p = Path(csv_path)
    write_header = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# Module-level global populated by worker initializer
_WORKER_MATRIX: np.ndarray | None = None


def _init_worker(matrix: np.ndarray) -> None:
    global _WORKER_MATRIX
    _WORKER_MATRIX = matrix


def _run_one(
    whitening_id: str,
    n_components: int | None,
    use_scaler: bool,
    params: dict,
) -> dict:
    """Worker task: whiten matrix, run clustering, compute metrics, return row dict."""
    if _WORKER_MATRIX is None:
        raise RuntimeError("_WORKER_MATRIX not initialized — must be called via ProcessPoolExecutor with initializer=_init_worker")

    from hdbscan import validity as hdbscan_validity
    from modules.clustering import compute_clusters

    # Cast params to correct types for compute_clusters
    typed: dict = {
        "umap_n_components": int(params["umap_n_components"]),
        "umap_n_neighbors": int(params["umap_n_neighbors"]),
        "umap_min_dist": float(params["umap_min_dist"]),
        "umap_metric": params["umap_metric"],
        "hdbscan_min_cluster_size": int(params["hdbscan_min_cluster_size"]),
        "hdbscan_cluster_selection_method": params["hdbscan_cluster_selection_method"],
        "hdbscan_metric": params["hdbscan_metric"],
    }

    try:
        matrix = _build_whitened_matrix(_WORKER_MATRIX, n_components, use_scaler)
        result = compute_clusters(matrix, return_nd_matrix=True, **typed)
    except Exception as exc:
        return {
            **params,
            "whitening": whitening_id,
            "n_clusters": 0,
            "noise_ratio": 1.0,
            "dbcv": None,
            "silhouette": None,
            "_error": str(exc),
        }

    labels = result.labels
    matrix_nd = result.matrix_nd.astype(np.float64)

    # DBCV
    dbcv = None
    if result.n_clusters >= 2:
        try:
            dbcv = float(hdbscan_validity.validity_index(matrix_nd, labels, metric=typed["hdbscan_metric"]))
        except Exception:
            dbcv = None

    # Silhouette on non-noise points
    sil = None
    non_noise_mask = labels != -1
    if result.n_clusters >= 2 and non_noise_mask.sum() > result.n_clusters:
        try:
            sil = float(silhouette_score(matrix_nd[non_noise_mask], labels[non_noise_mask], metric=typed["hdbscan_metric"]))
        except Exception:
            sil = None

    return {
        **params,
        "whitening": whitening_id,
        "n_clusters": result.n_clusters,
        "noise_ratio": round(result.noise_ratio, 4),
        "dbcv": round(dbcv, 4) if dbcv is not None else None,
        "silhouette": round(sil, 4) if sil is not None else None,
    }


def _print_summary(csv_path: str) -> None:
    """Print top-10 runs by DBCV (fallback: noise_ratio ASC), grouped by whitening."""
    if not Path(csv_path).exists():
        print("No results file found.")
        return

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No results to summarise.")
        return

    def sort_key(r: dict):
        dbcv_str = r.get("dbcv") or ""
        try:
            dbcv_val = float(dbcv_str)
        except ValueError:
            dbcv_val = -float("inf")
        return (-dbcv_val, float(r.get("noise_ratio") or 1.0))

    ranked = sorted(rows, key=sort_key)[:10]

    # Group by whitening strategy
    by_strategy: dict[str, list[dict]] = {}
    for r in ranked:
        by_strategy.setdefault(r["whitening"], []).append(r)

    print("\n── Top-10 runs by DBCV ────────────────────────────────────────────────")
    fmt = "{:<20} {:>10} {:>10} {:>8} {:>10} {:>12}"
    print(fmt.format("whitening", "n_clusters", "noise_ratio", "dbcv", "silhouette", "umap_metric"))
    print("─" * 74)
    for strategy, strategy_rows in by_strategy.items():
        for r in strategy_rows:
            print(fmt.format(
                strategy,
                r.get("n_clusters", ""),
                r.get("noise_ratio", ""),
                r.get("dbcv") or "n/a",
                r.get("silhouette") or "n/a",
                r.get("umap_metric", ""),
            ))
    print("─" * 74)
    print(f"Full results: {csv_path}\n")


def main() -> None:
    from modules.clustering import load_user_matrix

    matrix, user_pks = load_user_matrix("audio")
    if matrix.shape[0] == 0:
        print("No audio embeddings found in database.")
        sys.exit(1)

    print(f"Loaded audio matrix: {matrix.shape[0]} users × {matrix.shape[1]} dims")

    if not PARAMS_CSV.exists():
        print(f"Params CSV not found: {PARAMS_CSV}")
        sys.exit(1)

    with open(PARAMS_CSV, newline="") as f:
        param_rows = list(csv.DictReader(f))
    print(f"Param combos: {len(param_rows)}  |  Whitening variants: {len(WHITENING_SPECS)}")

    # Build full work list
    all_tasks = [
        (spec_id, n_comp, use_sc, params)
        for spec_id, n_comp, use_sc in WHITENING_SPECS
        for params in param_rows
    ]
    total = len(all_tasks)

    # Resume: skip already-done
    done = _load_done(str(OUTPUT_CSV), PARAM_COLS)
    tasks = [
        t for t in all_tasks
        if (t[0],) + tuple(t[3][c] for c in PARAM_COLS) not in done
    ]
    skipped = total - len(tasks)
    if skipped:
        print(f"Resuming: {skipped}/{total} runs already done, {len(tasks)} remaining")

    if not tasks:
        print("All runs complete.")
        _print_summary(str(OUTPUT_CSV))
        return

    completed = skipped

    print(f"Running {len(tasks)} tasks with {MAX_WORKERS} workers...\n")

    # Use fork so workers inherit the parent's memory space — avoids spawn's
    # requirement for the __main__ module to be importable from a file, which
    # breaks when running from a stdin script or via importlib.exec_module.
    _mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=_mp_ctx, initializer=_init_worker, initargs=(matrix,)) as pool:
        futures = {
            pool.submit(_run_one, spec_id, n_comp, use_sc, params): (spec_id, params)
            for spec_id, n_comp, use_sc, params in tasks
        }
        for future in as_completed(futures):
            spec_id, params = futures[future]
            completed += 1
            try:
                row = future.result()
            except Exception as exc:
                print(f"[{spec_id}] WORKER ERROR: {exc}", flush=True)
                continue

            error = row.pop("_error", None)
            _append_row(str(OUTPUT_CSV), row)

            status = f"n_clusters={row['n_clusters']} noise={float(row['noise_ratio']):.1%}"
            if row['dbcv'] is not None:
                status += f" dbcv={row['dbcv']}"
            if error:
                status += f" ERROR={error}"
            print(f"[{spec_id} | {completed}/{total}] {status}", flush=True)

    _print_summary(str(OUTPUT_CSV))


if __name__ == "__main__":
    main()
