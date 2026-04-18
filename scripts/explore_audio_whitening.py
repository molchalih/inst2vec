"""Explore PCA whitening strategies for audio embeddings.

Runs UMAP+HDBSCAN for 13 whitening variants × 64 param combos (832 total)
using ProcessPoolExecutor(max_workers=8). Appends results to
scripts/audio_whitening_results.csv. Safe to interrupt and resume.

Usage:
    python scripts/explore_audio_whitening.py
"""
from __future__ import annotations

import csv
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    ("whiten_512", 512, False),
    ("whiten_1024", 1024, False),
    ("scale_whiten_32", 32, True),
    ("scale_whiten_64", 64, True),
    ("scale_whiten_128", 128, True),
    ("scale_whiten_256", 256, True),
    ("scale_whiten_512", 512, True),
    ("scale_whiten_1024", 1024, True),
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
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
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
    from hdbscan import validity as hdbscan_validity
    from modules.clustering import compute_clusters

    matrix = _build_whitened_matrix(_WORKER_MATRIX, n_components, use_scaler)

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
            dbcv = float(hdbscan_validity.validity_index(matrix_nd, labels))
        except Exception:
            dbcv = None

    # Silhouette on non-noise points
    sil = None
    non_noise_mask = labels != -1
    if result.n_clusters >= 2 and non_noise_mask.sum() > result.n_clusters:
        try:
            sil = float(silhouette_score(matrix_nd[non_noise_mask], labels[non_noise_mask]))
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
