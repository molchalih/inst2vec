"""Seed stability + cross-method agreement.

`compute_seed_stability` re-runs top-N viable configs across multiple seeds
and computes pairwise ARI / NMI between their cluster assignments. The
result lands in a side table `stability` so the analyzer can join on it.

`compute_cross_method_ari` takes top-K configs per algorithm, re-runs each
once, and emits a (K * algos) × (K * algos) ARI matrix.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
from itertools import combinations
from typing import Any

import numpy as np

from scripts.cluster_lab import db as cdb

STABILITY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stability (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm TEXT NOT NULL,
    reducer TEXT NOT NULL,
    config_key TEXT NOT NULL,
    n_seeds INTEGER NOT NULL,
    mean_ari REAL,
    std_ari REAL,
    mean_nmi REAL,
    std_nmi REAL,
    median_n_clusters REAL,
    std_n_clusters REAL,
    UNIQUE (algorithm, reducer, config_key)
);

CREATE TABLE IF NOT EXISTS cross_method (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label_a TEXT NOT NULL,
    label_b TEXT NOT NULL,
    ari REAL,
    nmi REAL,
    UNIQUE (label_a, label_b)
);
"""


def init_stability_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(STABILITY_SCHEMA_SQL)


# ── label producers ───────────────────────────────────────────────────────────


def _run_and_return_labels(
    algorithm: str, reducer: str, matrix: np.ndarray, cfg: dict
) -> np.ndarray | None:
    """Run a config and return only the labels (not the full result dict).

    For stability we need labels themselves. We monkey around the runners by
    re-running them but intercepting labels: in practice, easier to just call
    the algorithm directly here to skip the metric pass and capture labels.
    """
    try:
        if algorithm == "hdbscan" and reducer == "umap":
            return _umap_hdbscan_labels(matrix, cfg)
        if algorithm == "hdbscan" and reducer == "pca":
            return _pca_hdbscan_labels(matrix, cfg)
        if algorithm == "hdbscan" and reducer == "none":
            return _hdbscan_direct_labels(matrix, cfg)
        if algorithm == "kmeans":
            return _kmeans_labels(matrix, cfg)
        if algorithm == "gmm":
            return _gmm_labels(matrix, cfg)
        if algorithm == "agglomerative":
            return _agglomerative_labels(matrix, cfg)
        if algorithm == "spectral":
            return _spectral_labels(matrix, cfg)
    except Exception:
        return None
    return None


def _maybe_normalize(matrix: np.ndarray, normalized: int | bool) -> np.ndarray:
    if int(normalized):
        from scripts.cluster_lab.loader import l2_normalize

        return l2_normalize(matrix)
    return matrix


def _umap_hdbscan_labels(matrix: np.ndarray, cfg: dict) -> np.ndarray:
    import hdbscan
    from umap import UMAP

    mat = _maybe_normalize(matrix, cfg.get("normalized", 0))
    nd = UMAP(
        n_components=int(cfg["umap_n_components"]),
        n_neighbors=int(cfg["umap_n_neighbors"]),
        min_dist=float(cfg["umap_min_dist"]),
        metric=str(cfg["umap_metric"]),
        init="random",
        random_state=int(cfg.get("random_state", 42)),
        n_jobs=1,
    ).fit_transform(mat)
    ms = cfg.get("hdbscan_min_samples")
    return hdbscan.HDBSCAN(
        min_cluster_size=int(cfg["hdbscan_min_cluster_size"]),
        min_samples=int(ms) if ms is not None else None,
        cluster_selection_method=str(cfg["hdbscan_cluster_selection_method"]),
        metric="euclidean",
    ).fit_predict(nd)


def _pca_hdbscan_labels(matrix: np.ndarray, cfg: dict) -> np.ndarray:
    import hdbscan
    from sklearn.decomposition import PCA

    mat = _maybe_normalize(matrix, cfg.get("normalized", 0))
    nd = PCA(
        n_components=int(cfg["pca_n_components"]),
        random_state=int(cfg.get("random_state", 42)),
    ).fit_transform(mat)
    ms = cfg.get("hdbscan_min_samples")
    return hdbscan.HDBSCAN(
        min_cluster_size=int(cfg["hdbscan_min_cluster_size"]),
        min_samples=int(ms) if ms is not None else None,
        cluster_selection_method=str(cfg["hdbscan_cluster_selection_method"]),
        metric="euclidean",
    ).fit_predict(nd)


def _hdbscan_direct_labels(matrix: np.ndarray, cfg: dict) -> np.ndarray:
    import hdbscan

    mat = _maybe_normalize(matrix, cfg.get("normalized", 0))
    ms = cfg.get("hdbscan_min_samples")
    return hdbscan.HDBSCAN(
        min_cluster_size=int(cfg["hdbscan_min_cluster_size"]),
        min_samples=int(ms) if ms is not None else None,
        cluster_selection_method=str(cfg["hdbscan_cluster_selection_method"]),
        metric="euclidean",
    ).fit_predict(mat)


def _kmeans_labels(matrix: np.ndarray, cfg: dict) -> np.ndarray:
    from sklearn.cluster import KMeans

    mat = _maybe_normalize(matrix, cfg.get("normalized", 0))
    return KMeans(
        n_clusters=int(cfg["k"]),
        random_state=int(cfg.get("random_state", 42)),
        n_init=10,
    ).fit_predict(mat)


def _gmm_labels(matrix: np.ndarray, cfg: dict) -> np.ndarray:
    from sklearn.mixture import GaussianMixture

    mat = _maybe_normalize(matrix, cfg.get("normalized", 0))
    return GaussianMixture(
        n_components=int(cfg["k"]),
        covariance_type=str(cfg.get("covariance_type", "full")),
        random_state=int(cfg.get("random_state", 42)),
        max_iter=200,
        reg_covar=1e-4,
    ).fit_predict(mat)


def _agglomerative_labels(matrix: np.ndarray, cfg: dict) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    mat = _maybe_normalize(matrix, cfg.get("normalized", 0))
    linkage = str(cfg.get("linkage", "ward"))
    metric = str(cfg.get("distance_metric", "euclidean"))
    return AgglomerativeClustering(
        n_clusters=int(cfg["k"]),
        linkage=linkage,
        metric=metric if linkage != "ward" else "euclidean",
    ).fit_predict(mat)


def _spectral_labels(matrix: np.ndarray, cfg: dict) -> np.ndarray:
    from sklearn.cluster import SpectralClustering

    mat = _maybe_normalize(matrix, cfg.get("normalized", 0))
    affinity = str(cfg.get("affinity", "nearest_neighbors"))
    kwargs: dict[str, Any] = {
        "n_clusters": int(cfg["k"]),
        "affinity": affinity,
        "random_state": int(cfg.get("random_state", 42)),
        "assign_labels": "kmeans",
    }
    if affinity == "nearest_neighbors":
        kwargs["n_neighbors"] = int(cfg.get("n_neighbors", 10))
    return SpectralClustering(**kwargs).fit_predict(mat)


# ── pairwise ARI/NMI ──────────────────────────────────────────────────────────


def _pairwise_metrics(
    labels_list: list[np.ndarray],
) -> tuple[float, float, float, float]:
    """Mean/std ARI + mean/std NMI across all pairs."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    aris = []
    nmis = []
    for a, b in combinations(labels_list, 2):
        if a is None or b is None:
            continue
        aris.append(adjusted_rand_score(a, b))
        nmis.append(normalized_mutual_info_score(a, b))
    if not aris:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(aris)),
        float(np.std(aris)),
        float(np.mean(nmis)),
        float(np.std(nmis)),
    )


# ── seed stability ────────────────────────────────────────────────────────────


def _config_key(cfg: dict, exclude: tuple = ("random_state",)) -> str:
    """Stable string identifier for everything except the seed."""
    sub = {k: v for k, v in cfg.items() if k not in exclude}
    return cdb.config_hash(sub)


def compute_seed_stability(
    matrix: np.ndarray,
    top_configs: list[tuple[str, str, dict]],
    seeds: list[int],
    db_path: str,
    max_workers: int = 5,
    verbose: bool = True,
) -> list[dict]:
    """For each base config, recompute clusterings across `seeds`.

    Each base config produces `len(seeds)` label arrays; we compute pairwise
    ARI/NMI and the median/std of `n_clusters` per group.
    """
    conn = cdb.connect(db_path)
    init_stability_schema(conn)

    # Build full task list
    tasks: list[tuple[str, str, dict]] = []
    base_keys: list[str] = []
    for algo, reducer, base_cfg in top_configs:
        key = _config_key({**base_cfg, "algorithm": algo, "reducer": reducer})
        base_keys.append(key)
        for s in seeds:
            cfg = {**base_cfg, "random_state": s}
            tasks.append((algo, reducer, cfg))

    if verbose:
        print(
            f"[stability] {len(top_configs)} base configs × {len(seeds)} seeds = {len(tasks)} runs"
        )

    matrix = np.ascontiguousarray(matrix)
    matrix_bytes = matrix.tobytes()
    matrix_shape = matrix.shape
    matrix_dtype = str(matrix.dtype)
    ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
    labels_by_idx: dict[int, np.ndarray | None] = {}
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_stab_worker_init,
        initargs=(matrix_bytes, matrix_shape, matrix_dtype),
    ) as pool:
        futs = {
            pool.submit(_stab_worker_run, i, algo, red, cfg): i
            for i, (algo, red, cfg) in enumerate(tasks)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                labels = fut.result()
            except Exception:
                labels = None
            labels_by_idx[i] = labels

    # Group labels by base config
    out_rows = []
    n_seeds = len(seeds)
    for i, (algo, reducer, _base_cfg) in enumerate(top_configs):
        slice_labels = [labels_by_idx.get(i * n_seeds + j) for j in range(n_seeds)]
        valid = [lbl for lbl in slice_labels if lbl is not None]
        if len(valid) < 2:
            mean_ari = std_ari = mean_nmi = std_nmi = float("nan")
        else:
            mean_ari, std_ari, mean_nmi, std_nmi = _pairwise_metrics(valid)
        n_clusters_list = [
            len({int(x) for x in lbl.tolist() if x >= 0}) for lbl in valid
        ]
        med_nc = float(np.median(n_clusters_list)) if n_clusters_list else float("nan")
        std_nc = float(np.std(n_clusters_list)) if n_clusters_list else float("nan")
        key = base_keys[i]
        out_rows.append(
            {
                "algorithm": algo,
                "reducer": reducer,
                "config_key": key,
                "n_seeds": len(valid),
                "mean_ari": mean_ari,
                "std_ari": std_ari,
                "mean_nmi": mean_nmi,
                "std_nmi": std_nmi,
                "median_n_clusters": med_nc,
                "std_n_clusters": std_nc,
            }
        )
        conn.execute(
            "INSERT OR REPLACE INTO stability "
            "(algorithm, reducer, config_key, n_seeds, mean_ari, std_ari, mean_nmi, std_nmi, median_n_clusters, std_n_clusters) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                algo,
                reducer,
                key,
                len(valid),
                mean_ari,
                std_ari,
                mean_nmi,
                std_nmi,
                med_nc,
                std_nc,
            ),
        )
    conn.close()
    return out_rows


# Workers (top-level for picklability):
_STAB_MAT: list[np.ndarray] = []


def _stab_worker_init(matrix_bytes: bytes, shape: tuple, dtype: str) -> None:
    arr = np.frombuffer(matrix_bytes, dtype=np.dtype(dtype)).reshape(shape)
    _STAB_MAT.append(arr)


def _stab_worker_run(
    idx: int, algorithm: str, reducer: str, cfg: dict
) -> np.ndarray | None:
    matrix = _STAB_MAT[0]
    return _run_and_return_labels(algorithm, reducer, matrix, cfg)


from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

# ── cross-method ARI ──────────────────────────────────────────────────────────


def compute_cross_method_ari(
    matrix: np.ndarray,
    labelled_configs: list[tuple[str, str, str, dict]],
    db_path: str,
    max_workers: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray | None]]:
    """`labelled_configs` is a list of (label, algorithm, reducer, cfg).

    Returns the N×N ARI matrix; also persists into the cross_method table.
    """
    conn = cdb.connect(db_path)
    init_stability_schema(conn)

    matrix = np.ascontiguousarray(matrix)
    matrix_bytes = matrix.tobytes()
    matrix_shape = matrix.shape
    matrix_dtype = str(matrix.dtype)
    ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
    labels_list: list[np.ndarray | None] = [None] * len(labelled_configs)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_stab_worker_init,
        initargs=(matrix_bytes, matrix_shape, matrix_dtype),
    ) as pool:
        futs = {
            pool.submit(_stab_worker_run, i, algo, red, cfg): i
            for i, (_label, algo, red, cfg) in enumerate(labelled_configs)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                labels_list[i] = fut.result()
            except Exception:
                labels_list[i] = None

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    n = len(labelled_configs)
    ari = np.zeros((n, n), dtype=float)
    nmi = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            la, lb = labels_list[i], labels_list[j]
            if la is None or lb is None:
                ari[i, j] = nmi[i, j] = float("nan")
            elif i == j:
                ari[i, j] = nmi[i, j] = 1.0
            else:
                ari[i, j] = adjusted_rand_score(la, lb)
                nmi[i, j] = normalized_mutual_info_score(la, lb)

    for i, j in combinations(range(n), 2):
        conn.execute(
            "INSERT OR REPLACE INTO cross_method (label_a, label_b, ari, nmi) VALUES (?, ?, ?, ?)",
            (
                labelled_configs[i][0],
                labelled_configs[j][0],
                float(ari[i, j]),
                float(nmi[i, j]),
            ),
        )
    conn.close()
    return ari, nmi, labels_list
