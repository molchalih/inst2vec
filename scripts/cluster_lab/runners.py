"""Algorithm runners for the cluster lab.

Each runner takes the raw user-embedding matrix plus a config dict and
returns a `result` dict suitable for db.insert_row(). On failure the
result has `error` populated and metric fields set to None.

All runners score internal metrics in the space the clusters live in:
  - UMAP+HDBSCAN: in the UMAP n-D output (matches the production pipeline).
  - PCA+HDBSCAN: in the PCA-reduced space.
  - everything else: in the (optionally L2-normalized) embedding space.

For high-dimensional embeddings, scoring silhouette / DBCV on the full
matrix can be slow but is the honest comparison for "no-reducer"
algorithms; we keep it.
"""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np

from scripts.cluster_lab import db as cdb
from scripts.cluster_lab.loader import l2_normalize
from scripts.cluster_lab.metrics import all_metrics


def _maybe_normalize(matrix: np.ndarray, normalized: bool | int) -> np.ndarray:
    if int(normalized):
        return l2_normalize(matrix)
    return matrix


def _empty_result(
    algorithm: str, reducer: str, config: dict, error: str | None = None
) -> dict:
    cfg = {**config, "algorithm": algorithm, "reducer": reducer}
    row = {
        "config_hash": cdb.config_hash(cfg),
        "algorithm": algorithm,
        "reducer": reducer,
        "normalized": int(config.get("normalized", 0)),
        "embedding_case": config.get("embedding_case", "sandwich"),
        "random_state": config.get("random_state"),
        "n_clusters": None,
        "noise_ratio": None,
        "n_singletons": None,
        "min_size": None,
        "median_size": None,
        "max_size": None,
        "dbcv": None,
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "error": error,
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
        if k in config:
            row[k] = config[k]
    return row


# ── UMAP + HDBSCAN ────────────────────────────────────────────────────────────


def run_umap_hdbscan(matrix: np.ndarray, config: dict[str, Any]) -> dict:
    try:
        import hdbscan
        from umap import UMAP

        mat = _maybe_normalize(matrix, config.get("normalized", 0))
        reducer = UMAP(
            n_components=int(config["umap_n_components"]),
            n_neighbors=int(config["umap_n_neighbors"]),
            min_dist=float(config["umap_min_dist"]),
            metric=str(config["umap_metric"]),
            init="random",
            random_state=int(config.get("random_state", 42)),
            n_jobs=1,
        )
        nd = np.asarray(reducer.fit_transform(mat))
        min_samples = config.get("hdbscan_min_samples")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(config["hdbscan_min_cluster_size"]),
            min_samples=int(min_samples) if min_samples is not None else None,
            cluster_selection_method=str(config["hdbscan_cluster_selection_method"]),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(nd)
        metrics = all_metrics(
            nd, labels, silhouette_metric="euclidean", dbcv_metric="euclidean"
        )
    except Exception:
        return _empty_result("hdbscan", "umap", config, error=traceback.format_exc())

    out = _empty_result("hdbscan", "umap", config)
    out.update(metrics)
    return out


# ── PCA + HDBSCAN ─────────────────────────────────────────────────────────────


def run_pca_hdbscan(matrix: np.ndarray, config: dict[str, Any]) -> dict:
    try:
        import hdbscan
        from sklearn.decomposition import PCA

        mat = _maybe_normalize(matrix, config.get("normalized", 0))
        pca = PCA(
            n_components=int(config["pca_n_components"]),
            random_state=int(config.get("random_state", 42)),
        )
        nd = pca.fit_transform(mat)
        min_samples = config.get("hdbscan_min_samples")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(config["hdbscan_min_cluster_size"]),
            min_samples=int(min_samples) if min_samples is not None else None,
            cluster_selection_method=str(config["hdbscan_cluster_selection_method"]),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(nd)
        metrics = all_metrics(nd, labels)
    except Exception:
        return _empty_result("hdbscan", "pca", config, error=traceback.format_exc())

    out = _empty_result("hdbscan", "pca", config)
    out.update(metrics)
    return out


# ── direct HDBSCAN on (normalized) embeddings ─────────────────────────────────


def run_hdbscan_direct(matrix: np.ndarray, config: dict[str, Any]) -> dict:
    try:
        import hdbscan

        mat = _maybe_normalize(matrix, config.get("normalized", 0))
        min_samples = config.get("hdbscan_min_samples")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(config["hdbscan_min_cluster_size"]),
            min_samples=int(min_samples) if min_samples is not None else None,
            cluster_selection_method=str(config["hdbscan_cluster_selection_method"]),
            metric="euclidean",
        )
        labels = clusterer.fit_predict(mat)
        metrics = all_metrics(mat, labels)
    except Exception:
        return _empty_result("hdbscan", "none", config, error=traceback.format_exc())

    out = _empty_result("hdbscan", "none", config)
    out.update(metrics)
    return out


# ── KMeans ────────────────────────────────────────────────────────────────────


def run_kmeans(matrix: np.ndarray, config: dict[str, Any]) -> dict:
    try:
        from sklearn.cluster import KMeans

        mat = _maybe_normalize(matrix, config.get("normalized", 0))
        km = KMeans(
            n_clusters=int(config["k"]),
            random_state=int(config.get("random_state", 42)),
            n_init=10,
        )
        labels = km.fit_predict(mat)
        metrics = all_metrics(mat, labels)
    except Exception:
        return _empty_result("kmeans", "none", config, error=traceback.format_exc())

    out = _empty_result("kmeans", "none", config)
    out.update(metrics)
    return out


# ── Gaussian Mixture ──────────────────────────────────────────────────────────


def run_gmm(matrix: np.ndarray, config: dict[str, Any]) -> dict:
    try:
        from sklearn.mixture import GaussianMixture

        mat = _maybe_normalize(matrix, config.get("normalized", 0))
        gmm = GaussianMixture(
            n_components=int(config["k"]),
            covariance_type=str(config.get("covariance_type", "full")),
            random_state=int(config.get("random_state", 42)),
            max_iter=200,
            reg_covar=1e-4,
        )
        labels = gmm.fit_predict(mat)
        metrics = all_metrics(mat, labels)
    except Exception:
        return _empty_result("gmm", "none", config, error=traceback.format_exc())

    out = _empty_result("gmm", "none", config)
    out.update(metrics)
    return out


# ── Agglomerative ─────────────────────────────────────────────────────────────


def run_agglomerative(matrix: np.ndarray, config: dict[str, Any]) -> dict:
    try:
        from sklearn.cluster import AgglomerativeClustering

        mat = _maybe_normalize(matrix, config.get("normalized", 0))
        linkage = str(config.get("linkage", "ward"))
        metric = str(config.get("distance_metric", "euclidean"))
        # sklearn renamed `affinity` → `metric` ≥1.4; keep parameter `metric`.
        agg = AgglomerativeClustering(
            n_clusters=int(config["k"]),
            linkage=linkage,
            metric=metric if linkage != "ward" else "euclidean",
        )
        labels = agg.fit_predict(mat)
        metrics = all_metrics(
            mat, labels, silhouette_metric=metric if linkage != "ward" else "euclidean"
        )
    except Exception:
        return _empty_result(
            "agglomerative", "none", config, error=traceback.format_exc()
        )

    out = _empty_result("agglomerative", "none", config)
    out.update(metrics)
    return out


# ── Spectral ──────────────────────────────────────────────────────────────────


def run_spectral(matrix: np.ndarray, config: dict[str, Any]) -> dict:
    try:
        from sklearn.cluster import SpectralClustering

        mat = _maybe_normalize(matrix, config.get("normalized", 0))
        affinity = str(config.get("affinity", "nearest_neighbors"))
        kwargs: dict[str, Any] = {
            "n_clusters": int(config["k"]),
            "affinity": affinity,
            "random_state": int(config.get("random_state", 42)),
            "assign_labels": "kmeans",
        }
        if affinity == "nearest_neighbors":
            kwargs["n_neighbors"] = int(config.get("n_neighbors", 10))
        sc = SpectralClustering(**kwargs)
        labels = sc.fit_predict(mat)
        metrics = all_metrics(mat, labels)
    except Exception:
        return _empty_result("spectral", "none", config, error=traceback.format_exc())

    out = _empty_result("spectral", "none", config)
    out.update(metrics)
    return out


# ── Dispatch ──────────────────────────────────────────────────────────────────


_DISPATCH = {
    ("hdbscan", "umap"): run_umap_hdbscan,
    ("hdbscan", "pca"): run_pca_hdbscan,
    ("hdbscan", "none"): run_hdbscan_direct,
    ("kmeans", "none"): run_kmeans,
    ("gmm", "none"): run_gmm,
    ("agglomerative", "none"): run_agglomerative,
    ("spectral", "none"): run_spectral,
}


def get_runner(algorithm: str, reducer: str):
    return _DISPATCH[(algorithm, reducer)]


def run(
    algorithm: str, reducer: str, matrix: np.ndarray, config: dict[str, Any]
) -> dict:
    return get_runner(algorithm, reducer)(matrix, config)
