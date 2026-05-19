"""Per-run cluster metrics with NaN-safe wrappers."""

from __future__ import annotations

import warnings

import numpy as np


def cluster_summary(labels: np.ndarray) -> dict:
    """Cluster-count, noise, and size statistics. Negative labels = noise."""
    labels = np.asarray(labels)
    unique = [int(lbl) for lbl in set(labels.tolist()) if lbl >= 0]
    n_clusters = len(unique)
    n_total = len(labels)
    n_noise = int(np.sum(labels == -1))
    noise_ratio = n_noise / n_total if n_total else 0.0
    sizes = sorted(
        (int(np.sum(labels == lbl)) for lbl in unique),
        reverse=True,
    )
    n_singletons = int(sum(1 for s in sizes if s == 1))
    if sizes:
        min_size = sizes[-1]
        max_size = sizes[0]
        median_size = int(sizes[len(sizes) // 2])
    else:
        min_size = max_size = median_size = 0
    return {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "n_singletons": n_singletons,
        "min_size": min_size,
        "median_size": median_size,
        "max_size": max_size,
    }


def _scored_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of points that contribute to internal scores (non-noise)."""
    return np.asarray(labels) >= 0


def _has_two_clusters(labels: np.ndarray) -> bool:
    mask = _scored_mask(labels)
    if mask.sum() < 2:
        return False
    return len(set(labels[mask].tolist())) >= 2


def safe_dbcv(
    matrix: np.ndarray, labels: np.ndarray, metric: str = "euclidean"
) -> float | None:
    """hdbscan.validity_index with all guards. Returns None on failure."""
    try:
        import hdbscan
    except Exception:
        return None
    if not _has_two_clusters(labels):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = hdbscan.validity.validity_index(
                matrix.astype(np.float64),
                np.asarray(labels),
                metric=metric,
            )
        if val is None or not np.isfinite(val):
            return None
        return float(val)
    except Exception:
        return None


def safe_silhouette(
    matrix: np.ndarray, labels: np.ndarray, metric: str = "euclidean"
) -> float | None:
    from sklearn.metrics import silhouette_score

    if not _has_two_clusters(labels):
        return None
    try:
        mask = _scored_mask(labels)
        return float(silhouette_score(matrix[mask], labels[mask], metric=metric))
    except Exception:
        return None


def safe_calinski_harabasz(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    from sklearn.metrics import calinski_harabasz_score

    if not _has_two_clusters(labels):
        return None
    try:
        mask = _scored_mask(labels)
        return float(calinski_harabasz_score(matrix[mask], labels[mask]))
    except Exception:
        return None


def safe_davies_bouldin(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    from sklearn.metrics import davies_bouldin_score

    if not _has_two_clusters(labels):
        return None
    try:
        mask = _scored_mask(labels)
        return float(davies_bouldin_score(matrix[mask], labels[mask]))
    except Exception:
        return None


def all_metrics(
    matrix_for_scoring: np.ndarray,
    labels: np.ndarray,
    silhouette_metric: str = "euclidean",
    dbcv_metric: str = "euclidean",
) -> dict:
    """Bundle of all per-run metrics."""
    out = cluster_summary(labels)
    out["dbcv"] = safe_dbcv(matrix_for_scoring, labels, metric=dbcv_metric)
    out["silhouette"] = safe_silhouette(
        matrix_for_scoring, labels, metric=silhouette_metric
    )
    out["calinski_harabasz"] = safe_calinski_harabasz(matrix_for_scoring, labels)
    out["davies_bouldin"] = safe_davies_bouldin(matrix_for_scoring, labels)
    return out
