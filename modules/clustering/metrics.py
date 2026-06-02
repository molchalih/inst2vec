"""Per-cluster quality metrics derived from a champion ``ClusterResult``.

Computed once, during the assign stage, alongside the ``UserCluster`` rows —
the persistence and per-cluster DBCV scores need the in-memory HDBSCAN
clusterer and the pass-1 UMAP matrix, neither of which is persisted, so they
cannot be recovered at report time. Everything here is pure: no DB, no I/O.

Per cluster (noise label ``-1`` excluded):
- ``n_users``           — members assigned to the cluster
- ``mean_centrality``   — mean HDBSCAN soft membership of its members
- ``high_conf_fraction``— share of members with centrality > threshold
- ``persistence``       — HDBSCAN-native cluster stability (condensed-tree)
- ``dbcv``              — per-cluster density-based validity (``validity_index``)
"""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan.validity
import numpy as np

from modules.clustering.core import (
    DEFAULT_HDBSCAN_METRIC,
    ClusterResult,
    resolve_hdbscan_metric,
)

# Members above this soft-membership probability count as "high confidence".
HIGH_CONFIDENCE_THRESHOLD = 0.8


@dataclass(frozen=True)
class ClusterMetricRow:
    """One per-cluster quality record (one non-noise HDBSCAN cluster)."""

    cluster_id: int
    n_users: int
    mean_centrality: float
    high_conf_fraction: float
    persistence: float | None
    dbcv: float | None


def _non_noise_labels(labels: np.ndarray) -> list[int]:
    return sorted({int(lbl) for lbl in labels} - {-1})


def per_cluster_dbcv(
    matrix_nd: np.ndarray, labels: np.ndarray, metric: str
) -> dict[int, float]:
    """Per-cluster DBCV via ``hdbscan.validity.validity_index``.

    Returns ``{cluster_id: score}``. The validity index returns one score per
    sorted non-noise label; we map those back to the cluster ids. On any
    degenerate input (single cluster, too-few points) the index raises — we
    treat that as "no per-cluster DBCV available" and return an empty dict.
    """
    try:
        _, per_cluster = hdbscan.validity.validity_index(
            matrix_nd.astype(np.float64),
            labels,
            metric=metric,
            per_cluster_scores=True,
        )
    except Exception:
        return {}
    out: dict[int, float] = {}
    for idx, cid in enumerate(_non_noise_labels(labels)):
        if idx < len(per_cluster):
            score = float(per_cluster[idx])
            if np.isfinite(score):
                out[cid] = score
    return out


def compute_cluster_metrics(
    result: ClusterResult,
    *,
    metric: str | None = None,
    high_conf_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> list[ClusterMetricRow]:
    """One ``ClusterMetricRow`` per non-noise cluster, ordered by cluster id.

    ``dbcv`` is ``None`` for every cluster when ``result.matrix_nd`` is absent
    (the assign stage requests it; callers that don't simply get no DBCV).
    ``persistence`` is ``None`` when the result carries no persistence vector
    or the cluster id falls outside it.
    """
    labels = result.labels
    centralities = result.centralities
    has_centrality = centralities.size == labels.size
    persistence = result.cluster_persistence
    resolved_metric = resolve_hdbscan_metric(metric or DEFAULT_HDBSCAN_METRIC)

    dbcv_by_cluster = (
        per_cluster_dbcv(result.matrix_nd, labels, resolved_metric)
        if result.matrix_nd is not None
        else {}
    )

    rows: list[ClusterMetricRow] = []
    for cid in _non_noise_labels(labels):
        mask = labels == cid
        n_users = int(mask.sum())
        if has_centrality:
            member_c = centralities[mask]
            mean_centrality = float(member_c.mean())
            high_conf_fraction = float((member_c > high_conf_threshold).mean())
        else:
            mean_centrality = 0.0
            high_conf_fraction = 0.0
        pers = float(persistence[cid]) if cid < persistence.size else None
        rows.append(
            ClusterMetricRow(
                cluster_id=cid,
                n_users=n_users,
                mean_centrality=mean_centrality,
                high_conf_fraction=high_conf_fraction,
                persistence=pers,
                dbcv=dbcv_by_cluster.get(cid),
            )
        )
    return rows


__all__ = (
    "HIGH_CONFIDENCE_THRESHOLD",
    "ClusterMetricRow",
    "compute_cluster_metrics",
    "per_cluster_dbcv",
)
