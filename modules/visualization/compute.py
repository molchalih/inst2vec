"""Pure compute helpers for the visualization stage.

No DB, no filesystem — functions take primitive inputs and return
primitive outputs / dataclasses, so they are trivially unit-tested.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.database import UserCluster, VisualizationCluster, VisualizationUser


@dataclass(frozen=True)
class Ellipse:
    cx: float
    cy: float
    rx: float
    ry: float
    angle: float  # radians; principal axis from +x


def fit_cluster_ellipse(xs: np.ndarray, ys: np.ndarray, sigma: float = 2.0) -> Ellipse:
    """2σ covariance ellipse over a cluster's member positions.

    Degenerate input (n<2) returns a tiny non-zero ellipse at the
    centroid so downstream consumers always see rx, ry > 0.
    """
    n = len(xs)
    cx = float(xs.mean())
    cy = float(ys.mean())
    if n < 2:
        return Ellipse(cx=cx, cy=cy, rx=1e-6, ry=1e-6, angle=0.0)
    cov = np.cov(np.vstack([xs, ys]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    rx = float(sigma * np.sqrt(max(float(eigvals[0]), 0.0)))
    ry = float(sigma * np.sqrt(max(float(eigvals[1]), 0.0)))
    angle = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    return Ellipse(cx=cx, cy=cy, rx=rx, ry=ry, angle=angle)


@dataclass(frozen=True)
class CasePayload:
    case: str
    label: str
    users: list[VisualizationUser]
    clusters: list[VisualizationCluster]


def build_case_payload(
    case: str, label: str, user_rows: Sequence[UserCluster]
) -> CasePayload:
    """Group UserCluster rows by cluster_id, fit ellipses for real
    clusters (id >= 0), skip noise (id == -1) entirely from the cluster
    table. Noise users still appear in `users` with cluster_id = -1.
    """
    users = [
        VisualizationUser(
            user_id=r.user_id,
            embedding_case=case,
            x=float(r.umap_x),
            y=float(r.umap_y),
            cluster_id=int(r.cluster_id),
        )
        for r in user_rows
    ]
    by_cluster: dict[int, list[UserCluster]] = defaultdict(list)
    for r in user_rows:
        if r.cluster_id >= 0:
            by_cluster[int(r.cluster_id)].append(r)
    clusters: list[VisualizationCluster] = []
    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        xs = np.fromiter(
            (float(m.umap_x) for m in members), dtype=np.float64, count=len(members)
        )
        ys = np.fromiter(
            (float(m.umap_y) for m in members), dtype=np.float64, count=len(members)
        )
        e = fit_cluster_ellipse(xs, ys)
        clusters.append(
            VisualizationCluster(
                embedding_case=case,
                cluster_id=cid,
                cx=e.cx,
                cy=e.cy,
                rx=e.rx,
                ry=e.ry,
                angle=e.angle,
                size=len(members),
                label=f"Cluster {cid + 1}",
            )
        )
    return CasePayload(case=case, label=label, users=users, clusters=clusters)
