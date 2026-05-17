"""Paper-facing cluster plot generators for Quarto and pipeline reuse."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sqlalchemy.orm import Session

from core.database import UserCluster
from modules.clustering import DEFAULT_CASES

__all__ = ("cluster_plot_figure_for_case",)


def _validate_case(case: str) -> str:
    if case not in DEFAULT_CASES:
        raise ValueError(f"unknown embedding case: {case}")
    return case


def _plot_case(case: str, rows: list) -> Figure:
    """Build a scatter figure for one embedding case from UserCluster-like rows."""
    xs = np.array([r.umap_x for r in rows], dtype=np.float32)
    ys = np.array([r.umap_y for r in rows], dtype=np.float32)
    labels = np.array([r.cluster_id for r in rows], dtype=np.int32)

    fig, ax = plt.subplots(figsize=(10, 8))

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            xs[noise_mask],
            ys[noise_mask],
            c="lightgray",
            s=12,
            alpha=0.5,
            linewidths=0,
            label="noise",
            zorder=1,
        )

    unique_clusters = sorted(set(labels[~noise_mask].tolist()))
    cmap = plt.colormaps.get_cmap("tab20")
    for i, cid in enumerate(unique_clusters):
        mask = labels == cid
        ax.scatter(
            xs[mask],
            ys[mask],
            color=cmap(i % 20),
            s=18,
            alpha=0.8,
            linewidths=0,
            label=f"cluster {cid}",
            zorder=2,
        )

    n_clusters = len(unique_clusters)
    ax.set_title(f"{case}; ({n_clusters} clusters)")
    ax.set_aspect("equal", "datalim")
    fig.tight_layout()
    return fig


def cluster_plot_figure_for_case(eng, case: str, *, title_label: str | None = None):
    case = _validate_case(case)
    with Session(eng) as session:
        rows = (
            session.query(UserCluster).filter(UserCluster.embedding_case == case).all()
        )

    if not rows:
        raise ValueError(f"no cluster rows found for embedding case: {case}")

    return _plot_case(title_label or case, rows)
