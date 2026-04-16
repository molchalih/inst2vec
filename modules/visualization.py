"""Save UMAP cluster scatter plots to data/plots/ as PNG files."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modules.database import get_session, UserCluster

PLOTS_DIR = "data/plots"


def _plot_case(case: str, rows: list) -> plt.Figure:
    """Build a scatter figure for one embedding case from UserCluster-like rows."""
    xs = np.array([r.umap_x for r in rows], dtype=np.float32)
    ys = np.array([r.umap_y for r in rows], dtype=np.float32)
    labels = np.array([r.cluster_id for r in rows], dtype=np.int32)

    fig, ax = plt.subplots(figsize=(10, 8))

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            xs[noise_mask], ys[noise_mask],
            c="lightgray", s=12, alpha=0.5, linewidths=0,
            label="noise", zorder=1,
        )

    unique_clusters = sorted(set(labels[~noise_mask].tolist()))
    cmap = plt.colormaps.get_cmap("tab20")
    for i, cid in enumerate(unique_clusters):
        mask = labels == cid
        ax.scatter(
            xs[mask], ys[mask],
            color=cmap(i % 20), s=18, alpha=0.8, linewidths=0,
            label=f"cluster {cid}", zorder=2,
        )

    n_clusters = len(unique_clusters)
    ax.set_title(f"UMAP — {case} embeddings ({n_clusters} clusters)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", markerscale=2, fontsize=7, ncol=2)
    ax.set_aspect("equal", "datalim")
    fig.tight_layout()
    return fig


def plot_clusters() -> None:
    """Load user_clusters from DB and save one PNG per embedding case."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    session = get_session()
    try:
        cases = sorted(
            r[0]
            for r in session.query(UserCluster.embedding_case).distinct().all()
        )
        for case in cases:
            rows = (
                session.query(UserCluster)
                .filter(UserCluster.embedding_case == case)
                .all()
            )
            if not rows:
                continue
            fig = _plot_case(case, rows)
            path = os.path.join(PLOTS_DIR, f"clusters_{case}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            n_clusters = len({r.cluster_id for r in rows if r.cluster_id >= 0})
            noise = sum(1 for r in rows if r.cluster_id == -1)
            print(f"[viz] saved {path} ({n_clusters} clusters, {noise} noise points)")
    finally:
        session.close()
