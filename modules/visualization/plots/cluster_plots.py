"""Pipeline entry point: render and save UMAP cluster scatter plots as PNGs."""

import os

import matplotlib.pyplot as plt

from core.console import log
from core.database import UserCluster, get_session
from modules.visualization.plots.cluster_case_plots import cluster_plot_figure_for_case

__all__ = ("plot_clusters",)


def plot_clusters(plots_dir: str) -> None:
    """Load user_clusters from DB and save one PNG per embedding case."""
    os.makedirs(plots_dir, exist_ok=True)
    session = get_session()
    try:
        eng = session.get_bind()
        cases = sorted(
            r[0] for r in session.query(UserCluster.embedding_case).distinct().all()
        )
        for case in cases:
            fig = cluster_plot_figure_for_case(eng, case)
            path = os.path.join(plots_dir, f"clusters_{case}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            log("viz", f"saved {path}", level="ok")
    finally:
        session.close()
