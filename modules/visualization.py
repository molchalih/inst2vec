"""Save UMAP cluster scatter plots to data/plots/ as PNG files."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generators.cluster_case_plots import cluster_plot_figure_for_case
from modules.database import get_session, UserCluster
from modules.console import log

PLOTS_DIR = "data/plots"


def plot_clusters() -> None:
    """Load user_clusters from DB and save one PNG per embedding case."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    session = get_session()
    try:
        eng = session.get_bind()
        cases = sorted(
            r[0]
            for r in session.query(UserCluster.embedding_case).distinct().all()
        )
        for case in cases:
            fig = cluster_plot_figure_for_case(eng, case)
            path = os.path.join(PLOTS_DIR, f"clusters_{case}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            log("viz", f"saved {path}", level="ok")
    finally:
        session.close()
