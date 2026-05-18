"""Plot subpackage: scatter plots produced by the pipeline and by Quarto."""

import matplotlib

matplotlib.use("Agg")

from core.config import Secrets, Settings
from modules.visualization.plots.cluster_plots import plot_clusters

__all__ = ("plot_clusters", "run")


def run(settings: Settings, secrets: Secrets) -> None:
    """Render scatter plots for assigned clusters."""
    plot_clusters(plots_dir=settings.paths.plots_dir)
