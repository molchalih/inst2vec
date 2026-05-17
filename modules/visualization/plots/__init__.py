"""Plot subpackage: scatter plots produced by the pipeline and by Quarto."""

import matplotlib

matplotlib.use("Agg")

from modules.visualization.plots.cluster_plots import plot_clusters

__all__ = ("plot_clusters",)
