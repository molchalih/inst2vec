import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace

from modules.visualization import _plot_case


def _rows(xs, ys, labels):
    return [SimpleNamespace(umap_x=x, umap_y=y, cluster_id=c)
            for x, y, c in zip(xs, ys, labels)]


def test_plot_case_returns_figure():
    rows = _rows([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0, 0, 1])
    fig = _plot_case("video", rows)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_case_handles_noise_points():
    rows = _rows([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [-1, 0, 1])
    fig = _plot_case("sandwich", rows)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_case_all_noise():
    rows = _rows([0.0, 1.0], [0.0, 1.0], [-1, -1])
    fig = _plot_case("audio", rows)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_case_single_cluster():
    rows = _rows([0.1, 0.2], [0.3, 0.4], [0, 0])
    fig = _plot_case("video", rows)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_case_title_contains_case_name():
    rows = _rows([0.0], [0.0], [0])
    fig = _plot_case("sandwich", rows)
    assert "sandwich" in fig.axes[0].get_title()
    plt.close(fig)
