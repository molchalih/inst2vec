import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import matplotlib

matplotlib.use("Agg")
from types import SimpleNamespace
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from generators.cluster_case_plots import _plot_case
from modules.visualization import plot_clusters


def _rows(xs, ys, labels):
    return [
        SimpleNamespace(umap_x=x, umap_y=y, cluster_id=c)
        for x, y, c in zip(xs, ys, labels, strict=False)
    ]


def test_plot_case_returns_figure():
    rows = _rows([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0, 0, 1])
    fig = _plot_case("video", rows)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_case_handles_noise_points():
    rows = _rows([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [-1, 0, 1])
    fig = _plot_case("sandwich", rows)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_case_all_noise():
    rows = _rows([0.0, 1.0], [0.0, 1.0], [-1, -1])
    fig = _plot_case("audio", rows)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_case_single_cluster():
    rows = _rows([0.1, 0.2], [0.3, 0.4], [0, 0])
    fig = _plot_case("video", rows)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_case_title_contains_case_name():
    rows = _rows([0.0], [0.0], [0])
    fig = _plot_case("sandwich", rows)
    assert "sandwich" in fig.axes[0].get_title()
    plt.close(fig)


def test_plot_clusters_uses_shared_cluster_plot_generator(monkeypatch, tmp_path):
    fake_rows = [
        SimpleNamespace(embedding_case="audio"),
        SimpleNamespace(embedding_case="video"),
        SimpleNamespace(embedding_case="sandwich"),
    ]

    query = MagicMock()
    query.distinct.return_value.all.return_value = [
        ("audio",),
        ("video",),
        ("sandwich",),
    ]
    query.filter.return_value.all.side_effect = [
        [fake_rows[0]],
        [fake_rows[1]],
        [fake_rows[2]],
    ]

    fake_session = MagicMock()
    fake_session.query.return_value = query

    called_cases = []

    def fake_cluster_plot_figure_for_case(eng, case: str, *, title_label=None):
        called_cases.append((eng, case, title_label))
        return plt.figure()

    monkeypatch.setattr("modules.visualization.get_session", lambda: fake_session)
    monkeypatch.setattr("modules.visualization.PLOTS_DIR", str(tmp_path))
    monkeypatch.setattr(
        "modules.visualization.cluster_plot_figure_for_case",
        fake_cluster_plot_figure_for_case,
    )

    plot_clusters()

    assert called_cases == [
        (fake_session.get_bind.return_value, "audio", None),
        (fake_session.get_bind.return_value, "sandwich", None),
        (fake_session.get_bind.return_value, "video", None),
    ]
