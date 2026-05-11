import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.database import Base, User, UserCluster


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _seed_clusters(eng):
    with Session(eng) as s:
        s.add_all(
            [
                User(
                    pk=1, username="alpha", parse_status="success", user_disqualified=0
                ),
                User(
                    pk=2, username="beta", parse_status="success", user_disqualified=0
                ),
                UserCluster(
                    user_pk=1,
                    embedding_case="audio",
                    cluster_id=0,
                    umap_x=0.1,
                    umap_y=0.2,
                ),
                UserCluster(
                    user_pk=2,
                    embedding_case="audio",
                    cluster_id=-1,
                    umap_x=0.3,
                    umap_y=0.4,
                ),
                UserCluster(
                    user_pk=1,
                    embedding_case="video",
                    cluster_id=1,
                    umap_x=1.1,
                    umap_y=1.2,
                ),
                UserCluster(
                    user_pk=2,
                    embedding_case="video",
                    cluster_id=1,
                    umap_x=1.3,
                    umap_y=1.4,
                ),
                UserCluster(
                    user_pk=1,
                    embedding_case="sandwich",
                    cluster_id=2,
                    umap_x=2.1,
                    umap_y=2.2,
                ),
                UserCluster(
                    user_pk=2,
                    embedding_case="sandwich",
                    cluster_id=2,
                    umap_x=2.3,
                    umap_y=2.4,
                ),
            ]
        )
        s.commit()


def test_cluster_plot_figure_for_case_returns_matplotlib_figure():
    eng = _make_engine()
    _seed_clusters(eng)

    from generators.cluster_case_plots import cluster_plot_figure_for_case

    fig = cluster_plot_figure_for_case(eng, "audio")

    assert isinstance(fig, Figure)
    assert "audio" in fig.axes[0].get_title().lower()
    plt.close(fig)


def test_cluster_plot_figure_for_case_hides_cluster_legend():
    eng = _make_engine()
    _seed_clusters(eng)

    from generators.cluster_case_plots import cluster_plot_figure_for_case

    fig = cluster_plot_figure_for_case(eng, "audio")

    assert fig.axes[0].get_legend() is None
    plt.close(fig)


def test_cluster_plot_figure_for_case_rejects_unknown_case():
    eng = _make_engine()

    from generators.cluster_case_plots import cluster_plot_figure_for_case

    with pytest.raises(ValueError, match="unknown embedding case"):
        cluster_plot_figure_for_case(eng, "bogus")


def test_render_audio_cluster_plot_returns_figure():
    eng = _make_engine()
    _seed_clusters(eng)

    from docs.quarto_helpers import render_audio_cluster_plot

    fig = render_audio_cluster_plot(eng=eng)

    assert isinstance(fig, Figure)
    assert "audio" in fig.axes[0].get_title().lower()
    plt.close(fig)


def test_render_multimodal_cluster_plot_maps_to_sandwich_case(monkeypatch):
    import docs.quarto_helpers as quarto_helpers

    calls = {}

    def fake_render(case: str, *, eng=None, title_label=None):
        calls["case"] = case
        calls["eng"] = eng
        calls["title_label"] = title_label
        return "figure"

    monkeypatch.setattr(quarto_helpers, "render_cluster_plot_by_case", fake_render)

    rendered = quarto_helpers.render_multimodal_cluster_plot(eng="db")

    assert rendered == "figure"
    assert calls == {"case": "sandwich", "eng": "db", "title_label": "multimodal"}
