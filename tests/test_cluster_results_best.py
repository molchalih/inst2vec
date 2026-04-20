import os
import sys

from IPython.display import Markdown
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.database import Base, ClusterRun


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _run_row(
    *,
    embedding_case: str,
    umap_n_neighbors: int,
    dbcv: float,
    param_plateau_score: float,
    silhouette: float,
    n_clusters: int,
    noise_ratio: float,
    disqualified: int = 0,
    in_current_grid: int = 1,
) -> ClusterRun:
    return ClusterRun(
        embedding_case=embedding_case,
        umap_n_components=15,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=0.0,
        umap_metric="cosine",
        umap2d_n_neighbors=15,
        umap2d_min_dist=0.1,
        umap2d_metric="cosine",
        hdbscan_min_cluster_size=20,
        hdbscan_min_samples=None,
        hdbscan_cluster_selection_method="eom",
        hdbscan_metric="euclidean",
        random_state=42,
        n_clusters=n_clusters,
        noise_ratio=noise_ratio,
        min_size=1,
        median_size=2,
        max_size=5,
        disqualified=disqualified,
        dbcv=dbcv,
        silhouette=silhouette,
        param_plateau_score=param_plateau_score,
        in_current_grid=in_current_grid,
    )


def test_best_run_to_markdown_uses_validation_pick_for_case(monkeypatch):
    monkeypatch.setenv("VALIDATION_PLATEAU_DROP_THRESHOLD", "0.05")
    eng = _make_engine()

    with Session(eng) as s:
        s.add(
            _run_row(
                embedding_case="audio",
                umap_n_neighbors=10,
                dbcv=0.90,
                param_plateau_score=0.80,
                silhouette=0.10,
                n_clusters=6,
                noise_ratio=0.20,
            )
        )
        s.add(
            _run_row(
                embedding_case="audio",
                umap_n_neighbors=20,
                dbcv=0.82,
                param_plateau_score=0.79,
                silhouette=0.25,
                n_clusters=4,
                noise_ratio=0.10,
            )
        )
        s.commit()

    from generators.cluster_results_best import best_run_to_markdown

    out = best_run_to_markdown(eng, "audio")

    assert out.startswith("| Field | Value |")
    assert "| $\\mathrm{DBCV}^*$ | 0.8200 |" in out
    assert "| $n_{\\mathrm{UMAP}}^*$ | 20 |" in out


def test_best_runs_all_to_markdown_unified_table(monkeypatch):
    monkeypatch.setenv("VALIDATION_PLATEAU_DROP_THRESHOLD", "0.05")
    eng = _make_engine()

    with Session(eng) as s:
        s.add(
            _run_row(
                embedding_case="audio",
                umap_n_neighbors=10,
                dbcv=0.90,
                param_plateau_score=0.80,
                silhouette=0.10,
                n_clusters=6,
                noise_ratio=0.20,
            )
        )
        s.add(
            _run_row(
                embedding_case="audio",
                umap_n_neighbors=20,
                dbcv=0.82,
                param_plateau_score=0.79,
                silhouette=0.25,
                n_clusters=4,
                noise_ratio=0.10,
            )
        )
        s.add(
            _run_row(
                embedding_case="video",
                umap_n_neighbors=11,
                dbcv=0.70,
                param_plateau_score=0.68,
                silhouette=0.21,
                n_clusters=5,
                noise_ratio=0.12,
            )
        )
        s.add(
            _run_row(
                embedding_case="sandwich",
                umap_n_neighbors=12,
                dbcv=0.55,
                param_plateau_score=0.52,
                silhouette=0.15,
                n_clusters=3,
                noise_ratio=0.08,
            )
        )
        s.commit()

    from generators.cluster_results_best import best_runs_all_to_markdown

    out = best_runs_all_to_markdown(eng, cases=("audio", "video", "sandwich"))

    assert "| Field | audio | video | sandwich |" in out
    assert "| $\\mathrm{DBCV}^*$ | 0.8200 | 0.7000 | 0.5500 |" in out


def test_render_best_cluster_run_returns_markdown_object(monkeypatch):
    monkeypatch.setenv("VALIDATION_PLATEAU_DROP_THRESHOLD", "0.05")
    eng = _make_engine()

    with Session(eng) as s:
        s.add(
            _run_row(
                embedding_case="video",
                umap_n_neighbors=11,
                dbcv=0.70,
                param_plateau_score=0.68,
                silhouette=0.21,
                n_clusters=5,
                noise_ratio=0.12,
            )
        )
        s.commit()

    from docs.quarto_helpers import render_best_cluster_run
    from generators.cluster_results_best import best_runs_all_to_markdown

    rendered = render_best_cluster_run(eng=eng)

    assert isinstance(rendered, Markdown)
    assert rendered.data == best_runs_all_to_markdown(
        eng,
        cases=("audio", "video", "sandwich"),
    )


def test_best_run_to_markdown_delegates_selection(monkeypatch):
    eng = _make_engine()
    with Session(eng) as s:
        row = _run_row(
            embedding_case="audio",
            umap_n_neighbors=20,
            dbcv=0.82,
            param_plateau_score=0.79,
            silhouette=0.25,
            n_clusters=4,
            noise_ratio=0.10,
        )
        s.add(row)
        s.commit()
        row_id = row.id

    def fake_select(session, case, threshold=None):
        return session.get(ClusterRun, row_id)

    monkeypatch.setattr("generators.cluster_results_best.select_best_cluster_run", fake_select)

    from generators.cluster_results_best import best_run_to_markdown

    out = best_run_to_markdown(eng, "audio")

    assert "| $\\mathrm{DBCV}^*$ | 0.8200 |" in out
