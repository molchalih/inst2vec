import os
import sys

from IPython.display import Markdown
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.database import Base, ClusterRun


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _run_row(
    *,
    embedding_case: str,
    umap_n_neighbors: int,
    dbcv: float | None,
    silhouette: float | None,
    n_clusters: int,
    noise_ratio: float,
    passes_validation: bool | None = True,
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
        hdbscan_min_cluster_size=15,
        hdbscan_min_samples=None,
        hdbscan_cluster_selection_method="eom",
        hdbscan_metric="euclidean",
        random_state=42,
        n_clusters=n_clusters,
        noise_ratio=noise_ratio,
        min_size=1,
        median_size=2,
        max_size=5,
        dbcv=dbcv,
        silhouette=silhouette,
        passes_validation=passes_validation,
    )


def test_summarize_all_to_markdown_unified_table():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(
            _run_row(
                embedding_case="audio",
                umap_n_neighbors=10,
                dbcv=0.5,
                silhouette=0.0,
                n_clusters=2,
                noise_ratio=0.1,
                passes_validation=True,
            )
        )
        s.add(
            _run_row(
                embedding_case="video",
                umap_n_neighbors=11,
                dbcv=0.9,
                silhouette=0.3,
                n_clusters=4,
                noise_ratio=0.0,
                passes_validation=True,
            )
        )
        s.add(
            _run_row(
                embedding_case="sandwich",
                umap_n_neighbors=12,
                dbcv=0.7,
                silhouette=0.2,
                n_clusters=3,
                noise_ratio=0.05,
                passes_validation=True,
            )
        )
        s.commit()

    from docs.reporting.tables.clustering_all import (
        summarize_all_to_markdown,
    )

    out = summarize_all_to_markdown(
        eng,
        cases=("audio", "video", "sandwich"),
    )

    assert "| Metric | audio | video | sandwich |" in out
    assert "| runs | 1 | 1 | 1 |" in out
    assert "| runs filtered | 1 | 1 | 1 |" in out
    assert "| mean DBCV | 0.500 | 0.900 | 0.700 |" in out
    assert "| mean silhouette | 0.000 | 0.300 | 0.200 |" in out


def test_render_clustering_summary_returns_markdown_object():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(
            _run_row(
                embedding_case="audio",
                umap_n_neighbors=10,
                dbcv=0.5,
                silhouette=0.1,
                n_clusters=2,
                noise_ratio=0.1,
                passes_validation=True,
            )
        )
        s.add(
            _run_row(
                embedding_case="video",
                umap_n_neighbors=11,
                dbcv=0.9,
                silhouette=0.3,
                n_clusters=4,
                noise_ratio=0.0,
                passes_validation=True,
            )
        )
        s.add(
            _run_row(
                embedding_case="sandwich",
                umap_n_neighbors=12,
                dbcv=0.7,
                silhouette=0.2,
                n_clusters=3,
                noise_ratio=0.05,
                passes_validation=True,
            )
        )
        s.commit()

    from docs.quarto_helpers import render_clustering_summary
    from docs.reporting.tables.clustering_all import (
        summarize_all_to_markdown,
    )

    rendered = render_clustering_summary(eng=eng)

    assert isinstance(rendered, Markdown)
    assert rendered.data == summarize_all_to_markdown(
        eng,
        cases=("video", "sandwich", "audio"),
    )


def test_summarize_all_to_markdown_raises_on_empty_cases():
    eng = _make_engine()

    from docs.reporting.tables.clustering_all import (
        summarize_all_to_markdown,
    )

    try:
        summarize_all_to_markdown(eng, cases=())
    except ValueError as exc:
        assert str(exc) == "cases must contain at least one embedding case"
    else:
        raise AssertionError("Expected ValueError for empty cases tuple")
