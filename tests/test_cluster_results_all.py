import os
import sys

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
    dbcv: float | None,
    silhouette: float | None,
    n_clusters: int,
    noise_ratio: float,
    disqualified: int | None = None,
    in_current_grid: int | None = None,
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
        disqualified=disqualified,
        in_current_grid=in_current_grid,
    )


def test_summarize_audio_fixed_format():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(
            _run_row(
                embedding_case="audio",
                umap_n_neighbors=10,
                dbcv=0.2,
                silhouette=-0.5,
                n_clusters=2,
                noise_ratio=0.1,
            )
        )
        s.add(
            _run_row(
                embedding_case="audio",
                umap_n_neighbors=11,
                dbcv=0.0,
                silhouette=0.1,
                n_clusters=3,
                noise_ratio=0.2,
            )
        )
        s.add(
            _run_row(
                embedding_case="video",
                umap_n_neighbors=12,
                dbcv=0.9,
                silhouette=0.5,
                n_clusters=5,
                noise_ratio=0.0,
            )
        )
        s.commit()

    from generators.cluster_results_all import summarize_to_lines

    lines = summarize_to_lines(eng, case="audio", include_filtered=True)
    out = "\n".join(lines) + "\n"
    expected = """case: audio
n_runs: 2
dbcv_mean: 0.100
dbcv_std: 0.141
dbcv_min: 0.000
dbcv_max: 0.200
silhouette_mean: -0.200
silhouette_std: 0.424
k_mean: 2.500
k_median: 2.500
pct_k_le_2: 50.0
pct_k_le_3: 100.0
noise_pct_mean: 15.000
noise_pct_std: 7.071
pct_negative_silhouette: 50.0
pct_dbcv_lt_0_1: 50.0
pct_dbcv_lt_0: 0.0
"""
    assert out == expected


def test_exclude_filtered_drops_rows():
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
                disqualified=0,
                in_current_grid=1,
            )
        )
        s.add(
            _run_row(
                embedding_case="audio",
                umap_n_neighbors=11,
                dbcv=0.9,
                silhouette=0.0,
                n_clusters=4,
                noise_ratio=0.0,
                disqualified=1,
                in_current_grid=1,
            )
        )
        s.add(
            _run_row(
                embedding_case="audio",
                umap_n_neighbors=12,
                dbcv=0.1,
                silhouette=0.0,
                n_clusters=2,
                noise_ratio=0.2,
                disqualified=0,
                in_current_grid=0,
            )
        )
        s.commit()

    from generators.cluster_results_all import summarize_to_lines

    all_lines = summarize_to_lines(eng, case="audio", include_filtered=True)
    filt_lines = summarize_to_lines(eng, case="audio", include_filtered=False)
    assert any("n_runs: 3" in ln for ln in all_lines)
    assert any("n_runs: 1" in ln for ln in filt_lines)
