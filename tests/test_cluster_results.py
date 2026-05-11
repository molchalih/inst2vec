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
    dbcv: float | None,
    silhouette: float | None,
    n_clusters: int,
    noise_ratio: float,
    param_plateau_score: float | None = None,
    disqualified: int | None = 0,
    in_current_grid: int | None = 1,
) -> ClusterRun:
    return ClusterRun(
        embedding_case=embedding_case,
        umap_n_components=15,
        umap_n_neighbors=10,
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
        disqualified=disqualified,
        dbcv=dbcv,
        silhouette=silhouette,
        param_plateau_score=param_plateau_score,
        in_current_grid=in_current_grid,
    )


def test_get_plateau_drop_threshold_uses_default(monkeypatch):
    monkeypatch.delenv("VALIDATION_PLATEAU_DROP_THRESHOLD", raising=False)
    from modules.cluster_results import get_plateau_drop_threshold

    assert get_plateau_drop_threshold() == 0.05


def test_pick_best_cluster_run_rejects_sharp_peak(monkeypatch):
    monkeypatch.setenv("VALIDATION_PLATEAU_DROP_THRESHOLD", "0.05")
    from modules.cluster_results import pick_best_cluster_run

    best = pick_best_cluster_run(
        [
            _run_row(
                embedding_case="video",
                dbcv=0.90,
                silhouette=0.10,
                n_clusters=6,
                noise_ratio=0.20,
                param_plateau_score=0.80,
            ),
            _run_row(
                embedding_case="video",
                dbcv=0.82,
                silhouette=0.25,
                n_clusters=4,
                noise_ratio=0.10,
                param_plateau_score=0.79,
            ),
        ]
    )

    assert best is not None
    assert best.dbcv == 0.82


def test_list_eligible_best_rows_filters_like_validation():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(
            _run_row(
                embedding_case="audio",
                dbcv=0.7,
                silhouette=0.2,
                n_clusters=4,
                noise_ratio=0.1,
                param_plateau_score=0.68,
            )
        )
        s.add(
            _run_row(
                embedding_case="audio",
                dbcv=0.9,
                silhouette=0.4,
                n_clusters=5,
                noise_ratio=0.1,
                param_plateau_score=0.85,
                disqualified=1,
            )
        )
        s.add(
            _run_row(
                embedding_case="audio",
                dbcv=0.6,
                silhouette=0.1,
                n_clusters=3,
                noise_ratio=0.2,
                param_plateau_score=0.55,
                in_current_grid=0,
            )
        )
        s.commit()

        from modules.cluster_results import list_eligible_best_rows

        rows = list_eligible_best_rows(s, "audio")

    assert len(rows) == 1
    assert rows[0].dbcv == 0.7


def test_pick_best_falls_back_when_all_sharp_peaks(monkeypatch):
    monkeypatch.setenv("VALIDATION_PLATEAU_DROP_THRESHOLD", "0.05")
    from modules.cluster_results import pick_best_cluster_run

    r1 = _run_row(
        embedding_case="video",
        dbcv=0.9,
        silhouette=0.1,
        n_clusters=5,
        noise_ratio=0.1,
        param_plateau_score=0.1,
    )
    r2 = _run_row(
        embedding_case="video",
        dbcv=0.7,
        silhouette=0.2,
        n_clusters=5,
        noise_ratio=0.1,
        param_plateau_score=0.2,
    )
    best = pick_best_cluster_run([r1, r2])
    assert best is not None
    assert best.dbcv == 0.9


def test_summarize_case_rows_counts_and_means():
    from modules.cluster_results import summarize_case_rows

    rows = [
        _run_row(
            embedding_case="audio",
            dbcv=0.2,
            silhouette=-0.5,
            n_clusters=2,
            noise_ratio=0.1,
            disqualified=0,
            in_current_grid=1,
        ),
        _run_row(
            embedding_case="audio",
            dbcv=0.9,
            silhouette=0.0,
            n_clusters=4,
            noise_ratio=0.0,
            disqualified=1,
            in_current_grid=1,
        ),
        _run_row(
            embedding_case="audio",
            dbcv=0.1,
            silhouette=0.0,
            n_clusters=2,
            noise_ratio=0.2,
            disqualified=0,
            in_current_grid=0,
        ),
    ]
    out = summarize_case_rows(rows)
    assert out["n_runs"] == "3"
    assert out["n_filtered"] == "1"
    assert out["dbcv_mean"] == "0.400"
    assert out["dbcv_std"] == "0.436"
    assert out["silhouette_mean"] == "-0.167"
    assert out["k_median"] == "2.000"
    assert out["noise_pct_mean"] == "10.000"


def test_select_best_cluster_run_matches_pick_best(monkeypatch):
    monkeypatch.setenv("VALIDATION_PLATEAU_DROP_THRESHOLD", "0.05")
    eng = _make_engine()
    with Session(eng) as s:
        s.add(
            _run_row(
                embedding_case="video",
                dbcv=0.90,
                silhouette=0.10,
                n_clusters=6,
                noise_ratio=0.20,
                param_plateau_score=0.80,
            )
        )
        s.add(
            _run_row(
                embedding_case="video",
                dbcv=0.82,
                silhouette=0.25,
                n_clusters=4,
                noise_ratio=0.10,
                param_plateau_score=0.79,
            )
        )
        s.commit()

        from modules.cluster_results import (
            list_eligible_best_rows,
            pick_best_cluster_run,
            select_best_cluster_run,
        )

        by_rows = pick_best_cluster_run(list_eligible_best_rows(s, "video"))
        by_query = select_best_cluster_run(s, "video")

    assert by_rows is not None
    assert by_query is not None
    assert by_rows.dbcv == by_query.dbcv
