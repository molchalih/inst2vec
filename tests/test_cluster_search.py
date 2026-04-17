import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.database import ClusterRun


def test_cluster_run_tablename():
    assert ClusterRun.__tablename__ == "cluster_runs"


def test_cluster_run_columns():
    cols = {c.key for c in ClusterRun.__table__.columns}
    assert cols == {
        "id", "embedding_case",
        "umap_n_components", "umap_n_neighbors", "umap_min_dist", "umap_metric",
        "umap2d_n_neighbors", "umap2d_min_dist", "umap2d_metric",
        "hdbscan_min_cluster_size", "hdbscan_min_samples",
        "hdbscan_cluster_selection_method", "hdbscan_metric",
        "random_state",
        "n_clusters", "noise_ratio", "min_size", "median_size", "max_size",
        "created_at",
    }


def test_cluster_run_unique_constraint():
    constraint_names = {c.name for c in ClusterRun.__table__.constraints}
    assert "uq_cluster_runs_params" in constraint_names


def test_cluster_run_hdbscan_min_samples_nullable():
    col = ClusterRun.__table__.c["hdbscan_min_samples"]
    assert col.nullable is True
