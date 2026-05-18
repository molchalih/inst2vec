from core.config import Secrets, Settings
from modules.clustering.assign import assign_clusters
from modules.clustering.core import (  # noqa: F401  -- re-exports
    DEFAULT_HDBSCAN_METRIC,
    ClusterResult,
    cluster_users,
    compute_clusters,
    load_user_matrix,
    resolve_hdbscan_metric,
    resolve_umap2d_params,
)
from modules.clustering.results import (  # noqa: F401  -- re-exports
    get_plateau_drop_threshold,
    list_best_candidate_rows,
    list_case_rows,
    pick_best_cluster_run,
    select_best_cluster_run,
    summarize_case_rows,
)
from modules.clustering.search import run_cluster_search
from modules.clustering.validation import validate_clustering
from modules.embeddings.cases import default_cases  # noqa: F401  -- canonical re-export


def run_search(settings: Settings, secrets: Secrets) -> None:
    """UMAP + HDBSCAN grid search over embedding cases."""
    workers = getattr(settings.search, "clustering_grid_workers", 1)
    run_cluster_search(settings=settings, clustering_grid_workers=workers)


def run_validation(settings: Settings, secrets: Secrets) -> None:
    """DBCV + silhouette validation with plateau detection."""
    workers = getattr(settings.search, "clustering_grid_workers", 1)
    validate_clustering(settings=settings, clustering_grid_workers=workers)


def run_assign(settings: Settings, secrets: Secrets) -> None:
    """Assign final cluster labels using the best run per case."""
    assign_clusters(settings=settings)
