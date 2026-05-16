from modules.clustering.core import (  # noqa: F401  -- re-exports
    DEFAULT_HDBSCAN_METRIC,
    ClusterResult,
    cluster_users,
    compute_clusters,
    load_user_matrix,
    resolve_hdbscan_metric,
    resolve_umap2d_params,
)

# NOTE: list_eligible_best_rows is renamed to list_best_candidate_rows in Task 3.
from modules.clustering.results import (  # noqa: F401  -- re-exports
    DEFAULT_CASES,
    get_plateau_drop_threshold,
    list_case_rows,
    list_eligible_best_rows,
    pick_best_cluster_run,
    select_best_cluster_run,
    summarize_case_rows,
)
from modules.clustering.search import run_cluster_search  # noqa: F401
from modules.clustering.validation import validate_clustering  # noqa: F401
