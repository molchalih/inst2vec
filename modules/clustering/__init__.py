from modules.clustering.assign import assign_clusters  # noqa: F401
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
from modules.clustering.search import run_cluster_search  # noqa: F401
from modules.clustering.validation import validate_clustering  # noqa: F401
from modules.embeddings.cases import DEFAULT_CASES  # noqa: F401  -- canonical re-export
