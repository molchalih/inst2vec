"""Clustering stage public API."""

from core.config import Secrets, Settings
from modules.clustering.assign import assign_clusters
from modules.clustering.search import run_cluster_search
from modules.clustering.validation import validate_clustering


def run_search(settings: Settings, secrets: Secrets, cases: tuple[str, ...]) -> None:
    """UMAP + HDBSCAN grid search over embedding cases."""
    run_cluster_search(settings=settings, cases=cases)


def run_validation(
    settings: Settings, secrets: Secrets, cases: tuple[str, ...]
) -> None:
    """DBCV + silhouette validation with plateau detection."""
    validate_clustering(settings=settings, cases=cases)


def run_assign(settings: Settings, secrets: Secrets, cases: tuple[str, ...]) -> None:
    """Assign final cluster labels using the best run per case."""
    assign_clusters(settings=settings, cases=cases)


__all__ = [
    "assign_clusters",
    "run_assign",
    "run_cluster_search",
    "run_search",
    "run_validation",
    "validate_clustering",
]
