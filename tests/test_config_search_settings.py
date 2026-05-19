"""SearchSettings schema accepts hdbscan_min_samples list (new axis from cluster lab)."""

from core.config import SearchSettings


def test_search_settings_accepts_hdbscan_min_samples():
    s = SearchSettings(
        umap_n_components=[20],
        umap_n_neighbors=[10],
        umap_min_dist=[0.0],
        umap_metrics=["cosine"],
        umap2d_n_neighbors=15,
        umap2d_min_dist=0.1,
        umap2d_metrics=["cosine"],
        hdbscan_min_cluster_size=[15],
        hdbscan_min_samples=[5, 10, 15],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    assert s.hdbscan_min_samples == [5, 10, 15]


def test_search_settings_hdbscan_min_samples_defaults_to_empty():
    s = SearchSettings(
        umap_n_components=[20],
        umap_n_neighbors=[10],
        umap_min_dist=[0.0],
        umap_metrics=["cosine"],
        umap2d_n_neighbors=15,
        umap2d_min_dist=0.1,
        umap2d_metrics=["cosine"],
        hdbscan_min_cluster_size=[15],
        hdbscan_selection=["eom"],
        random_state=42,
    )
    assert s.hdbscan_min_samples == []
