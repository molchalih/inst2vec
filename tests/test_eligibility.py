from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.database import Base, ClusterRun, User


def test_user_and_cluster_run_default_to_pending():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)

    with Session(eng) as s:
        user = User(id=1)
        run = ClusterRun(
            embedding_case="video",
            umap_n_components=5,
            umap_n_neighbors=5,
            umap_min_dist=0.0,
            umap_metric="cosine",
            umap2d_n_neighbors=5,
            umap2d_min_dist=0.1,
            umap2d_metric="cosine",
            hdbscan_min_cluster_size=5,
            hdbscan_min_samples=None,
            hdbscan_cluster_selection_method="eom",
            hdbscan_metric="euclidean",
            random_state=42,
            n_clusters=2,
            noise_ratio=0.1,
            min_size=2,
            median_size=2,
            max_size=2,
        )
        s.add(user)
        s.add(run)
        s.commit()

        loaded = s.get(User, 1)
        assert loaded is not None
        assert loaded.is_eligible is None  # NULL = pending
        assert run.passes_validation is None  # NULL = pending
