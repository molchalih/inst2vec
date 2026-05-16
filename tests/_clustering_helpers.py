"""Private test helpers shared between cluster_search and cluster_validation tests.

Leading underscore ensures pytest does not collect this file as a test module
(default discovery pattern requires ``test_`` prefix).
"""

from types import SimpleNamespace


def _seed_search_dataset(n_users: int = 30, case: str = "video") -> None:
    """Seed Users + selected/downloaded Clips + UserEmbeddings."""
    import numpy as np

    from modules.database import (
        Base,
        Clip,
        ClusterRun,
        StageState,
        User,
        UserEmbedding,
        get_engine,
        get_session,
    )

    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for m in (ClusterRun, StageState, UserEmbedding, Clip, User):
            session.query(m).delete()
        session.commit()
        for uid in range(n_users):
            session.merge(User(id=uid))
            session.merge(
                Clip(
                    id=1000 + uid,
                    user_id=uid,
                    is_selected=True,
                    is_downloaded=True,
                )
            )
            session.merge(
                UserEmbedding(
                    user_id=uid,
                    embedding_case=case,
                    embedding=np.random.default_rng(uid)
                    .standard_normal(8)
                    .astype(np.float32)
                    .tobytes(),
                )
            )
        session.commit()
    finally:
        session.close()


def _mutate_one_embedding(case: str = "video") -> None:
    """Replace one UserEmbedding blob with new random bytes."""
    import numpy as np

    from modules.database import UserEmbedding, get_session

    session = get_session()
    try:
        row = (
            session.query(UserEmbedding)
            .filter_by(embedding_case=case)
            .order_by(UserEmbedding.user_id)
            .first()
        )
        row.embedding = (
            np.random.default_rng(9999).standard_normal(8).astype(np.float32).tobytes()
        )
        session.commit()
    finally:
        session.close()


def _make_minimal_search_settings(
    *, umap_n_components: list[int] | None = None
) -> object:
    """Tiny settings object with the fields _load_grid reads."""
    if umap_n_components is None:
        umap_n_components = [3]
    return SimpleNamespace(
        umap_n_components=umap_n_components,
        umap_n_neighbors=[5],
        umap_min_dist=[0.1],
        umap_metrics=["cosine"],
        umap2d_n_neighbors=5,
        umap2d_min_dist=0.1,
        umap2d_metrics=["cosine"],
        hdbscan_min_cluster_size=[5],
        hdbscan_selection=["eom"],
        random_state=42,
    )
