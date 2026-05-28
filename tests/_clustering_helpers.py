"""Private test helpers shared between cluster_search and cluster_validation tests.

Leading underscore ensures pytest does not collect this file as a test module
(default discovery pattern requires ``test_`` prefix).
"""

from types import SimpleNamespace

from sqlalchemy.orm import Session


def seed_audio_mir(
    session: Session, *, clip_id: int, is_music_detected: bool = True
) -> None:
    """Seed a minimal ``AudioMIR`` row sufficient for ``verbalize_mir``.

    Writes the smallest field set that yields a non-empty verbalization:
    genre/mood/instrument label strings and the ``is_music_detected`` flag.
    Read by both the audio-case input adapter (``build_audio_text``) and
    the sandwich case (``build_sandwich_text``).
    """
    from core.database import AudioMIR

    session.add(
        AudioMIR(
            clip_id=clip_id,
            is_music_detected=is_music_detected,
            genre_labels="lofi, downtempo",
            moodtheme_labels="relaxed, mellow",
            instrument_labels="piano",
        )
    )
    session.commit()


def _seed_search_dataset(n_users: int = 30, case: str = "video") -> None:
    """Seed Users + selected/downloaded Clips + UserEmbeddings."""
    import numpy as np

    from core.database import (
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

    from core.database import UserEmbedding, get_session

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
    """Tiny settings object for run_cluster_search tests (grid hyperparameters only)."""
    if umap_n_components is None:
        umap_n_components = [3]
    return SimpleNamespace(
        search=SimpleNamespace(
            umap_n_components=umap_n_components,
            umap_n_neighbors=[5],
            umap_min_dist=[0.1],
            umap_metrics=["cosine"],
            umap2d_n_neighbors=5,
            umap2d_min_dist=0.1,
            umap2d_metrics=["cosine"],
            hdbscan_min_cluster_size=[5],
            hdbscan_min_samples=[],
            hdbscan_selection=["eom"],
            hdbscan_max_cluster_frac=0.0,
            embedding_preprocess={},
            random_state=42,
        ),
    )


def _make_minimal_validation_settings(
    *,
    plateau_drop_threshold: float = 0.05,
    max_noise_ratio: float = 0.9,
    min_clusters: int = 1,
    max_clusters: int = 20,
    max_dominance: float = 1.0,
) -> object:
    """Tiny settings object for validate_clustering/assign_clusters tests.

    Carries a ``.search`` slice too, since validate_clustering/assign_clusters
    read the preprocess map and max_cluster_frac from there.
    """
    return SimpleNamespace(
        validation=SimpleNamespace(
            plateau_drop_threshold=plateau_drop_threshold,
            max_noise_ratio=max_noise_ratio,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            max_dominance=max_dominance,
        ),
        search=SimpleNamespace(
            hdbscan_max_cluster_frac=0.0,
            embedding_preprocess={},
        ),
    )
