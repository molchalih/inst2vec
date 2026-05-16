"""Analysis-scope filtering for clustering.load_user_matrix."""

from __future__ import annotations

import numpy as np

from modules.database import (
    Base,
    Clip,
    User,
    UserEmbedding,
    get_engine,
    get_session,
)


def _seed_user(session, user_id: int, *, is_selected: bool, is_downloaded: bool):
    session.merge(User(id=user_id))
    session.merge(
        Clip(
            id=10_000 + user_id,
            user_id=user_id,
            is_selected=is_selected,
            is_downloaded=is_downloaded,
        )
    )
    blob = np.ones(8, dtype=np.float32).tobytes()
    session.merge(
        UserEmbedding(user_id=user_id, embedding_case="video", embedding=blob)
    )


def _clear_main_db():
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for model in (UserEmbedding, Clip, User):
            session.query(model).delete()
        session.commit()
    finally:
        session.close()


def test_load_user_matrix_filters_by_clip_used_in_analysis():
    from modules.clustering import load_user_matrix

    _clear_main_db()
    session = get_session()
    try:
        _seed_user(session, 1, is_selected=True, is_downloaded=True)
        _seed_user(session, 2, is_selected=False, is_downloaded=True)
        _seed_user(session, 3, is_selected=True, is_downloaded=False)
        session.commit()
    finally:
        session.close()

    matrix, user_ids = load_user_matrix("video")
    assert user_ids == [1]
    assert matrix.shape == (1, 8)


def test_load_user_matrix_empty_when_no_analysis_users():
    from modules.clustering import load_user_matrix

    _clear_main_db()
    session = get_session()
    try:
        _seed_user(session, 1, is_selected=False, is_downloaded=False)
        session.commit()
    finally:
        session.close()

    matrix, user_ids = load_user_matrix("video")
    assert user_ids == []
    assert matrix.shape == (0, 0)


def test_load_user_matrix_ignores_other_embedding_cases():
    from modules.clustering import load_user_matrix

    _clear_main_db()
    session = get_session()
    try:
        _seed_user(session, 1, is_selected=True, is_downloaded=True)
        session.merge(
            UserEmbedding(
                user_id=1,
                embedding_case="audio",
                embedding=np.zeros(8, dtype=np.float32).tobytes(),
            )
        )
        session.commit()
    finally:
        session.close()

    matrix, user_ids = load_user_matrix("video")
    assert user_ids == [1]
    sums = matrix.sum(axis=1)
    assert sums[0] == 8.0  # video embedding (ones), not audio (zeros)
