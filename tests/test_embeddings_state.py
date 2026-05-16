import numpy as np

from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    User,
    get_engine,
    get_session,
)
from modules.embeddings.state import get_clip_embedding_rows_for_user_aggregation


def _make_blob(values: list[float]) -> bytes:
    """Helper to create embedding blob from float array."""
    return np.array(values, dtype=np.float32).tobytes()


def test_aggregation_excludes_orphan_rows():
    """Verify that clips deselected after embedding don't contaminate user means."""
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(Clip(id=11, user_id=1, is_selected=False, is_downloaded=True))
    session.merge(
        ClipEmbedding(
            clip_id=10, embedding_case="video", embedding=_make_blob([1.0, 2.0, 3.0])
        )
    )
    session.merge(
        ClipEmbedding(
            clip_id=11, embedding_case="video", embedding=_make_blob([4.0, 5.0, 6.0])
        )
    )
    session.commit()

    rows = get_clip_embedding_rows_for_user_aggregation(session, "video")
    user_ids_seen = {user_id for _, user_id in rows}
    clip_ids_seen = {
        ce.clip_id
        for ce in session.query(ClipEmbedding).filter_by(embedding_case="video")
    }
    assert user_ids_seen == {1}, "user 1 should contribute"
    assert clip_ids_seen == {10, 11}, "both embedding rows should still exist on disk"
    assert len(rows) == 1, "only clip 10 should be included in aggregation"
    session.close()
