import numpy as np

from modules import fingerprint as fp
from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    Music,
    User,
    get_engine,
    get_session,
)
from modules.embeddings.state import (
    dependency_rows_for_case,
    get_clip_embedding_rows_for_user_aggregation,
    get_embedded_source_hashes,
    per_clip_source_hashes_and_aggregate,
)


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

    rows = get_clip_embedding_rows_for_user_aggregation(
        session, "video", exclude_disqualified_users=False
    )
    user_ids_seen = {user_id for _, _, user_id in rows}
    clip_ids_seen = {
        ce.clip_id
        for ce in session.query(ClipEmbedding).filter_by(embedding_case="video")
    }
    assert user_ids_seen == {1}, "user 1 should contribute"
    assert clip_ids_seen == {10, 11}, "both embedding rows should still exist on disk"
    assert len(rows) == 1, "only clip 10 should be included in aggregation"
    assert rows[0][0] == 10, "row must carry clip_id for fingerprint use"
    session.close()


def test_get_embedded_source_hashes_returns_clip_id_to_hash_map():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(Clip(id=11, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(
        ClipEmbedding(
            clip_id=10,
            embedding_case="video",
            embedding=b"\x00" * 4,
            source_hash="abc",
        )
    )
    session.merge(
        ClipEmbedding(
            clip_id=11,
            embedding_case="video",
            embedding=b"\x00" * 4,
            source_hash=None,
        )
    )
    session.merge(
        ClipEmbedding(
            clip_id=10,
            embedding_case="audio",
            embedding=b"\x00" * 4,
            source_hash="zzz",
        )
    )
    session.commit()

    out = get_embedded_source_hashes(session, "video")
    assert out == {10: "abc", 11: None}
    session.close()


def test_per_clip_source_hashes_match_dependency_rows():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (ClipEmbedding, Clip, Music, User):
        session.query(model).delete()
    session.commit()

    session.merge(User(id=1, is_selected=True, is_eligible=True))
    session.merge(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
    session.merge(Clip(id=11, user_id=1, is_selected=True, is_downloaded=True))
    session.commit()

    per_clip, aggregate = per_clip_source_hashes_and_aggregate(
        session, "video", [10, 11]
    )

    # Per-clip hash must equal hash_rows of the single dependency row.
    rows = dependency_rows_for_case(session, "video", [10, 11])
    by_id = {r[0]: r for r in rows}
    assert per_clip == {
        10: fp.hash_rows([by_id[10]]),
        11: fp.hash_rows([by_id[11]]),
    }
    # Aggregate must equal hash_rows over the full ordered row list.
    assert aggregate == fp.hash_rows(rows)
    session.close()


def test_per_clip_source_hashes_with_no_candidates():
    Base.metadata.create_all(get_engine())
    session = get_session()
    per_clip, aggregate = per_clip_source_hashes_and_aggregate(session, "video", [])
    assert per_clip == {}
    assert aggregate == fp.hash_rows([])
    session.close()
