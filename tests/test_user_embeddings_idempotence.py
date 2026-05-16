"""Idempotence tests for embed_user_embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from modules.database import (
    Base,
    Clip,
    ClipEmbedding,
    StageState,
    User,
    UserEmbedding,
    get_engine,
    get_session,
)
from modules.embeddings.users import embed_user_embeddings


def _blob(values: list[float]) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


@pytest.fixture
def db_session():
    Base.metadata.create_all(get_engine())
    session = get_session()
    for model in (StageState, UserEmbedding, ClipEmbedding, Clip, User):
        session.query(model).delete()
    session.commit()
    yield session
    session.close()


def _seed_users_and_clips(session, pairs: list[tuple[int, int]]):
    """pairs: list of (user_id, clip_id). Inserts the parents."""
    user_ids = {u for u, _ in pairs}
    for uid in user_ids:
        session.merge(User(id=uid, is_selected=True, is_eligible=True))
    for uid, cid in pairs:
        session.merge(Clip(id=cid, user_id=uid, is_selected=True, is_downloaded=True))
    session.commit()


def _seed_clip_embeddings(session, case: str, items: list[tuple[int, list[float]]]):
    """items: list of (clip_id, vector). Replaces rows for that case."""
    for cid, vec in items:
        session.merge(
            ClipEmbedding(clip_id=cid, embedding_case=case, embedding=_blob(vec))
        )
    session.commit()


def _settings_stub(*, exclude_disqualified_users: bool = False):
    """Minimal settings stub: embed_user_embeddings reads
    ``settings.embeddings.exclude_disqualified_users`` and nothing else.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        embeddings=SimpleNamespace(
            exclude_disqualified_users=exclude_disqualified_users,
        )
    )


def test_first_run_aggregates_and_writes_stage_state(db_session):
    _seed_users_and_clips(db_session, [(1, 10), (1, 11), (2, 20)])
    _seed_clip_embeddings(
        db_session,
        "video",
        [(10, [1.0, 0.0]), (11, [3.0, 0.0]), (20, [0.0, 4.0])],
    )

    embed_user_embeddings(_settings_stub(), cases=["video"])

    s = db_session
    ue = {
        r.user_id: r for r in s.query(UserEmbedding).filter_by(embedding_case="video")
    }
    assert set(ue.keys()) == {1, 2}
    np.testing.assert_array_almost_equal(
        np.frombuffer(ue[1].embedding, dtype=np.float32), [2.0, 0.0]
    )
    np.testing.assert_array_almost_equal(
        np.frombuffer(ue[2].embedding, dtype=np.float32), [0.0, 4.0]
    )

    state = s.get(StageState, ("user_embeddings", "video"))
    assert state is not None
    assert state.data_hash and state.config_hash and state.dependency_hash


def test_rerun_with_identical_inputs_is_noop(db_session):
    _seed_users_and_clips(db_session, [(1, 10), (2, 20)])
    _seed_clip_embeddings(db_session, "video", [(10, [1.0]), (20, [2.0])])

    embed_user_embeddings(_settings_stub(), cases=["video"])
    first_updated = db_session.get(StageState, ("user_embeddings", "video")).updated_at

    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    second_updated = db_session.get(StageState, ("user_embeddings", "video")).updated_at

    # No-op: stage_state row not rewritten.
    assert first_updated == second_updated


def test_clip_embedding_change_triggers_user_recompute(db_session):
    _seed_users_and_clips(db_session, [(1, 10)])
    _seed_clip_embeddings(db_session, "video", [(10, [1.0, 0.0])])
    embed_user_embeddings(_settings_stub(), cases=["video"])

    row = (
        db_session.query(ClipEmbedding)
        .filter_by(clip_id=10, embedding_case="video")
        .one()
    )
    row.embedding = _blob([7.0, 0.0])
    db_session.commit()

    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    ue = (
        db_session.query(UserEmbedding)
        .filter_by(user_id=1, embedding_case="video")
        .one()
    )
    np.testing.assert_array_almost_equal(
        np.frombuffer(ue.embedding, dtype=np.float32), [7.0, 0.0]
    )


def test_clip_embedding_bytes_change_without_updated_at_change(db_session):
    """Bytes-only change must invalidate the user-stage dependency.

    Simulates the production hazard: when the clip-embeddings stage clears
    and re-inserts rows inside the same SQLite second, updated_at stays
    identical (second precision) while the bytes differ. The dependency
    hash must still flip on the bytes.
    """
    _seed_users_and_clips(db_session, [(1, 10)])
    _seed_clip_embeddings(db_session, "video", [(10, [1.0, 0.0])])
    embed_user_embeddings(_settings_stub(), cases=["video"])

    row = (
        db_session.query(ClipEmbedding)
        .filter_by(clip_id=10, embedding_case="video")
        .one()
    )
    pinned_updated_at = row.updated_at
    row.embedding = _blob([9.0, 0.0])
    row.updated_at = pinned_updated_at  # pin to simulate same-second precision
    db_session.commit()

    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    ue = (
        db_session.query(UserEmbedding)
        .filter_by(user_id=1, embedding_case="video")
        .one()
    )
    np.testing.assert_array_almost_equal(
        np.frombuffer(ue.embedding, dtype=np.float32), [9.0, 0.0]
    )


def test_new_clip_embedding_triggers_user_recompute(db_session):
    _seed_users_and_clips(db_session, [(1, 10), (1, 11)])
    _seed_clip_embeddings(db_session, "video", [(10, [2.0, 0.0])])
    embed_user_embeddings(_settings_stub(), cases=["video"])

    _seed_clip_embeddings(db_session, "video", [(11, [4.0, 0.0])])
    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    ue = (
        db_session.query(UserEmbedding)
        .filter_by(user_id=1, embedding_case="video")
        .one()
    )
    np.testing.assert_array_almost_equal(
        np.frombuffer(ue.embedding, dtype=np.float32), [3.0, 0.0]
    )


def test_deselecting_one_of_two_clips_triggers_user_recompute(db_session):
    """Per code review: when one of a user's clips is deselected but the
    user still has another selected clip, aggregation membership shifts
    but the fingerprint must also flip so the user vector recomputes."""
    _seed_users_and_clips(db_session, [(1, 10), (1, 11)])
    _seed_clip_embeddings(db_session, "video", [(10, [2.0, 0.0]), (11, [4.0, 0.0])])

    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    initial = np.frombuffer(
        db_session.query(UserEmbedding)
        .filter_by(user_id=1, embedding_case="video")
        .one()
        .embedding,
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(initial, [3.0, 0.0])

    # Deselect clip 11. The user keeps clip 10, so user_ids set is
    # unchanged; aggregation drops clip 11 → mean must shift.
    db_session.query(Clip).filter_by(id=11).update({"is_selected": False})
    db_session.commit()

    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    after = np.frombuffer(
        db_session.query(UserEmbedding)
        .filter_by(user_id=1, embedding_case="video")
        .one()
        .embedding,
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(
        after, [2.0, 0.0], err_msg="user vector must recompute when clip 11 deselected"
    )


def test_ineligible_user_excluded_when_exclude_disqualified(db_session):
    """Per code review: when exclude_disqualified_users=True, an
    ineligible user must NOT have a UserEmbedding row even if they still
    have selected+downloaded clips with embeddings."""
    _seed_users_and_clips(db_session, [(1, 10), (2, 20)])
    _seed_clip_embeddings(db_session, "video", [(10, [1.0]), (20, [9.0])])

    db_session.query(User).filter_by(id=2).update({"is_eligible": False})
    db_session.commit()

    embed_user_embeddings(
        _settings_stub(exclude_disqualified_users=True), cases=["video"]
    )
    db_session.expire_all()

    user_ids = {
        r.user_id
        for r in db_session.query(UserEmbedding).filter_by(embedding_case="video")
    }
    assert user_ids == {1}, (
        "user 2 is ineligible — their clip embedding must not be averaged in"
    )


def test_eligibility_flip_triggers_user_recompute(db_session):
    """Flipping a user from eligible to ineligible mid-cycle must flip
    the user-embedding fingerprint so the prior row gets cleared."""
    _seed_users_and_clips(db_session, [(1, 10), (2, 20)])
    _seed_clip_embeddings(db_session, "video", [(10, [1.0]), (20, [9.0])])

    settings = _settings_stub(exclude_disqualified_users=True)
    embed_user_embeddings(settings, cases=["video"])
    db_session.expire_all()
    assert {
        r.user_id
        for r in db_session.query(UserEmbedding).filter_by(embedding_case="video")
    } == {1, 2}

    db_session.query(User).filter_by(id=2).update({"is_eligible": False})
    db_session.commit()

    embed_user_embeddings(settings, cases=["video"])
    db_session.expire_all()
    assert {
        r.user_id
        for r in db_session.query(UserEmbedding).filter_by(embedding_case="video")
    } == {1}


def test_empty_inputs_writes_stage_state_and_skips_next(db_session):
    embed_user_embeddings(_settings_stub(), cases=["video"])
    state = db_session.get(StageState, ("user_embeddings", "video"))
    assert state is not None  # row written even for empty case
    first_updated = state.updated_at

    embed_user_embeddings(_settings_stub(), cases=["video"])
    db_session.expire_all()
    second_updated = db_session.get(StageState, ("user_embeddings", "video")).updated_at
    assert first_updated == second_updated
