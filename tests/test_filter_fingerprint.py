"""Fingerprint-gated idempotence for modules/filter.py::process_dataset.

Uses the conftest in-memory DB (NOT a fresh create_engine) so that
StageState rows land in the same engine as Clip/User/UserStats rows.
Each test wipes the relevant tables before seeding to provide isolation,
and again on teardown so we don't leak rows into later test modules.
"""

import pytest

from modules.config import FilterSettings
from modules.database import (
    Base,
    Clip,
    StageState,
    User,
    UserStats,
    get_engine,
    get_session,
)
from modules.filter import process_dataset


def _wipe() -> None:
    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        for m in (UserStats, StageState, Clip, User):
            session.query(m).delete()
        session.commit()
    finally:
        session.close()


@pytest.fixture(autouse=True)
def _isolate_db():
    _wipe()
    yield
    _wipe()


def _seed_dataset(*, n_users: int = 3, n_clips_per_user: int = 12) -> None:
    """Seed users + clips that pass all hard policies under default FilterSettings.

    With default cfg (min_video_duration=3, max_video_duration=80, min_taken_at=1640995200,
    creator_min_median_views=10000, min_eligible_clips_per_user=10),
    these clips are eligible and select_clips will pick selected_clips_per_user of them.
    """
    session = get_session()
    try:
        clip_id = 1000
        for uid in range(n_users):
            session.add(User(id=uid))
            for _ in range(n_clips_per_user):
                session.add(
                    Clip(
                        id=clip_id,
                        user_id=uid,
                        play_count=100_000,
                        video_duration=15.0,
                        taken_at=1_700_000_000,
                        video_url=f"https://example.test/{clip_id}.mp4",
                        like_count=1_000,
                    )
                )
                clip_id += 1
        session.commit()
    finally:
        session.close()


def _default_cfg() -> FilterSettings:
    return FilterSettings()


def test_creates_stage_state_row_on_first_run():
    _seed_dataset()
    process_dataset(_default_cfg())

    session = get_session()
    try:
        row = session.get(StageState, ("filter", "all"))
        assert row is not None
        assert row.data_hash
        assert row.config_hash
        assert row.dependency_hash
    finally:
        session.close()


def test_skips_on_unchanged_rerun():
    """Second call with identical inputs+cfg must not re-run the work.

    Verified by manually clearing every clip.is_selected after the first run
    and checking that the second run does NOT restore them.  (select_clips
    is deterministic given the seed, so a re-run would re-select the same
    clips; if our mutation persists, we skipped.)
    """
    _seed_dataset()
    cfg = _default_cfg()
    process_dataset(cfg)

    session = get_session()
    try:
        first_selected = session.query(Clip).filter(Clip.is_selected.is_(True)).count()
        assert first_selected > 0  # sanity: first run did something
        session.query(Clip).update({Clip.is_selected: False}, synchronize_session=False)
        session.commit()
    finally:
        session.close()

    process_dataset(cfg)

    session = get_session()
    try:
        second_selected = session.query(Clip).filter(Clip.is_selected.is_(True)).count()
    finally:
        session.close()

    assert second_selected == 0  # skipped → our mutation was preserved


def test_reruns_on_config_change():
    """Changing FilterSettings invalidates the fingerprint and recomputes."""
    _seed_dataset()
    process_dataset(_default_cfg())

    session = get_session()
    try:
        before_too_short = (
            session.query(Clip).filter(Clip.is_too_short.is_(True)).count()
        )
    finally:
        session.close()
    assert before_too_short == 0  # 15.0s clips are not too short under default cfg

    # Raise min_video_duration above every clip's duration.
    stricter = _default_cfg().model_copy(update={"min_video_duration": 999})
    process_dataset(stricter)

    session = get_session()
    try:
        after_too_short = (
            session.query(Clip).filter(Clip.is_too_short.is_(True)).count()
        )
    finally:
        session.close()
    assert after_too_short > 0  # recomputed under stricter cfg


def test_reruns_on_new_clip():
    """Adding a new clip invalidates the data hash and recomputes."""
    _seed_dataset()
    cfg = _default_cfg()
    process_dataset(cfg)

    # Sanity: confirm the fingerprint was written so we know we're testing
    # the stale-data path, not the "no prior state" path.
    session = get_session()
    try:
        assert session.get(StageState, ("filter", "all")) is not None
        # Add a new well-formed clip for an existing user.
        session.add(
            Clip(
                id=99_999,
                user_id=0,
                play_count=100_000,
                video_duration=20.0,
                taken_at=1_700_000_000,
                video_url="https://example.test/new.mp4",
                like_count=500,
            )
        )
        session.commit()
    finally:
        session.close()

    process_dataset(cfg)

    session = get_session()
    try:
        new_clip = session.get(Clip, 99_999)
        # If filter ran, the new clip got its is_garbage derived (False).
        # If filter skipped, is_garbage would still be NULL.
        assert new_clip.is_garbage is False
    finally:
        session.close()
