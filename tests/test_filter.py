from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.database import Base, Clip, User
from modules.filter import (
    _is_garbage,
    _is_too_long,
    _is_too_old,
    _is_too_short,
)


def test_clip_has_filter_columns():
    clip_cols = {c.key for c in Clip.__table__.columns}
    for col in [
        "is_garbage",
        "is_too_short",
        "is_too_long",
        "is_too_old",
        "is_low_percentile",
        "is_high_percentile",
        "is_creator_low_outlier",
        "log_plays",
        "creator_relative_robust_z",
        "is_preprocessed",
        "is_eligible",
        "is_selected",
    ]:
        assert col in clip_cols, f"Clip missing column: {col}"


def test_clip_no_low_play_count_column():
    clip_cols = {c.key for c in Clip.__table__.columns}
    assert "is_low_play_count" not in clip_cols


def test_user_has_filter_columns():
    user_cols = {c.key for c in User.__table__.columns}
    for col in [
        "is_low_plays_median",
        "is_not_enough_clips",
        "is_selected",
    ]:
        assert col in user_cols, f"User missing column: {col}"


def test_user_does_not_carry_creator_log_stats():
    user_cols = {c.key for c in User.__table__.columns}
    assert "log_plays_median" not in user_cols
    assert "log_plays_mad" not in user_cols


def test_clip_filter_columns_default_null():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(Clip(id=100, user_id=1))
        s.commit()
        clip = s.get(Clip, 100)
        assert clip.is_garbage is None
        assert clip.is_preprocessed is None
        assert clip.is_eligible is None
        assert clip.is_selected is None
        assert clip.log_plays is None


def test_user_filter_columns_default_null():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=2))
        s.commit()
        user = s.get(User, 2)
        assert user.is_low_plays_median is None
        assert user.is_not_enough_clips is None
        assert user.is_selected is None


def test_filter_settings_load():
    from modules.config import FilterSettings

    cfg = FilterSettings(
        min_video_duration=3,
        max_video_duration=80,
        min_taken_at=1640995200,
        creator_min_median_views=10000,
        min_eligible_clips_per_user=10,
        global_low_percentile=5,
        global_high_percentile=99,
        creator_low_z_threshold=-3.5,
        selection_pool_percent=0.20,
        selected_clips_per_user=10,
        selection_random_seed=42,
    )
    assert cfg.selected_clips_per_user == 10


def _make_clip(**kwargs):
    """Build a minimal Clip-like object via SimpleNamespace."""
    from types import SimpleNamespace

    defaults = dict(
        video_duration=10.0,
        taken_at=1700000000,
        play_count=5000,
        video_url="http://example.com/v.mp4",
        like_count=100,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_is_garbage_missing_video_duration():
    c = _make_clip(video_duration=None)
    assert _is_garbage(c) is True


def test_is_garbage_zero_video_duration():
    c = _make_clip(video_duration=0.0)
    assert _is_garbage(c) is True


def test_is_garbage_missing_taken_at():
    c = _make_clip(taken_at=None)
    assert _is_garbage(c) is True


def test_is_garbage_zero_taken_at():
    c = _make_clip(taken_at=0)
    assert _is_garbage(c) is True


def test_is_garbage_missing_play_count():
    c = _make_clip(play_count=None)
    assert _is_garbage(c) is True


def test_is_garbage_zero_play_count():
    c = _make_clip(play_count=0)
    assert _is_garbage(c) is True


def test_is_garbage_missing_download_url():
    c = _make_clip(video_url=None)
    assert _is_garbage(c) is True


def test_is_garbage_empty_download_url():
    c = _make_clip(video_url="")
    assert _is_garbage(c) is True


def test_is_garbage_missing_like_count():
    c = _make_clip(like_count=None)
    assert _is_garbage(c) is True


def test_is_garbage_valid_clip():
    c = _make_clip()
    assert _is_garbage(c) is False


def test_is_too_short_below():
    c = _make_clip(video_duration=2.9)
    assert _is_too_short(c, min_video_duration=3) is True


def test_is_too_short_at():
    c = _make_clip(video_duration=3.0)
    assert _is_too_short(c, min_video_duration=3) is False


def test_is_too_long_above():
    c = _make_clip(video_duration=80.1)
    assert _is_too_long(c, max_video_duration=80) is True


def test_is_too_long_at():
    c = _make_clip(video_duration=80.0)
    assert _is_too_long(c, max_video_duration=80) is False


def test_is_too_old_below():
    c = _make_clip(taken_at=1640995199)
    assert _is_too_old(c, min_taken_at=1640995200) is True


def test_is_too_old_at():
    c = _make_clip(taken_at=1640995200)
    assert _is_too_old(c, min_taken_at=1640995200) is False


def test_clip_exclusion_flags_tuple_contents():
    from modules.filter import CLIP_EXCLUSION_FLAGS

    assert CLIP_EXCLUSION_FLAGS == (
        "is_garbage",
        "is_too_short",
        "is_too_long",
        "is_too_old",
        "is_low_percentile",
        "is_high_percentile",
        "is_creator_low_outlier",
    )


def test_hard_clip_exclusion_flags_tuple_contents():
    from modules.filter import HARD_CLIP_EXCLUSION_FLAGS

    assert HARD_CLIP_EXCLUSION_FLAGS == (
        "is_garbage",
        "is_too_short",
        "is_too_long",
        "is_too_old",
    )


def test_soft_clip_exclusion_flags_tuple_contents():
    from modules.filter import SOFT_CLIP_EXCLUSION_FLAGS

    assert SOFT_CLIP_EXCLUSION_FLAGS == (
        "is_low_percentile",
        "is_high_percentile",
        "is_creator_low_outlier",
    )


def test_clip_exclusion_flags_is_hard_then_soft():
    from modules.filter import (
        CLIP_EXCLUSION_FLAGS,
        HARD_CLIP_EXCLUSION_FLAGS,
        SOFT_CLIP_EXCLUSION_FLAGS,
    )

    assert (
        *HARD_CLIP_EXCLUSION_FLAGS,
        *SOFT_CLIP_EXCLUSION_FLAGS,
    ) == CLIP_EXCLUSION_FLAGS


def test_user_exclusion_flags_tuple_contents():
    from modules.filter import USER_EXCLUSION_FLAGS

    assert USER_EXCLUSION_FLAGS == (
        "is_low_plays_median",
        "is_not_enough_clips",
    )


def _make_db():
    from modules.database import Base

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def test_flag_garbage_clips_sets_is_garbage():
    from modules.database import Clip, User
    from modules.filter import _flag_garbage_clips

    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1))
        # garbage: missing play_count
        s.add(
            Clip(
                id=1,
                user_id=1,
                video_duration=10.0,
                taken_at=1700000000,
                play_count=None,
                video_url="http://x.com/v.mp4",
                like_count=5,
            )
        )
        # valid clip
        s.add(
            Clip(
                id=2,
                user_id=1,
                video_duration=10.0,
                taken_at=1700000000,
                play_count=5000,
                video_url="http://x.com/v.mp4",
                like_count=5,
            )
        )
        s.commit()
        _flag_garbage_clips(s)
        s.commit()
        c1 = s.get(Clip, 1)
        c2 = s.get(Clip, 2)
        assert c1.is_garbage is True
        assert c2.is_garbage is False


def test_flag_basic_policy_clips():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _flag_basic_policy_clips

    eng = _make_db()
    cfg = FilterSettings(
        min_video_duration=3,
        max_video_duration=80,
        min_taken_at=1640995200,
    )
    with Session(eng) as s:
        s.add(User(id=1))
        # low play count
        s.add(
            Clip(
                id=1,
                user_id=1,
                video_duration=10.0,
                taken_at=1700000000,
                play_count=500,
                video_url="http://x.com/v.mp4",
                like_count=5,
            )
        )
        # too short
        s.add(
            Clip(
                id=2,
                user_id=1,
                video_duration=1.0,
                taken_at=1700000000,
                play_count=5000,
                video_url="http://x.com/v.mp4",
                like_count=5,
            )
        )
        # too long
        s.add(
            Clip(
                id=3,
                user_id=1,
                video_duration=90.0,
                taken_at=1700000000,
                play_count=5000,
                video_url="http://x.com/v.mp4",
                like_count=5,
            )
        )
        # too old
        s.add(
            Clip(
                id=4,
                user_id=1,
                video_duration=10.0,
                taken_at=1000000000,
                play_count=5000,
                video_url="http://x.com/v.mp4",
                like_count=5,
            )
        )
        # valid
        s.add(
            Clip(
                id=5,
                user_id=1,
                video_duration=10.0,
                taken_at=1700000000,
                play_count=5000,
                video_url="http://x.com/v.mp4",
                like_count=5,
            )
        )
        s.commit()
        _flag_basic_policy_clips(s, cfg)
        s.commit()
        clips = {c.id: c for c in s.query(Clip).all()}
        assert clips[2].is_too_short is True
        assert clips[3].is_too_long is True
        assert clips[4].is_too_old is True
        assert clips[5].is_too_short is False
        assert clips[5].is_too_long is False
        assert clips[5].is_too_old is False


def test_flag_low_median_creators():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _flag_low_median_creators

    eng = _make_db()
    cfg = FilterSettings(creator_min_median_views=10000)
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(User(id=2))
        # user 1: median plays = 2000 → low
        for i, plays in enumerate([1000, 2000, 3000], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    video_duration=10.0,
                    taken_at=1700000000,
                    play_count=plays,
                    video_url="http://x.com/v.mp4",
                    like_count=5,
                    is_garbage=False,
                    is_too_short=False,
                    is_too_long=False,
                    is_too_old=False,
                    is_preprocessed=True,
                )
            )
        # user 2: median plays = 50000 → fine
        for i, plays in enumerate([40000, 50000, 60000], start=10):
            s.add(
                Clip(
                    id=i,
                    user_id=2,
                    video_duration=10.0,
                    taken_at=1700000000,
                    play_count=plays,
                    video_url="http://x.com/v.mp4",
                    like_count=5,
                    is_garbage=False,
                    is_too_short=False,
                    is_too_long=False,
                    is_too_old=False,
                    is_preprocessed=True,
                )
            )
        s.commit()
        _flag_low_median_creators(s, cfg)
        s.commit()
        u1 = s.get(User, 1)
        u2 = s.get(User, 2)
        assert u1.is_low_plays_median is True
        assert u2.is_low_plays_median is False


def test_flag_low_median_creators_excludes_garbage():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _flag_low_median_creators

    eng = _make_db()
    cfg = FilterSettings(creator_min_median_views=10000)
    with Session(eng) as s:
        s.add(User(id=1))
        # high-play clips that are garbage — should be excluded from median calc
        for i, plays in enumerate([100000, 200000], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    video_duration=10.0,
                    taken_at=1700000000,
                    play_count=plays,
                    video_url="http://x.com/v.mp4",
                    like_count=5,
                    is_garbage=True,
                    is_too_short=False,
                    is_too_long=False,
                    is_too_old=False,
                    is_preprocessed=False,
                )
            )
        # low-play non-garbage clips
        for i, plays in enumerate([500, 600], start=10):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    video_duration=10.0,
                    taken_at=1700000000,
                    play_count=plays,
                    video_url="http://x.com/v.mp4",
                    like_count=5,
                    is_garbage=False,
                    is_too_short=False,
                    is_too_long=False,
                    is_too_old=False,
                    is_preprocessed=True,
                )
            )
        s.commit()
        _flag_low_median_creators(s, cfg)
        s.commit()
        u1 = s.get(User, 1)
        # median of [500, 600] = 550 < 10000 → low
        assert u1.is_low_plays_median is True


def test_flag_users_without_enough_clips():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _flag_users_without_enough_clips

    eng = _make_db()
    cfg = FilterSettings(min_eligible_clips_per_user=3)
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False))
        s.add(User(id=2, is_low_plays_median=False))
        # user 1 gets 2 surviving clips (below threshold of 3)
        for i in range(1, 3):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    video_duration=10.0,
                    taken_at=1700000000,
                    play_count=5000,
                    video_url="http://x.com/v.mp4",
                    like_count=5,
                    is_garbage=False,
                    is_too_short=False,
                    is_too_long=False,
                    is_too_old=False,
                    is_preprocessed=True,
                )
            )
        # user 2 gets 5 surviving clips (above threshold)
        for i in range(10, 15):
            s.add(
                Clip(
                    id=i,
                    user_id=2,
                    video_duration=10.0,
                    taken_at=1700000000,
                    play_count=5000,
                    video_url="http://x.com/v.mp4",
                    like_count=5,
                    is_garbage=False,
                    is_too_short=False,
                    is_too_long=False,
                    is_too_old=False,
                    is_preprocessed=True,
                )
            )
        s.commit()
        _flag_users_without_enough_clips(s, cfg)
        s.commit()
        u1 = s.get(User, 1)
        u2 = s.get(User, 2)
        assert u1.is_not_enough_clips is True
        assert u2.is_not_enough_clips is False


def test_flag_global_percentile_clips():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _flag_global_percentile_clips

    eng = _make_db()
    cfg = FilterSettings(global_low_percentile=20, global_high_percentile=80)
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        play_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        for i, plays in enumerate(play_counts, start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=plays,
                    video_duration=10.0,
                    taken_at=1700000000,
                    video_url="http://x.com/v.mp4",
                    like_count=5,
                    is_preprocessed=True,
                    is_garbage=False,
                    is_too_short=False,
                    is_too_long=False,
                    is_too_old=False,
                )
            )
        s.commit()
        _flag_global_percentile_clips(s, cfg)
        s.commit()
        clips = {c.id: c for c in s.query(Clip).all()}
        low_clips = [c for c in clips.values() if c.is_low_percentile]
        high_clips = [c for c in clips.values() if c.is_high_percentile]
        assert len(low_clips) >= 1
        assert len(high_clips) >= 1
        for c in clips.values():
            assert not (c.is_low_percentile and c.is_high_percentile)


def test_compute_creator_robust_stats():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _compute_creator_robust_stats

    eng = _make_db()
    cfg = FilterSettings(creator_low_z_threshold=-3.5)
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        plays = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        for i, p in enumerate(plays, start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=p,
                    video_duration=10.0,
                    taken_at=1700000000,
                    video_url="http://x.com/v.mp4",
                    like_count=5,
                    is_preprocessed=True,
                    is_low_percentile=False,
                    is_high_percentile=False,
                )
            )
        s.commit()
        _compute_creator_robust_stats(s, cfg)
        s.commit()
        clips = s.query(Clip).filter(Clip.user_id == 1).all()
        for c in clips:
            assert c.log_plays is not None
            assert c.creator_relative_robust_z is not None
            assert c.is_creator_low_outlier is not None


def test_compute_creator_robust_stats_mad_zero_no_crash():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _compute_creator_robust_stats

    eng = _make_db()
    cfg = FilterSettings(creator_low_z_threshold=-3.5)
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        for i in range(1, 6):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=5000,
                    video_duration=10.0,
                    taken_at=1700000000,
                    video_url="http://x.com/v.mp4",
                    like_count=5,
                    is_preprocessed=True,
                    is_low_percentile=False,
                    is_high_percentile=False,
                )
            )
        s.commit()
        _compute_creator_robust_stats(s, cfg)
        s.commit()
        clips = s.query(Clip).filter(Clip.user_id == 1).all()
        for c in clips:
            assert c.is_creator_low_outlier is False


def test_compute_creator_robust_stats_outlier_flagged():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _compute_creator_robust_stats

    eng = _make_db()
    cfg = FilterSettings(creator_low_z_threshold=-1.0)
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        plays = [10, 100, 1000, 10000, 100000]
        for i, p in enumerate(plays, start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=p,
                    video_duration=10.0,
                    taken_at=1700000000,
                    video_url="http://x.com/v.mp4",
                    like_count=5,
                    is_preprocessed=True,
                    is_low_percentile=False,
                    is_high_percentile=False,
                )
            )
        s.commit()
        _compute_creator_robust_stats(s, cfg)
        s.commit()
        c_low = s.get(Clip, 1)
        assert c_low.is_creator_low_outlier is True


def test_compute_creator_robust_stats_does_not_write_to_user():
    """User-level log-play stats live in UserStats; this stage must not touch User."""
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _compute_creator_robust_stats

    eng = _make_db()
    cfg = FilterSettings(creator_low_z_threshold=-3.5)
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        for i, p in enumerate([1000, 2000, 3000], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=p,
                    is_preprocessed=True,
                    is_low_percentile=False,
                    is_high_percentile=False,
                )
            )
        s.commit()
        _compute_creator_robust_stats(s, cfg)
        s.commit()
        u = s.get(User, 1)
        # User no longer carries creator-level stats
        assert not hasattr(u, "log_plays_median") or u.log_plays_median is None
        assert not hasattr(u, "log_plays_mad") or u.log_plays_mad is None


def test_compute_creator_robust_stats_skips_low_percentile_clips():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _compute_creator_robust_stats

    eng = _make_db()
    cfg = FilterSettings(creator_low_z_threshold=-3.5)
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        # Eligible-so-far
        for i, p in enumerate([1000, 2000, 3000], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=p,
                    is_preprocessed=True,
                    is_low_percentile=False,
                    is_high_percentile=False,
                )
            )
        # Soft-excluded — must not get robust-z assigned
        s.add(
            Clip(
                id=99,
                user_id=1,
                play_count=100,
                is_preprocessed=True,
                is_low_percentile=True,
                is_high_percentile=False,
            )
        )
        s.commit()
        _compute_creator_robust_stats(s, cfg)
        s.commit()
        skipped = s.get(Clip, 99)
        assert skipped.log_plays is None
        assert skipped.creator_relative_robust_z is None


def test_derive_eligibility_sets_all_clips():
    from modules.database import Clip, User
    from modules.filter import _derive_eligibility

    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        # eligible clip
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_preprocessed=True,
                is_garbage=False,
                is_too_old=False,
                is_too_long=False,
                is_too_short=False,
                is_low_percentile=False,
                is_creator_low_outlier=False,
            )
        )
        # disqualified: is_garbage
        s.add(
            Clip(
                id=2,
                user_id=1,
                is_preprocessed=False,
                is_garbage=True,
                is_too_old=False,
                is_too_long=False,
                is_too_short=False,
                is_low_percentile=False,
                is_creator_low_outlier=False,
            )
        )
        # disqualified: creator is_low_plays_median
        s.add(User(id=2, is_low_plays_median=True, is_not_enough_clips=False))
        s.add(
            Clip(
                id=3,
                user_id=2,
                is_preprocessed=True,
                is_garbage=False,
                is_too_old=False,
                is_too_long=False,
                is_too_short=False,
                is_low_percentile=False,
                is_creator_low_outlier=False,
            )
        )
        s.commit()
        _derive_eligibility(s)
        s.commit()
        c1 = s.get(Clip, 1)
        c2 = s.get(Clip, 2)
        c3 = s.get(Clip, 3)
        assert c1.is_eligible is True
        assert c2.is_eligible is False
        assert c3.is_eligible is False


def test_derive_eligibility_no_nulls_after_run():
    from modules.database import Clip, User
    from modules.filter import _derive_eligibility

    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_preprocessed=True,
                is_garbage=False,
                is_too_old=False,
                is_too_long=False,
                is_too_short=False,
                is_low_percentile=False,
                is_creator_low_outlier=False,
            )
        )
        s.commit()
        _derive_eligibility(s)
        s.commit()
        clips = s.query(Clip).all()
        for c in clips:
            assert c.is_eligible is not None


def _seed_eligible_clips(s, user_id: int, play_counts: list, id_offset: int = 0):
    """Add eligible clips with given play counts for a user."""
    from modules.database import Clip

    for i, plays in enumerate(play_counts):
        s.add(
            Clip(
                id=id_offset + i + 1,
                user_id=user_id,
                play_count=plays,
                video_duration=10.0,
                taken_at=1700000000,
                video_url="http://x.com/v.mp4",
                like_count=5,
                is_garbage=False,
                is_too_short=False,
                is_too_long=False,
                is_too_old=False,
                is_low_percentile=False,
                is_creator_low_outlier=False,
                is_eligible=True,
            )
        )


def test_select_clips_basic():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import select_clips

    eng = _make_db()
    cfg = FilterSettings(
        selection_pool_percent=0.50,
        selected_clips_per_user=2,
        selection_random_seed=42,
    )
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        _seed_eligible_clips(s, user_id=1, play_counts=[100 * i for i in range(1, 11)])
        s.commit()
        select_clips(s, cfg)
        s.commit()
        selected = [c for c in s.query(Clip).all() if c.is_selected]
        assert len(selected) == 2
        u = s.get(User, 1)
        assert u.is_selected is True


def test_select_clips_stable_across_new_users():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import select_clips

    eng = _make_db()
    cfg = FilterSettings(
        selection_pool_percent=1.0,
        selected_clips_per_user=3,
        selection_random_seed=42,
    )
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        _seed_eligible_clips(s, user_id=1, play_counts=[1000, 2000, 3000, 4000, 5000])
        s.commit()
        select_clips(s, cfg)
        s.commit()
        selected_run1 = {c.id for c in s.query(Clip).all() if c.is_selected}

    eng2 = _make_db()
    with Session(eng2) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        s.add(User(id=2, is_low_plays_median=False, is_not_enough_clips=False))
        _seed_eligible_clips(s, user_id=1, play_counts=[1000, 2000, 3000, 4000, 5000])
        _seed_eligible_clips(s, user_id=2, play_counts=[500, 600, 700], id_offset=100)
        s.commit()
        select_clips(s, cfg)
        s.commit()
        selected_run2 = {
            c.id for c in s.query(Clip).filter(Clip.user_id == 1).all() if c.is_selected
        }

    assert selected_run1 == selected_run2


def test_select_clips_user_with_no_eligible_clips_not_selected():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import select_clips

    eng = _make_db()
    cfg = FilterSettings(
        selection_pool_percent=0.5,
        selected_clips_per_user=2,
        selection_random_seed=42,
    )
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=False, is_not_enough_clips=False))
        s.add(
            Clip(
                id=1,
                user_id=1,
                play_count=5000,
                video_duration=10.0,
                taken_at=1700000000,
                video_url="http://x.com/v.mp4",
                like_count=5,
                is_garbage=True,
                is_eligible=False,
            )
        )
        s.commit()
        select_clips(s, cfg)
        s.commit()
        u = s.get(User, 1)
        assert u.is_selected is False


def test_not_enough_clips_independent_of_low_plays_median():
    """A user flagged as low_plays_median MUST still get is_not_enough_clips
    computed from preprocessed-clip count, not short-circuited to False."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1, is_low_plays_median=True))
        s.commit()

        from modules.config import FilterSettings
        from modules.filter import _flag_users_without_enough_clips

        cfg = FilterSettings(min_eligible_clips_per_user=5)
        _flag_users_without_enough_clips(s, cfg)

        u = s.get(User, 1)
        # User has zero preprocessed clips → flagged True regardless of low_plays_median.
        assert u.is_not_enough_clips is True


def test_flag_users_without_enough_clips_second_pass_counts_after_soft_exclusions():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _flag_users_without_enough_clips

    eng = _make_db()
    cfg = FilterSettings(min_eligible_clips_per_user=3)
    with Session(eng) as s:
        s.add(User(id=1))
        # Five preprocessed clips, three of which are soft-excluded as
        # is_low_percentile=True → only 2 remain, below the threshold of 3.
        s.add(Clip(id=1, user_id=1, is_preprocessed=True, is_low_percentile=False))
        s.add(Clip(id=2, user_id=1, is_preprocessed=True, is_low_percentile=False))
        s.add(Clip(id=3, user_id=1, is_preprocessed=True, is_low_percentile=True))
        s.add(Clip(id=4, user_id=1, is_preprocessed=True, is_low_percentile=True))
        s.add(Clip(id=5, user_id=1, is_preprocessed=True, is_low_percentile=True))
        s.commit()
        _flag_users_without_enough_clips(s, cfg)
        s.commit()
        assert s.get(User, 1).is_not_enough_clips is True


def test_user_stats_table_columns():
    from modules.database import UserStats

    cols = {c.key for c in UserStats.__table__.columns}
    expected = {
        "user_id",
        "n_clips",
        "median_plays",
        "mean_plays",
        "max_plays",
        "min_plays",
        "mean_log_plays",
        "median_log_plays",
        "log_plays_std",
        "log_plays_mad",
        "top_to_median_plays_ratio",
        "share_of_plays_from_top_clip",
        "median_video_duration",
        "mean_video_duration",
        "oldest_clip_taken_at",
        "newest_clip_taken_at",
        "clip_time_span_days",
        "approx_clips_per_week",
        "computed_at",
    }
    assert expected.issubset(cols), f"UserStats missing columns: {expected - cols}"


def test_user_stats_user_id_is_primary_key():
    from modules.database import UserStats

    pk_cols = {c.key for c in UserStats.__table__.primary_key.columns}
    assert pk_cols == {"user_id"}


def test_user_stats_insert_roundtrip():
    from modules.database import Base, User, UserStats

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(UserStats(user_id=1, n_clips=3, median_plays=5000.0))
        s.commit()
        row = s.get(UserStats, 1)
        assert row is not None
        assert row.n_clips == 3
        assert row.median_plays == 5000.0
        assert row.computed_at is not None


def test_has_any_flag_only_true_counts():
    from types import SimpleNamespace

    from modules.filter import _has_any_flag

    flags = ("a", "b", "c")
    assert _has_any_flag(SimpleNamespace(a=None, b=None, c=None), flags) is False
    assert _has_any_flag(SimpleNamespace(a=False, b=False, c=False), flags) is False
    assert _has_any_flag(SimpleNamespace(a=False, b=True, c=False), flags) is True
    # Missing attributes are treated as not-set
    assert _has_any_flag(SimpleNamespace(), flags) is False


def test_preprocessed_clips_filters_by_is_preprocessed_true():
    from types import SimpleNamespace

    from modules.filter import _preprocessed_clips

    user = SimpleNamespace(
        clips=[
            SimpleNamespace(is_preprocessed=True),
            SimpleNamespace(is_preprocessed=False),
            SimpleNamespace(is_preprocessed=None),
            SimpleNamespace(is_preprocessed=True),
        ]
    )
    assert len(_preprocessed_clips(user)) == 2


def test_eligible_clips_filters_by_is_eligible_true():
    from types import SimpleNamespace

    from modules.filter import _eligible_clips

    user = SimpleNamespace(
        clips=[
            SimpleNamespace(is_eligible=True),
            SimpleNamespace(is_eligible=False),
            SimpleNamespace(is_eligible=None),
        ]
    )
    assert len(_eligible_clips(user)) == 1


def test_selected_clips_filters_by_is_selected_true():
    from types import SimpleNamespace

    from modules.filter import _selected_clips

    user = SimpleNamespace(
        clips=[
            SimpleNamespace(is_selected=True),
            SimpleNamespace(is_selected=False),
            SimpleNamespace(is_selected=None),
        ]
    )
    assert len(_selected_clips(user)) == 1


def test_derive_preprocessing_status_marks_clean_clips_true():
    from modules.database import Clip, User
    from modules.filter import _derive_preprocessing_status

    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_garbage=False,
                is_too_short=False,
                is_too_long=False,
                is_too_old=False,
            )
        )
        s.commit()
        _derive_preprocessing_status(s)
        s.commit()
        assert s.get(Clip, 1).is_preprocessed is True


def test_derive_preprocessing_status_marks_garbage_clips_false():
    from modules.database import Clip, User
    from modules.filter import _derive_preprocessing_status

    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(
            Clip(
                id=1,
                user_id=1,
                is_garbage=True,
                is_too_short=False,
                is_too_long=False,
                is_too_old=False,
            )
        )
        s.commit()
        _derive_preprocessing_status(s)
        s.commit()
        assert s.get(Clip, 1).is_preprocessed is False


def test_derive_preprocessing_status_handles_none_hard_flags_as_not_excluded():
    from modules.database import Clip, User
    from modules.filter import _derive_preprocessing_status

    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(Clip(id=1, user_id=1))  # all flags None
        s.commit()
        _derive_preprocessing_status(s)
        s.commit()
        # None != True for each flag, so the clip is preprocessed.
        assert s.get(Clip, 1).is_preprocessed is True


def test_flag_low_median_creators_uses_preprocessed_clips_only():
    """Garbage clips with extreme play counts must not skew the median."""
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import _flag_low_median_creators

    eng = _make_db()
    cfg = FilterSettings(creator_min_median_views=10000)
    with Session(eng) as s:
        s.add(User(id=1))
        # Garbage clips with extreme plays — must be ignored
        for i, plays in enumerate([1_000_000, 2_000_000], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=plays,
                    is_preprocessed=False,
                )
            )
        # Preprocessed low-play clips
        for i, plays in enumerate([500, 600], start=10):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=plays,
                    is_preprocessed=True,
                )
            )
        s.commit()
        _flag_low_median_creators(s, cfg)
        s.commit()
        assert s.get(User, 1).is_low_plays_median is True


def test_calculate_user_stats_inserts_row_per_user_with_preprocessed_clips():
    from modules.database import Clip, User, UserStats
    from modules.filter import calculate_user_stats

    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1))
        for i, plays in enumerate([1000, 2000, 3000, 4000, 5000], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=plays,
                    video_duration=10.0,
                    taken_at=1700000000 + i * 86400,
                    is_preprocessed=True,
                )
            )
        # Non-preprocessed clip must be excluded from stats
        s.add(Clip(id=99, user_id=1, play_count=999999, is_preprocessed=False))
        s.commit()
        calculate_user_stats(s)
        s.commit()
        row = s.get(UserStats, 1)
        assert row is not None
        assert row.n_clips == 5
        assert row.median_plays == 3000.0
        assert row.max_plays == 5000
        assert row.min_plays == 1000
        assert row.mean_plays == 3000.0


def test_calculate_user_stats_skips_users_with_no_preprocessed_clips():
    from modules.database import Clip, User, UserStats
    from modules.filter import calculate_user_stats

    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(Clip(id=1, user_id=1, play_count=1000, is_preprocessed=False))
        s.commit()
        calculate_user_stats(s)
        s.commit()
        assert s.get(UserStats, 1) is None


def test_calculate_user_stats_is_idempotent_and_recomputes():
    """Running twice produces the same final row (upsert/recompute semantics)."""
    from modules.database import Clip, User, UserStats
    from modules.filter import calculate_user_stats

    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1))
        for i, plays in enumerate([1000, 2000, 3000], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=plays,
                    video_duration=10.0,
                    taken_at=1700000000,
                    is_preprocessed=True,
                )
            )
        s.commit()
        calculate_user_stats(s)
        s.commit()
        first = s.get(UserStats, 1)
        n1, med1 = first.n_clips, first.median_plays
        calculate_user_stats(s)
        s.commit()
        second = s.get(UserStats, 1)
        assert second.n_clips == n1
        assert second.median_plays == med1
        # Exactly one row per user, no duplicates
        assert s.query(UserStats).filter_by(user_id=1).count() == 1


def test_calculate_user_stats_share_and_ratio():
    from modules.database import Clip, User, UserStats
    from modules.filter import calculate_user_stats

    eng = _make_db()
    with Session(eng) as s:
        s.add(User(id=1))
        # plays = [1, 1, 1, 7] → max=7, median=1, sum=10
        for i, plays in enumerate([1, 1, 1, 7], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=plays,
                    video_duration=10.0,
                    taken_at=1700000000,
                    is_preprocessed=True,
                )
            )
        s.commit()
        calculate_user_stats(s)
        s.commit()
        row = s.get(UserStats, 1)
        assert row.top_to_median_plays_ratio == 7.0
        assert abs(row.share_of_plays_from_top_clip - 0.7) < 1e-9


def test_calculate_user_stats_time_span_and_cadence():
    from modules.database import Clip, User, UserStats
    from modules.filter import calculate_user_stats

    eng = _make_db()
    week_secs = 7 * 86400
    with Session(eng) as s:
        s.add(User(id=1))
        # Two clips, exactly one week apart → span=7 days, ≈2 clips/week
        s.add(
            Clip(
                id=1,
                user_id=1,
                play_count=1000,
                video_duration=10.0,
                taken_at=1700000000,
                is_preprocessed=True,
            )
        )
        s.add(
            Clip(
                id=2,
                user_id=1,
                play_count=2000,
                video_duration=10.0,
                taken_at=1700000000 + week_secs,
                is_preprocessed=True,
            )
        )
        s.commit()
        calculate_user_stats(s)
        s.commit()
        row = s.get(UserStats, 1)
        assert row.oldest_clip_taken_at == 1700000000
        assert row.newest_clip_taken_at == 1700000000 + week_secs
        assert row.clip_time_span_days == 7.0
        assert row.approx_clips_per_week == 2.0


def test_process_dataset_end_to_end():
    from modules.config import FilterSettings
    from modules.database import Clip, User
    from modules.filter import process_dataset

    eng = _make_db()
    cfg = FilterSettings(
        min_video_duration=3,
        max_video_duration=80,
        min_taken_at=1600000000,
        creator_min_median_views=1000,
        min_eligible_clips_per_user=2,
        global_low_percentile=0,
        global_high_percentile=100,
        creator_low_z_threshold=-3.5,
        selection_pool_percent=1.0,
        selected_clips_per_user=2,
        selection_random_seed=42,
    )
    with Session(eng) as s:
        s.add(User(id=1))
        for i, plays in enumerate([2000, 3000, 4000, 5000, 6000], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=plays,
                    video_duration=15.0,
                    taken_at=1700000000,
                    video_url="http://x.com/v.mp4",
                    like_count=10,
                )
            )
        s.add(
            Clip(
                id=99,
                user_id=1,
                play_count=None,
                video_duration=None,
                taken_at=None,
                video_url=None,
                like_count=None,
            )
        )
        s.commit()

    process_dataset(cfg, engine=eng)

    with Session(eng) as s:
        clips = s.query(Clip).all()
        for c in clips:
            assert c.is_eligible is not None
            assert c.is_preprocessed is not None
        garbage = s.get(Clip, 99)
        assert garbage.is_preprocessed is False
        assert garbage.is_eligible is False
        healthy = [c for c in clips if c.id != 99]
        assert all(c.is_eligible is True for c in healthy)
        assert s.get(User, 1).is_selected is True


def test_process_dataset_is_idempotent():
    from modules.config import FilterSettings
    from modules.database import Clip, User, UserStats
    from modules.filter import process_dataset

    eng = _make_db()
    cfg = FilterSettings(
        min_video_duration=3,
        max_video_duration=80,
        min_taken_at=1600000000,
        creator_min_median_views=1000,
        min_eligible_clips_per_user=2,
        global_low_percentile=0,
        global_high_percentile=100,
        creator_low_z_threshold=-3.5,
        selection_pool_percent=1.0,
        selected_clips_per_user=2,
        selection_random_seed=42,
    )
    with Session(eng) as s:
        s.add(User(id=1))
        for i, plays in enumerate([2000, 3000, 4000, 5000, 6000], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=plays,
                    video_duration=15.0,
                    taken_at=1700000000,
                    video_url="http://x.com/v.mp4",
                    like_count=10,
                )
            )
        s.commit()

    process_dataset(cfg, engine=eng)
    with Session(eng) as s:
        flags_run1 = [
            (c.id, c.is_preprocessed, c.is_eligible, c.is_selected)
            for c in s.query(Clip).order_by(Clip.id).all()
        ]
        stats_run1 = {row.user_id: row.n_clips for row in s.query(UserStats).all()}

    process_dataset(cfg, engine=eng)
    with Session(eng) as s:
        flags_run2 = [
            (c.id, c.is_preprocessed, c.is_eligible, c.is_selected)
            for c in s.query(Clip).order_by(Clip.id).all()
        ]
        stats_run2 = {row.user_id: row.n_clips for row in s.query(UserStats).all()}
        assert s.query(UserStats).filter_by(user_id=1).count() == 1

    assert flags_run1 == flags_run2
    assert stats_run1 == stats_run2


def test_process_dataset_runs_not_enough_clips_after_robust_stats():
    """Same intent as the old test: the second pass of _flag_users_without_enough_clips
    must run inside process_dataset so that users dropping below the threshold after
    soft exclusions are correctly flagged."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1))
        s.add(
            Clip(
                id=1,
                user_id=1,
                video_duration=10.0,
                taken_at=1700000000,
                play_count=100_000,
                video_url="https://x/high",
                like_count=10,
            )
        )
        for i in range(2, 11):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    video_duration=10.0,
                    taken_at=1700000000,
                    play_count=1000,
                    video_url=f"https://x/{i}",
                    like_count=10,
                )
            )
        s.commit()

    from modules.config import FilterSettings
    from modules.filter import process_dataset

    cfg = FilterSettings(
        min_eligible_clips_per_user=8,
        creator_min_median_views=500,
        creator_low_z_threshold=0.5,
    )
    process_dataset(cfg, engine=eng)

    with Session(eng) as s:
        u = s.get(User, 1)
        clips = s.query(Clip).filter(Clip.user_id == 1).all()
        surviving = [
            c
            for c in clips
            if c.is_preprocessed is True
            and not any(
                [c.is_low_percentile, c.is_high_percentile, c.is_creator_low_outlier]
            )
        ]
        assert u.is_not_enough_clips == (
            len(surviving) < cfg.min_eligible_clips_per_user
        )


def test_reset_dataset_processing_state_clears_all_derived_fields():
    from modules.config import FilterSettings
    from modules.database import Clip, User, UserStats
    from modules.filter import _reset_dataset_processing_state, process_dataset

    eng = _make_db()
    cfg = FilterSettings(
        min_video_duration=3,
        max_video_duration=80,
        min_taken_at=1600000000,
        creator_min_median_views=1000,
        min_eligible_clips_per_user=2,
        global_low_percentile=0,
        global_high_percentile=100,
        creator_low_z_threshold=-3.5,
        selection_pool_percent=1.0,
        selected_clips_per_user=2,
        selection_random_seed=42,
    )
    with Session(eng) as s:
        s.add(User(id=1))
        for i, plays in enumerate([2000, 3000, 4000], start=1):
            s.add(
                Clip(
                    id=i,
                    user_id=1,
                    play_count=plays,
                    video_duration=15.0,
                    taken_at=1700000000,
                    video_url="http://x/",
                    like_count=10,
                )
            )
        s.commit()

    process_dataset(cfg, engine=eng)
    with Session(eng) as s:
        _reset_dataset_processing_state(s)
        s.commit()
        for c in s.query(Clip).all():
            for col in [
                "is_garbage",
                "is_too_short",
                "is_too_long",
                "is_too_old",
                "is_low_percentile",
                "is_high_percentile",
                "is_creator_low_outlier",
                "log_plays",
                "creator_relative_robust_z",
                "is_preprocessed",
                "is_eligible",
                "is_selected",
            ]:
                assert getattr(c, col) is None, f"{col} not reset on clip {c.id}"
        for u in s.query(User).all():
            assert u.is_low_plays_median is None
            assert u.is_not_enough_clips is None
            assert u.is_selected is None
        assert s.query(UserStats).count() == 0
