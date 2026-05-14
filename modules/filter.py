from __future__ import annotations

import math
import random
import statistics
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from modules.config import FilterSettings
from modules.database import Clip, User

CLIP_EXCLUSION_FLAGS: tuple[str, ...] = (
    "is_garbage",
    "is_too_short",
    "is_too_long",
    "is_too_old",
    "is_low_percentile",
    "is_high_percentile",
    "is_creator_low_outlier",
)

USER_EXCLUSION_FLAGS: tuple[str, ...] = (
    "is_low_plays_median",
    "is_not_enough_clips",
)


def _has_clip_exclusion(clip: Any) -> bool:
    return any(getattr(clip, flag, None) is True for flag in CLIP_EXCLUSION_FLAGS)


def _has_user_exclusion(user: Any) -> bool:
    return any(getattr(user, flag, None) is True for flag in USER_EXCLUSION_FLAGS)


def _surviving_clips(user: Any) -> list:
    return [clip for clip in user.clips if not _has_clip_exclusion(clip)]


def _count_surviving_clips(user: Any) -> int:
    return len(_surviving_clips(user))


def _is_garbage(clip: Any) -> bool:
    if not clip.video_duration or clip.video_duration <= 0:
        return True
    if not clip.taken_at or clip.taken_at <= 0:
        return True
    if not clip.play_count or clip.play_count <= 0:
        return True
    if not clip.video_url or not str(clip.video_url).strip():
        return True
    return clip.like_count is None


def _is_too_short(clip: Any, *, min_video_duration: float) -> bool:
    return (clip.video_duration or 0) < min_video_duration


def _is_too_long(clip: Any, *, max_video_duration: float) -> bool:
    return (clip.video_duration or 0) > max_video_duration


def _is_too_old(clip: Any, *, min_taken_at: int) -> bool:
    return (clip.taken_at or 0) < min_taken_at


def _flag_garbage_clips(session: Session) -> None:
    clips = session.query(Clip).all()
    for clip in clips:
        clip.is_garbage = _is_garbage(clip)


def _flag_basic_policy_clips(session: Session, cfg: FilterSettings) -> None:
    for clip in session.query(Clip).all():
        clip.is_too_short = _is_too_short(
            clip, min_video_duration=cfg.min_video_duration
        )
        clip.is_too_long = _is_too_long(
            clip, max_video_duration=cfg.max_video_duration
        )
        clip.is_too_old = _is_too_old(clip, min_taken_at=cfg.min_taken_at)


def _flag_low_median_creators(session: Session, cfg: FilterSettings) -> None:
    for user in session.query(User).all():
        surviving_plays = [
            clip.play_count
            for clip in _surviving_clips(user)
            if clip.play_count is not None
        ]
        if not surviving_plays:
            user.is_low_plays_median = True
            continue
        median_plays = statistics.median(surviving_plays)
        user.is_low_plays_median = median_plays < cfg.creator_min_median_views


def _flag_users_without_enough_clips(session: Session, cfg: FilterSettings) -> None:
    for user in session.query(User).all():
        user.is_not_enough_clips = (
            _count_surviving_clips(user) < cfg.min_eligible_clips_per_user
        )


def _flag_global_percentile_clips(session: Session, cfg: FilterSettings) -> None:
    for clip in session.query(Clip).all():
        clip.is_low_percentile = False
        clip.is_high_percentile = False

    surviving = [
        clip
        for user in session.query(User).all()
        if not _has_user_exclusion(user)
        for clip in _surviving_clips(user)
    ]

    if not surviving:
        return

    plays = np.array([c.play_count for c in surviving], dtype=float)
    low_boundary = float(np.percentile(plays, cfg.global_low_percentile))
    high_boundary = float(np.percentile(plays, cfg.global_high_percentile))

    for clip in surviving:
        clip.is_low_percentile = clip.play_count < low_boundary
        clip.is_high_percentile = clip.play_count > high_boundary


def _median_absolute_deviation(values: list[float]) -> float:
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def _compute_creator_robust_stats(session: Session, cfg: FilterSettings) -> None:
    for user in session.query(User).all():
        user.log_plays_median = None
        user.log_plays_mad = None
        for clip in user.clips:
            clip.log_plays = None
            clip.creator_relative_robust_z = None
            clip.is_creator_low_outlier = False

        if _has_user_exclusion(user):
            continue
        surviving = _surviving_clips(user)
        if not surviving:
            continue

        log_plays_vals = [math.log1p(c.play_count) for c in surviving]
        creator_median = statistics.median(log_plays_vals)
        mad = _median_absolute_deviation(log_plays_vals)

        user.log_plays_median = creator_median
        user.log_plays_mad = mad

        for clip in surviving:
            lp = math.log1p(clip.play_count)
            clip.log_plays = lp
            if mad > 0:
                clip.creator_relative_robust_z = 0.6745 * (lp - creator_median) / mad
            else:
                clip.creator_relative_robust_z = 0.0
            clip.is_creator_low_outlier = (
                clip.creator_relative_robust_z < cfg.creator_low_z_threshold
            )


def _derive_eligibility(session: Session) -> None:
    user_map = {u.id: u for u in session.query(User).all()}
    for clip in session.query(Clip).all():
        user = user_map.get(clip.user_id)
        if user is None:
            clip.is_eligible = False
            continue
        clip.is_eligible = not (
            _has_clip_exclusion(clip) or _has_user_exclusion(user)
        )


def select_clips_for_embedding(session: Session, cfg: FilterSettings) -> None:
    for clip in session.query(Clip).all():
        clip.is_selected = False
    for user in session.query(User).all():
        user.is_selected = False

    users = session.query(User).all()
    for user in users:
        eligible = sorted(
            [c for c in user.clips if c.is_eligible],
            key=lambda c: c.play_count or 0,
            reverse=True,
        )
        if not eligible:
            continue

        pool_size = max(1, math.ceil(len(eligible) * cfg.selection_pool_percent))
        pool = eligible[:pool_size]

        n = min(cfg.selected_clips_per_user, len(pool))
        rng = random.Random(f"{cfg.selection_random_seed}:{user.id}")
        chosen = rng.sample(pool, n)

        for clip in chosen:
            clip.is_selected = True
        user.is_selected = True


def preprocess_new_data(
    cfg: FilterSettings,
    *,
    engine=None,
) -> None:
    from modules.database import get_engine

    eng = engine or get_engine()
    with Session(eng) as session:
        for clip in session.query(Clip).all():
            clip.is_garbage = None
            clip.is_too_short = None
            clip.is_too_long = None
            clip.is_too_old = None
            clip.is_low_percentile = None
            clip.is_high_percentile = None
            clip.is_creator_low_outlier = None
            clip.log_plays = None
            clip.creator_relative_robust_z = None
            clip.is_eligible = None
            clip.is_selected = None
        for user in session.query(User).all():
            user.is_low_plays_median = None
            user.is_not_enough_clips = None
            user.is_selected = None
            user.log_plays_median = None
            user.log_plays_mad = None

        _flag_garbage_clips(session)
        _flag_basic_policy_clips(session, cfg)
        _flag_low_median_creators(session, cfg)
        _flag_users_without_enough_clips(session, cfg)
        _flag_global_percentile_clips(session, cfg)
        _compute_creator_robust_stats(session, cfg)
        _flag_users_without_enough_clips(session, cfg)
        _derive_eligibility(session)
        select_clips_for_embedding(session, cfg)

        session.commit()
