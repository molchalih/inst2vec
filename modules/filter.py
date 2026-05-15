from __future__ import annotations

import math
import random
import statistics
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from modules.config import FilterSettings
from modules.database import Clip, User, UserStats

HARD_CLIP_EXCLUSION_FLAGS: tuple[str, ...] = (
    "is_garbage",
    "is_too_short",
    "is_too_long",
    "is_too_old",
)

SOFT_CLIP_EXCLUSION_FLAGS: tuple[str, ...] = (
    "is_low_percentile",
    "is_high_percentile",
    "is_creator_low_outlier",
)

CLIP_EXCLUSION_FLAGS: tuple[str, ...] = (
    *HARD_CLIP_EXCLUSION_FLAGS,
    *SOFT_CLIP_EXCLUSION_FLAGS,
)

USER_EXCLUSION_FLAGS: tuple[str, ...] = (
    "is_low_plays_median",
    "is_not_enough_clips",
)


def _has_any_flag(obj: Any, flags: tuple[str, ...]) -> bool:
    return any(getattr(obj, flag, None) is True for flag in flags)


def _preprocessed_clips(user: Any) -> list:
    return [clip for clip in user.clips if clip.is_preprocessed is True]


def _eligible_clips(user: Any) -> list:
    return [clip for clip in user.clips if clip.is_eligible is True]


def _selected_clips(user: Any) -> list:
    return [clip for clip in user.clips if clip.is_selected is True]


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
        clip.is_too_long = _is_too_long(clip, max_video_duration=cfg.max_video_duration)
        clip.is_too_old = _is_too_old(clip, min_taken_at=cfg.min_taken_at)


def _derive_preprocessing_status(session: Session) -> None:
    for clip in session.query(Clip).all():
        clip.is_preprocessed = not _has_any_flag(clip, HARD_CLIP_EXCLUSION_FLAGS)


def _flag_low_median_creators(session: Session, cfg: FilterSettings) -> None:
    for user in session.query(User).all():
        plays = [
            clip.play_count
            for clip in _preprocessed_clips(user)
            if clip.play_count is not None
        ]
        if not plays:
            user.is_low_plays_median = True
            continue
        user.is_low_plays_median = (
            statistics.median(plays) < cfg.creator_min_median_views
        )


def _flag_users_without_enough_clips(session: Session, cfg: FilterSettings) -> None:
    for user in session.query(User).all():
        count = sum(
            1
            for clip in _preprocessed_clips(user)
            if not _has_any_flag(clip, SOFT_CLIP_EXCLUSION_FLAGS)
        )
        user.is_not_enough_clips = count < cfg.min_eligible_clips_per_user


def _flag_global_percentile_clips(session: Session, cfg: FilterSettings) -> None:
    for clip in session.query(Clip).all():
        clip.is_low_percentile = False
        clip.is_high_percentile = False

    population = [
        clip
        for user in session.query(User).all()
        if not _has_any_flag(user, USER_EXCLUSION_FLAGS)
        for clip in _preprocessed_clips(user)
    ]
    if not population:
        return

    plays = np.array([c.play_count for c in population], dtype=float)
    low_boundary = float(np.percentile(plays, cfg.global_low_percentile))
    high_boundary = float(np.percentile(plays, cfg.global_high_percentile))

    for clip in population:
        clip.is_low_percentile = clip.play_count < low_boundary
        clip.is_high_percentile = clip.play_count > high_boundary


def _median_absolute_deviation(values: list[float]) -> float:
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def _compute_creator_robust_stats(session: Session, cfg: FilterSettings) -> None:
    # Reset state-dependent clip fields before recomputing so the result is
    # determined only by the current (preprocessed, soft-flag) state.
    for clip in session.query(Clip).all():
        clip.log_plays = None
        clip.creator_relative_robust_z = None
        clip.is_creator_low_outlier = False

    for user in session.query(User).all():
        if _has_any_flag(user, USER_EXCLUSION_FLAGS):
            continue
        candidates = [
            clip
            for clip in _preprocessed_clips(user)
            if not _has_any_flag(clip, SOFT_CLIP_EXCLUSION_FLAGS)
        ]
        if not candidates:
            continue

        log_plays_vals = [math.log1p(c.play_count) for c in candidates]
        creator_median = statistics.median(log_plays_vals)
        mad = _median_absolute_deviation(log_plays_vals)

        for clip in candidates:
            lp = math.log1p(clip.play_count)
            clip.log_plays = lp
            z = 0.6745 * (lp - creator_median) / mad if mad > 0 else 0.0
            clip.creator_relative_robust_z = z
            clip.is_creator_low_outlier = z < cfg.creator_low_z_threshold


def _derive_eligibility(session: Session) -> None:
    user_map = {u.id: u for u in session.query(User).all()}
    for clip in session.query(Clip).all():
        user = user_map.get(clip.user_id)
        if user is None:
            clip.is_eligible = False
            continue
        clip.is_eligible = (
            clip.is_preprocessed is True
            and not _has_any_flag(clip, SOFT_CLIP_EXCLUSION_FLAGS)
            and not _has_any_flag(user, USER_EXCLUSION_FLAGS)
        )


def _derive_user_eligibility(session: Session) -> None:
    for user in session.query(User).all():
        user.is_eligible = not _has_any_flag(user, USER_EXCLUSION_FLAGS)


def select_clips(session: Session, cfg: FilterSettings) -> None:
    for clip in session.query(Clip).all():
        clip.is_selected = False
    for user in session.query(User).all():
        user.is_selected = False

    for user in session.query(User).all():
        eligible = sorted(
            _eligible_clips(user),
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


def calculate_user_stats(session: Session) -> None:
    session.query(UserStats).delete()
    session.flush()

    for user in session.query(User).all():
        clips = _preprocessed_clips(user)
        if not clips:
            continue

        plays = [c.play_count for c in clips]
        log_plays_vals = [math.log1p(p) for p in plays]
        durations = [c.video_duration for c in clips if c.video_duration is not None]
        taken_ats = [c.taken_at for c in clips if c.taken_at is not None]

        n = len(clips)
        max_plays = max(plays)
        median_plays = statistics.median(plays)
        total_plays = sum(plays)
        ratio = max_plays / median_plays if median_plays > 0 else None
        share_top = max_plays / total_plays if total_plays > 0 else None

        oldest = min(taken_ats) if taken_ats else None
        newest = max(taken_ats) if taken_ats else None
        span_days = (
            (newest - oldest) / 86400.0
            if oldest is not None and newest is not None
            else None
        )
        per_week = (
            n / (span_days / 7.0) if span_days is not None and span_days > 0 else None
        )

        session.add(
            UserStats(
                user_id=user.id,
                n_clips=n,
                median_plays=float(median_plays),
                mean_plays=float(statistics.mean(plays)),
                max_plays=max_plays,
                min_plays=min(plays),
                mean_log_plays=float(statistics.mean(log_plays_vals)),
                median_log_plays=float(statistics.median(log_plays_vals)),
                log_plays_std=(
                    float(statistics.pstdev(log_plays_vals))
                    if len(log_plays_vals) > 1
                    else 0.0
                ),
                log_plays_mad=float(_median_absolute_deviation(log_plays_vals)),
                top_to_median_plays_ratio=ratio,
                share_of_plays_from_top_clip=share_top,
                median_video_duration=(
                    float(statistics.median(durations)) if durations else None
                ),
                mean_video_duration=(
                    float(statistics.mean(durations)) if durations else None
                ),
                oldest_clip_taken_at=oldest,
                newest_clip_taken_at=newest,
                clip_time_span_days=span_days,
                approx_clips_per_week=per_week,
            )
        )


def _reset_dataset_processing_state(session: Session) -> None:
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
        clip.is_preprocessed = None
        clip.is_eligible = None
        clip.is_selected = None
    for user in session.query(User).all():
        user.is_low_plays_median = None
        user.is_not_enough_clips = None
        user.is_selected = None
        user.is_eligible = None
    session.query(UserStats).delete()


def _hard_preprocess(session: Session, cfg: FilterSettings) -> None:
    _flag_garbage_clips(session)
    _flag_basic_policy_clips(session, cfg)
    _derive_preprocessing_status(session)
    _flag_low_median_creators(session, cfg)
    _flag_users_without_enough_clips(session, cfg)


def _soft_preprocess(session: Session, cfg: FilterSettings) -> None:
    _flag_global_percentile_clips(session, cfg)
    _compute_creator_robust_stats(session, cfg)
    _flag_users_without_enough_clips(session, cfg)
    _derive_eligibility(session)
    _derive_user_eligibility(session)


def _random_sample(session: Session, cfg: FilterSettings) -> None:
    select_clips(session, cfg)


def process_dataset(
    cfg: FilterSettings,
    *,
    engine=None,
) -> None:
    from modules.database import get_engine

    eng = engine or get_engine()
    with Session(eng) as session:
        _reset_dataset_processing_state(session)

        _hard_preprocess(session, cfg)
        calculate_user_stats(session)
        _soft_preprocess(session, cfg)
        _random_sample(session, cfg)

        session.commit()
