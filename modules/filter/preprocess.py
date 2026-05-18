from __future__ import annotations

import math
import statistics

import numpy as np
from sqlalchemy.orm import Session

from core.config import FilterSettings
from core.database import Clip, ClipFilterScratch, User
from modules.filter.predicates import (
    _is_garbage,
    _is_too_long,
    _is_too_old,
    _is_too_short,
    has_any_flag,
    is_creator_low_outlier,
    preprocessed_clips,
)
from modules.filter.state import (
    HARD_CLIP_EXCLUSION_FLAGS,
    SOFT_CLIP_EXCLUSION_FLAGS,
    USER_EXCLUSION_FLAGS,
)
from modules.filter.stats import _median_absolute_deviation


def flag_garbage_clips(session: Session) -> None:
    clips = session.query(Clip).all()
    for clip in clips:
        clip.is_garbage = _is_garbage(clip)


def flag_basic_policy_clips(session: Session, cfg: FilterSettings) -> None:
    for clip in session.query(Clip).all():
        clip.is_too_short = _is_too_short(
            clip, min_video_duration=cfg.min_video_duration
        )
        clip.is_too_long = _is_too_long(clip, max_video_duration=cfg.max_video_duration)
        clip.is_too_old = _is_too_old(clip, min_taken_at=cfg.min_taken_at)


def derive_preprocessing_status(session: Session) -> None:
    for clip in session.query(Clip).all():
        clip.is_preprocessed = not has_any_flag(clip, HARD_CLIP_EXCLUSION_FLAGS)


def flag_low_median_creators(session: Session, cfg: FilterSettings) -> None:
    for user in session.query(User).all():
        plays = [
            clip.play_count
            for clip in preprocessed_clips(user)
            if clip.play_count is not None
        ]
        if not plays:
            user.is_low_plays_median = True
            continue
        user.is_low_plays_median = (
            statistics.median(plays) < cfg.creator_min_median_views
        )


def flag_users_without_enough_preprocessed_clips(
    session: Session, cfg: FilterSettings
) -> None:
    for user in session.query(User).all():
        count = sum(
            1
            for clip in preprocessed_clips(user)
            if not has_any_flag(clip, SOFT_CLIP_EXCLUSION_FLAGS)
        )
        user.is_not_enough_preprocessed = count < cfg.min_eligible_clips_per_user


def flag_users_without_enough_eligible_clips(
    session: Session, cfg: FilterSettings
) -> None:
    for user in session.query(User).all():
        count = sum(
            1
            for clip in preprocessed_clips(user)
            if not has_any_flag(clip, SOFT_CLIP_EXCLUSION_FLAGS)
            and not is_creator_low_outlier(session, clip)
        )
        user.is_not_enough_eligible = count < cfg.min_eligible_clips_per_user


def flag_global_percentile_clips(session: Session, cfg: FilterSettings) -> None:
    for clip in session.query(Clip).all():
        clip.is_low_percentile = False
        clip.is_high_percentile = False

    population = [
        clip
        for user in session.query(User).all()
        if not has_any_flag(user, USER_EXCLUSION_FLAGS)
        for clip in preprocessed_clips(user)
    ]
    if not population:
        return

    plays = np.array([c.play_count for c in population], dtype=float)
    low_boundary = float(np.percentile(plays, cfg.global_low_percentile))
    high_boundary = float(np.percentile(plays, cfg.global_high_percentile))

    for clip in population:
        clip.is_low_percentile = clip.play_count < low_boundary
        clip.is_high_percentile = clip.play_count > high_boundary


def compute_creator_robust_stats(session: Session, cfg: FilterSettings) -> None:
    session.query(ClipFilterScratch).delete()

    for user in session.query(User).all():
        if has_any_flag(user, USER_EXCLUSION_FLAGS):
            continue
        candidates = [
            clip
            for clip in preprocessed_clips(user)
            if not has_any_flag(clip, SOFT_CLIP_EXCLUSION_FLAGS)
        ]
        if not candidates:
            continue

        log_plays_vals = [math.log1p(c.play_count) for c in candidates]
        creator_median = statistics.median(log_plays_vals)
        mad = _median_absolute_deviation(log_plays_vals)

        for clip in candidates:
            lp = math.log1p(clip.play_count)
            z = 0.6745 * (lp - creator_median) / mad if mad > 0 else 0.0
            session.add(
                ClipFilterScratch(
                    clip_id=clip.id,
                    log_plays=lp,
                    creator_relative_robust_z=z,
                    is_creator_low_outlier=z < cfg.creator_low_z_threshold,
                )
            )


def derive_eligibility(session: Session) -> None:
    user_map = {u.id: u for u in session.query(User).all()}
    for clip in session.query(Clip).all():
        user = user_map.get(clip.user_id)
        if user is None:
            clip.is_eligible = False
            continue
        clip.is_eligible = (
            clip.is_preprocessed is True
            and not has_any_flag(clip, SOFT_CLIP_EXCLUSION_FLAGS)
            and not is_creator_low_outlier(session, clip)
            and not has_any_flag(user, USER_EXCLUSION_FLAGS)
        )


def derive_user_eligibility(session: Session) -> None:
    for user in session.query(User).all():
        user.is_eligible = not has_any_flag(user, USER_EXCLUSION_FLAGS)


def _hard_preprocess(session: Session, cfg: FilterSettings) -> None:
    flag_garbage_clips(session)
    flag_basic_policy_clips(session, cfg)
    derive_preprocessing_status(session)
    flag_low_median_creators(session, cfg)
    flag_users_without_enough_preprocessed_clips(session, cfg)


def _soft_preprocess(session: Session, cfg: FilterSettings) -> None:
    flag_global_percentile_clips(session, cfg)
    compute_creator_robust_stats(session, cfg)
    flag_users_without_enough_eligible_clips(session, cfg)
    derive_eligibility(session)
    derive_user_eligibility(session)
