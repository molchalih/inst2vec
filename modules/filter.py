from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from modules.config import FilterSettings
from modules.database import Clip, User


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


def _is_low_play_count(clip: Any, *, min_play_count: int) -> bool:
    return (clip.play_count or 0) < min_play_count


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
    clips = session.query(Clip).all()
    for clip in clips:
        clip.is_low_play_count = _is_low_play_count(clip, min_play_count=cfg.min_play_count)
        clip.is_too_short = _is_too_short(clip, min_video_duration=cfg.min_video_duration)
        clip.is_too_long = _is_too_long(clip, max_video_duration=cfg.max_video_duration)
        clip.is_too_old = _is_too_old(clip, min_taken_at=cfg.min_taken_at)


def _flag_low_median_creators(session: Session, cfg: FilterSettings) -> None:
    users = session.query(User).all()
    for user in users:
        surviving_plays = [
            c.play_count
            for c in user.clips
            if not c.is_garbage
            and not c.is_low_play_count
            and not c.is_too_short
            and not c.is_too_long
            and not c.is_too_old
            and c.play_count is not None
        ]
        if not surviving_plays:
            user.is_low_plays_median = True
            continue
        median_plays = statistics.median(surviving_plays)
        user.is_low_plays_median = median_plays < cfg.creator_min_median_views


def _flag_users_without_enough_clips(session: Session, cfg: FilterSettings) -> None:
    users = session.query(User).all()
    for user in users:
        if user.is_low_plays_median:
            user.is_not_enough_clips = False
            continue
        count = sum(
            1
            for c in user.clips
            if not c.is_garbage
            and not c.is_low_play_count
            and not c.is_too_short
            and not c.is_too_long
            and not c.is_too_old
        )
        user.is_not_enough_clips = count < cfg.min_eligible_clips_per_user


def _surviving_clips_for_percentile(session: Session) -> list:
    return [
        c
        for u in session.query(User).all()
        if not u.is_low_plays_median and not u.is_not_enough_clips
        for c in u.clips
        if not c.is_garbage
        and not c.is_low_play_count
        and not c.is_too_short
        and not c.is_too_long
        and not c.is_too_old
        and c.play_count is not None
    ]


def _flag_global_percentile_clips(session: Session, cfg: FilterSettings) -> None:
    surviving = _surviving_clips_for_percentile(session)
    if not surviving:
        return
    plays = np.array([c.play_count for c in surviving], dtype=float)
    low_boundary = float(np.percentile(plays, cfg.global_low_percentile))
    high_boundary = float(np.percentile(plays, cfg.global_high_percentile))

    for c in session.query(Clip).all():
        c.is_low_percentile = False
        c.is_high_percentile = False

    surviving_ids = {c.id for c in surviving}
    for c in session.query(Clip).all():
        if c.id not in surviving_ids:
            continue
        c.is_low_percentile = c.play_count < low_boundary
        c.is_high_percentile = c.play_count > high_boundary
