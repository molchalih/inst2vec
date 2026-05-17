from __future__ import annotations

from sqlalchemy.orm import Session

from core.database import Clip, User, UserStats

STAGE = "filter"
SCOPE = "all"

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
