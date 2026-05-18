from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def has_any_flag(obj: Any, flags: tuple[str, ...]) -> bool:
    return any(getattr(obj, flag, None) is True for flag in flags)


def preprocessed_clips(user: Any) -> list:
    return [clip for clip in user.clips if clip.is_preprocessed is True]


def eligible_clips(user: Any) -> list:
    return [clip for clip in user.clips if clip.is_eligible is True]


def selected_clips(user: Any) -> list:
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


def is_creator_low_outlier(session: Session, clip: Any) -> bool:
    """Return True iff the scratch row marks this clip as a creator-relative low outlier."""
    from core.database import ClipFilterScratch

    row = session.get(ClipFilterScratch, clip.id)
    return bool(row and row.is_creator_low_outlier)
