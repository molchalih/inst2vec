from __future__ import annotations

from typing import Any


def _is_garbage(clip: Any) -> bool:
    if not clip.video_duration or clip.video_duration <= 0:
        return True
    if not clip.taken_at or clip.taken_at <= 0:
        return True
    if not clip.play_count or clip.play_count <= 0:
        return True
    if not clip.download_url or not str(clip.download_url).strip():
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
