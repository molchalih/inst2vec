"""Adaptive video sampling and token-budget fallback for embedding."""

from __future__ import annotations

import subprocess


def probe_duration_seconds(path: str) -> float | None:
    """Return video duration in seconds via ffprobe, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    raw = (result.stdout or "").strip()
    if not raw:
        return None
    try:
        duration = float(raw)
    except ValueError:
        return None
    return duration if duration > 0 else None


def adaptive_sampling(
    path: str, adaptive_max_frames: int, adaptive_default_fps: float
) -> tuple[float, int, float | None]:
    """Choose (fps, max_frames, duration) from probed clip duration."""
    duration = probe_duration_seconds(path)
    if duration is None:
        return adaptive_default_fps, adaptive_max_frames, None
    if duration < 15:
        return 3.0, adaptive_max_frames, duration
    if duration <= 45:
        return 2.0, adaptive_max_frames, duration
    return 1.0, adaptive_max_frames, duration


def frame_retry_schedule(initial_max_frames: int) -> list[int]:
    """Descending frame-cap schedule for token-budget retries."""
    caps = [initial_max_frames, 64, 48, 32, 24, 16]
    unique: list[int] = []
    seen: set[int] = set()
    for c in caps:
        if c <= initial_max_frames and c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def is_token_mismatch_error(exc: Exception) -> bool:
    """True iff `exc` is a recognized Qwen video-token-budget mismatch."""
    msg = str(exc)
    return (
        "Mismatch in `video` token count" in msg
        or "Likely due to `truncation='max_length'" in msg
    )
