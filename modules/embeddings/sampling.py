"""Adaptive video sampling and token-budget fallback for embedding."""

from __future__ import annotations

import subprocess
from typing import Literal, overload


@overload
def probe_duration_seconds(path: str, *, strict: Literal[True]) -> float: ...
@overload
def probe_duration_seconds(
    path: str, *, strict: Literal[False] = False
) -> float | None: ...
def probe_duration_seconds(path: str, *, strict: bool = False) -> float | None:
    """Return video duration in seconds via ffprobe.

    strict=False (default): swallow all errors and return None when the
    duration cannot be determined.
    strict=True: re-raise the underlying exception (matches the
    previous gemini-side behavior — used for the gemini upload path
    where a missing duration is fatal).
    """
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
            check=strict,
            timeout=10,
        )
    except Exception:
        if strict:
            raise
        return None
    if result.returncode != 0:
        if strict:
            raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
        return None
    raw = (result.stdout or "").strip()
    if not raw:
        if strict:
            raise RuntimeError("ffprobe returned empty duration")
        return None
    try:
        duration = float(raw)
    except ValueError:
        if strict:
            raise
        return None
    if duration > 0:
        return duration
    if strict:
        raise RuntimeError(f"ffprobe returned non-positive duration: {duration}")
    return None


def adaptive_sampling(
    path: str, adaptive_max_frames: int, adaptive_default_fps: float
) -> tuple[float, int]:
    """Choose (fps, max_frames) from probed clip duration."""
    duration = probe_duration_seconds(path)
    if duration is None:
        return adaptive_default_fps, adaptive_max_frames
    if duration < 15:
        return 3.0, adaptive_max_frames
    if duration <= 45:
        return 2.0, adaptive_max_frames
    return 1.0, adaptive_max_frames


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
