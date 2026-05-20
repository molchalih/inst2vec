"""Shared ffmpeg/ffprobe subprocess helpers.

Used by speech VAD, audio extraction, and any future stage that shells
out to ffmpeg. Failure path is uniform: ``run_ffmpeg`` returns False on
non-zero exit or timeout; callers decide whether to raise.
"""

from __future__ import annotations

import subprocess


def run_ffmpeg(cmd: list[str], *, timeout: int) -> bool:
    """Run ``cmd`` (list, no shell). Return True on exit code 0, else False."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def has_audio_stream(path: str, *, timeout: int = 10) -> bool:
    """Return True iff ``path`` contains at least one audio stream.

    Uses ``ffprobe -select_streams a`` to list audio-stream indexes; any
    non-empty stdout means an audio stream is present. Returns False on
    missing file, ffprobe failure, timeout, or no audio streams. Never
    raises — callers treat False as "skip audio work for this clip."
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if result.returncode != 0:
        return False
    return bool((result.stdout or "").strip())
