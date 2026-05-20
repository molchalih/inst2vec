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


def probe_audio_stream(path: str, *, timeout: int = 10) -> bool | None:
    """Tri-state audio-stream probe.

    Returns True if ``path`` has at least one audio stream, False if the
    probe succeeded and confirmed no audio streams, and None if ffprobe
    itself failed (missing binary, timeout, non-zero exit). The None case
    is transient and callers MUST treat it as retryable — never as a
    definitive "no audio" verdict.
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
        return None
    if result.returncode != 0:
        return None
    return bool((result.stdout or "").strip())


def has_audio_stream(path: str, *, timeout: int = 10) -> bool:
    """Return True iff ``path`` is confirmed to contain an audio stream.

    Thin wrapper over :func:`probe_audio_stream` that collapses the
    tri-state result to a boolean: probe failures (None) and confirmed
    no-audio (False) both return False. Use this only at call sites
    where the false branch is non-terminal (i.e. retried on next run);
    otherwise call :func:`probe_audio_stream` directly.
    """
    return probe_audio_stream(path, timeout=timeout) is True
