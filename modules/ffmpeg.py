"""Shared ffmpeg subprocess helper.

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
