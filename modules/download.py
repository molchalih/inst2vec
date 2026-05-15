"""Download stage: fetch profile pics, thumbnails, and videos for selected users/clips.

NOTE: This module will be completely rewritten in the download refactor.
The current version is stubbed to avoid using the removed Download model.
"""

from __future__ import annotations

from modules.console import log

SCOPE = "download"


def download_files(
    batch_size: int,
    max_clips: int,
    max_attempts: int,
    retry_delay: int,
    profile_pic_dir: str,
    thumbnail_dir: str,
    video_dir: str,
) -> None:
    """Stub: real implementation coming in Task 10 of the download refactor."""
    log(SCOPE, "stubbed (removed Download model, awaiting Task 10 rewrite)")
