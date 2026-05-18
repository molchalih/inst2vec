"""Download stage: fetch profile pics, thumbnails, and videos for selected users/clips.

Concurrency model:
    - Work submitted to a ThreadPoolExecutor (pool size = concurrency).
    - Workers do pure I/O (HTTP + atomic file write); no DB access.
    - Main thread consumes `as_completed` results and updates Clip.is_downloaded.
    - SQLite stays single-writer; no contention.

State semantics on Clip.is_downloaded:
    NULL  → selected, not yet attempted
    True  → video downloaded successfully
    False → video failed all retries (or video_url was None)

Re-running the pipeline is the resume mechanism. NULL clips are retried; True/False are skipped.
"""

from __future__ import annotations

import contextlib
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from core.config import DownloadSettings, PathsSettings
from core.console import log, progress
from core.database import (
    Clip,
    User,
    get_profile_pic_url,
    get_session,
)

SCOPE = "download"

_COMMIT_EVERY = 20


def fetch_file(
    url: str,
    path: str,
    max_attempts: int,
    retry_delay: int,
    retry_jitter: int,
) -> bool:
    """Download `url` to `path` with retries + jittered backoff + atomic rename.

    Returns True on success, False if all attempts failed.
    """
    tmp = path + ".part"
    for attempt in range(max_attempts):
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30)
            r.raise_for_status()
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, path)
            return True
        except Exception:
            if os.path.exists(tmp):
                with contextlib.suppress(OSError):
                    os.remove(tmp)
            if attempt < max_attempts - 1:
                time.sleep(retry_delay + random.uniform(0, retry_jitter))
    return False


def download_files(
    download: DownloadSettings,
    paths: PathsSettings,
    *,
    retry_failed: bool = False,
) -> None:
    max_attempts = download.max_attempts
    retry_delay = download.retry_delay
    retry_jitter = download.retry_jitter
    concurrency = download.concurrency
    profile_pic_dir = paths.profile_pic_dir
    thumbnail_dir = paths.thumbnail_dir
    video_dir = paths.video_dir

    for d in (profile_pic_dir, thumbnail_dir, video_dir):
        os.makedirs(d, exist_ok=True)

    session = get_session()
    try:
        users = session.query(User).filter(User.is_selected.is_(True)).all()

        # 1. Resolve work items
        profile_jobs: list[tuple[str, str]] = []  # (url, path)
        thumbnail_jobs: list[tuple[str, str]] = []
        video_jobs: list[tuple[int, str, str]] = []  # (clip_id, url, path)
        clips_missing_url: list[int] = []

        for user in users:
            pic_path = os.path.join(profile_pic_dir, f"{user.id}.jpg")
            if not os.path.exists(pic_path):
                pic_url = get_profile_pic_url(user.id)
                if pic_url:
                    profile_jobs.append((pic_url, pic_path))

            for clip in user.clips:
                if not clip.is_selected:
                    continue
                # Thumbnails use disk existence as resume signal — check for all selected clips.
                thumb_path = os.path.join(thumbnail_dir, f"{clip.id}.jpg")
                if not os.path.exists(thumb_path) and clip.thumbnail_url:
                    thumbnail_jobs.append((clip.thumbnail_url, thumb_path))
                # Video gated on is_downloaded=NULL; when retry_failed=True also
                # pick up False rows so the next run re-attempts them.
                if clip.is_downloaded is True:
                    continue
                if clip.is_downloaded is False and not retry_failed:
                    continue
                if clip.video_url is None:
                    clips_missing_url.append(clip.id)
                    continue
                video_path = os.path.join(video_dir, f"{clip.id}.mp4")
                if not os.path.exists(video_path):
                    video_jobs.append((clip.id, clip.video_url, video_path))
                else:
                    # File already on disk from prior run: flag as success.
                    clip.is_downloaded = True

        # 2. Mark URL-less clips as terminal failures and commit immediately
        if clips_missing_url:
            for cid in clips_missing_url:
                clip = session.get(Clip, cid)
                clip.is_downloaded = False
            session.commit()
            log(
                SCOPE,
                f"{len(clips_missing_url)} clips have no video_url — marked failed",
            )

        total_jobs = len(profile_jobs) + len(thumbnail_jobs) + len(video_jobs)
        if total_jobs == 0:
            log(SCOPE, "nothing to download")
            session.commit()
            return

        log(
            SCOPE,
            f"jobs: {len(video_jobs)} video, {len(thumbnail_jobs)} thumb, "
            f"{len(profile_jobs)} pic",
        )

        # 3. Dispatch all work to the pool
        video_failed = 0
        thumb_failed = 0
        pic_failed = 0
        committed_since = 0

        with (
            progress(total_jobs, "Downloading") as advance,
            ThreadPoolExecutor(max_workers=concurrency) as pool,
        ):
            future_meta: dict = {}

            for url, path in profile_jobs:
                fut = pool.submit(
                    fetch_file, url, path, max_attempts, retry_delay, retry_jitter
                )
                future_meta[fut] = ("pic", None)

            for url, path in thumbnail_jobs:
                fut = pool.submit(
                    fetch_file, url, path, max_attempts, retry_delay, retry_jitter
                )
                future_meta[fut] = ("thumb", None)

            for clip_id, url, path in video_jobs:
                fut = pool.submit(
                    fetch_file, url, path, max_attempts, retry_delay, retry_jitter
                )
                future_meta[fut] = ("video", clip_id)

            for fut in as_completed(future_meta):
                kind, clip_id = future_meta[fut]
                ok = fut.result()

                if kind == "video":
                    clip = session.get(Clip, clip_id)
                    clip.is_downloaded = ok
                    committed_since += 1
                    if not ok:
                        video_failed += 1
                        log(SCOPE, f"video {clip_id} failed", level="warn")
                    if committed_since >= _COMMIT_EVERY:
                        session.commit()
                        committed_since = 0
                elif kind == "thumb" and not ok:
                    thumb_failed += 1
                elif kind == "pic" and not ok:
                    pic_failed += 1

                advance()

        session.commit()
        log(
            SCOPE,
            f"done: {len(video_jobs) - video_failed} videos OK, {video_failed} failed, "
            f"{thumb_failed} thumbs failed, {pic_failed} pics failed",
        )
    finally:
        session.close()
