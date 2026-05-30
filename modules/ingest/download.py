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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

import httpx

from core.concurrency import retry_with_backoff
from core.config import DownloadSettings, PathsSettings
from core.console import progress
from core.database import (
    Clip,
    User,
    get_profile_pic_url,
    get_session,
)
from core.log import StageResult, event, item, stage

_COMMIT_EVERY = 20


class FetchResult(NamedTuple):
    ok: bool
    status: int | None
    size: int | None
    duration: float
    err: str | None


def fetch_file(
    url: str,
    path: str,
    max_attempts: int,
    retry_delay: int,
    retry_jitter: int,
    *,
    min_bytes: int = 0,
) -> FetchResult:
    """Download `url` to `path` with retries + jittered backoff + atomic rename.

    If `min_bytes > 0`, any response whose body is smaller than that threshold
    is treated as a transient failure (the `.part` file is deleted and the
    attempt is retried).  After all attempts are exhausted the result has
    ``ok=False`` and ``err`` set to the ``repr`` of the last failure (for the
    too-small case, a ``ValueError`` whose message is
    ``response_too_small: <actual> < <min>``).
    """
    tmp = path + ".part"
    t0 = time.perf_counter()
    result: dict[str, int | None] = {"status": None, "size": None}

    def _attempt() -> None:
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30)
            result["status"] = r.status_code
            r.raise_for_status()
            actual = len(r.content)
            if min_bytes > 0 and actual < min_bytes:
                raise ValueError(f"response_too_small: {actual} < {min_bytes}")
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, path)
            result["size"] = actual
        except Exception:
            if os.path.exists(tmp):
                with contextlib.suppress(OSError):
                    os.remove(tmp)
            raise

    try:
        retry_with_backoff(
            _attempt,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            retry_jitter=retry_jitter,
        )
        return FetchResult(
            ok=True,
            status=result["status"],
            size=result["size"],
            duration=time.perf_counter() - t0,
            err=None,
        )
    except Exception as e:
        return FetchResult(
            ok=False,
            status=result["status"],
            size=None,
            duration=time.perf_counter() - t0,
            err=repr(e),
        )


@stage("ingest:download")
def download_files(
    download: DownloadSettings,
    paths: PathsSettings,
    *,
    retry_failed: bool = False,
) -> StageResult:
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
        profile_jobs: list[tuple[str, str, int]] = []  # (url, path, user_id)
        thumbnail_jobs: list[tuple[str, str, int]] = []  # (url, path, clip_id)
        video_jobs: list[tuple[int, str, str]] = []  # (clip_id, url, path)
        clips_missing_url: list[int] = []

        for user in users:
            pic_path = os.path.join(profile_pic_dir, f"{user.id}.jpg")
            if not os.path.exists(pic_path):
                pic_url = get_profile_pic_url(user.id)
                if pic_url:
                    profile_jobs.append((pic_url, pic_path, user.id))

            for clip in user.clips:
                if not clip.is_selected:
                    continue
                thumb_path = str(paths.thumbnail_for(clip.id))
                if not os.path.exists(thumb_path) and clip.thumbnail_url:
                    thumbnail_jobs.append((clip.thumbnail_url, thumb_path, clip.id))
                if clip.is_downloaded is True:
                    continue
                if clip.is_downloaded is False and not retry_failed:
                    continue
                if clip.video_url is None:
                    clips_missing_url.append(clip.id)
                    continue
                video_path = str(paths.video_for(clip.id))
                if not os.path.exists(video_path):
                    video_jobs.append((clip.id, clip.video_url, video_path))
                else:
                    clip.is_downloaded = True

        # 2. Mark URL-less clips as terminal failures and commit immediately
        if clips_missing_url:
            for cid in clips_missing_url:
                clip = session.get(Clip, cid)
                clip.is_downloaded = False
            session.commit()
            for cid in clips_missing_url:
                event("GET", f"clip_{cid}", result="ERR", stats={"err": "no video_url"})

        total_jobs = len(profile_jobs) + len(thumbnail_jobs) + len(video_jobs)
        if total_jobs == 0:
            session.commit()
            return StageResult(downloaded=0, skipped=0, failed=len(clips_missing_url))

        event(
            "SCAN",
            "clips",
            stats={
                "todo": total_jobs,
                "videos": len(video_jobs),
                "thumbs": len(thumbnail_jobs),
                "profiles": len(profile_jobs),
            },
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

            for url, path, user_id in profile_jobs:
                fut = pool.submit(
                    fetch_file,
                    url,
                    path,
                    max_attempts,
                    retry_delay,
                    retry_jitter,
                    min_bytes=256,
                )
                future_meta[fut] = ("pic", user_id)

            for url, path, clip_id in thumbnail_jobs:
                fut = pool.submit(
                    fetch_file,
                    url,
                    path,
                    max_attempts,
                    retry_delay,
                    retry_jitter,
                    min_bytes=256,
                )
                future_meta[fut] = ("thumb", clip_id)

            for clip_id, url, path in video_jobs:
                fut = pool.submit(
                    fetch_file,
                    url,
                    path,
                    max_attempts,
                    retry_delay,
                    retry_jitter,
                    min_bytes=1024,
                )
                future_meta[fut] = ("video", clip_id)

            for fut in as_completed(future_meta):
                kind, item_id = future_meta[fut]
                fetch = fut.result()

                if kind == "video":
                    target = f"clip_{item_id}"
                elif kind == "thumb":
                    target = f"thumb_{item_id}"
                else:
                    target = f"profile_{item_id}"

                with item("GET", target) as t:
                    if not fetch.ok:
                        raise RuntimeError(fetch.err or "unknown")
                    t.stats(size=fetch.size or 0)

                if kind == "video":
                    clip = session.get(Clip, item_id)
                    clip.is_downloaded = fetch.ok
                    committed_since += 1
                    if t.failed:
                        video_failed += 1
                    if committed_since >= _COMMIT_EVERY:
                        session.commit()
                        committed_since = 0
                elif kind == "thumb" and t.failed:
                    thumb_failed += 1
                elif kind == "pic" and t.failed:
                    pic_failed += 1

                advance()

        session.commit()
    finally:
        session.close()

    downloaded = len(video_jobs) - video_failed
    return StageResult(
        downloaded=downloaded,
        skipped=0,
        failed=video_failed + len(clips_missing_url),
    )
