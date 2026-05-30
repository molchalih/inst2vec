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


class _DownloadJobs(NamedTuple):
    profile_jobs: list[tuple[str, str, int]]  # (url, path, user_id)
    thumbnail_jobs: list[tuple[str, str, int]]  # (url, path, clip_id)
    video_jobs: list[tuple[int, str, str]]  # (clip_id, url, path)
    clips_missing_url: list[int]


def _resolve_clip_jobs(
    clip,
    paths: PathsSettings,
    *,
    retry_failed: bool,
    thumbnail_jobs: list[tuple[str, str, int]],
    video_jobs: list[tuple[int, str, str]],
    clips_missing_url: list[int],
) -> None:
    """Append this clip's thumbnail/video work (or terminal/skip state).

    Mutates the three job lists in place and may set ``clip.is_downloaded``
    when the video file already exists on disk.
    """
    # DB is the source of truth: a clip whose video is recorded as
    # downloaded is trusted as complete, so it incurs zero
    # filesystem stats on rerun (its thumbnail was fetched in the
    # same prior run). This is the steady-state fast path.
    if clip.is_downloaded is True:
        return
    if clip.is_downloaded is False and not retry_failed:
        return
    thumb_path = str(paths.thumbnail_for(clip.id))
    if not os.path.exists(thumb_path) and clip.thumbnail_url:
        thumbnail_jobs.append((clip.thumbnail_url, thumb_path, clip.id))
    if clip.video_url is None:
        clips_missing_url.append(clip.id)
        return
    video_path = str(paths.video_for(clip.id))
    if not os.path.exists(video_path):
        video_jobs.append((clip.id, clip.video_url, video_path))
    else:
        clip.is_downloaded = True


def _resolve_jobs(users, paths: PathsSettings, *, retry_failed: bool) -> _DownloadJobs:
    """Resolve all profile/thumbnail/video work items for the selected users."""
    profile_jobs: list[tuple[str, str, int]] = []
    thumbnail_jobs: list[tuple[str, str, int]] = []
    video_jobs: list[tuple[int, str, str]] = []
    clips_missing_url: list[int] = []

    for user in users:
        pic_path = os.path.join(paths.profile_pic_dir, f"{user.id}.jpg")
        if not os.path.exists(pic_path):
            pic_url = get_profile_pic_url(user.id)
            if pic_url:
                profile_jobs.append((pic_url, pic_path, user.id))

        for clip in user.clips:
            if not clip.is_selected:
                continue
            _resolve_clip_jobs(
                clip,
                paths,
                retry_failed=retry_failed,
                thumbnail_jobs=thumbnail_jobs,
                video_jobs=video_jobs,
                clips_missing_url=clips_missing_url,
            )

    return _DownloadJobs(profile_jobs, thumbnail_jobs, video_jobs, clips_missing_url)


def _mark_missing_url_terminal(session, clips_missing_url: list[int]) -> None:
    """Mark URL-less clips as terminal failures and commit immediately."""
    if not clips_missing_url:
        return
    for cid in clips_missing_url:
        clip = session.get(Clip, cid)
        clip.is_downloaded = False
    session.commit()
    for cid in clips_missing_url:
        event("GET", f"clip_{cid}", result="ERR", stats={"err": "no video_url"})


def _dispatch_jobs(
    pool: ThreadPoolExecutor,
    jobs: _DownloadJobs,
    *,
    max_attempts: int,
    retry_delay: int,
    retry_jitter: int,
) -> dict:
    """Submit every job to the pool. Returns {future: (kind, item_id)}."""
    future_meta: dict = {}
    # Normalise to a uniform (url, path, kind, item_id, min_bytes) shape first.
    # video_jobs is (clip_id, url, path); the other two are (url, path, id).
    normalized: list[tuple[str, str, str, int, int]] = []
    for url, path, item_id in jobs.profile_jobs:
        normalized.append((url, path, "pic", item_id, 256))
    for url, path, item_id in jobs.thumbnail_jobs:
        normalized.append((url, path, "thumb", item_id, 256))
    for clip_id, url, path in jobs.video_jobs:
        normalized.append((url, path, "video", clip_id, 1024))
    for url, path, kind, item_id, min_bytes in normalized:
        fut = pool.submit(
            fetch_file,
            url,
            path,
            max_attempts,
            retry_delay,
            retry_jitter,
            min_bytes=min_bytes,
        )
        future_meta[fut] = (kind, item_id)
    return future_meta


def _target_for(kind: str, item_id: int) -> str:
    if kind == "video":
        return f"clip_{item_id}"
    if kind == "thumb":
        return f"thumb_{item_id}"
    return f"profile_{item_id}"


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

    for d in (paths.profile_pic_dir, paths.thumbnail_dir, paths.video_dir):
        os.makedirs(d, exist_ok=True)

    session = get_session()
    try:
        users = session.query(User).filter(User.is_selected.is_(True)).all()

        # 1. Resolve work items
        jobs = _resolve_jobs(users, paths, retry_failed=retry_failed)

        # 2. Mark URL-less clips as terminal failures and commit immediately
        _mark_missing_url_terminal(session, jobs.clips_missing_url)

        total_jobs = (
            len(jobs.profile_jobs) + len(jobs.thumbnail_jobs) + len(jobs.video_jobs)
        )
        if total_jobs == 0:
            session.commit()
            return StageResult(
                downloaded=0, skipped=0, failed=len(jobs.clips_missing_url)
            )

        event(
            "SCAN",
            "clips",
            stats={
                "todo": total_jobs,
                "videos": len(jobs.video_jobs),
                "thumbs": len(jobs.thumbnail_jobs),
                "profiles": len(jobs.profile_jobs),
            },
        )

        # 3. Dispatch all work to the pool
        video_failed = 0
        committed_since = 0

        with (
            progress(total_jobs, "Downloading") as advance,
            ThreadPoolExecutor(max_workers=concurrency) as pool,
        ):
            future_meta = _dispatch_jobs(
                pool,
                jobs,
                max_attempts=max_attempts,
                retry_delay=retry_delay,
                retry_jitter=retry_jitter,
            )

            for fut in as_completed(future_meta):
                kind, item_id = future_meta[fut]
                fetch = fut.result()

                with item("GET", _target_for(kind, item_id)) as t:
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

                advance()

        session.commit()
    finally:
        session.close()

    downloaded = len(jobs.video_jobs) - video_failed
    return StageResult(
        downloaded=downloaded,
        skipped=0,
        failed=video_failed + len(jobs.clips_missing_url),
    )
