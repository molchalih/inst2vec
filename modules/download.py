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

from modules import fingerprint as fp
from modules.config import DownloadSettings, PathsSettings
from modules.console import log, progress
from modules.database import (
    Base,
    Clip,
    User,
    get_engine,
    get_profile_pic_url,
    get_session,
)
from modules.ffmpeg import run_ffmpeg

SCOPE = "download"

_COMMIT_EVERY = 20

AUDIO_EXTRACT_STAGE = "audio_extract"
AUDIO_EXTRACT_SCOPE = "default"


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


def download_files(download: DownloadSettings, paths: PathsSettings) -> None:
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
                # Video gated on is_downloaded=NULL only.
                if clip.is_downloaded is not None:
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


def extract_audio(
    video_path: str,
    audio_path: str,
    *,
    bitrate_kbps: int,
    sample_rate_hz: int,
    timeout_s: int,
) -> bool:
    """Extract mp3 audio from ``video_path`` to ``audio_path``.

    Idempotent: returns True without invoking ffmpeg when ``audio_path``
    exists and is at least as new as ``video_path``.
    """
    if (
        os.path.exists(audio_path)
        and os.path.exists(video_path)
        and os.path.getmtime(audio_path) >= os.path.getmtime(video_path)
    ):
        return True
    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{bitrate_kbps}k",
        "-ar",
        str(sample_rate_hz),
        audio_path,
    ]
    return run_ffmpeg(cmd, timeout=timeout_s)


def _video_stat(video_dir: str, clip_id: int) -> tuple[int, int]:
    p = os.path.join(video_dir, f"{clip_id}.mp4")
    if not os.path.exists(p):
        return (-1, -1)
    st = os.stat(p)
    return (st.st_size, st.st_mtime_ns)


def extract_audio_stage(settings) -> None:
    """Extract mp3 audio for every downloaded clip into ``paths.audio_dir``.

    No-op when ``embeddings.gemini_enabled`` is False — gemini_mm is the
    only consumer today.
    """
    if not settings.embeddings.gemini_enabled:
        log(AUDIO_EXTRACT_STAGE, "disabled — skipping")
        return

    Base.metadata.create_all(get_engine())
    session = get_session()
    try:
        clips = (
            session.query(Clip)
            .filter(Clip.is_downloaded.is_(True))
            .order_by(Clip.id)
            .all()
        )
        if not clips:
            log(AUDIO_EXTRACT_STAGE, "no downloaded clips — nothing to do")
            return

        ids = [c.id for c in clips]
        video_dir = settings.paths.video_dir
        audio_dir = settings.paths.audio_dir
        os.makedirs(audio_dir, exist_ok=True)

        current = fp.Fingerprint(
            data=fp.hash_rows((cid,) for cid in ids),
            config=fp.hash_text(
                f"bitrate={settings.embeddings.audio_bitrate_kbps}"
                f"|sr={settings.embeddings.audio_sample_rate_hz}"
                f"|codec=libmp3lame"
            ),
            dependency=fp.hash_rows(_video_stat(video_dir, cid) for cid in ids),
        )
        if not fp.is_stale(session, AUDIO_EXTRACT_STAGE, AUDIO_EXTRACT_SCOPE, current):
            log(AUDIO_EXTRACT_STAGE, "fingerprint match — skipping")
            return

        failures = 0
        with progress(len(clips), "Extracting audio") as advance:
            for clip in clips:
                video_path = os.path.join(video_dir, f"{clip.id}.mp4")
                audio_path = os.path.join(audio_dir, f"{clip.id}.mp3")
                if not os.path.exists(video_path):
                    failures += 1
                    advance(detail=f"✗ {clip.id} (no video)")
                    continue
                ok = extract_audio(
                    video_path,
                    audio_path,
                    bitrate_kbps=settings.embeddings.audio_bitrate_kbps,
                    sample_rate_hz=settings.embeddings.audio_sample_rate_hz,
                    timeout_s=settings.embeddings.audio_extract_timeout_s,
                )
                if ok:
                    advance(detail=f"✓ {clip.id}")
                else:
                    failures += 1
                    advance(detail=f"✗ {clip.id}")

        if failures == 0:
            fp.mark_complete(session, AUDIO_EXTRACT_STAGE, AUDIO_EXTRACT_SCOPE, current)
            session.commit()
            log(AUDIO_EXTRACT_STAGE, "done", level="ok")
        else:
            log(
                AUDIO_EXTRACT_STAGE,
                f"{failures}/{len(clips)} failed — leaving stage stale for retry",
                level="warn",
            )
    finally:
        session.close()
