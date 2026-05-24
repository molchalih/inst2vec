"""Upload selected + downloaded clip videos to the object store.

Pipeline stage. Idempotent: the bucket is authoritative — ``Clip.is_uploaded``
is rewritten from the real post-verify state on every run, so a deleted or
half-uploaded object self-heals on the next run.  When the storage bucket is
unconfigured (empty string in settings), the stage is a no-op — used for
fully-local runs that never talk to the GPU pod.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import Secrets, Settings
from core.console import log, progress
from core.database import Clip, get_session
from core.storage import get_object_store

__all__ = ["run"]

STAGE = "upload"

# Heartbeat cadence: emit a running PUT summary line every N completed clips so
# a long batch shows progress between SCAN and SEAL (mirrors other stages).
_LOG_EVERY = 200


def verify_outcome(store, key: str, local_size: int) -> str:
    """Compare the bucket object against the local file by size.

    Returns one of: ``present`` (object exists, size matches — nothing to do),
    ``absent`` (no object — upload), ``mismatch`` (size differs — re-upload).
    The bucket is authoritative: this is the source of truth for "available to
    pods", not the ``is_uploaded`` flag.
    """
    meta = store.head(key)
    if meta is None:
        return "absent"
    return "present" if meta["size"] == local_size else "mismatch"


def process_clip_upload(store, key: str, local_path: str) -> str:
    """Verify one clip against the bucket and upload if needed. Pure of DB
    state so it is safe to run in a worker thread; the caller updates the DB.

    Returns: ``ok`` (already present), ``uploaded`` (put done), ``missing``
    (no local file), ``failed`` (HEAD or PUT raised). A transient store error on
    a single clip degrades to ``failed`` (re-tried next run) rather than aborting
    the whole concurrent batch."""
    if not os.path.exists(local_path):
        return "missing"
    try:
        outcome = verify_outcome(store, key, os.path.getsize(local_path))
        if outcome == "present":
            return "ok"
        store.put(local_path, key)
    except Exception:
        return "failed"
    return "uploaded"


def run_uploads(
    store,
    clips,
    key_for,
    *,
    workers: int,
    on_result: Callable[[int, str], None] | None = None,
) -> dict[int, bool]:
    """Verify+upload ``clips`` (list of (clip_id, local_path)) concurrently.

    Network I/O runs in a thread pool; no DB access here. Returns
    {clip_id: available}, where ``available`` means the object is confirmed
    present in the bucket (ok or uploaded) — the value to store in
    ``Clip.is_uploaded``.

    ``on_result(clip_id, outcome)`` is called on the calling thread as each
    clip completes (``outcome`` is one of ok/uploaded/missing/failed), for
    progress + logging; it never affects the returned availability map."""
    available: dict[int, bool] = {}
    if not clips:
        return available
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = {
            pool.submit(process_clip_upload, store, key_for(cid), path): cid
            for cid, path in clips
        }
        for fut in as_completed(futures):
            cid = futures[fut]
            outcome = fut.result()
            available[cid] = outcome in ("ok", "uploaded")
            if on_result is not None:
                on_result(cid, outcome)
    return available


def upload_videos(settings, secrets) -> None:
    """Verify selected+downloaded clips against the bucket and upload the
    missing/changed ones concurrently. The bucket is authoritative:
    ``Clip.is_uploaded`` is rewritten from the real post-verify state, so a
    deleted/half-uploaded object self-heals on the next run."""
    if not settings.storage.bucket:
        log(STAGE, "SKIP", "bucket", "none")
        return

    store = get_object_store(settings, secrets)
    video_dir = settings.paths.video_dir
    workers = settings.download.concurrency

    session = get_session()
    t_stage = time.perf_counter()
    try:
        candidates = (
            session.query(Clip)
            .filter(Clip.is_selected.is_(True), Clip.is_downloaded.is_(True))
            .all()
        )
        if not candidates:
            return
        log(STAGE, "SCAN", "clips", "ok", stats={"todo": len(candidates)})

        clips = [(c.id, os.path.join(video_dir, f"{c.id}.mp4")) for c in candidates]
        key_for = store.key_for_clip
        total = len(clips)
        counts = {"up": 0, "skip": 0, "fail": 0}
        done = 0

        with progress(total, "Uploading") as advance:

            def on_result(cid: int, outcome: str) -> None:
                nonlocal done
                done += 1
                if outcome == "uploaded":
                    counts["up"] += 1
                elif outcome == "ok":
                    counts["skip"] += 1
                else:  # missing | failed
                    counts["fail"] += 1
                    err = "no local file" if outcome == "missing" else "head/put failed"
                    log(STAGE, "PUT", f"clips/{cid}.mp4", "ERR", stats={"err": err})
                advance(1)
                if done % _LOG_EVERY == 0:
                    log(STAGE, "PUT", "clips", "ok", stats={"done": done, **counts})

            available = run_uploads(
                store, clips, key_for, workers=workers, on_result=on_result
            )

        uploaded = absent = 0
        for clip in candidates:
            ok = available.get(clip.id, False)
            if bool(clip.is_uploaded) != ok:
                clip.is_uploaded = ok
            if ok:
                uploaded += 1
            else:
                absent += 1
        session.commit()
    finally:
        session.close()

    log(
        STAGE,
        "SEAL",
        "upload",
        "ok",
        stats={
            "available": uploaded,
            "absent": absent,
            "time": time.perf_counter() - t_stage,
        },
    )


def run(settings: Settings, secrets: Secrets) -> None:
    """Upload selected clip videos to the object store."""
    upload_videos(settings, secrets)
