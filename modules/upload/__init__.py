"""Upload selected + downloaded clip videos to the object store.

Pipeline stage. Idempotent: each clip is uploaded at most once; the
``Clip.is_uploaded`` flag is the source of truth for "already in
object storage." When the storage bucket is unconfigured (empty
string in settings), the stage is a no-op — used for fully-local
runs that never talk to the GPU pod.
"""

from __future__ import annotations

import os
import time

from core.config import Secrets, Settings
from core.console import log, progress
from core.database import Clip, get_session
from core.storage import get_object_store

__all__ = ["run"]

STAGE = "upload"


def upload_videos(settings, secrets) -> None:
    """Upload every selected + downloaded + not-yet-uploaded clip's video."""
    if not settings.storage.bucket:
        log(STAGE, "SKIP", "bucket", "none")
        return

    store = get_object_store(settings, secrets)
    bucket = settings.storage.bucket
    video_dir = settings.paths.video_dir

    session = get_session()
    uploaded = 0
    missing = 0
    failed = 0
    t_stage = time.perf_counter()
    try:
        candidates = (
            session.query(Clip)
            .filter(
                Clip.is_selected.is_(True),
                Clip.is_downloaded.is_(True),
                (Clip.is_uploaded.is_(None)) | (Clip.is_uploaded.is_(False)),
            )
            .all()
        )

        if not candidates:
            return

        log(STAGE, "SCAN", "clips", "ok", stats={"todo": len(candidates)})

        with progress(len(candidates), "Uploading clips") as advance:
            for clip in candidates:
                path = os.path.join(video_dir, f"{clip.id}.mp4")
                key = store.key_for_clip(clip.id)
                target = f"s3://{bucket}/{key}"
                if not os.path.exists(path):
                    missing += 1
                    log(
                        STAGE,
                        "PUT",
                        target,
                        "ERR",
                        stats={"err": "missing on disk"},
                    )
                    advance(detail=f"✗ {clip.id} (missing on disk)")
                    continue
                size = os.path.getsize(path)
                t0 = time.perf_counter()
                try:
                    store.put(path, key)
                except Exception as e:
                    failed += 1
                    log(
                        STAGE,
                        "PUT",
                        target,
                        "ERR",
                        stats={"err": f"{type(e).__name__}: {e}"},
                    )
                    advance(detail=f"✗ {clip.id} ({type(e).__name__})")
                    continue
                clip.is_uploaded = True
                session.commit()
                uploaded += 1
                log(
                    STAGE,
                    "PUT",
                    target,
                    "ok",
                    stats={"time": time.perf_counter() - t0, "size": size},
                )
                advance(detail=f"✓ {clip.id}")

    finally:
        session.close()

    log(
        STAGE,
        "SEAL",
        "upload",
        "ok",
        stats={
            "done": uploaded,
            "err": failed,
            "missing": missing,
            "time": time.perf_counter() - t_stage,
        },
    )


def run(settings: Settings, secrets: Secrets) -> None:
    """Upload selected clip videos to the object store."""
    upload_videos(settings, secrets)
