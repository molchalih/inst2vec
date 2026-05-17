"""Upload selected + downloaded clip videos to the object store.

Pipeline stage. Idempotent: each clip is uploaded at most once; the
``Clip.is_uploaded`` flag is the source of truth for "already in
object storage." When the storage bucket is unconfigured (empty
string in settings), the stage is a no-op — used for fully-local
runs that never talk to the GPU pod.
"""

from __future__ import annotations

import os

from modules.console import log, progress
from modules.database import Clip, get_session
from modules.storage import get_object_store

STAGE = "upload"


def upload_videos(settings, secrets) -> None:
    """Upload every selected + downloaded + not-yet-uploaded clip's video."""
    if not settings.storage.bucket:
        log(STAGE, "storage.bucket not set — skipping upload stage")
        return

    store = get_object_store(settings, secrets)
    video_dir = settings.paths.video_dir

    session = get_session()
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
            log(STAGE, "nothing to upload")
            return

        log(STAGE, f"{len(candidates)} clip(s) to upload")

        with progress(len(candidates), "Uploading clips") as advance:
            for clip in candidates:
                path = os.path.join(video_dir, f"{clip.id}.mp4")
                if not os.path.exists(path):
                    advance(detail=f"✗ {clip.id} (missing on disk)")
                    continue
                key = store.key_for_clip(clip.id)
                try:
                    store.put(path, key)
                except Exception as e:
                    advance(detail=f"✗ {clip.id} ({type(e).__name__})")
                    continue
                clip.is_uploaded = True
                session.commit()
                advance(detail=f"✓ {clip.id}")

    finally:
        session.close()

    log(STAGE, "done", level="ok")
