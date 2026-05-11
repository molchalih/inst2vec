import os
import time

import httpx

from modules.console import log, progress
from modules.database import Download, User, get_session
from modules.identity import get_profile_pic_url

SCOPE = "download"

DIRS = {
    "profile_pic": "data/source/profile_pics",
    "thumbnail": "data/source/thumbnails",
    "video": "data/source/videos",
}
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5))
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", 5))
MAX_ATTEMPTS = int(os.environ.get("MAX_DOWNLOAD_ATTEMPTS", 3))
RETRY_DELAY = int(os.environ.get("DOWNLOAD_RETRY_DELAY", 2))


def _download(url, path):
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            return True
        except Exception:
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
    return False


def _try_download(session, entity_id, file_type, url):
    if (
        session.query(Download)
        .filter_by(entity_id=entity_id, file_type=file_type)
        .first()
    ):
        return

    if not url:
        session.add(
            Download(
                entity_id=entity_id,
                file_type=file_type,
                success=False,
                parse_available=True,
            )
        )
        return

    ext = "mp4" if file_type == "video" else "jpg"
    path = os.path.join(DIRS[file_type], f"{entity_id}.{ext}")

    if os.path.exists(path):
        session.add(
            Download(
                entity_id=entity_id,
                file_type=file_type,
                success=True,
                parse_available=True,
            )
        )
        return

    ok = _download(url, path)
    session.add(
        Download(
            entity_id=entity_id,
            file_type=file_type,
            success=ok,
            parse_available=True,
        )
    )


def download_files():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    session = get_session()
    done_ids = session.query(Download.entity_id).filter(
        Download.file_type == "profile_pic"
    )
    users = (
        session.query(User)
        .filter(
            ~User.id.in_(done_ids),
            (User.user_disqualified.is_(None)) | (User.user_disqualified == 0),
        )
        .limit(BATCH_SIZE)
        .all()
    )

    if not users:
        session.close()
        return

    log(SCOPE, f"{len(users)} users to download")
    with progress(len(users), "Downloading") as advance:
        for user in users:
            pic_url = get_profile_pic_url(user.id)
            _try_download(session, user.id, "profile_pic", pic_url)
            for clip in user.clips[: MAX_CLIPS or None]:
                if clip.disqualified == 1:
                    continue
                _try_download(session, clip.id, "thumbnail", clip.thumbnail_url)
                _try_download(session, clip.id, "video", clip.video_url)
            session.commit()
            advance(detail=str(user.id))

    session.close()
