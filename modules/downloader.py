import os
import time

import httpx

from modules.database import get_session, User, Download

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


def _try_download(session, entity_pk, file_type, url):
    if session.query(Download).filter_by(entity_pk=entity_pk, file_type=file_type).first():
        return

    if not url:
        session.add(Download(entity_pk=entity_pk, file_type=file_type, success=False, parse_available=False))
        return

    ext = "mp4" if file_type == "video" else "jpg"
    path = os.path.join(DIRS[file_type], f"{entity_pk}.{ext}")

    if os.path.exists(path):
        session.add(Download(entity_pk=entity_pk, file_type=file_type, success=True, parse_available=True))
        return

    ok = _download(url, path)
    session.add(Download(
        entity_pk=entity_pk, file_type=file_type,
        success=ok, parse_available=ok,
    ))
    print(f"  {'ok' if ok else 'FAILED'} {file_type}/{entity_pk}")


def download_files():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    session = get_session()
    done_pks = session.query(Download.entity_pk).filter(Download.file_type == "profile_pic")
    users = session.query(User).filter(~User.pk.in_(done_pks)).limit(BATCH_SIZE).all()

    for user in users:
        print(f"[download] {user.username}")
        _try_download(session, user.pk, "profile_pic", user.profile_pic_url)
        for clip in user.clips[:MAX_CLIPS or None]:
            _try_download(session, clip.pk, "thumbnail", clip.thumbnail_url)
            _try_download(session, clip.pk, "video", clip.video_url)
        session.commit()

    session.close()
