import os
import time

import httpx

from modules.console import log, progress
from modules.database import Download, User, get_session
from modules.identity import get_profile_pic_url

SCOPE = "download"


def _download(url, path, max_attempts: int, retry_delay: int):
    for attempt in range(max_attempts):
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            return True
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(retry_delay)
    return False


def _try_download(
    session, entity_id, file_type, url, dirs: dict, max_attempts: int, retry_delay: int
):
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
    path = os.path.join(dirs[file_type], f"{entity_id}.{ext}")

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

    ok = _download(url, path, max_attempts, retry_delay)
    session.add(
        Download(
            entity_id=entity_id,
            file_type=file_type,
            success=ok,
            parse_available=True,
        )
    )


def download_files(
    batch_size: int,
    max_clips: int,
    max_attempts: int,
    retry_delay: int,
    profile_pic_dir: str,
    thumbnail_dir: str,
    video_dir: str,
) -> None:
    dirs = {
        "profile_pic": profile_pic_dir,
        "thumbnail": thumbnail_dir,
        "video": video_dir,
    }
    for d in dirs.values():
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
        .limit(batch_size)
        .all()
    )

    if not users:
        session.close()
        return

    log(SCOPE, f"{len(users)} users to download")
    with progress(len(users), "Downloading") as advance:
        for user in users:
            pic_url = get_profile_pic_url(user.id)
            _try_download(
                session,
                user.id,
                "profile_pic",
                pic_url,
                dirs,
                max_attempts,
                retry_delay,
            )
            for clip in user.clips[: max_clips or None]:
                if clip.disqualified == 1:
                    continue
                _try_download(
                    session,
                    clip.id,
                    "thumbnail",
                    clip.thumbnail_url,
                    dirs,
                    max_attempts,
                    retry_delay,
                )
                _try_download(
                    session,
                    clip.id,
                    "video",
                    clip.video_url,
                    dirs,
                    max_attempts,
                    retry_delay,
                )
            session.commit()
            advance(detail=str(user.id))

    session.close()
