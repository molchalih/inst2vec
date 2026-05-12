import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from hikerapi import Client

from modules.console import log, progress
from modules.database import Clip, User, get_session
from modules.identity import (
    get_api_pk,
    get_or_create_clip_identity,
    get_username,
    update_user_identity,
)

SCOPE = "fetch_profiles"

_FETCH_RETRY_DELAYS_SEC = [0, 30, 60, 90]


def _fetch_clips(cl: Any, user: User, session: Any, max_clips: int) -> int:
    if user.clips:
        return 0

    api_pk = get_api_pk(user.id)
    if api_pk is None:
        return 0

    data = cl.user_clips_v2(str(api_pk))
    items = data["response"]["items"]
    items.sort(key=lambda x: x["media"].get("play_count") or 0, reverse=True)

    count = 0
    for item in items[: max_clips or None]:
        m = item["media"]
        clip_api_pk = int(m["pk"])
        clip_id = get_or_create_clip_identity(clip_api_pk)

        if session.query(Clip).filter_by(id=clip_id).first():
            continue

        cap = m.get("caption") or {}

        session.add(
            Clip(
                id=clip_id,
                user_id=user.id,
                thumbnail_url=m.get("thumbnail_url"),
                video_url=m.get("video_url"),
                caption_text=cap.get("text"),
                caption_translation=cap.get("text_translation"),
                comment_count=m.get("comment_count"),
                reshare_count=m.get("reshare_count"),
                like_count=m.get("like_count"),
                play_count=m.get("play_count"),
            )
        )
        count += 1
    return count


def fetch_profiles(
    batch_size: int,
    max_clips: int,
    hiker_api_key: str,
) -> None:
    cl = Client(token=hiker_api_key)
    session = get_session()

    users = (
        session.query(User)
        .filter(
            (User.parse_status.is_(None)) | (User.parse_status == "pending"),
        )
        .limit(batch_size)
        .all()
    )

    parsed = skipped = failed = 0

    if not users:
        log(SCOPE, "no new users provided", level="warn")
        session.close()
        return

    log(SCOPE, f"{len(users)} users to process")

    with progress(len(users), "Fetching profiles") as advance:
        for user in users:
            user_id = user.id
            username = get_username(user_id)
            for attempt in range(4):
                if _FETCH_RETRY_DELAYS_SEC[attempt]:
                    time.sleep(_FETCH_RETRY_DELAYS_SEC[attempt])
                try:
                    data = cl.user_by_username_v1(username)
                    info = data.get("user", data)

                    update_user_identity(
                        user_id,
                        api_pk=info["pk"],
                        full_name=info.get("full_name"),
                        city_name=info.get("city_name"),
                        profile_pic_url=info.get("profile_pic_url"),
                        profile_pic_url_hd=info.get("profile_pic_url_hd"),
                    )

                    user.following_count = info.get("following_count")

                    clips_count = _fetch_clips(cl, user, session, max_clips)
                    user.parse_status = "success"
                    session.commit()
                    parsed += 1
                    advance(
                        detail=f"{username} ({info.get('full_name')}, {clips_count} clips)"
                    )
                    break
                except Exception:
                    session.rollback()
                    user = session.query(User).filter_by(id=user_id).one()
                    if attempt == 3:
                        user.parse_status = "failed"
                        session.commit()
                        failed += 1
                        advance(detail=f"{username} — error")

            time.sleep(0.3)

    session.close()

    total = parsed + skipped + failed
    log(
        SCOPE,
        f"done — total: {total}, parsed: {parsed}, skipped: {skipped}, failed: {failed}",
        level="ok",
    )
