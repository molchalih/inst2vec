import os
import time

from hikerapi import Client

from modules.console import log, progress
from modules.database import Clip, User, get_session

SCOPE = "fetch_profiles"

HIKER_TOKEN = os.environ.get("HIKER_API_KEY", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5))
MAX_CLIPS = int(os.environ.get("MAX_CLIPS", 5))

_FETCH_RETRY_DELAYS_SEC = [0, 30, 60, 90]


def _fetch_clips(cl: Client, user: User, session) -> int:
    if user.clips:
        return 0

    data = cl.user_clips_v2(str(user.pk))
    items = data["response"]["items"]
    items.sort(key=lambda x: x["media"].get("play_count") or 0, reverse=True)

    count = 0
    for item in items[: MAX_CLIPS or None]:
        m = item["media"]
        clip_pk = int(m["pk"])
        if session.query(Clip).filter_by(pk=clip_pk).first():
            continue

        cap = m.get("caption") or {}

        session.add(
            Clip(
                pk=clip_pk,
                user_pk=user.pk,
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


def fetch_profiles():
    cl = Client(token=HIKER_TOKEN)
    session = get_session()

    users = (
        session.query(User)
        .filter(
            (User.parse_status.is_(None)) | (User.parse_status == "pending"),
        )
        .limit(BATCH_SIZE)
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
            username = user.username
            for attempt in range(4):
                if _FETCH_RETRY_DELAYS_SEC[attempt]:
                    time.sleep(_FETCH_RETRY_DELAYS_SEC[attempt])
                try:
                    data = cl.user_by_username_v1(username)
                    info = data.get("user", data)

                    user.pk = info["pk"]
                    user.full_name = info.get("full_name")
                    user.profile_pic_url = info.get("profile_pic_url")
                    user.profile_pic_url_hd = info.get("profile_pic_url_hd")
                    user.following_count = info.get("following_count")
                    user.city_name = info.get("city_name")

                    clips_count = _fetch_clips(cl, user, session)
                    user.parse_status = "success"
                    session.commit()
                    parsed += 1
                    advance(
                        detail=f"{username} ({user.full_name}, {clips_count} clips)"
                    )
                    break
                except Exception:
                    session.rollback()
                    user = session.query(User).filter_by(username=username).one()
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
