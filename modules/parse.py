import time
from typing import Any

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


def _fetch_clips(cl: Any, user: User, session: Any) -> int:
    if user.clips:
        return 0

    api_pk = get_api_pk(user.id)
    if api_pk is None:
        return 0

    data = cl.user_clips_v2(str(api_pk))
    items = data["response"]["items"]
    items.sort(key=lambda x: x["media"].get("play_count") or 0, reverse=True)

    count = 0
    for item in items:
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


def _process_user(cl: Any, user: User, session: Any) -> dict[str, Any]:
    username = get_username(user.id)
    data = cl.user_by_username_v1(username)
    info = data.get("user", data)

    update_user_identity(
        user.id,
        api_pk=info["pk"],
        full_name=info.get("full_name"),
        city_name=info.get("city_name"),
        profile_pic_url=info.get("profile_pic_url"),
        profile_pic_url_hd=info.get("profile_pic_url_hd"),
    )

    user.following_count = info.get("following_count")
    clips_count = _fetch_clips(cl, user, session)
    user.parse_status = "success"

    return {
        "username": username,
        "full_name": info.get("full_name"),
        "clips_count": clips_count,
    }


def fetch_profiles(
    hiker_api_key: str,
) -> None:
    cl = Client(token=hiker_api_key)
    session = get_session()

    users = (
        session.query(User)
        .filter(
            (User.parse_status.is_(None)) | (User.parse_status == "pending"),
        )
        .all()
    )

    if not users:
        log(SCOPE, "no new users provided", level="warn")
        session.close()
        return

    log(SCOPE, f"{len(users)} users to process")

    parsed = skipped = failed = 0

    with progress(len(users), "Fetching profiles") as advance:
        for user in users:
            for attempt in range(3):
                if attempt:
                    time.sleep(30)
                try:
                    result = _process_user(cl, user, session)
                    session.commit()
                    parsed += 1
                    advance(
                        detail=f"{result['username']} ({result['full_name']}, {result['clips_count']} clips)"
                    )
                    break
                except Exception:
                    session.rollback()
                    user = session.query(User).filter_by(id=user.id).one()
                    if attempt == 2:
                        user.parse_status = "failed"
                        session.commit()
                        failed += 1
                        advance(detail=f"{get_username(user.id)} — error")

    session.close()

    total = parsed + skipped + failed
    log(
        SCOPE,
        f"done — total: {total}, parsed: {parsed}, skipped: {skipped}, failed: {failed}",
        level="ok",
    )
