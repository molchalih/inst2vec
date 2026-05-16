import time
from typing import Any

from hikerapi import Client

from modules.console import log, progress
from modules.database import (
    Clip,
    User,
    get_api_pk,
    get_or_create_clip_identity,
    get_session,
    get_username,
    update_user_identity,
)

SCOPE = "fetch_profiles"


def _fetch_clips(cl: Any, user: User, session: Any) -> None:
    api_pk = get_api_pk(user.id)
    if api_pk is None:
        return

    all_items: list[Any] = []
    next_page_id: str | None = None

    for _ in range(5):
        if next_page_id is not None:
            data = cl.user_clips_v2(str(api_pk), next_page_id)
        else:
            data = cl.user_clips_v2(str(api_pk))
        page = data["response"]
        all_items.extend(page.get("items", []))
        next_page_id = data.get("next_page_id") or None
        if not next_page_id:
            break

    all_items.sort(key=lambda x: x["media"].get("play_count") or 0, reverse=True)

    for item in all_items:
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
                video_duration=m.get("video_duration"),
                taken_at=m.get("taken_at"),
            )
        )


def _process_user(cl: Any, user: User, session: Any) -> None:
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
    user.follower_count = info.get("follower_count")
    _fetch_clips(cl, user, session)
    user.parse_status = "success"


def fetch_profiles(
    hiker_api_key: str,
) -> None:
    cl = Client(token=hiker_api_key)
    session = get_session()

    users = session.query(User).filter(User.parse_status.is_(None)).all()

    if not users:
        log(SCOPE, "no new users provided", level="warn")
        session.close()
        return

    log(SCOPE, f"{len(users)} users to process")

    parsed = skipped = failed = 0
    total_users = len(users)

    with progress(total_users, "Fetching profiles") as advance:
        for user in users:
            for attempt in range(3):
                if attempt:
                    time.sleep(30)
                try:
                    _process_user(cl, user, session)
                    session.commit()
                    parsed += 1
                    advance(detail=f"{parsed}/{total_users}")
                    break
                except Exception as e:
                    log(SCOPE, f"error fetching user: {e}", level="err")
                    session.rollback()
                    user = session.query(User).filter_by(id=user.id).one()
                    if attempt == 2:
                        user.parse_status = "failed"
                        session.commit()
                        failed += 1
                        advance(detail=f"{parsed}/{total_users} ({failed} failed)")

    session.close()

    total = parsed + skipped + failed
    log(
        SCOPE,
        f"done — total: {total}, parsed: {parsed}, skipped: {skipped}, failed: {failed}",
        level="ok",
    )
