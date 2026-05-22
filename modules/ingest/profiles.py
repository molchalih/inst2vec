import time
from typing import Any

from hikerapi import Client

from core.console import log, progress
from core.database import (
    Clip,
    User,
    allocate_clip_identity,
    get_api_pk,
    get_session,
    get_username,
    update_user_identity,
)

SCOPE = "hiker"


def _fetch_clips(cl: Any, user: User, session: Any) -> None:
    api_pk = get_api_pk(user.id)
    if api_pk is None:
        return

    all_items: list[Any] = []
    next_page_id: str | None = None

    for _ in range(5):
        t0 = time.perf_counter()
        if next_page_id is not None:
            data = cl.user_clips_v2(str(api_pk), next_page_id)
        else:
            data = cl.user_clips_v2(str(api_pk))
        page = data["response"]
        items = page.get("items", [])
        log(
            SCOPE,
            "GET",
            f"user_clips/{api_pk}",
            "200",
            stats={"time": time.perf_counter() - t0, "clips": len(items)},
        )
        all_items.extend(items)
        next_page_id = data.get("next_page_id") or None
        if not next_page_id:
            break

    all_items.sort(key=lambda x: x["media"].get("play_count") or 0, reverse=True)

    for item in all_items:
        m = item["media"]
        clip_api_pk = int(m["pk"])
        with allocate_clip_identity(clip_api_pk) as clip_id:
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
            session.flush()


def _process_user(cl: Any, user: User, session: Any) -> None:
    username = get_username(user.id)
    t0 = time.perf_counter()
    data = cl.user_by_username_v1(username)
    log(
        SCOPE,
        "GET",
        f"user_by_username/{username}",
        "200",
        stats={"time": time.perf_counter() - t0},
    )
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
        session.close()
        return

    log(SCOPE, "SCAN", "users", "ok", stats={"todo": len(users)})

    parsed = failed = 0
    total_users = len(users)
    t_stage = time.perf_counter()

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
                    log(
                        SCOPE,
                        "GET",
                        f"user/{user.id}",
                        "ERR",
                        stats={"err": str(e), "attempt": attempt + 1},
                    )
                    session.rollback()
                    user = session.query(User).filter_by(id=user.id).one()
                    if attempt == 2:
                        user.parse_status = "failed"
                        session.commit()
                        failed += 1
                        advance(detail=f"{parsed}/{total_users} ({failed} failed)")

    session.close()

    log(
        SCOPE,
        "SEAL",
        "profiles",
        "ok",
        stats={
            "ok": parsed,
            "err": failed,
            "time": time.perf_counter() - t_stage,
        },
    )
