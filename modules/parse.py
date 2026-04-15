import os
import time

from hikerapi import Client
from httpx import ConnectError, TimeoutException

from modules.database import get_session, User, Clip, Download

HIKER_TOKEN = os.environ.get("HIKER_API_KEY", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 5))


def _is_parsed(user: User) -> bool:
    return all([
        user.full_name is not None,
        user.profile_pic_url is not None,
        user.following_count is not None,
    ])


MAX_CLIPS = int(os.environ.get("MAX_CLIPS", 5))


def _fetch_clips(cl: Client, user: User, session) -> int:
    if user.clips:
        return 0

    data = cl.user_clips_v2(user.pk)
    items = data["response"]["items"]
    items.sort(key=lambda x: x["media"].get("play_count") or 0, reverse=True)

    count = 0
    for item in items[:MAX_CLIPS or None]:
        m = item["media"]
        clip_pk = int(m["pk"])
        if session.query(Clip).filter_by(pk=clip_pk).first():
            continue

        cap = m.get("caption") or {}

        session.add(Clip(
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
        ))
        count += 1
    return count


def fetch_profiles():
    cl = Client(token=HIKER_TOKEN)
    session = get_session()

    failed_pks = session.query(Download.entity_pk).filter(
        Download.file_type == "profile_pic",
        Download.parse_available.is_(False),
    )
    users = (
        session.query(User)
        .filter(~User.pk.in_(failed_pks))
        .limit(BATCH_SIZE)
        .all()
    )

    parsed = 0
    skipped = 0
    failed = 0

    for i, user in enumerate(users, 1):
        if _is_parsed(user):
            skipped += 1
            continue

        print(f"[{i}/{len(users)}] {user.username} — ", end="", flush=True)

        try:
            data = cl.user_by_username_v1(user.username)
            info = data.get("user", data)

            user.pk = info["pk"]
            user.full_name = info.get("full_name")
            user.profile_pic_url = info.get("profile_pic_url")
            user.profile_pic_url_hd = info.get("profile_pic_url_hd")
            user.following_count = info.get("following_count")
            user.city_name = info.get("city_name")

            clips_count = _fetch_clips(cl, user, session)
            print(f"ok ({user.full_name}, {user.following_count} following, {clips_count} clips)")
            parsed += 1

        except (ConnectError, TimeoutException) as e:
            print(f"network error ({e})")
            session.merge(Download(entity_pk=user.pk, file_type="profile_pic", parse_available=False))
            failed += 1

        except Exception as e:
            print(f"error ({e})")
            session.merge(Download(entity_pk=user.pk, file_type="profile_pic", parse_available=False))
            failed += 1

        time.sleep(0.3)

    session.commit()
    session.close()

    total = parsed + skipped + failed
    print(f"\nDone — total: {total}, parsed: {parsed}, skipped: {skipped}, failed: {failed}")
