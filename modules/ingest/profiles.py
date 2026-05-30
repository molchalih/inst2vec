import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, NamedTuple

from hikerapi import Client

from core.concurrency import retry_with_backoff
from core.config import ParseSettings, Secrets, Settings
from core.console import progress
from core.database import (
    Clip,
    User,
    allocate_clip_identity,
    get_session,
    get_username,
    update_user_identity,
)
from core.log import StageResult, event, item, stage

_MAX_CLIP_PAGES = 5


class ProfileResult(NamedTuple):
    ok: bool
    info: dict | None  # user_by_username_v1 payload
    items: list[dict]  # paginated, play_count-sorted clip media items
    duration: float
    err: str | None


def _fetch_profile(
    cl: Any,
    username: str,
    *,
    max_attempts: int,
    retry_delay: int,
    retry_jitter: int,
) -> ProfileResult:
    """Pure-I/O worker: fetch a user's profile + clips. No DB access.

    Wrapped in retry_with_backoff; never raises - returns ok=False on failure.
    """
    t0 = time.perf_counter()

    def _fetch() -> tuple[dict, list[dict]]:
        data = cl.user_by_username_v1(username)
        info = data.get("user", data)
        api_pk = info["pk"]

        all_items: list[Any] = []
        next_page_id: str | None = None
        for _ in range(_MAX_CLIP_PAGES):
            if next_page_id is not None:
                page_data = cl.user_clips_v2(str(api_pk), next_page_id)
            else:
                page_data = cl.user_clips_v2(str(api_pk))
            page = page_data["response"]
            all_items.extend(page.get("items", []))
            next_page_id = page_data.get("next_page_id") or None
            if not next_page_id:
                break

        all_items.sort(key=lambda x: x["media"].get("play_count") or 0, reverse=True)
        return info, all_items

    try:
        info, items = retry_with_backoff(
            _fetch,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            retry_jitter=retry_jitter,
        )
        return ProfileResult(
            ok=True,
            info=info,
            items=items,
            duration=time.perf_counter() - t0,
            err=None,
        )
    except Exception as e:
        return ProfileResult(
            ok=False,
            info=None,
            items=[],
            duration=time.perf_counter() - t0,
            err=repr(e),
        )


def _persist_profile(user: User, result: ProfileResult, session: Any) -> None:
    """Main-thread DB write for one successful profile fetch."""
    info = result.info
    assert info is not None
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

    for clip_item in result.items:
        m = clip_item["media"]
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

    user.parse_status = "success"


@stage("ingest:profiles")
def run_profiles(settings: Settings, secrets: Secrets) -> StageResult:
    """Fetch Instagram profiles + clips metadata via HikerAPI."""
    parse = settings.parse
    hiker_api_key = secrets.hiker_api_key
    cl = Client(token=hiker_api_key)
    session = get_session()

    users = session.query(User).filter(User.parse_status.is_(None)).all()
    if not users:
        session.close()
        return StageResult(done=0, failed=0)

    # Resolve usernames up front (only identity-DB reads before dispatch).
    jobs = [(user.id, get_username(user.id)) for user in users]

    total_users = len(jobs)
    event("SCAN", "users", stats={"todo": total_users})

    done = failed = 0

    try:
        with (
            progress(total_users, "Fetching profiles") as advance,
            ThreadPoolExecutor(max_workers=parse.concurrency) as pool,
        ):
            future_user = {
                pool.submit(
                    _fetch_profile,
                    cl,
                    username,
                    max_attempts=parse.max_attempts,
                    retry_delay=parse.retry_delay,
                    retry_jitter=parse.retry_jitter,
                ): user_id
                for user_id, username in jobs
            }

            for fut in as_completed(future_user):
                user_id = future_user[fut]
                result = fut.result()
                with item("GET", f"user_{user_id}") as t:
                    if not result.ok:
                        raise RuntimeError(result.err or "unknown")
                    user = session.query(User).filter_by(id=user_id).one()
                    _persist_profile(user, result, session)
                    session.commit()
                    t.stats(clips=len(result.items))
                if t.failed:
                    failed += 1
                    # _persist_profile / commit may have left the session in a
                    # failed-transaction state; rollback before re-querying so
                    # we don't trip PendingRollbackError and lose the chance to
                    # mark this user failed (legacy fetch_profiles did the same).
                    session.rollback()
                    try:
                        user = session.query(User).filter_by(id=user_id).one()
                        user.parse_status = "failed"
                        session.commit()
                    except Exception:
                        session.rollback()
                else:
                    done += 1
                advance(detail=f"{done}/{total_users} ({failed} failed)")
    finally:
        session.close()

    return StageResult(done=done, failed=failed)


def fetch_profiles(
    hiker_api_key: str,
    parse: ParseSettings | None = None,
) -> None:
    """Backward-compatible shim used by tests and legacy callers."""
    if parse is None:
        parse = ParseSettings()

    cl = Client(token=hiker_api_key)
    session = get_session()

    users = session.query(User).filter(User.parse_status.is_(None)).all()
    if not users:
        session.close()
        return

    # Resolve usernames up front (only identity-DB reads before dispatch).
    jobs = [(user.id, get_username(user.id)) for user in users]

    total_users = len(jobs)

    try:
        with (
            progress(total_users, "Fetching profiles") as advance,
            ThreadPoolExecutor(max_workers=parse.concurrency) as pool,
        ):
            future_user = {
                pool.submit(
                    _fetch_profile,
                    cl,
                    username,
                    max_attempts=parse.max_attempts,
                    retry_delay=parse.retry_delay,
                    retry_jitter=parse.retry_jitter,
                ): user_id
                for user_id, username in jobs
            }

            for fut in as_completed(future_user):
                user_id = future_user[fut]
                result = fut.result()
                try:
                    user = session.query(User).filter_by(id=user_id).one()
                    if result.ok:
                        _persist_profile(user, result, session)
                        session.commit()
                    else:
                        user.parse_status = "failed"
                        session.commit()
                except Exception:
                    session.rollback()
                    try:
                        user = session.query(User).filter_by(id=user_id).one()
                        user.parse_status = "failed"
                        session.commit()
                    except Exception:
                        session.rollback()
                advance()
    finally:
        session.close()
