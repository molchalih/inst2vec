import inspect
from unittest.mock import MagicMock

import pytest

from core.database import (
    AudioMIR,
    Clip,
    ClipEmbedding,
    StageState,
    User,
    get_or_create_user_identity,
    get_session,
    update_user_identity,
)
from modules.ingest import profiles as parse_mod


@pytest.fixture(autouse=True)
def _clean_db():
    """Clear shared main-DB state so prior tests cannot pollute identity allocations."""
    session = get_session()
    for model in (StageState, ClipEmbedding, AudioMIR, Clip, User):
        session.query(model).delete()
    session.commit()
    session.close()
    yield


def test_fetch_profiles_accepts_explicit_params():
    sig = inspect.signature(parse_mod.fetch_profiles)
    assert "hiker_api_key" in sig.parameters


def _make_user_with_identity(username: str, api_pk: int):
    """Create a User in main DB and a UserIdentity with api_pk in identity DB."""
    user_id = get_or_create_user_identity(username)
    update_user_identity(
        user_id,
        api_pk=api_pk,
        full_name=None,
        city_name=None,
        profile_pic_url=None,
        profile_pic_url_hd=None,
    )
    session = get_session()
    user = User(id=user_id)
    session.add(user)
    session.commit()
    return user_id, session


def _make_clip_item(
    pk: int,
    play_count: int = 0,
    video_duration: float = 12.5,
    taken_at: int = 1700000000,
) -> dict:
    return {
        "media": {
            "pk": pk,
            "thumbnail_url": f"https://example.com/{pk}.jpg",
            "video_url": f"https://example.com/{pk}.mp4",
            "caption": {"text": "hello", "text_translation": None},
            "comment_count": 0,
            "reshare_count": 0,
            "like_count": 10,
            "play_count": play_count,
            "video_duration": video_duration,
            "taken_at": taken_at,
        }
    }


def test_fetch_profile_returns_info_and_sorted_items_without_db_writes():
    """The worker is pure I/O: it returns data and touches no DB."""
    cl = MagicMock()
    cl.user_by_username_v1.return_value = {
        "user": {"pk": 222001, "full_name": "W", "follower_count": 9}
    }
    cl.user_clips_v2.return_value = {
        "response": {
            "items": [
                _make_clip_item(pk=1, play_count=10),
                _make_clip_item(pk=2, play_count=99),
                _make_clip_item(pk=3, play_count=50),
            ]
        },
        "next_page_id": None,
    }

    result = parse_mod._fetch_profile(
        cl, "worker_user", max_attempts=1, retry_delay=0, retry_jitter=0
    )

    assert result.ok is True
    assert result.info["pk"] == 222001
    # sorted by play_count desc
    assert [it["media"]["pk"] for it in result.items] == [2, 3, 1]
    # clips fetched using info["pk"], not an identity-DB lookup
    assert cl.user_clips_v2.call_args_list[0] == (("222001",),)


def test_fetch_profile_paginates_up_to_5_pages():
    pages = [
        {
            "response": {"items": [_make_clip_item(pk=30001 + i) for i in range(12)]},
            "next_page_id": f"page{p + 2}" if p < 4 else None,
        }
        for p in range(5)
    ]
    cl = MagicMock()
    cl.user_by_username_v1.return_value = {"user": {"pk": 222002}}
    cl.user_clips_v2.side_effect = pages

    result = parse_mod._fetch_profile(
        cl, "paginate_user", max_attempts=1, retry_delay=0, retry_jitter=0
    )

    assert result.ok is True
    assert len(result.items) == 60
    assert cl.user_clips_v2.call_count == 5


def test_fetch_profile_returns_err_on_failure(monkeypatch):
    monkeypatch.setattr(parse_mod.time, "sleep", lambda _: None)
    cl = MagicMock()
    cl.user_by_username_v1.side_effect = RuntimeError("api down")

    result = parse_mod._fetch_profile(
        cl, "broken_user", max_attempts=2, retry_delay=0, retry_jitter=0
    )

    assert result.ok is False
    assert result.info is None
    assert "api down" in result.err
    assert cl.user_by_username_v1.call_count == 2  # retried


def test_persist_profile_writes_user_and_clips():
    user_id, session = _make_user_with_identity("persist_user", api_pk=0)
    info = {
        "pk": 333001,
        "full_name": "Persist",
        "city_name": None,
        "profile_pic_url": None,
        "profile_pic_url_hd": None,
        "following_count": 7,
        "follower_count": 4242,
    }
    items = [_make_clip_item(pk=55001, video_duration=8.3, taken_at=1710000000)]
    result = parse_mod.ProfileResult(
        ok=True, info=info, items=items, duration=0.0, err=None
    )

    user = session.query(User).filter_by(id=user_id).one()
    parse_mod._persist_profile(user, result, session)
    session.commit()

    refreshed = session.query(User).filter_by(id=user_id).one()
    assert refreshed.follower_count == 4242
    assert refreshed.parse_status == "success"
    clip = session.query(Clip).filter_by(user_id=user_id).one()
    assert clip.video_duration == pytest.approx(8.3)
    assert clip.taken_at == 1710000000
    session.close()


def test_fetch_profiles_persist_failure_does_not_abort_batch(monkeypatch):
    """A main-thread DB failure for one user must not prevent other users from being persisted."""
    # Set up two users; user_id_bad will have _persist_profile raise, user_id_good will succeed.
    user_id_good, session_good = _make_user_with_identity("batch_good", api_pk=444001)
    session_good.close()
    user_id_bad, session_bad = _make_user_with_identity("batch_bad", api_pk=444002)
    session_bad.close()

    # Build a FakeClient that returns valid payloads for both usernames.
    user_data = {
        "batch_good": {
            "user": {
                "pk": 444001,
                "full_name": "Good",
                "city_name": None,
                "profile_pic_url": None,
                "profile_pic_url_hd": None,
                "following_count": 1,
                "follower_count": 1,
            }
        },
        "batch_bad": {
            "user": {
                "pk": 444002,
                "full_name": "Bad",
                "city_name": None,
                "profile_pic_url": None,
                "profile_pic_url_hd": None,
                "following_count": 2,
                "follower_count": 2,
            }
        },
    }

    class FakeClient:
        def user_by_username_v1(self, username):
            return user_data[username]

        def user_clips_v2(self, pk, *args, **kwargs):
            return {"response": {"items": []}}

    monkeypatch.setattr(parse_mod, "Client", lambda token: FakeClient())
    monkeypatch.setattr(parse_mod.time, "sleep", lambda _: None)

    # Patch _persist_profile: raise for user_id_bad (keyed on result.info["pk"]), pass otherwise.
    _real_persist = parse_mod._persist_profile

    def _patched_persist(user, result, session):
        if result.info and result.info["pk"] == 444002:
            raise RuntimeError("boom")
        _real_persist(user, result, session)

    monkeypatch.setattr(parse_mod, "_persist_profile", _patched_persist)

    # fetch_profiles must not raise even though one user's persist blows up.
    parse_mod.fetch_profiles(hiker_api_key="test_key")

    # Good user must end up as "success".
    session = get_session()
    good = session.query(User).filter_by(id=user_id_good).one()
    bad = session.query(User).filter_by(id=user_id_bad).one()
    session.close()

    assert good.parse_status == "success"
    assert bad.parse_status == "failed"
