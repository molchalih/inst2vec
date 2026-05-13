import inspect
from unittest.mock import MagicMock

import pytest

from modules import parse as parse_mod
from modules.database import Clip, User, get_session
from modules.identity import (
    get_or_create_user_identity,
    update_user_identity,
)


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


def test_process_user_stores_follower_count():
    user_id, session = _make_user_with_identity("test_follower_user", api_pk=111001)

    cl = MagicMock()
    cl.user_by_username_v1.return_value = {
        "user": {
            "pk": 111001,
            "full_name": "Test User",
            "city_name": None,
            "profile_pic_url": None,
            "profile_pic_url_hd": None,
            "following_count": 50,
            "follower_count": 1234,
        }
    }
    cl.user_clips_v2.return_value = {"response": {"items": [], "next_page_id": None}}

    user = session.query(User).filter_by(id=user_id).one()
    parse_mod._process_user(cl, user, session)
    session.commit()

    refreshed = session.query(User).filter_by(id=user_id).one()
    assert refreshed.follower_count == 1234
    session.close()


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


def test_fetch_clips_stores_video_duration_and_taken_at():
    user_id, session = _make_user_with_identity("test_duration_user", api_pk=111002)

    cl = MagicMock()
    cl.user_clips_v2.return_value = {
        "response": {
            "items": [
                _make_clip_item(pk=20001, video_duration=8.3, taken_at=1710000000)
            ],
            "next_page_id": None,
        }
    }

    user = session.query(User).filter_by(id=user_id).one()
    parse_mod._fetch_clips(cl, user, session)
    session.commit()

    clip = session.query(Clip).filter_by(user_id=user_id).one()
    assert clip.video_duration == pytest.approx(8.3)
    assert clip.taken_at == 1710000000
    session.close()


def test_fetch_clips_paginates_up_to_5_pages():
    user_id, session = _make_user_with_identity("test_paginate_user", api_pk=111003)

    pages = [
        {"items": [_make_clip_item(pk=30001 + i) for i in range(12)], "next_page_id": "page2"},
        {"items": [_make_clip_item(pk=30013 + i) for i in range(12)], "next_page_id": "page3"},
        {"items": [_make_clip_item(pk=30025 + i) for i in range(12)], "next_page_id": "page4"},
        {"items": [_make_clip_item(pk=30037 + i) for i in range(12)], "next_page_id": "page5"},
        {"items": [_make_clip_item(pk=30049 + i) for i in range(12)], "next_page_id": None},
    ]
    cl = MagicMock()
    cl.user_clips_v2.side_effect = [{"response": p} for p in pages]

    user = session.query(User).filter_by(id=user_id).one()
    parse_mod._fetch_clips(cl, user, session)
    session.commit()

    clips = session.query(Clip).filter_by(user_id=user_id).all()
    assert len(clips) == 60
    assert cl.user_clips_v2.call_count == 5
    session.close()


def test_fetch_clips_stops_when_no_next_page():
    user_id, session = _make_user_with_identity("test_no_next_user", api_pk=111004)

    cl = MagicMock()
    cl.user_clips_v2.return_value = {
        "response": {
            "items": [_make_clip_item(pk=40001 + i) for i in range(12)],
            "next_page_id": None,
        }
    }

    user = session.query(User).filter_by(id=user_id).one()
    parse_mod._fetch_clips(cl, user, session)
    session.commit()

    clips = session.query(Clip).filter_by(user_id=user_id).all()
    assert len(clips) == 12
    assert cl.user_clips_v2.call_count == 1
    session.close()


def test_fetch_clips_first_call_has_no_page_id():
    user_id, session = _make_user_with_identity("test_first_call_user", api_pk=111005)

    cl = MagicMock()
    cl.user_clips_v2.return_value = {
        "response": {"items": [], "next_page_id": None}
    }

    user = session.query(User).filter_by(id=user_id).one()
    parse_mod._fetch_clips(cl, user, session)

    first_call_args = cl.user_clips_v2.call_args_list[0]
    assert first_call_args == (("111005",),)
    session.close()
