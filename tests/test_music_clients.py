"""Tests for modules.music.clients: SpotifyClient and ReccoBeatsClient."""

import inspect

import httpx
import pytest

from modules.music.clients import ReccoBeatsClient, SpotifyClient, TransientError


def test_spotify_client_takes_credentials():
    """SpotifyClient.__init__ accepts client_id and client_secret."""
    sig = inspect.signature(SpotifyClient.__init__)
    assert "client_id" in sig.parameters
    assert "client_secret" in sig.parameters
    assert "token_skew" in sig.parameters
    assert "search_limit" in sig.parameters


def test_reccobeats_client_takes_batch_params():
    """ReccoBeatsClient.__init__ accepts batch configuration parameters."""
    sig = inspect.signature(ReccoBeatsClient.__init__)
    assert "batch" in sig.parameters
    assert "delay_min" in sig.parameters
    assert "delay_max" in sig.parameters
    assert "timeout" in sig.parameters


# ── retry behavior tests ──────────────────────────────────────────────────────


class _StubHttp:
    """Records requests; returns a sequence of (status, payload) responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        return self._next("POST")

    def get(self, *args, **kwargs):
        return self._next("GET")

    def request(self, method, *args, **kwargs):
        return self._next(method)

    def _next(self, method):
        self.calls += 1
        if not self._responses:
            raise httpx.NetworkError("out of stubs")
        status, payload = self._responses.pop(0)
        if isinstance(payload, Exception):
            raise payload
        req = httpx.Request(method, "https://stub/")
        return httpx.Response(status, json=payload, request=req)


def _spotify(http, max_attempts=3):
    return SpotifyClient(
        http,
        client_id="id",
        client_secret="secret",
        token_skew=30,
        search_limit=5,
        search_timeout=1.0,
        max_attempts=max_attempts,
        retry_delay=0.0,
        retry_jitter=0.0,
    )


def test_spotify_search_returns_id_on_clean_success():
    http = _StubHttp(
        [
            (200, {"access_token": "tok", "expires_in": 3600}),
            (200, {"tracks": {"items": [{"id": "abc"}]}}),
        ]
    )
    assert _spotify(http).search_id("a", "t") == "abc"


def test_spotify_search_returns_none_on_clean_no_match():
    http = _StubHttp(
        [
            (200, {"access_token": "tok", "expires_in": 3600}),
            (200, {"tracks": {"items": []}}),
        ]
    )
    assert _spotify(http).search_id("a", "t") is None


def test_spotify_search_retries_on_5xx_then_succeeds():
    http = _StubHttp(
        [
            (200, {"access_token": "tok", "expires_in": 3600}),
            (503, {}),
            (200, {"tracks": {"items": [{"id": "abc"}]}}),
        ]
    )
    assert _spotify(http).search_id("a", "t") == "abc"


def test_spotify_search_raises_transient_after_exhaustion():
    http = _StubHttp(
        [
            (200, {"access_token": "tok", "expires_in": 3600}),
            (503, {}),
            (503, {}),
            (503, {}),
        ]
    )
    with pytest.raises(TransientError):
        _spotify(http, max_attempts=3).search_id("a", "t")


def test_spotify_search_returns_none_on_4xx_non_429():
    http = _StubHttp(
        [
            (200, {"access_token": "tok", "expires_in": 3600}),
            (400, {}),
        ]
    )
    assert _spotify(http).search_id("a", "t") is None


def _rb(http, max_attempts=3, batch=2):
    return ReccoBeatsClient(
        http,
        batch=batch,
        delay_min=0.0,
        delay_max=0.0,
        timeout=1.0,
        max_attempts=max_attempts,
        retry_delay=0.0,
        retry_jitter=0.0,
    )


def test_rb_get_ids_returns_matched_dict():
    http = _StubHttp(
        [
            (
                200,
                {
                    "content": [
                        {"href": "/track/aaa", "id": "rb-aaa"},
                        {"href": "/track/bbb", "id": "rb-bbb"},
                    ]
                },
            ),
        ]
    )
    out = _rb(http, batch=10).get_ids(["aaa", "bbb"])
    assert out == {"aaa": "rb-aaa", "bbb": "rb-bbb"}


def test_rb_get_ids_omits_missing_ids():
    http = _StubHttp(
        [
            (200, {"content": [{"href": "/track/aaa", "id": "rb-aaa"}]}),
        ]
    )
    out = _rb(http, batch=10).get_ids(["aaa", "bbb"])
    assert out == {"aaa": "rb-aaa"}


def test_rb_get_ids_retries_batch_on_5xx():
    http = _StubHttp(
        [
            (503, {}),
            (200, {"content": [{"href": "/track/aaa", "id": "rb-aaa"}]}),
        ]
    )
    out = _rb(http, batch=10).get_ids(["aaa"])
    assert out == {"aaa": "rb-aaa"}


def test_rb_get_ids_drops_batch_after_exhausted_retries():
    http = _StubHttp(
        [
            (503, {}),
            (503, {}),
            (503, {}),
            (200, {"content": [{"href": "/track/bbb", "id": "rb-bbb"}]}),
        ]
    )
    out = _rb(http, max_attempts=3, batch=1).get_ids(["aaa", "bbb"])
    assert out == {"bbb": "rb-bbb"}  # first batch exhausted, second succeeded


def test_rb_upload_features_returns_dict_on_success(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF")
    http = _StubHttp([(200, {"tempo": 120.0, "energy": 0.5})])
    out = _rb(http).upload_features(audio)
    assert out == {"tempo": 120.0, "energy": 0.5}


def test_rb_upload_features_raises_transient_after_exhaustion(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF")
    http = _StubHttp([(503, {}), (503, {}), (503, {})])
    with pytest.raises(TransientError):
        _rb(http, max_attempts=3).upload_features(audio)


def test_rb_upload_features_returns_none_on_4xx_non_429(tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"RIFF")
    http = _StubHttp([(400, {})])
    out = _rb(http).upload_features(audio)
    assert out is None
