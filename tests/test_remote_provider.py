"""Tests for RemoteQwenProvider — HTTP client side of the GPU pod link."""

import json

import httpx
import pytest

from modules.embeddings.providers import RemoteEmbedError, RemoteQwenProvider
from modules.embeddings.sampling import is_token_mismatch_error


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("modules.embeddings.providers.time.sleep", lambda *_: None)


class _FakeStore:
    """Minimal ObjectStore stand-in: clip 123 → 'signed-url-1' / 'signed-url-2'."""

    def __init__(self):
        self.calls = 0

    def key_for_clip(self, clip_id: int) -> str:
        return f"videos/{clip_id}.mp4"

    def signed_get(self, key: str, ttl_s: int | None = None) -> str:
        self.calls += 1
        return f"signed-url-{self.calls}"


def _provider(transport: httpx.MockTransport, store=None, max_retries: int = 3):
    return RemoteQwenProvider(
        url="https://pod.example",
        token="tok",
        storage=store or _FakeStore(),
        timeout_s=5,
        max_retries=max_retries,
        _client=httpx.Client(transport=transport),
    )


def test_happy_path_video_payload():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(
            200, json={"embedding": [0.1, 0.2, 0.3], "dim": 3, "took_ms": 1}
        )

    store = _FakeStore()
    p = _provider(httpx.MockTransport(handler), store=store)
    out = p.embed(
        {
            "case": "video",
            "clip_id": 123,
            "video": "/local/123.mp4",
            "fps": 1.0,
            "max_frames": 32,
        }
    )

    assert out == [[0.1, 0.2, 0.3]]
    assert seen["body"]["video_url"] == "signed-url-1"
    assert "video" not in seen["body"]  # local path is stripped
    assert seen["body"]["case"] == "video"
    assert seen["body"]["fps"] == 1.0
    assert seen["body"]["max_frames"] == 32
    assert seen["auth"] == "Bearer tok"


def test_happy_path_audio_payload_no_video_url():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"embedding": [0.5], "dim": 1, "took_ms": 1})

    store = _FakeStore()
    p = _provider(httpx.MockTransport(handler), store=store)
    out = p.embed({"case": "audio", "clip_id": 7, "text": "hi", "instruction": "ix"})
    assert out == [[0.5]]
    assert "video_url" not in seen["body"]
    assert seen["body"]["text"] == "hi"
    assert store.calls == 0  # no signed URL needed for audio


def test_retries_on_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"error": "transient"})
        return httpx.Response(200, json={"embedding": [1.0], "dim": 1, "took_ms": 1})

    p = _provider(httpx.MockTransport(handler), max_retries=3)
    out = p.embed({"case": "audio", "clip_id": 1, "text": "x", "instruction": "y"})
    assert out == [[1.0]]
    assert calls["n"] == 3


def test_5xx_exhausts_retries_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    p = _provider(httpx.MockTransport(handler), max_retries=2)
    with pytest.raises(RemoteEmbedError):
        p.embed({"case": "audio", "clip_id": 1, "text": "x", "instruction": "y"})


def test_signed_url_expired_regenerates_and_retries_once():
    store = _FakeStore()
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(502, json={"error": "signed_url_expired"})
        return httpx.Response(200, json={"embedding": [9.0], "dim": 1, "took_ms": 1})

    p = _provider(httpx.MockTransport(handler), store=store, max_retries=1)
    out = p.embed(
        {
            "case": "video",
            "clip_id": 123,
            "video": "/local/123.mp4",
            "fps": 1.0,
            "max_frames": 32,
        }
    )
    assert out == [[9.0]]
    assert store.calls == 2  # initial + refresh


def test_token_mismatch_raises_runner_recognizable_exception():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "error": "token_mismatch",
                "detail": "Mismatch in `video` token count vs payload",
            },
        )

    p = _provider(httpx.MockTransport(handler), max_retries=1)
    with pytest.raises(Exception) as ei:
        p.embed(
            {
                "case": "video",
                "clip_id": 1,
                "video": "/local/1.mp4",
                "fps": 1.0,
                "max_frames": 64,
            }
        )
    assert is_token_mismatch_error(ei.value)
