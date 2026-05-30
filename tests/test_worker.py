from __future__ import annotations

import threading
import time

import httpx

from modules.embeddings.broker import JobBroker, make_job
from modules.embeddings.cases import CASE_REGISTRY
from modules.embeddings.worker import (
    HttpJobSource,
    LocalJobSource,
    _resolve_audio_path,
    _resolve_video_path,
    embed_with_token_fallback,
    run_worker,
)


class _StubProvider:
    """Returns a fixed 2-float vector; records payloads it saw."""

    def __init__(self):
        self.payloads: list[dict] = []
        self.lock = threading.Lock()

    def embed(self, payload):
        with self.lock:
            self.payloads.append(dict(payload))
        time.sleep(0.005)
        return [[1.0, 2.0]]


def test_embed_with_token_fallback_injects_case_and_clip_id():
    p = _StubProvider()
    embed_with_token_fallback(
        p,
        CASE_REGISTRY["spoken"],
        clip_id=42,
        text="hi",
        video_path=None,
        audio_path=None,
        fps=None,
        max_frames=None,
    )
    embed_with_token_fallback(
        p,
        CASE_REGISTRY["video"],
        clip_id=42,
        text=None,
        video_path="/v/42.mp4",
        audio_path=None,
        fps=1.0,
        max_frames=32,
    )
    assert p.payloads[0]["case"] == "spoken" and p.payloads[0]["clip_id"] == 42
    assert p.payloads[1]["case"] == "video" and p.payloads[1]["clip_id"] == 42


def test_token_mismatch_retries_smaller_frame_caps():
    class _Flaky:
        def __init__(self):
            self.caps_seen: list[int] = []

        def embed(self, payload):
            self.caps_seen.append(payload["max_frames"])
            if payload["max_frames"] > 48:
                raise RuntimeError("Mismatch in `video` token count: too many")
            return [[1.0, 2.0]]

    p = _Flaky()
    blob = embed_with_token_fallback(
        p,
        CASE_REGISTRY["video"],
        clip_id=1,
        text=None,
        video_path="/v/1.mp4",
        audio_path=None,
        fps=2.0,
        max_frames=96,
    )
    assert blob is not None
    assert p.caps_seen[0] == 96 and p.caps_seen[-1] <= 48


def test_http_job_source_lease_complete_roundtrip():
    state = {"leased": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/lease"):
            if state["leased"]:
                return httpx.Response(410, json={"status": "drained"})
            state["leased"] = True
            return httpx.Response(
                200,
                json={
                    "lease_id": "L1",
                    "job": make_job(
                        clip_id=7,
                        case="sandwich",
                        text="hi",
                        video_key=None,
                        fps=None,
                        max_frames=None,
                        remote_eligible=True,
                    ),
                },
            )
        if request.url.path.endswith("/complete"):
            body = request.read().decode()
            assert '"lease_id": "L1"' in body or '"lease_id":"L1"' in body
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(request.url.path)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    src = HttpJobSource(
        base_url="http://pod", token="t", timeout_s=5, max_retries=2, _client=client
    )
    leased = src.lease(served_only=True)
    assert leased.job["clip_id"] == 7
    src.complete(leased.lease_id, [1.0, 2.0])
    # next lease -> drained
    assert src.lease(served_only=True) == "drained"


def test_worker_resolves_audio_path_for_audio_key(tmp_path):
    (tmp_path / "5.mp3").write_bytes(b"x")
    b = JobBroker(lease_ttl_s=600, max_attempts=3)
    b.add(
        make_job(
            clip_id=5,
            case="auditory",
            text=None,
            video_key=None,
            fps=None,
            max_frames=None,
            remote_eligible=False,
            audio_key="5.mp3",
        )
    )
    b.producer_done()
    p = _StubProvider()
    run_worker(
        LocalJobSource(b),
        provider=p,
        video_root="/x",
        audio_root=str(tmp_path),
        inflight=1,
        served_only=False,
        poll_idle_s=0.01,
    )
    assert p.payloads[0]["audio_path"] == str(tmp_path / "5.mp3")
    assert p.payloads[0]["case"] == "auditory"


def test_lease_raises_clear_error_on_unauthorized():
    import pytest

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "unauthorized"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    src = HttpJobSource(
        base_url="http://pod", token="bad", timeout_s=5, max_retries=0, _client=client
    )
    with pytest.raises(RuntimeError, match="401"):
        src.lease(served_only=True)


def test_resolve_accepts_bare_filenames():
    assert _resolve_video_path("/v", "1.mp4") == "/v/1.mp4"
    assert _resolve_audio_path("/a", "1.mp3") == "/a/1.mp3"
    assert _resolve_video_path("/v", None) is None
    assert _resolve_audio_path(None, "1.mp3") is None
    assert _resolve_audio_path("/a", None) is None


def test_resolve_returns_absolute_path_for_relative_root():
    # qwen-vl-utils turns the path into a "file://" URI before decoding. A
    # relative root yields the malformed "file://data/source/videos/1.mp4"
    # (parsed host=data) that torchcodec cannot open, even though the file
    # exists. The worker must hand the model an absolute path.
    import os

    vp = _resolve_video_path("data/source/videos", "1.mp4")
    assert vp is not None and os.path.isabs(vp)
    assert vp == os.path.abspath("data/source/videos/1.mp4")

    ap = _resolve_audio_path("data/source/audio", "1.mp3")
    assert ap is not None and os.path.isabs(ap)
    assert ap == os.path.abspath("data/source/audio/1.mp3")


def test_resolve_rejects_keys_that_escape_the_media_root():
    import pytest

    for bad in ("../secrets.mp4", "videos/1.mp4", "/etc/passwd", "..", "."):
        with pytest.raises(ValueError):
            _resolve_video_path("/v", bad)
        with pytest.raises(ValueError):
            _resolve_audio_path("/a", bad)


def test_complete_logs_and_does_not_raise_on_server_error(monkeypatch):
    warned = []
    import modules.embeddings.worker as worker_mod

    monkeypatch.setattr(worker_mod, "log", lambda *a, **kw: warned.append((a, kw)))

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"detail": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    src = HttpJobSource(
        base_url="http://pod", token="t", timeout_s=5, max_retries=0, _client=client
    )
    # Must not raise; the lease will be reaped + retried orchestrator-side.
    src.complete("L1", [1.0, 2.0])
    assert calls["n"] == 1
    assert warned, "a failed /complete must be logged, not silently dropped"
