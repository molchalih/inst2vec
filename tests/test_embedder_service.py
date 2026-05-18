"""Tests for the FastAPI embedder service. Uses a fake provider — no model load."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from services.embedder import app as app_module

    fake = MagicMock()
    fake.embed.return_value = [[0.1, 0.2, 0.3]]
    with (
        patch.object(app_module, "_get_provider", return_value=fake),
        patch.object(
            app_module, "_resolve_video_url", side_effect=lambda url: "/tmp/x.mp4"
        ),
    ):
        app_module._reset_for_tests(token="tok")
        yield TestClient(app_module.app)


def test_healthz_returns_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_embed_requires_bearer_token(client):
    r = client.post(
        "/embed",
        json={"case": "audio", "clip_id": 1, "text": "x", "instruction": "i"},
    )
    assert r.status_code == 401


def test_embed_audio_returns_vector(client):
    r = client.post(
        "/embed",
        headers={"Authorization": "Bearer tok"},
        json={"case": "audio", "clip_id": 1, "text": "x", "instruction": "i"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["embedding"] == [0.1, 0.2, 0.3]
    assert body["dim"] == 3


def test_embed_video_downloads_url_to_temp_file_and_passes_path():
    """The handler should swap video_url → local path before calling the provider."""
    from services.embedder import app as app_module

    fake = MagicMock()
    fake.embed.return_value = [[0.5]]
    with (
        patch.object(app_module, "_get_provider", return_value=fake),
        patch.object(
            app_module, "_resolve_video_url", return_value="/tmp/123.mp4"
        ) as rv,
    ):
        app_module._reset_for_tests(token="tok")
        c = TestClient(app_module.app)
        r = c.post(
            "/embed",
            headers={"Authorization": "Bearer tok"},
            json={
                "case": "video",
                "clip_id": 123,
                "video_url": "https://r2/123.mp4?sig",
                "fps": 1.0,
                "max_frames": 32,
            },
        )
        assert r.status_code == 200
        rv.assert_called_once_with("https://r2/123.mp4?sig")
        call_payload = fake.embed.call_args[0][0]
        assert call_payload["video"] == "/tmp/123.mp4"
        assert "video_url" not in call_payload


def test_embed_returns_structured_token_mismatch():
    """Qwen video-token mismatch must surface as 422 {error: token_mismatch},
    not a generic 500, so RemoteQwenProvider's frame-cap fallback engages.
    """
    from services.embedder import app as app_module

    fake = MagicMock()
    fake.embed.side_effect = RuntimeError(
        "Mismatch in `video` token count vs payload size"
    )
    with (
        patch.object(app_module, "_get_provider", return_value=fake),
        patch.object(app_module, "_resolve_video_url", return_value="/tmp/1.mp4"),
    ):
        app_module._reset_for_tests(token="tok")
        c = TestClient(app_module.app)
        r = c.post(
            "/embed",
            headers={"Authorization": "Bearer tok"},
            json={
                "case": "video",
                "clip_id": 1,
                "video_url": "https://r2/1.mp4",
                "fps": 1.0,
                "max_frames": 64,
            },
        )
        assert r.status_code == 422
        assert r.json()["error"] == "token_mismatch"


def test_embed_rejects_unknown_case(client):
    """A case string outside CASE_REGISTRY must produce a 422 with a
    clear validation message — not silent passthrough."""
    r = client.post(
        "/embed",
        headers={"Authorization": "Bearer tok"},
        json={"case": "telepathy", "clip_id": 1, "text": "x", "instruction": "i"},
    )
    assert r.status_code == 422, r.text
    assert "unknown case" in r.text.lower()


def test_embed_rejects_unsupported_case(client):
    """A case registered in CASE_REGISTRY but not served by the Qwen pod
    (e.g. gemini) must produce a 422 — not an unhandled 500 from the
    payload builder that requires audio_path."""
    r = client.post(
        "/embed",
        headers={"Authorization": "Bearer tok"},
        json={"case": "gemini", "clip_id": 1, "text": "x"},
    )
    assert r.status_code == 422, r.text
    assert "unsupported case" in r.text.lower()
