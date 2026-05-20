"""Tests for the upload_features upstream-vs-rejection distinction."""

from __future__ import annotations

import httpx
import pytest

from modules.music.clients import (
    ReccoBeatsClient,
    TransientError,
    UpstreamAnalysisError,
)


def _rb(http: httpx.Client, **overrides) -> ReccoBeatsClient:
    base = dict(
        batch=20,
        delay_min=0.0,
        delay_max=0.0,
        timeout=5.0,
        max_attempts=3,
        retry_delay=0.0,
        retry_jitter=0.0,
    )
    base.update(overrides)
    return ReccoBeatsClient(http, **base)


def test_rb_upload_features_raises_upstream_on_400(tmp_path):
    """A non-429 4xx from the analyzer must raise UpstreamAnalysisError
    carrying the HTTP status and the body's `error` field — NOT return
    None — so the call site can leave the row retryable and distinguish
    the failure from a clean per-file rejection."""
    body = {
        "timestamp": "2026-05-20T20:46:10.107+00:00",
        "error": "Get audio features fail",
        "path": "uri=/v1/analysis/audio-features",
        "status": 4002,
    }
    transport = httpx.MockTransport(lambda req: httpx.Response(400, json=body))
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"00")
    with (
        httpx.Client(transport=transport) as http,
        pytest.raises(UpstreamAnalysisError) as excinfo,
    ):
        _rb(http).upload_features(audio)
    exc = excinfo.value
    assert exc.status_code == 400
    assert exc.error == "Get audio features fail"


def test_rb_upload_features_raises_upstream_on_415(tmp_path):
    """4xx other than 429 must all surface as UpstreamAnalysisError so
    the call site keeps a single recovery path; the per-file vs upstream
    distinction lives in logs, not in row state."""
    transport = httpx.MockTransport(
        lambda req: httpx.Response(415, json={"error": "Unsupported file type"})
    )
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"00")
    with (
        httpx.Client(transport=transport) as http,
        pytest.raises(UpstreamAnalysisError) as excinfo,
    ):
        _rb(http).upload_features(audio)
    assert excinfo.value.status_code == 415


def test_rb_upload_features_still_transient_on_5xx_exhaustion(tmp_path):
    """5xx exhaustion must remain TransientError (unchanged behavior).
    Only non-429 4xx is reclassified by this change."""
    transport = httpx.MockTransport(lambda req: httpx.Response(503))
    audio = tmp_path / "x.mp3"
    audio.write_bytes(b"00")
    with (
        httpx.Client(transport=transport) as http,
        pytest.raises(TransientError),
    ):
        _rb(http, max_attempts=2).upload_features(audio)
