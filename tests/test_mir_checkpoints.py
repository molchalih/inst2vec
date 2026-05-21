"""Tests for ensure_checkpoints bootstrap behaviour."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest


@pytest.fixture
def mir_settings(tmp_path):
    from core.config import MirSettings

    return MirSettings(
        model_dir=str(tmp_path / "models"),
        download_concurrency=2,
        http_timeout=2.0,
    )


def _stub_transport(status: int, payload: bytes = b"x" * 16) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=payload)

    return httpx.MockTransport(handler)


def test_manifest_total_18_entries(mir_settings):
    from modules.mir.checkpoints import _manifest

    items = _manifest(mir_settings)
    assert len(items) == 18
    # 1 MAEST + 1 EffNet + 16 heads
    targets = [str(p) for (_url, p) in items]
    assert any("discogs-maest-30s-pw-519l-1.pb" in t for t in targets)
    assert any("discogs-effnet-bs64-1.pb" in t for t in targets)
    assert any("mtg_jamendo_moodtheme-discogs-effnet-1.pb" in t for t in targets)
    assert any("mtg_jamendo_instrument-discogs-effnet-1.pb" in t for t in targets)
    assert any("approachability_regression-discogs-effnet-1.pb" in t for t in targets)


def test_all_present_skips_downloads(monkeypatch, mir_settings):
    from modules.mir import checkpoints

    Path(mir_settings.model_dir).mkdir(parents=True, exist_ok=True)
    for _url, target in checkpoints._manifest(mir_settings):
        Path(target).write_bytes(b"already-there")

    calls = {"n": 0}

    def fake_make_client(timeout):  # pragma: no cover - should not run
        calls["n"] += 1
        raise AssertionError("should not open client when all files exist")

    monkeypatch.setattr(checkpoints, "_make_client", fake_make_client)
    checkpoints.ensure_checkpoints(mir_settings)
    assert calls["n"] == 0


def test_missing_files_are_fetched_in_parallel(monkeypatch, mir_settings):
    from modules.mir import checkpoints

    monkeypatch.setattr(
        checkpoints,
        "_make_client",
        lambda timeout: httpx.Client(transport=_stub_transport(200)),
    )
    checkpoints.ensure_checkpoints(mir_settings)

    for _url, target in checkpoints._manifest(mir_settings):
        assert Path(target).read_bytes() == b"x" * 16
        assert not Path(str(target) + ".part").exists()


def test_failed_download_cleans_partfile_and_raises(monkeypatch, mir_settings):
    from modules.mir import checkpoints

    monkeypatch.setattr(
        checkpoints,
        "_make_client",
        lambda timeout: httpx.Client(transport=_stub_transport(500, b"oops")),
    )
    with pytest.raises(RuntimeError):
        checkpoints.ensure_checkpoints(mir_settings)

    for _url, target in checkpoints._manifest(mir_settings):
        assert not Path(str(target) + ".part").exists()
        assert not Path(target).exists()
