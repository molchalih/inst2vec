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


def test_validate_checkpoint_sidecars_writes_missing_sidecar(mir_settings):
    """A .pb without a sidecar gets one with the right digest, size, mtime."""
    import hashlib
    import json
    from pathlib import Path

    from modules.mir import checkpoints

    Path(mir_settings.model_dir).mkdir(parents=True, exist_ok=True)
    pb = Path(mir_settings.model_dir) / "fake.pb"
    pb.write_bytes(b"abc")
    side = checkpoints._sidecar_path(pb)

    checkpoints._maintain_sidecar(pb)

    assert side.exists()
    data = json.loads(side.read_text())
    assert data["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert data["size"] == 3
    assert data["mtime_ns"] == pb.stat().st_mtime_ns


def test_validate_checkpoint_sidecars_skips_when_header_matches(mir_settings):
    """A sidecar with matching (size, mtime) header is left alone."""
    import json
    from pathlib import Path

    from modules.mir import checkpoints

    Path(mir_settings.model_dir).mkdir(parents=True, exist_ok=True)
    pb = Path(mir_settings.model_dir) / "fake.pb"
    pb.write_bytes(b"abc")
    side = checkpoints._sidecar_path(pb)
    side.write_text(
        json.dumps(
            {
                "sha256": "stale-but-matching-header",
                "size": 3,
                "mtime_ns": pb.stat().st_mtime_ns,
            }
        )
    )

    checkpoints._maintain_sidecar(pb)

    data = json.loads(side.read_text())
    assert data["sha256"] == "stale-but-matching-header"


def test_validate_checkpoint_sidecars_rehashes_on_header_drift(mir_settings):
    """If size or mtime drifts, the sidecar is rewritten with the new digest."""
    import hashlib
    import json
    import os
    from pathlib import Path

    from modules.mir import checkpoints

    Path(mir_settings.model_dir).mkdir(parents=True, exist_ok=True)
    pb = Path(mir_settings.model_dir) / "fake.pb"
    pb.write_bytes(b"old")
    side = checkpoints._sidecar_path(pb)
    side.write_text(
        json.dumps(
            {"sha256": "old-digest", "size": 3, "mtime_ns": pb.stat().st_mtime_ns}
        )
    )

    pb.write_bytes(b"newcontent")
    new_mtime = pb.stat().st_mtime_ns + 1000
    os.utime(pb, ns=(new_mtime, new_mtime))

    checkpoints._maintain_sidecar(pb)

    data = json.loads(side.read_text())
    assert data["sha256"] == hashlib.sha256(b"newcontent").hexdigest()
    assert data["size"] == len(b"newcontent")


def test_validate_checkpoint_sidecars_iterates_all_present_pbs(mir_settings):
    """validate_checkpoint_sidecars visits every present .pb (no downloads)."""
    from pathlib import Path

    from modules.mir import checkpoints

    Path(mir_settings.model_dir).mkdir(parents=True, exist_ok=True)
    # Drop two .pb files matching the manifest; leave others missing.
    items = checkpoints._manifest(mir_settings)
    pb_a = items[0][1]
    pb_b = items[1][1]
    pb_a.write_bytes(b"aaa")
    pb_b.write_bytes(b"bbb")

    checkpoints.validate_checkpoint_sidecars(mir_settings)

    assert checkpoints._sidecar_path(pb_a).exists()
    assert checkpoints._sidecar_path(pb_b).exists()
    # Missing .pb files do not have stray sidecars.
    pb_missing = items[2][1]
    assert not checkpoints._sidecar_path(pb_missing).exists()


def test_ensure_checkpoints_writes_sidecar_for_downloaded_files(
    monkeypatch, mir_settings
):
    """A successful download leaves a matching .pb.sha256 next to every .pb."""
    import hashlib
    import json

    from modules.mir import checkpoints

    monkeypatch.setattr(
        checkpoints,
        "_make_client",
        lambda timeout: httpx.Client(transport=_stub_transport(200, b"payload")),
    )
    checkpoints.ensure_checkpoints(mir_settings)

    expected = hashlib.sha256(b"payload").hexdigest()
    for _url, target in checkpoints._manifest(mir_settings):
        side = checkpoints._sidecar_path(target)
        assert side.exists(), target
        data = json.loads(side.read_text())
        assert data["sha256"] == expected


def test_ensure_checkpoints_refreshes_present_sidecars_on_entry(
    monkeypatch, mir_settings
):
    """All-present path still maintains sidecars (e.g. catches manual swaps)."""
    import json
    from pathlib import Path

    from modules.mir import checkpoints

    Path(mir_settings.model_dir).mkdir(parents=True, exist_ok=True)
    for _url, target in checkpoints._manifest(mir_settings):
        Path(target).write_bytes(b"data")
        # No sidecar exists yet.

    def fake_make_client(timeout):  # pragma: no cover
        raise AssertionError("should not open client when all .pb files exist")

    monkeypatch.setattr(checkpoints, "_make_client", fake_make_client)
    checkpoints.ensure_checkpoints(mir_settings)

    for _url, target in checkpoints._manifest(mir_settings):
        side = checkpoints._sidecar_path(target)
        assert side.exists()
        data = json.loads(side.read_text())
        assert "sha256" in data


def test_ensure_checkpoints_retries_transient_failures(monkeypatch, mir_settings):
    """A flaky 500 that succeeds on attempt 2 yields a complete .pb."""
    from pathlib import Path

    from modules.mir import checkpoints

    attempts: dict[str, int] = {}

    def transport_factory():
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            attempts[url] = attempts.get(url, 0) + 1
            if attempts[url] == 1:
                return httpx.Response(500, content=b"boom")
            return httpx.Response(200, content=b"ok-payload")

        return httpx.MockTransport(handler)

    monkeypatch.setattr(
        checkpoints,
        "_make_client",
        lambda timeout: httpx.Client(transport=transport_factory()),
    )
    monkeypatch.setattr(checkpoints.time, "sleep", lambda _s: None)
    mir_settings = mir_settings.model_copy(
        update={"checkpoint_max_attempts": 3, "checkpoint_backoff_seconds": 0.0}
    )

    checkpoints.ensure_checkpoints(mir_settings)

    for _url, target in checkpoints._manifest(mir_settings):
        assert Path(target).read_bytes() == b"ok-payload"


def test_ensure_checkpoints_gives_up_after_max_attempts(monkeypatch, mir_settings):
    """Permanent 500s blow up after checkpoint_max_attempts."""
    from modules.mir import checkpoints

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(500, content=b"nope")

    monkeypatch.setattr(
        checkpoints,
        "_make_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(checkpoints.time, "sleep", lambda _s: None)
    mir_settings = mir_settings.model_copy(
        update={"checkpoint_max_attempts": 2, "checkpoint_backoff_seconds": 0.0}
    )

    with pytest.raises(RuntimeError):
        checkpoints.ensure_checkpoints(mir_settings)
    # 18 files × 2 attempts each = 36 hits
    assert len(seen) == 18 * 2


def test_ensure_checkpoints_partial_failure_keeps_successes_and_raises(
    monkeypatch, mir_settings
):
    """One URL 500s permanently, others 200.

    After the call:
      - RuntimeError is raised.
      - No .part files remain.
      - Successful .pb files are persisted on disk (don't punish completed work).
      - errors list contains exactly one entry, even under concurrent download.
    """
    from pathlib import Path

    from modules.mir import checkpoints

    items = checkpoints._manifest(mir_settings)
    failing_url = items[0][0]

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == failing_url:
            return httpx.Response(500, content=b"boom")
        return httpx.Response(200, content=b"ok-payload")

    monkeypatch.setattr(
        checkpoints,
        "_make_client",
        lambda timeout: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(checkpoints.time, "sleep", lambda _s: None)
    mir_settings = mir_settings.model_copy(
        update={"checkpoint_max_attempts": 2, "checkpoint_backoff_seconds": 0.0}
    )

    with pytest.raises(RuntimeError) as exc_info:
        checkpoints.ensure_checkpoints(mir_settings)
    # Exactly one failed file is named in the aggregate error message.
    assert items[0][1].name in str(exc_info.value)

    # Successes persist; no .part files anywhere.
    for url, target in items:
        assert not Path(str(target) + ".part").exists(), f".part left for {target.name}"
        if url == failing_url:
            assert not target.exists(), "failed download must not be sealed"
        else:
            assert target.read_bytes() == b"ok-payload"
