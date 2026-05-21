"""Parallel bootstrap downloader for MIR Essentia model graphs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from core.config import MirSettings
from core.console import log

_LOG_CKPT = "mir:checkpoint"

_MAEST_BASE = "https://essentia.upf.edu/models/feature-extractors/maest"
_EFFNET_BASE = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet"
_HEAD_BASE = "https://essentia.upf.edu/models/classification-heads"

# (subdir, filename) for every EffNet classification/regression head.
_EFFNET_HEAD_FILES: tuple[tuple[str, str], ...] = (
    ("approachability", "approachability_regression-discogs-effnet-1.pb"),
    ("engagement", "engagement_regression-discogs-effnet-1.pb"),
    ("danceability", "danceability-discogs-effnet-1.pb"),
    ("mood_aggressive", "mood_aggressive-discogs-effnet-1.pb"),
    ("mood_happy", "mood_happy-discogs-effnet-1.pb"),
    ("mood_party", "mood_party-discogs-effnet-1.pb"),
    ("mood_relaxed", "mood_relaxed-discogs-effnet-1.pb"),
    ("mood_sad", "mood_sad-discogs-effnet-1.pb"),
    ("mood_acoustic", "mood_acoustic-discogs-effnet-1.pb"),
    ("mood_electronic", "mood_electronic-discogs-effnet-1.pb"),
    ("voice_instrumental", "voice_instrumental-discogs-effnet-1.pb"),
    ("gender", "gender-discogs-effnet-1.pb"),
    ("timbre", "timbre-discogs-effnet-1.pb"),
    ("tonal_atonal", "tonal_atonal-discogs-effnet-1.pb"),
    ("mtg_jamendo_moodtheme", "mtg_jamendo_moodtheme-discogs-effnet-1.pb"),
    ("mtg_jamendo_instrument", "mtg_jamendo_instrument-discogs-effnet-1.pb"),
)


def _manifest(mir: MirSettings) -> list[tuple[str, Path]]:
    """(url, target_path) pairs for every checkpoint required by run_mir."""
    root = Path(mir.model_dir)
    items: list[tuple[str, Path]] = [
        (
            f"{_MAEST_BASE}/{mir.maest_checkpoint}",
            root / mir.maest_checkpoint,
        ),
        (
            f"{_EFFNET_BASE}/{mir.effnet_checkpoint}",
            root / mir.effnet_checkpoint,
        ),
    ]
    for subdir, filename in _EFFNET_HEAD_FILES:
        items.append((f"{_HEAD_BASE}/{subdir}/{filename}", root / filename))
    return items


def _make_client(timeout: float) -> httpx.Client:
    """Indirection to make tests easy to monkeypatch."""
    return httpx.Client(timeout=timeout)


def ensure_checkpoints(mir: MirSettings) -> None:
    """Download every required ``.pb`` graph into ``mir.model_dir``.

    All-present -> returns immediately. Missing -> parallel HTTP GETs with
    atomic ``.part`` -> rename. Any failure aborts and removes ``.part``
    files. Idempotent.
    """
    items = _manifest(mir)
    missing = [(u, p) for (u, p) in items if not p.exists()]
    if not missing:
        log(_LOG_CKPT, "SCAN", "checkpoints", "ok", stats={"missing": 0})
        validate_checkpoint_sidecars(mir)
        return

    log(_LOG_CKPT, "SCAN", "checkpoints", "ok", stats={"missing": len(missing)})
    os.makedirs(mir.model_dir, exist_ok=True)
    errors: list[str] = []
    errors_lock = threading.Lock()

    def _fetch(client: httpx.Client, url: str, target: Path) -> None:
        tmp = Path(str(target) + ".part")
        last_err: Exception | None = None
        for attempt in range(mir.checkpoint_max_attempts):
            try:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_bytes():
                            f.write(chunk)
                os.replace(tmp, target)
                log(
                    "mir:checkpoint",
                    "GET",
                    target.name,
                    "ok",
                    stats={"attempts": attempt + 1},
                )
                return
            except (httpx.HTTPError, OSError) as exc:
                last_err = exc
                with contextlib.suppress(OSError):
                    tmp.unlink()
                if attempt + 1 < mir.checkpoint_max_attempts:
                    time.sleep(mir.checkpoint_backoff_seconds * (2**attempt))
        with errors_lock:
            errors.append(f"{target.name}: {last_err}")
        log(
            "mir:checkpoint",
            "GET",
            target.name,
            "ERR",
            stats={"err": str(last_err), "attempts": mir.checkpoint_max_attempts},
        )

    with (
        _make_client(mir.http_timeout) as client,
        ThreadPoolExecutor(max_workers=mir.download_concurrency) as pool,
    ):
        futs = [pool.submit(_fetch, client, u, p) for (u, p) in missing]
        for fut in as_completed(futs):
            fut.result()

    if errors:
        log(_LOG_CKPT, "SEAL", "checkpoints", "ERR", stats={"err": len(errors)})
        raise RuntimeError("ensure_checkpoints failed:\n  " + "\n  ".join(errors))
    log(_LOG_CKPT, "SEAL", "checkpoints", "ok", stats={"got": len(missing)})
    validate_checkpoint_sidecars(mir)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar_path(pb_path: Path) -> Path:
    return pb_path.parent / (pb_path.name + ".sha256")


def _maintain_sidecar(pb_path: Path) -> None:
    """Ensure ``<pb>.sha256`` matches the current (size, mtime) of ``pb_path``.

    Re-hashes and rewrites the sidecar when missing, unreadable, or when the
    stored (size, mtime) header differs from the .pb on disk. Idempotent.
    """
    side = _sidecar_path(pb_path)
    st = pb_path.stat()
    if side.exists():
        try:
            data = json.loads(side.read_text())
            if (
                isinstance(data, dict)
                and data.get("size") == st.st_size
                and data.get("mtime_ns") == st.st_mtime_ns
                and isinstance(data.get("sha256"), str)
            ):
                return
        except (json.JSONDecodeError, OSError):
            pass
    digest = _sha256_file(pb_path)
    side.write_text(
        json.dumps(
            {"sha256": digest, "size": st.st_size, "mtime_ns": st.st_mtime_ns},
            sort_keys=True,
        )
    )


def validate_checkpoint_sidecars(mir: MirSettings) -> None:
    """Refresh sidecars for every .pb present in ``mir.model_dir``.

    Cheap: a noop when sidecar headers match the current files. Designed to
    run at MIR stage entry so manual .pb swaps surface as fingerprint drift
    on the next run.
    """
    for _url, target in _manifest(mir):
        if target.exists():
            _maintain_sidecar(target)
