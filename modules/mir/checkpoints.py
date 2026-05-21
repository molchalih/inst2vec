"""Parallel bootstrap downloader for MIR Essentia model graphs."""

from __future__ import annotations

import contextlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from core.config import MirSettings
from core.console import log

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
        log("mir:checkpoint", "SCAN", "checkpoints", "ok", stats={"missing": 0})
        return

    log("mir:checkpoint", "SCAN", "checkpoints", "ok", stats={"missing": len(missing)})
    os.makedirs(mir.model_dir, exist_ok=True)
    errors: list[str] = []

    def _fetch(client: httpx.Client, url: str, target: Path) -> None:
        tmp = Path(str(target) + ".part")
        try:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
            os.replace(tmp, target)
            log("mir:checkpoint", "GET", target.name, "ok")
        except (httpx.HTTPError, OSError) as exc:
            with contextlib.suppress(OSError):
                tmp.unlink()
            errors.append(f"{target.name}: {exc}")
            log(
                "mir:checkpoint",
                "GET",
                target.name,
                "ERR",
                stats={"err": str(exc)},
            )

    with (
        _make_client(mir.http_timeout) as client,
        ThreadPoolExecutor(max_workers=mir.download_concurrency) as pool,
    ):
        futs = [pool.submit(_fetch, client, u, p) for (u, p) in missing]
        for fut in as_completed(futs):
            fut.result()

    if errors:
        log("mir:checkpoint", "SEAL", "checkpoints", "ERR", stats={"err": len(errors)})
        raise RuntimeError("ensure_checkpoints failed:\n  " + "\n  ".join(errors))
    log("mir:checkpoint", "SEAL", "checkpoints", "ok", stats={"got": len(missing)})
