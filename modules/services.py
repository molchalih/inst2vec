"""Shared services: structured logging, Spotify and ReccoBeats HTTP clients."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

# ── logging ───────────────────────────────────────────────────────────────────

from modules.console import log  # noqa: F401 — re-exported for existing callers


# ── utilities ─────────────────────────────────────────────────────────────────

def chunked(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _spotify_id_from_href(href: Optional[str]) -> Optional[str]:
    if not href:
        return None
    parts = [p for p in urlparse(href).path.split("/") if p]
    return parts[-1] if len(parts) >= 2 and parts[-2] == "track" else None


# ── Spotify ───────────────────────────────────────────────────────────────────

_SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
_TOKEN_SKEW = int(os.environ.get("SPOTIFY_TOKEN_SKEW_SECONDS", 30))
_SEARCH_LIMIT = int(os.environ.get("SPOTIFY_SEARCH_LIMIT", 5))
_SEARCH_TIMEOUT = float(os.environ.get("SPOTIFY_REQUEST_TIMEOUT", 8))


class SpotifyClient:
    """Client-credentials OAuth Spotify track search with automatic token refresh."""

    def __init__(self, http: httpx.Client):
        self._http = http
        self._token: Optional[str] = None
        self._expires_at = 0.0

    def _refresh(self) -> bool:
        cid = os.environ.get("SPOTIFY_CLIENT_ID", "")
        secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
        if not cid or not secret:
            return False
        try:
            r = self._http.post(
                _SPOTIFY_TOKEN_URL,
                data={"grant_type": "client_credentials", "client_id": cid, "client_secret": secret},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=_SEARCH_TIMEOUT,
            )
            r.raise_for_status()
            p = r.json()
        except Exception:
            return False
        self._token = p.get("access_token")
        self._expires_at = time.time() + max(0, int(p.get("expires_in", 3600)) - _TOKEN_SKEW)
        return bool(self._token)

    def search_id(self, artist: str, track: str) -> Optional[str]:
        """Return the Spotify track ID for the best match, or None."""
        if not track.strip():
            return None
        if not self._token or time.time() >= self._expires_at:
            if not self._refresh():
                return None
        q = f"track:{track.strip()}"
        if artist.strip():
            q += f" artist:{artist.strip()}"
        try:
            r = self._http.get(
                _SPOTIFY_SEARCH_URL,
                params={"q": q, "type": "track", "limit": _SEARCH_LIMIT},
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=_SEARCH_TIMEOUT,
            )
            r.raise_for_status()
            items = r.json().get("tracks", {}).get("items", [])
        except Exception:
            return None
        return items[0].get("id") if items else None


# ── ReccoBeats ────────────────────────────────────────────────────────────────

_RB_TRACK_URL = "https://api.reccobeats.com/v1/track"
_RB_FEATURES_URL = "https://api.reccobeats.com/v1/audio-features"
_RB_ANALYSIS_URL = "https://api.reccobeats.com/v1/analysis/audio-features"
_RB_BATCH = int(os.environ.get("RECCOBEATS_BATCH_SIZE", 20))
_RB_DELAY_MIN = float(os.environ.get("RECCOBEATS_DELAY_MIN", 2))
_RB_DELAY_MAX = float(os.environ.get("RECCOBEATS_DELAY_MAX", 3))
_RB_TIMEOUT = float(os.environ.get("MUSIC_HTTP_TIMEOUT", 20))

# on_batch(batch_idx, total_batches, matched_in_batch)
OnBatch = Callable[[int, int, int], None]


class ReccoBeatsClient:
    """Batched ReccoBeats client: track ID lookup, catalog features, upload analysis."""

    def __init__(self, http: httpx.Client):
        self._http = http

    def _sleep(self) -> None:
        time.sleep(random.uniform(_RB_DELAY_MIN, _RB_DELAY_MAX))

    def get_ids(self, spotify_ids: list[str], on_batch: Optional[OnBatch] = None) -> dict[str, str]:
        """Return {spotify_id: reccobeats_id} for all matched tracks."""
        out: dict[str, str] = {}
        batches = list(chunked(spotify_ids, _RB_BATCH))
        for i, batch in enumerate(batches, 1):
            matched = 0
            try:
                r = self._http.get(_RB_TRACK_URL, params={"ids": ",".join(batch)}, timeout=_RB_TIMEOUT)
                r.raise_for_status()
                for item in r.json().get("content", []):
                    sid = _spotify_id_from_href(item.get("href"))
                    rid = item.get("id")
                    if sid and rid:
                        out[sid] = rid
                        matched += 1
            except Exception:
                pass
            if on_batch:
                on_batch(i, len(batches), matched)
            self._sleep()
        return out

    def get_features(self, rb_ids: list[str], on_batch: Optional[OnBatch] = None) -> dict[str, dict]:
        """Return {reccobeats_id: feature_payload} for matched tracks."""
        out: dict[str, dict] = {}
        batches = list(chunked(rb_ids, _RB_BATCH))
        for i, batch in enumerate(batches, 1):
            matched = 0
            try:
                r = self._http.get(_RB_FEATURES_URL, params={"ids": ",".join(batch)}, timeout=_RB_TIMEOUT)
                r.raise_for_status()
                for item in r.json().get("content", []):
                    rid = item.get("id")
                    if rid:
                        out[rid] = item
                        matched += 1
            except Exception:
                pass
            if on_batch:
                on_batch(i, len(batches), matched)
            self._sleep()
        return out

    def upload_features(self, audio: Path) -> Optional[dict]:
        """POST a short audio file to the analysis endpoint; return feature payload or None."""
        mime = "audio/wav" if audio.suffix.lower() == ".wav" else "audio/mpeg"
        try:
            with audio.open("rb") as fh:
                r = self._http.post(
                    _RB_ANALYSIS_URL,
                    files={"audioFile": (audio.name, fh, mime)},
                    timeout=_RB_TIMEOUT,
                )
                r.raise_for_status()
                p = r.json()
        except Exception:
            return None
        return p if isinstance(p, dict) else None
