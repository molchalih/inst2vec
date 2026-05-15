"""Shared services: Spotify and ReccoBeats HTTP clients."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import httpx

# ── utilities ─────────────────────────────────────────────────────────────────


class TransientError(Exception):
    """Raised when a service-client method exhausts its retry budget on transient errors."""


def _is_transient_http(exc: Exception) -> bool:
    if isinstance(
        exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError)
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def chunked(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _spotify_id_from_href(href: str | None) -> str | None:
    if not href:
        return None
    parts = [p for p in urlparse(href).path.split("/") if p]
    return parts[-1] if len(parts) >= 2 and parts[-2] == "track" else None


# ── Spotify ───────────────────────────────────────────────────────────────────

_SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"


class SpotifyClient:
    """Client-credentials OAuth Spotify track search with automatic token refresh and retries."""

    def __init__(
        self,
        http: httpx.Client,
        client_id: str,
        client_secret: str,
        token_skew: int = 30,
        search_limit: int = 5,
        search_timeout: float = 8.0,
        max_attempts: int = 3,
        retry_delay: float = 1.0,
        retry_jitter: float = 1.5,
    ):
        self._http = http
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_skew = token_skew
        self._search_limit = search_limit
        self._search_timeout = search_timeout
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay
        self._retry_jitter = retry_jitter
        self._token: str | None = None
        self._expires_at = 0.0

    def _sleep(self) -> None:
        time.sleep(self._retry_delay + random.uniform(0, self._retry_jitter))

    def _refresh(self) -> bool:
        if not self._client_id or not self._client_secret:
            return False
        try:
            r = self._http.post(
                _SPOTIFY_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self._search_timeout,
            )
            r.raise_for_status()
            p = r.json()
        except Exception:
            return False
        self._token = p.get("access_token")
        self._expires_at = time.time() + max(
            0, int(p.get("expires_in", 3600)) - self._token_skew
        )
        return bool(self._token)

    def search_id(self, artist: str, track: str) -> str | None:
        """Return Spotify track ID, None on clean no-match, or raise TransientError."""
        if not track.strip():
            return None
        q = f"track:{track.strip()}"
        if artist.strip():
            q += f" artist:{artist.strip()}"

        last_exc: Exception | None = None
        for attempt in range(self._max_attempts):
            if (
                not self._token or time.time() >= self._expires_at
            ) and not self._refresh():
                last_exc = TransientError("token refresh failed")
                if attempt < self._max_attempts - 1:
                    self._sleep()
                continue
            try:
                r = self._http.get(
                    _SPOTIFY_SEARCH_URL,
                    params={"q": q, "type": "track", "limit": self._search_limit},
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=self._search_timeout,
                )
                r.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if _is_transient_http(exc):
                    last_exc = exc
                    if attempt < self._max_attempts - 1:
                        self._sleep()
                    continue
                return None  # permanent 4xx → clean no-match
            except Exception as exc:
                if _is_transient_http(exc):
                    last_exc = exc
                    if attempt < self._max_attempts - 1:
                        self._sleep()
                    continue
                return None
            items = r.json().get("tracks", {}).get("items", [])
            return items[0].get("id") if items else None

        raise TransientError(f"spotify search exhausted: {last_exc!r}")


# ── ReccoBeats ────────────────────────────────────────────────────────────────

_RB_TRACK_URL = "https://api.reccobeats.com/v1/track"
_RB_FEATURES_URL = "https://api.reccobeats.com/v1/audio-features"
_RB_ANALYSIS_URL = "https://api.reccobeats.com/v1/analysis/audio-features"

# on_batch(batch_idx, total_batches, matched_in_batch)
OnBatch = Callable[[int, int, int], None]


class ReccoBeatsClient:
    """Batched ReccoBeats client: track ID lookup, catalog features, upload analysis."""

    def __init__(
        self,
        http: httpx.Client,
        batch: int = 20,
        delay_min: float = 2.0,
        delay_max: float = 3.0,
        timeout: float = 20.0,
    ):
        self._http = http
        self._batch = batch
        self._delay_min = delay_min
        self._delay_max = delay_max
        self._timeout = timeout

    def _sleep(self) -> None:
        time.sleep(random.uniform(self._delay_min, self._delay_max))

    def get_ids(
        self, spotify_ids: list[str], on_batch: OnBatch | None = None
    ) -> dict[str, str]:
        """Return {spotify_id: reccobeats_id} for all matched tracks."""
        out: dict[str, str] = {}
        batches = list(chunked(spotify_ids, self._batch))
        for i, batch in enumerate(batches, 1):
            matched = 0
            try:
                r = self._http.get(
                    _RB_TRACK_URL,
                    params={"ids": ",".join(batch)},
                    timeout=self._timeout,
                )
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

    def get_features(
        self, rb_ids: list[str], on_batch: OnBatch | None = None
    ) -> dict[str, dict]:
        """Return {reccobeats_id: feature_payload} for matched tracks."""
        out: dict[str, dict] = {}
        batches = list(chunked(rb_ids, self._batch))
        for i, batch in enumerate(batches, 1):
            matched = 0
            try:
                r = self._http.get(
                    _RB_FEATURES_URL,
                    params={"ids": ",".join(batch)},
                    timeout=self._timeout,
                )
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

    def upload_features(self, audio: Path) -> dict | None:
        """POST a short audio file to the analysis endpoint; return feature payload or None."""
        mime = "audio/wav" if audio.suffix.lower() == ".wav" else "audio/mpeg"
        try:
            with audio.open("rb") as fh:
                r = self._http.post(
                    _RB_ANALYSIS_URL,
                    files={"audioFile": (audio.name, fh, mime)},
                    timeout=self._timeout,
                )
                r.raise_for_status()
                p = r.json()
        except Exception:
            return None
        return p if isinstance(p, dict) else None
