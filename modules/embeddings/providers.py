"""Embedding providers.

A provider is a thin wrapper around an underlying embedding model that
exposes a uniform ``.embed(payload: dict) -> list`` interface. Providers
are not interchangeable backends for the same ``embedding_case`` — each
case in the registry binds to one provider configuration, because
different providers produce incompatible vector spaces.
"""

from __future__ import annotations

import time
from typing import Protocol

import httpx


class Provider(Protocol):
    def embed(self, payload: dict): ...


class LocalQwenProvider:
    """Wraps the local Qwen3VLEmbedder behind the Provider protocol."""

    def __init__(
        self,
        *,
        model_path: str,
        max_length: int,
        max_frames: int | None = None,
        fps: float | None = None,
    ) -> None:
        from modules.external.qwen3_vl_embedding import Qwen3VLEmbedder

        kwargs: dict = {
            "model_name_or_path": model_path,
            "max_length": max_length,
        }
        if max_frames is not None:
            kwargs["max_frames"] = max_frames
        if fps is not None:
            kwargs["fps"] = fps
        self._model = Qwen3VLEmbedder(**kwargs)

    def embed(self, payload: dict):
        """Process a single payload and return the model's output list."""
        return self._model.process([payload])


# Used to bridge the pod's structured `token_mismatch` JSON error into a
# Python exception whose `str()` satisfies the runner's existing predicate
# `modules.embeddings.sampling.is_token_mismatch_error`.
_TOKEN_MISMATCH_MARKER = "Mismatch in `video` token count"


class RemoteEmbedError(RuntimeError):
    """Raised when the GPU pod returns a non-recoverable error."""


class RemoteQwenProvider:
    """HTTP client for the embedder pod (`services/embedder/`).

    Translates local-style payloads to remote payloads (replacing
    ``"video": <local-path>`` with ``"video_url": <signed-url>``),
    posts to ``POST /embed``, retries transient 5xx/timeout, and
    refreshes signed URLs on expiry.
    """

    def __init__(
        self,
        *,
        url: str,
        token: str,
        storage,
        timeout_s: int,
        max_retries: int,
        _client: httpx.Client | None = None,
    ) -> None:
        self._url = url.rstrip("/") + "/embed"
        self._headers = {"Authorization": f"Bearer {token}"}
        self._storage = storage
        self._timeout = timeout_s
        self._max_retries = max_retries
        self._client = _client or httpx.Client(timeout=timeout_s)

    def embed(self, payload: dict):
        clip_id = payload["clip_id"]
        body, signed_key = self._build_body(payload, clip_id)
        attempts = 0
        signed_refreshes = 0

        while True:
            attempts += 1
            try:
                resp = self._client.post(
                    self._url, headers=self._headers, json=body, timeout=self._timeout
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempts > self._max_retries:
                    raise RemoteEmbedError(f"transport error: {e}") from e
                time.sleep(_backoff(attempts))
                continue

            if 200 <= resp.status_code < 300:
                data = resp.json()
                return [data["embedding"]]

            err = _safe_error(resp)

            if err == "signed_url_expired" and signed_key and signed_refreshes == 0:
                signed_refreshes += 1
                body["video_url"] = self._storage.signed_get(signed_key)
                continue

            if err == "token_mismatch":
                detail = (resp.json() or {}).get("detail", "")
                raise RemoteEmbedError(f"{_TOKEN_MISMATCH_MARKER}: {detail}")

            if resp.status_code >= 500:
                if attempts > self._max_retries:
                    raise RemoteEmbedError(
                        f"5xx after {attempts} attempt(s): {resp.status_code} {err}"
                    )
                time.sleep(_backoff(attempts))
                continue

            raise RemoteEmbedError(f"{resp.status_code}: {err}")

    # ── internals ────────────────────────────────────────────────────────────

    def _build_body(self, payload: dict, clip_id: int) -> tuple[dict, str | None]:
        body = dict(payload)
        signed_key: str | None = None
        if "video" in body:
            body.pop("video")
            signed_key = self._storage.key_for_clip(clip_id)
            body["video_url"] = self._storage.signed_get(signed_key)
        return body, signed_key


def _backoff(attempt: int) -> float:
    return min(2 ** (attempt - 1), 30)


def _safe_error(resp: httpx.Response) -> str:
    try:
        return (resp.json() or {}).get("error", "")
    except Exception:
        return resp.text[:200]
