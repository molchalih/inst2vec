"""FastAPI app for the embedder service.

Endpoints:
  GET  /healthz   — liveness + model status
  POST /embed     — single-clip embedding

Auth: bearer token from `EMBEDDER_TOKEN` env. Single shared secret.

The service reuses ``modules.embeddings.providers.LocalQwenProvider``
and ``modules.embeddings.cases.CASE_REGISTRY[case].payload_builder``
so the dict handed to the model is bit-for-bit identical to a local
pipeline run.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
import urllib.request
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from modules.embeddings.cases import CASE_REGISTRY
from modules.embeddings.providers import LocalQwenProvider
from modules.embeddings.sampling import is_token_mismatch_error

# ── module-level state (constructed at startup, swappable in tests) ─────────

_token: str = ""
_provider: LocalQwenProvider | None = None


def _reset_for_tests(token: str) -> None:
    """Test helper: clear cached provider and set the auth token."""
    global _token, _provider
    _token = token
    _provider = None


def _get_provider() -> LocalQwenProvider:
    """Return the singleton model provider; load on first call."""
    global _provider
    if _provider is None:
        _provider = LocalQwenProvider(
            model_path=os.environ.get(
                "MODEL_PATH", "/workspace/models/Qwen3-VL-Embedding-8B"
            ),
            max_length=int(os.environ.get("EMBED_MAX_LENGTH", "32768")),
        )
    return _provider


def _resolve_video_url(url: str) -> str:
    """Download `url` to a temp file; return the local path."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path


# ── request / response models ───────────────────────────────────────────────


class EmbedRequest(BaseModel):
    case: Literal["video", "sandwich", "audio"]
    clip_id: int
    video_url: str | None = None
    text: str | None = None
    instruction: str | None = None
    fps: float | None = None
    max_frames: int | None = None


class EmbedResponse(BaseModel):
    embedding: list[float]
    dim: int
    took_ms: int


# ── app ─────────────────────────────────────────────────────────────────────


app = FastAPI(title="inst2vec embedder")


def _check_auth(authorization: str | None = Header(default=None)) -> None:
    token = os.environ.get("EMBEDDER_TOKEN", _token)
    if not token:
        raise HTTPException(status_code=500, detail="EMBEDDER_TOKEN not set")
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "model_loaded": _provider is not None,
        "gpu": os.environ.get("GPU_LABEL", "unknown"),
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, _: None = Depends(_check_auth)) -> EmbedResponse:
    t0 = time.monotonic()

    spec = CASE_REGISTRY[req.case]

    local_video_path: str | None = None
    if req.video_url:
        local_video_path = _resolve_video_url(req.video_url)

    try:
        payload = spec.payload_builder(
            None,
            req.text,
            local_video_path,
            req.fps,
            req.max_frames,
        )
        try:
            out = _get_provider().embed(payload)
        except Exception as e:
            if is_token_mismatch_error(e):
                return JSONResponse(
                    status_code=422,
                    content={"error": "token_mismatch", "detail": str(e)},
                )
            raise
    finally:
        if local_video_path and os.path.exists(local_video_path):
            with contextlib.suppress(OSError):
                os.remove(local_video_path)

    took_ms = int((time.monotonic() - t0) * 1000)
    vec = list(out[0])
    return EmbedResponse(embedding=vec, dim=len(vec), took_ms=took_ms)
