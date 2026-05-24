"""Embedding providers.

A provider is a thin wrapper around an underlying embedding model that
exposes a uniform ``.embed(payload: dict) -> list`` interface. Providers
are not interchangeable backends for the same ``embedding_case`` — each
case in the registry binds to one provider configuration, because
different providers produce incompatible vector spaces.
"""

from __future__ import annotations

import contextlib
import io
from typing import Protocol


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
        from qwen_vl_utils.vision_process import get_video_reader_backend

        from core.vendor.qwen3_vl_embedding import Qwen3VLEmbedder

        # qwen-vl-utils prints "qwen-vl-utils using <backend> to read video."
        # to stderr the first time `get_video_reader_backend()` runs; the
        # function is `lru_cache(maxsize=1)`, so pre-warming it with stderr
        # captured here means subsequent decodes hit the cache and never
        # reach our log stream.
        with contextlib.redirect_stderr(io.StringIO()):
            get_video_reader_backend()

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
