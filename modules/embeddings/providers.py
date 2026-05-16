"""Embedding providers.

A provider is a thin wrapper around an underlying embedding model that
exposes a uniform ``.embed(payload: dict) -> list`` interface. Providers
are not interchangeable backends for the same ``embedding_case`` — each
case in the registry binds to one provider configuration, because
different providers produce incompatible vector spaces.
"""

from __future__ import annotations

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
