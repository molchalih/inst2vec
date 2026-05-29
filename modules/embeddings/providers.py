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
import threading
from collections.abc import Callable
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

    def embed_batch(self, payloads: list[dict]):
        """Coalesced multi-payload forward. Returns a (N, dim) embeddings
        tensor aligned with ``payloads``. The Qwen3VLEmbedder already
        accepts a list of payloads natively (left-padded across the batch
        with the right padding side for attention-mask-aware pooling);
        this just exposes that path to the worker. Caller must ensure all
        payloads route through this same provider (one case = one
        provider, enforced by the worker grouping)."""
        return self._model.process(payloads)


class ProviderRouter:
    """Provider that dispatches ``embed(payload)`` to a per-case backend.

    Backends are built lazily on first use (double-checked locking) so a
    case with no work never loads its model, and concurrent ``inflight``
    lanes don't race the build. The actual ``embed`` runs outside the lock.
    """

    def __init__(self, factories: dict[str, Callable[[], Provider]]) -> None:
        self._factories = factories
        self._instances: dict[str, Provider] = {}
        self._lock = threading.Lock()

    def _resolve(self, case: str) -> Provider:
        prov = self._instances.get(case)
        if prov is None:
            with self._lock:
                prov = self._instances.get(case)
                if prov is None:
                    prov = self._factories[case]()
                    self._instances[case] = prov
        return prov

    def embed(self, payload: dict):
        return self._resolve(payload["case"]).embed(payload)

    def embed_batch(self, payloads: list[dict]):
        if not payloads:
            return []
        case = payloads[0]["case"]
        assert all(p["case"] == case for p in payloads), (
            "ProviderRouter.embed_batch requires a single shared case"
        )
        prov = self._resolve(case)
        fn = getattr(prov, "embed_batch", None)
        if fn is None:
            # Backend doesn't expose a batched path (e.g. HTTP/ONNX); fall
            # back to single-payload calls. Result shape stays aligned with
            # ``payloads`` so the worker can index by position.
            return [prov.embed(p)[0] for p in payloads]
        return fn(payloads)
