"""Gemini Embedding 2 multimodal provider.

Single-call multimodal embedding: text + video + audio → one vector.
``google.genai`` is imported lazily inside ``__init__`` so disabled
installs do not require the optional dependency to be present.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from modules.embeddings.sampling import probe_duration_seconds


class GeminiClipTooLongError(Exception):
    """Raised when video or audio exceeds configured caps. Pre-upload."""


class GeminiOutputDimMismatch(Exception):
    """Returned vector length disagrees with configured output_dim."""


@dataclass(frozen=True)
class GeminiSecrets:
    api_key: str


def _is_retriable(exc: Exception) -> bool:
    code = getattr(exc, "status_code", None)
    if code is None:
        # Fall back to checking common transient keywords.
        text = str(exc).lower()
        return any(
            s in text
            for s in ("timeout", "temporarily", "unavailable", "deadline", "reset")
        )
    return code == 429 or 500 <= int(code) < 600


def _retry(call, *, max_retries: int, base_delay: float = 1.0, max_delay: float = 60.0):
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:
            if not _is_retriable(exc) or attempt == max_retries:
                raise
            last = exc
            delay = min(max_delay, base_delay * (2**attempt))
            time.sleep(delay)
    raise last  # unreachable


class GeminiMultimodalProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        output_dim: int,
        max_video_seconds: int,
        max_audio_seconds: int,
        request_timeout_s: int,
        max_retries: int,
        client: object | None = None,  # test injection
    ) -> None:
        self.model = model
        self.output_dim = output_dim
        self.max_video_seconds = max_video_seconds
        self.max_audio_seconds = max_audio_seconds
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries

        if client is not None:
            self._client = client
            return
        from google import genai  # lazy: only when actually used

        self._client = genai.Client(api_key=api_key)

    def embed(self, payload: dict) -> list[list[float]]:
        """Embed one clip. Returns ``[vector]`` (single-element list)."""
        video_path = payload["video_path"]
        audio_path = payload["audio_path"]
        text = payload["text"]

        v_dur = probe_duration_seconds(video_path, strict=True)
        if v_dur > self.max_video_seconds:
            raise GeminiClipTooLongError(
                f"video {v_dur:.1f}s > cap {self.max_video_seconds}s"
            )
        a_dur = probe_duration_seconds(audio_path, strict=True)
        if a_dur > self.max_audio_seconds:
            raise GeminiClipTooLongError(
                f"audio {a_dur:.1f}s > cap {self.max_audio_seconds}s"
            )

        return self._upload_and_embed(video_path, audio_path, text)

    def _upload_and_embed(self, video_path, audio_path, text):
        import torch  # local import is fine; torch is a project dep

        t0 = time.time()
        video_file = self._client.files.upload(file=video_path)
        audio_file = self._client.files.upload(file=audio_path)

        contents, config = self._build_request(text, video_file, audio_file)
        response = _retry(
            lambda: self._client.models.embed_content(
                model=self.model, contents=contents, config=config
            ),
            max_retries=self.max_retries,
        )

        values = list(response.embeddings[0].values)
        if len(values) != self.output_dim:
            raise GeminiOutputDimMismatch(
                f"expected {self.output_dim}-d vector, got {len(values)}-d"
            )
        vector = torch.tensor(values, dtype=torch.float32)

        # Best-effort observability; never the cause of a failure.
        try:
            elapsed = time.time() - t0
            bytes_up = os.path.getsize(video_path) + os.path.getsize(audio_path)
            print(f"[gemini] bytes_uploaded={bytes_up} embed_seconds={elapsed:.2f}")
        except OSError:
            pass

        return [vector]

    def _build_request(self, text, video_file, audio_file):
        from google.genai import types

        return (
            [
                text,
                types.Part.from_uri(
                    file_uri=video_file.uri,
                    mime_type=video_file.mime_type or "video/mp4",
                ),
                types.Part.from_uri(
                    file_uri=audio_file.uri,
                    mime_type=audio_file.mime_type or "audio/mpeg",
                ),
            ],
            types.EmbedContentConfig(output_dimensionality=self.output_dim),
        )
