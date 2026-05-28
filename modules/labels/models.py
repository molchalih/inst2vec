"""Thin wrapper exposing the lazy Qwen3-VL-Instruct singleton in the
project's own namespace. Keeps the vendor adapter import-isolated to a
single file the rest of the stage doesn't need to know about.
"""

from __future__ import annotations

from pathlib import Path

from core.config import LabelsSettings


class LabelsGenerator:
    """Lazy local-GPU labeller. Instantiate via ``LabelsGenerator.lazy``."""

    def __init__(self, labels: LabelsSettings) -> None:
        self._labels = labels
        self._impl = None

    @classmethod
    def lazy(cls, labels: LabelsSettings) -> LabelsGenerator:
        return cls(labels)

    def _ensure_impl(self):
        if self._impl is None:
            from core.vendor.qwen3_vl_instruct import Qwen3VLInstructGenerator

            self._impl = Qwen3VLInstructGenerator.load_once(
                self._labels.model_path,
                frame_count=self._labels.frame_count,
                max_new_tokens=self._labels.max_new_tokens,
                generation_seed=self._labels.generation_seed,
            )
        return self._impl

    def run(self, video_path: str | Path, prompt: str) -> str:
        return self._ensure_impl().run(str(video_path), prompt)

    def run_many(self, video_paths: list[str | Path], prompt: str) -> list[str]:
        """Batched counterpart to ``run`` — same prompt, multiple videos.

        Returns one decoded string per input path. See
        ``Qwen3VLInstructGenerator.run_many`` for determinism caveats.
        """
        return self._ensure_impl().run_many([str(p) for p in video_paths], prompt)

    def run_text(self, prompt: str, *, max_new_tokens: int) -> str:
        return self._ensure_impl().run_text(prompt, max_new_tokens=max_new_tokens)

    def unload(self) -> None:
        if self._impl is not None:
            from core.vendor.qwen3_vl_instruct import Qwen3VLInstructGenerator

            Qwen3VLInstructGenerator.unload()
            self._impl = None
