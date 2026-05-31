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

    def prepare_many(self, video_paths: list[str | Path], prompt: str):
        """CPU half of ``run_many`` — frame decode + tokenize. Returns a
        processor BatchFeature (CPU). Pairs with ``generate_from_inputs``
        to overlap CPU prep with GPU generation across batches.
        """
        return self._ensure_impl().prepare_many([str(p) for p in video_paths], prompt)

    def generate_from_inputs(self, inputs) -> list[str]:
        """GPU half of ``run_many``. Pairs with ``prepare_many``."""
        return self._ensure_impl().generate_from_inputs(inputs)

    def run_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        return self._ensure_impl().run_text(
            prompt,
            max_new_tokens=max_new_tokens,
            seed=seed,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

    def unload(self) -> None:
        if self._impl is not None:
            from core.vendor.qwen3_vl_instruct import Qwen3VLInstructGenerator

            Qwen3VLInstructGenerator.unload()
            self._impl = None

    def reclaim_memory(self) -> None:
        if self._impl is not None:
            from core.vendor.qwen3_vl_instruct import Qwen3VLInstructGenerator

            Qwen3VLInstructGenerator.reclaim_memory()


class ClusterLabelsGenerator:
    """Lazy local-GPU cluster labeller — Qwen3-30B-A3B Int4 + schema decoding."""

    def __init__(self, labels: LabelsSettings) -> None:
        self._labels = labels
        self._impl = None

    @classmethod
    def lazy(cls, labels: LabelsSettings) -> ClusterLabelsGenerator:
        return cls(labels)

    def _ensure_impl(self):
        if self._impl is None:
            from core.vendor.qwen3_text import Qwen3TextGenerator

            self._impl = Qwen3TextGenerator.load_once(
                self._labels.cluster_model_path,
                max_new_tokens=self._labels.cluster_max_new_tokens,
                generation_seed=self._labels.generation_seed,
                gpu_memory_utilization=self._labels.cluster_gpu_memory_utilization,
                max_model_len=self._labels.cluster_max_model_len,
                enforce_eager=self._labels.cluster_enforce_eager,
            )
        return self._impl

    def run_text_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        seeds: list[int | None],
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        schema: dict | None = None,
    ) -> list[str]:
        return self._ensure_impl().run_text_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            seeds=seeds,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            schema=schema,
        )

    def unload(self) -> None:
        if self._impl is not None:
            from core.vendor.qwen3_text import Qwen3TextGenerator

            Qwen3TextGenerator.unload()
            self._impl = None

    def reclaim_memory(self) -> None:
        if self._impl is not None:
            from core.vendor.qwen3_text import Qwen3TextGenerator

            Qwen3TextGenerator.reclaim_memory()
