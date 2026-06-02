"""vLLM-backed Qwen3-VL-Instruct generator — video/text + prompt → raw text.

Lazy-loaded singleton mirroring ``core/vendor/qwen3_text.py`` but for the
vision-language Instruct model. Runs vLLM's in-process offline ``LLM`` (no
server, no docker) with optional structured-output JSON decoding, and exposes a
batched ``run_many`` so the per-clip pass can tag many clips in one scheduling
pass (vLLM's continuous batching beats the per-clip HF generate path).

The HF ``AutoModelForImageTextToText`` + ``torch.inference_mode`` decode path is
gone — vLLM ingests the checkpoint directly and batches internally, so the old
``prepare_many`` / ``generate_from_inputs`` split is no longer needed.

Structured-output API name moved across vLLM versions
(``StructuredOutputsParams`` in vLLM >= 0.11, ``GuidedDecodingParams`` before),
so it is resolved at load time.

Note: ``qwen_vl_utils.process_vision_info``'s ``return_video_kwargs`` argument
does not exist in all versions of the package, so we unpack its return value
defensively (it may yield 2 or 3 values) rather than hard-depending on a
specific signature.

Excluded from ruff/ty via the ``core/vendor`` directory exclude.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from qwen_vl_utils.vision_process import process_vision_info

_NOT_LOADED = "load_once must be called first"


def _ensure_local_model(model_path: str) -> None:
    """Materialize ``model_path`` from the HF hub when the directory is empty.

    HF ``from_pretrained`` rejects bare paths like ``./models/foo`` when the
    folder is missing — it falls back to repo-id validation. Derive the repo id
    from the basename (``Qwen/<basename>``) and snapshot the weights into the
    requested location so later runs are zero-cost.
    """
    path = Path(model_path)
    if path.is_dir() and any(path.iterdir()):
        return
    repo_id = f"Qwen/{path.name}"
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(path),
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
    )


def _vision_video_item(messages):
    """Return ``(video_item, video_kwargs)`` for one video message.

    vLLM's Qwen3-VL multimodal parser requires per-video METADATA: each video
    entry in ``multi_modal_data`` must be a ``(frames, metadata)`` tuple, not a
    bare array (a bare array yields ``metadata=None`` and the parser raises
    "Video metadata is required but not found in mm input"). ``qwen_vl_utils``
    produces that tuple when called with ``return_video_metadata=True``;
    ``return_video_kwargs=True`` additionally yields processor kwargs (fps, …)
    that we forward via ``mm_processor_kwargs``.
    """
    _images, videos, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True
    )
    item = videos[0] if videos else None
    return item, (video_kwargs or None)


@dataclass
class _Loaded:
    llm: object  # vllm.LLM
    processor: object
    so_factory: object  # builds the structured-output params object from a schema
    so_kwarg: str  # SamplingParams kwarg name for that object


class Qwen3VLInstructGenerator:
    _loaded: Optional[_Loaded] = None

    def __init__(self):
        self.frame_count: int = 0
        self.max_new_tokens: int = 0
        self.generation_seed: int = 0

    @classmethod
    def load_once(
        cls,
        model_path: str,
        *,
        frame_count: int,
        max_new_tokens: int,
        generation_seed: int,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 32768,
        enforce_eager: bool = True,
    ) -> "Qwen3VLInstructGenerator":
        if cls._loaded is None:
            import vllm.sampling_params as svp
            from vllm import LLM

            _ensure_local_model(model_path)
            processor = AutoProcessor.from_pretrained(model_path)
            llm = LLM(
                model=model_path,
                dtype="bfloat16",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                limit_mm_per_prompt={"image": 0, "video": 1},
            )
            if hasattr(svp, "StructuredOutputsParams"):
                so_factory = lambda schema: svp.StructuredOutputsParams(json=schema)  # noqa: E731
                so_kwarg = "structured_outputs"
            else:
                so_factory = lambda schema: svp.GuidedDecodingParams(  # noqa: E731
                    json=schema, backend="xgrammar"
                )
                so_kwarg = "guided_decoding"
            cls._loaded = _Loaded(
                llm=llm, processor=processor, so_factory=so_factory, so_kwarg=so_kwarg
            )
        self = cls()
        self.frame_count = frame_count
        self.max_new_tokens = max_new_tokens
        self.generation_seed = generation_seed
        return self

    @classmethod
    def unload(cls) -> None:
        cls._loaded = None
        gc.collect()
        # Best-effort vLLM distributed teardown so a later model can load.
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def reclaim_memory(cls) -> None:
        """Release cached allocator blocks without unloading weights."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sampling(self, *, max_new_tokens, seed, do_sample, temperature, top_p, schema):
        from vllm import SamplingParams

        ld = self._loaded
        kwargs: dict = {"max_tokens": max_new_tokens, "seed": seed}
        # do_sample=False -> deterministic greedy regardless of the checkpoint's
        # generation_config defaults; sampling pins temperature/top_p.
        kwargs["temperature"] = temperature if do_sample else 0.0
        if do_sample:
            kwargs["top_p"] = top_p
        if schema is not None:
            kwargs[ld.so_kwarg] = ld.so_factory(schema)
        return SamplingParams(**kwargs)

    def _video_prompt(self, video_path: str, prompt: str):
        """Build ``(chat_text, video_item, video_kwargs)`` for one request.

        ``max_frames`` + ``fps=1`` caps the sampled frame budget to
        ``frame_count`` (mirroring the prior HF behaviour). ``video_item`` is
        the ``(frames, metadata)`` tuple vLLM's Qwen3-VL parser requires.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "max_frames": self.frame_count,
                        "fps": 1,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self._loaded.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        video_item, video_kwargs = _vision_video_item(messages)
        return chat_text, video_item, video_kwargs

    def run(self, video_path: str, prompt: str, *, schema: dict | None = None) -> str:
        return self.run_many([video_path], prompt, schema=schema)[0]

    def run_many(
        self, video_paths: list[str], prompt: str, *, schema: dict | None = None
    ) -> list[str]:
        """Batched video generation — same prompt, N video paths.

        One vLLM request per video; greedy decoding (``do_sample=False``) with
        the configured base seed. Returns decoded strings aligned with
        ``video_paths``; vLLM returns outputs in input order.
        """
        assert self._loaded is not None, _NOT_LOADED
        if not video_paths:
            return []
        reqs = []
        for vp in video_paths:
            chat_text, video_item, video_kwargs = self._video_prompt(vp, prompt)
            req = {"prompt": chat_text, "multi_modal_data": {"video": video_item}}
            if video_kwargs:
                req["mm_processor_kwargs"] = video_kwargs
            reqs.append(req)
        sampling = self._sampling(
            max_new_tokens=self.max_new_tokens,
            seed=self.generation_seed,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            schema=schema,
        )
        outputs = self._loaded.llm.generate(reqs, sampling, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def _chat_text(self, prompt: str) -> str:
        return self._loaded.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def run_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        schema: dict | None = None,
    ) -> str:
        return self.run_text_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            seeds=[seed],
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            schema=schema,
        )[0]

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
        """Text-only generation — one completion per prompt in a single pass.

        ``seeds`` is per-prompt (``None`` -> the configured base seed). vLLM
        returns outputs in input order.
        """
        assert self._loaded is not None, _NOT_LOADED
        assert len(prompts) == len(seeds), "prompts and seeds length mismatch"
        texts = [self._chat_text(p) for p in prompts]
        sampling = [
            self._sampling(
                max_new_tokens=max_new_tokens,
                seed=(s if s is not None else self.generation_seed),
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                schema=schema,
            )
            for s in seeds
        ]
        outputs = self._loaded.llm.generate(texts, sampling, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]
