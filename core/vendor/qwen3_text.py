"""Qwen3-30B-A3B-GPTQ-Int4 cluster-label generator, backed by vLLM (offline).

Lazy-loaded singleton mirroring ``core/vendor/qwen3_vl_instruct.py`` but for the
text-only MoE cluster model. Runs vLLM's in-process offline ``LLM`` (no server,
no docker) with structured-output JSON decoding so output is schema-valid, and
exposes a batched ``run_text_batch`` so the cluster pass can label many clusters
in one scheduling pass (vLLM's fused MoE + continuous batching is ~20-40x the
per-expert HF path). Excluded from ruff/ty via the ``core/vendor`` exclude.

Why vLLM and not transformers+gptqmodel: this checkpoint stores unfused
per-expert weights that transformers' ``Qwen3MoeForCausalLM`` mis-loads, and the
per-expert HF decode is launch-overhead-bound (~3.6 tok/s). vLLM ingests the
checkpoint directly and runs fused Marlin MoE kernels with overlapped structured
decoding. Structured-output API name moved across versions
(``StructuredOutputsParams`` in vLLM >= 0.11, ``GuidedDecodingParams`` before),
so it is resolved at load time.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def _ensure_local_model(model_path: str) -> str:
    path = Path(model_path)
    if path.is_dir() and any(path.iterdir()):
        return model_path
    repo_id = f"Qwen/{path.name}"
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id, local_dir=str(path),
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
    )
    return model_path


@dataclass
class _Loaded:
    llm: object  # vllm.LLM
    tokenizer: object
    so_factory: object  # builds the structured-output params object from a schema
    so_kwarg: str  # SamplingParams kwarg name for that object


class Qwen3TextGenerator:
    _loaded: Optional[_Loaded] = None

    def __init__(self):
        self.max_new_tokens: int = 0
        self.generation_seed: int = 0

    @classmethod
    def load_once(
        cls,
        model_path: str,
        *,
        max_new_tokens: int,
        generation_seed: int,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 20480,
        enforce_eager: bool = True,
    ) -> "Qwen3TextGenerator":
        if cls._loaded is None:
            import vllm.sampling_params as svp
            from vllm import LLM

            model_path = _ensure_local_model(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            llm = LLM(
                model=model_path,
                dtype="float16",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
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
                llm=llm, tokenizer=tokenizer, so_factory=so_factory, so_kwarg=so_kwarg
            )
        self = cls()
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _sampling_params(self, *, max_new_tokens, seed, do_sample, temperature, top_p, schema):
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

    def _chat_text(self, prompt: str) -> str:
        return self._loaded.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    def run_text(self, prompt: str, *, max_new_tokens: int, seed: int | None = None,
                 do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0,
                 schema: dict | None = None) -> str:
        return self.run_text_batch(
            [prompt], max_new_tokens=max_new_tokens, seeds=[seed], do_sample=do_sample,
            temperature=temperature, top_p=top_p, schema=schema,
        )[0]

    def run_text_batch(self, prompts: list[str], *, max_new_tokens: int,
                       seeds: list[int | None], do_sample: bool = False,
                       temperature: float = 1.0, top_p: float = 1.0,
                       schema: dict | None = None) -> list[str]:
        """Generate one completion per prompt in a single vLLM scheduling pass.

        ``seeds`` is per-prompt (``None`` -> the configured base seed), so each
        cluster keeps its own seed-escalation across retries. vLLM returns
        outputs in input order.
        """
        assert self._loaded is not None, "load_once must be called first"
        assert len(prompts) == len(seeds), "prompts and seeds length mismatch"
        texts = [self._chat_text(p) for p in prompts]
        sampling = [
            self._sampling_params(
                max_new_tokens=max_new_tokens,
                seed=(s if s is not None else self.generation_seed),
                do_sample=do_sample, temperature=temperature, top_p=top_p, schema=schema,
            )
            for s in seeds
        ]
        outputs = self._loaded.llm.generate(texts, sampling, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]
