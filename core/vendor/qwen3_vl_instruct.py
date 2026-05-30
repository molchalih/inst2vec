"""Qwen3-VL-Instruct adapter — frames + prompt → raw text.

Lazy-loaded singleton. Mirrors ``core/vendor/qwen3_vl_embedding.py``:
- ``Qwen3VLInstructGenerator.load_once(model_path, ...)`` instantiates the
  HF model + processor on first call.
- ``.run(video_path, prompt) -> str`` returns the raw decoded model output
  for a single video.
- ``.unload()`` releases the model + CUDA cache.

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
from transformers import AutoModelForImageTextToText, AutoProcessor
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


@dataclass
class _Loaded:
    model: object
    processor: object
    device: str


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
    ) -> "Qwen3VLInstructGenerator":
        if cls._loaded is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                # Enable TF32 for the fp32 matmuls (norm/rotary). Cheap, no
                # numerical surprises at this scale.
                torch.set_float32_matmul_precision("high")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            _ensure_local_model(model_path)
            processor = AutoProcessor.from_pretrained(model_path)
            # Qwen3-VL is image-text-to-text (vision-language conditional
            # generation); AutoModelForCausalLM does not register a class
            # for the `qwen3_vl` config in transformers>=5.8.
            # FA2 only on CUDA + bf16/fp16; fall back to SDPA on CPU.
            attn_impl = "flash_attention_2" if device == "cuda" else "sdpa"
            model = (
                AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    attn_implementation=attn_impl,
                )
                .to(device)
                .eval()
            )
            cls._loaded = _Loaded(model=model, processor=processor, device=device)
        self = cls()
        self.frame_count = frame_count
        self.max_new_tokens = max_new_tokens
        self.generation_seed = generation_seed
        return self

    @classmethod
    def unload(cls) -> None:
        cls._loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def reclaim_memory(cls) -> None:
        """Release cached allocator blocks without unloading weights.

        Cluster-pass generations build up large KV caches and prefill
        tensors. After each generation the tensors are freed but the
        CUDA allocator holds the blocks, fragmenting VRAM over the
        course of a per-case loop. Call between generations to give the
        allocator back contiguous space.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        """Text-only generation — no vision branch, no `process_vision_info`.

        ``seed`` overrides the instance's default ``generation_seed``
        for this single call. ``do_sample=True`` switches from greedy
        to nucleus sampling — required for ``seed`` to actually affect
        the output (greedy decoding never consults the RNG, so different
        seeds produce identical outputs). The cluster pass uses
        sampling so per-attempt seed variation can recover from
        validation failures; everything else stays greedy for full
        determinism.
        """
        assert self._loaded is not None, _NOT_LOADED
        torch.manual_seed(seed if seed is not None else self.generation_seed)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        text = self._loaded.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._loaded.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self._loaded.device)
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            # Greedy decoding on long structured JSON outputs is prone
            # to repetition loops (the model emits near-identical array
            # entries until ``max_new_tokens`` is hit). Block any
            # 10-gram from appearing twice. Vision branches (``run`` /
            # ``run_many``) keep plain greedy because their outputs are
            # short and the n-gram block can hurt observable-tag arrays
            # where short repeats are valid.
            "no_repeat_ngram_size": 10,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        with torch.inference_mode():
            out = self._loaded.model.generate(**inputs, **gen_kwargs)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self._loaded.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def run(self, video_path: str, prompt: str) -> str:
        assert self._loaded is not None, _NOT_LOADED
        torch.manual_seed(self.generation_seed)
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
        text = self._loaded.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = self._loaded.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(self._loaded.device)
        with torch.inference_mode():
            out = self._loaded.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        decoded = self._loaded.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return decoded

    def prepare_many(self, video_paths: list[str], prompt: str):
        """CPU-only batch prep: decode frames, tokenize, pad. Returns a
        processor BatchFeature on CPU (or ``None`` for an empty input).
        Split off from ``run_many`` so callers can overlap this on a
        background thread while the GPU runs the previous batch.
        """
        assert self._loaded is not None, _NOT_LOADED
        if not video_paths:
            return None
        all_messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(vp),
                            "max_frames": self.frame_count,
                            "fps": 1,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for vp in video_paths
        ]
        texts = [
            self._loaded.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in all_messages
        ]
        flat_videos = []
        for m in all_messages:
            _images, videos = process_vision_info(m)
            flat_videos.append(videos[0])
        # Decoder LMs require left-padding for batched generation so the
        # first ``new`` token lives at column ``prefill``. Right-padding
        # leaves padding tokens between the prefill and the generation
        # head, corrupting the attention pattern.
        self._loaded.processor.tokenizer.padding_side = "left"
        return self._loaded.processor(
            text=texts,
            images=None,
            videos=flat_videos,
            padding=True,
            return_tensors="pt",
        )

    def generate_from_inputs(self, inputs) -> list[str]:
        """GPU half of ``run_many``. Pairs with ``prepare_many``."""
        assert self._loaded is not None, _NOT_LOADED
        if inputs is None:
            return []
        torch.manual_seed(self.generation_seed)
        inputs = inputs.to(self._loaded.device)
        with torch.inference_mode():
            out = self._loaded.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prefix = inputs["input_ids"].shape[1]
        trimmed = out[:, prefix:]
        decoded = self._loaded.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return list(decoded)

    def run_many(self, video_paths: list[str], prompt: str) -> list[str]:
        """Batched video generation. Same prompt, N video paths.

        Returns the decoded strings aligned with ``video_paths``. Greedy
        decoding (``do_sample=False``), left-padding for autoregressive
        decoder compatibility. Output is NOT byte-identical to per-clip
        ``run`` results — bf16 numerical noise diverges greedy paths once
        logits get close — but JSON-schema validity is unaffected.
        """
        return self.generate_from_inputs(self.prepare_many(video_paths, prompt))
