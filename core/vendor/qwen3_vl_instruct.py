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


def _ensure_local_model(model_path: str) -> str:
    """Materialize ``model_path`` from the HF hub when the directory is empty.

    HF ``from_pretrained`` rejects bare paths like ``./models/foo`` when the
    folder is missing — it falls back to repo-id validation. Derive the repo id
    from the basename (``Qwen/<basename>``) and snapshot the weights into the
    requested location so later runs are zero-cost.
    """
    path = Path(model_path)
    if path.is_dir() and any(path.iterdir()):
        return model_path
    repo_id = f"Qwen/{path.name}"
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(path),
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
    )
    return model_path


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
            model_path = _ensure_local_model(model_path)
            processor = AutoProcessor.from_pretrained(model_path)
            # Qwen3-VL is image-text-to-text (vision-language conditional
            # generation); AutoModelForCausalLM does not register a class
            # for the `qwen3_vl` config in transformers>=5.8.
            model = (
                AutoModelForImageTextToText.from_pretrained(
                    model_path,
                    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
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

    def run_text(self, prompt: str, *, max_new_tokens: int) -> str:
        """Text-only generation — no vision branch, no `process_vision_info`."""
        assert self._loaded is not None, "load_once must be called first"
        torch.manual_seed(self.generation_seed)
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
        with torch.inference_mode():
            out = self._loaded.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self._loaded.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def run(self, video_path: str, prompt: str) -> str:
        assert self._loaded is not None, "load_once must be called first"
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

    def run_many(self, video_paths: list[str], prompt: str) -> list[str]:
        """Batched video generation. Same prompt, N video paths.

        Returns the decoded strings aligned with ``video_paths``. Greedy
        decoding (``do_sample=False``), left-padding for autoregressive
        decoder compatibility. Output is NOT byte-identical to per-clip
        ``run`` results — bf16 numerical noise diverges greedy paths once
        logits get close — but JSON-schema validity is unaffected.
        """
        assert self._loaded is not None, "load_once must be called first"
        if not video_paths:
            return []
        torch.manual_seed(self.generation_seed)
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
        inputs = self._loaded.processor(
            text=texts,
            images=None,
            videos=flat_videos,
            padding=True,
            return_tensors="pt",
        ).to(self._loaded.device)
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
