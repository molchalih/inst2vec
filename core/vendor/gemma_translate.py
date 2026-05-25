"""Shared TranslateGemma pipeline wrapper for text translation."""
from __future__ import annotations

import os

import torch
from transformers import pipeline
from transformers.utils.logging import disable_progress_bar

# Silence the transformers "Loading weights" tqdm bar (added in transformers v5).
disable_progress_bar()


DEFAULT_MODEL_ID = os.environ.get("GEMMA_TRANSLATE_MODEL", "google/translategemma-4b-it")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("GEMMA_TRANSLATE_MAX_NEW_TOKENS", 200))


class GemmaTranslator:
    """Thin wrapper around HF image-text-to-text pipeline for TranslateGemma."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.model_id = model_id
        self.pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device=device,
            dtype=dtype,
        )

        # Keep generation config compatible with explicit max_new_tokens usage.
        if getattr(self.pipe.model, "generation_config", None) is not None:
            self.pipe.model.generation_config.max_length = None
            self.pipe.model.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id
        if getattr(self.pipe, "generation_config", None) is not None:
            self.pipe.generation_config.max_length = None
            self.pipe.generation_config.pad_token_id = self.pipe.tokenizer.eos_token_id

        # Decoder-only batched generation requires LEFT padding: with right
        # padding the pad tokens sit between the prompt and the first generated
        # token, so generation continues from pads and the output is garbage.
        # Harmless at batch=1 (no padding), required once translate_batch runs.
        self.pipe.tokenizer.padding_side = "left"
        processor = getattr(self.pipe, "processor", None)
        proc_tok = getattr(processor, "tokenizer", None)
        if proc_tok is not None:
            proc_tok.padding_side = "left"

        self.pad_token_id = self.pipe.tokenizer.eos_token_id
        self.device = device

    @staticmethod
    def _message(text: str, source_lang_code: str, target_lang_code: str) -> list:
        """One TranslateGemma chat conversation for a single source string."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": source_lang_code,
                        "target_lang_code": target_lang_code,
                        "text": text,
                    }
                ],
            }
        ]

    @staticmethod
    def _content(record) -> str:
        """Pull the assistant turn out of one pipeline output record.

        The pipeline wraps each conversation's result in a length-1 list for
        single inputs and may hand back the bare dict for batched inputs;
        accept both shapes."""
        rec = record[0] if isinstance(record, list) else record
        return rec["generated_text"][-1]["content"].strip()

    def translate_text(
        self,
        text: str,
        source_lang_code: str,
        target_lang_code: str = "en",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        """Translate a single text string with TranslateGemma chat format."""
        output = self.pipe(
            text=self._message(text, source_lang_code, target_lang_code),
            generate_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.pad_token_id,
            },
        )
        return self._content(output[0])

    def translate_batch(
        self,
        items: list[tuple[str, str, str]],
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        batch_size: int = 16,
    ) -> list[str]:
        """Translate many ``(text, source_lang_code, target_lang_code)`` tuples
        in one decode batch. Returns translations aligned to ``items``.

        Decode is memory-bandwidth bound at batch=1, so batching amortizes the
        per-step weight read across the batch and is the main throughput lever.
        """
        if not items:
            return []
        conversations = [self._message(text, src, dst) for (text, src, dst) in items]
        outputs = self.pipe(
            text=conversations,
            batch_size=batch_size,
            generate_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.pad_token_id,
            },
        )
        return [self._content(out) for out in outputs]
