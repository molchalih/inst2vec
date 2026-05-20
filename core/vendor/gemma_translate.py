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

        self.pad_token_id = self.pipe.tokenizer.eos_token_id
        self.device = device

    def translate_text(
        self,
        text: str,
        source_lang_code: str,
        target_lang_code: str = "en",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        """Translate a text string with TranslateGemma chat format."""
        messages = [
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

        output = self.pipe(
            text=messages,
            generate_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.pad_token_id,
            },
        )
        return output[0]["generated_text"][-1]["content"].strip()
