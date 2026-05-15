"""Shared state constants and text helpers for the captions package."""

from __future__ import annotations

import re

SCOPE_CLEAN: str = "clean_captions"
SCOPE_DETECT: str = "detect_caption_language"
SCOPE_TRANSLATE: str = "translate_captions"

_MENTION_RE = re.compile(r"@[\w.]+")


def clean_caption_text(text: str | None) -> str:
    """Strip @mentions, collapse whitespace/newlines. Deterministic, safe.

    Returns "" for None / empty / mentions-only input so callers can treat
    the result uniformly: store NULL on falsy, store the cleaned string
    otherwise.
    """
    if not text:
        return ""
    return " ".join(_MENTION_RE.sub("", text).split())
