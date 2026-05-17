"""Shared state constants and text helpers for the captions package."""

from __future__ import annotations

import re

from sqlalchemy.orm import Session

from modules.database import Clip, clip_used_in_analysis

SCOPE_CLEAN: str = "clean_captions"
SCOPE_DETECT: str = "detect_caption_language"
SCOPE_TRANSLATE: str = "translate_captions"

STAGE_CAPTIONS: str = "captions"
SCOPE_CAPTIONS: str = "all"

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


def reset_caption_outputs(session: Session) -> None:
    """NULL the three caption output columns on every eligible clip.

    Called from process_captions() when the caption-stage config
    fingerprint drifts. The row-level idempotence in clean/detect/translate
    repopulates the cleared columns on the next pass. caption_text is
    upstream input, never touched.
    """
    session.query(Clip).filter(*clip_used_in_analysis()).update(
        {
            Clip.caption_clean: None,
            Clip.caption_language: None,
            Clip.caption_translation: None,
        },
        synchronize_session=False,
    )
    session.commit()
