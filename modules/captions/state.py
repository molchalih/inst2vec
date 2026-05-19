"""Shared state constants and text helpers for the captions package."""

from __future__ import annotations

import json
import re

from sqlalchemy.orm import Session

from core.config import CaptionsSettings
from core.database import Clip
from core.pipeline import Stage

SCOPE_CLEAN: str = "clean_captions"
SCOPE_DETECT: str = "detect_caption_language"
SCOPE_TRANSLATE: str = "translate_captions"

STAGE_CAPTIONS: Stage = Stage.CAPTIONS
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
    """NULL the three caption output columns on every clip.

    Called from process_captions() when the caption-stage config
    fingerprint drifts. Resets all clips (not just currently-eligible
    ones) so clips that re-enter the selection pool on a later run
    can't carry stale derived fields produced under the previous
    config. The row-level idempotence in clean/detect/translate
    repopulates the cleared columns for eligible clips on the next
    pass. caption_text is upstream input, never touched.
    """
    session.query(Clip).update(
        {
            Clip.caption_clean: None,
            Clip.caption_language: None,
            Clip.caption_translation: None,
        },
        synchronize_session=False,
    )
    session.commit()


# Fields whose values can change caption outputs. commit_every is
# purely operational and intentionally excluded so changing it does
# not invalidate the stored fingerprint.
_CAPTIONS_CONFIG_FIELDS: tuple[str, ...] = (
    "translate_model",
    "translate_target_lang",
    "translation_max_chars",
    "translate_max_new_tokens",
)


def captions_config_payload(cfg: CaptionsSettings) -> str:
    """Stable JSON of the CaptionsSettings fields that affect caption outputs."""
    payload = {f: getattr(cfg, f) for f in _CAPTIONS_CONFIG_FIELDS}
    return json.dumps(payload, sort_keys=True, default=str)
