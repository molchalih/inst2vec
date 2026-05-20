"""Captions pipeline: clean → detect language → translate.

``process_captions`` is fingerprint-gated: on config drift it nulls every
caption output column on eligible clips, then lets the existing row-level
predicates in clean/detect/translate refill them. On a match (or no prior
state) it skips the reset and lets the existing row-level loops resume
any partial work.
"""

from __future__ import annotations

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from core import fingerprint as fp
from core.config import CaptionsSettings, Secrets, Settings
from core.database import get_engine
from modules.captions.clean import clean_captions
from modules.captions.detect import detect_caption_language
from modules.captions.state import (
    SCOPE_CAPTIONS,
    STAGE_CAPTIONS,
    captions_config_payload,
    reset_caption_outputs,
)
from modules.captions.translate import translate_captions

__all__ = [
    "clean_captions",
    "detect_caption_language",
    "run",
    "translate_captions",
]


def process_captions(cfg: CaptionsSettings, *, engine: Engine | None = None) -> None:
    """Run the full captions pipeline: clean → detect → translate.

    Wrapped in a config-only fingerprint gate. Each row-level stage opens
    its own ``with Session(engine) as session:`` block as before.
    ``engine=None`` resolves to the module-level engine via ``get_engine()``
    inside each stage.
    """
    eng = engine or get_engine()

    current = fp.Fingerprint(
        data=fp.hash_text(""),
        config=fp.hash_text(captions_config_payload(cfg)),
        dependency=fp.hash_text(""),
    )
    with Session(eng) as session:
        fp.gate(
            session,
            STAGE_CAPTIONS,
            SCOPE_CAPTIONS,
            current,
            reset_caption_outputs,
            log_scope="captions",
            drift_msg="resetting caption outputs",
        )
        session.commit()

    clean_captions(cfg, engine=eng)
    detect_caption_language(cfg, engine=eng)
    translate_captions(cfg, engine=eng)

    with Session(eng) as session:
        fp.mark_complete(session, STAGE_CAPTIONS, SCOPE_CAPTIONS, current)
        session.commit()


def run(settings: Settings, secrets: Secrets) -> None:
    """Captions clean + detect language + translate."""
    process_captions(settings.captions)
