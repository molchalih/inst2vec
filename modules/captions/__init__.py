"""Captions pipeline: clean → detect language → translate.

``process_captions`` is fingerprint-gated: on config drift it nulls every
caption output column on eligible clips, then lets the existing row-level
predicates in clean/detect/translate refill them. On a match (or no prior
state) it skips the reset and lets the existing row-level loops resume
any partial work.
"""

from __future__ import annotations

import json

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from modules import fingerprint as fp
from modules.captions.clean import clean_captions
from modules.captions.detect import detect_caption_language
from modules.captions.state import (
    SCOPE_CAPTIONS,
    STAGE_CAPTIONS,
    reset_caption_outputs,
)
from modules.captions.translate import translate_captions
from modules.config import CaptionsSettings
from modules.console import log
from modules.database import StageState, get_engine

__all__ = [
    "clean_captions",
    "detect_caption_language",
    "process_captions",
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
        config=fp.hash_text(json.dumps(cfg.model_dump(), sort_keys=True, default=str)),
        dependency=fp.hash_text(""),
    )
    with Session(eng) as session:
        stored = session.get(StageState, (STAGE_CAPTIONS, SCOPE_CAPTIONS))
        if stored is not None and stored.config_hash != current.config:
            diff = fp.describe_diff(session, STAGE_CAPTIONS, SCOPE_CAPTIONS, current)
            log("captions", f"config drift ({diff}) — resetting caption outputs")
            reset_caption_outputs(session)
        elif stored is None:
            log("captions", "no prior state — sealing on completion")
        else:
            log("captions", "fingerprint match — skipping reset")

    clean_captions(cfg, engine=eng)
    detect_caption_language(cfg, engine=eng)
    translate_captions(cfg, engine=eng)

    with Session(eng) as session:
        fp.mark_complete(session, STAGE_CAPTIONS, SCOPE_CAPTIONS, current)
        session.commit()
