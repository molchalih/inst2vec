"""Captions pipeline: clean → detect language → translate."""

from __future__ import annotations

from sqlalchemy.engine import Engine

from modules.captions.clean import clean_captions
from modules.captions.detect import detect_caption_language
from modules.captions.translate import translate_captions
from modules.config import CaptionsSettings

__all__ = [
    "clean_captions",
    "detect_caption_language",
    "process_captions",
    "translate_captions",
]


def process_captions(cfg: CaptionsSettings, *, engine: Engine | None = None) -> None:
    """Run the full captions pipeline: clean → detect → translate.

    Each stage opens its own ``with Session(engine) as session:`` block and
    can be invoked independently. ``engine=None`` resolves to the
    module-level engine via ``get_engine()`` inside each stage.
    """
    clean_captions(cfg, engine=engine)
    detect_caption_language(cfg, engine=engine)
    translate_captions(cfg, engine=engine)
