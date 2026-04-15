#!/usr/bin/env python
"""One-off backfill: strip @mentions and newlines from already-processed captions.

Cleans both caption_text and caption_translation for all existing rows.
Safe to run multiple times (skips rows that are already clean).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import re

from dotenv import load_dotenv
load_dotenv()

from modules.database import Clip, get_session

_MENTION_RE = re.compile(r"@[\w.]+")


def _clean(text: str) -> str:
    return " ".join(_MENTION_RE.sub("", text).split())

COMMIT_EVERY = 50

session = get_session()
clips = (
    session.query(Clip)
    .filter(Clip.caption_text.is_not(None), Clip.caption_text != "")
    .order_by(Clip.pk)
    .all()
)

print(f"Processing {len(clips)} clips…")
text_updated = translation_updated = 0

for i, clip in enumerate(clips, 1):
    clean_text = _clean(clip.caption_text)
    if clean_text != clip.caption_text:
        clip.caption_text = clean_text
        text_updated += 1

    if clip.caption_translation:
        clean_tr = _clean(clip.caption_translation)
        if clean_tr != clip.caption_translation:
            clip.caption_translation = clean_tr
            translation_updated += 1

    if i % COMMIT_EVERY == 0:
        session.commit()
        print(f"  {i}/{len(clips)} committed")

session.commit()
session.close()
print(f"Done — {text_updated} caption_text, {translation_updated} caption_translation updated")
