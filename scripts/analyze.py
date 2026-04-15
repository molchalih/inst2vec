#!/usr/bin/env python
"""Dataset analysis report: pipeline health + content statistics."""
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import func
from modules.database import User, Clip, Music, Download, get_session


def _pct(n, total):
    """Format n/total as a percentage string, or 'N/A' if total is zero."""
    return f"{100 * n / total:.1f}%" if total else "N/A"


def _p90(values):
    """Return 90th percentile of a list of numbers."""
    if not values:
        return 0
    sorted_v = sorted(values)
    idx = min(int(0.9 * len(sorted_v)), len(sorted_v) - 1)
    return sorted_v[idx]


def _header(title, char="="):
    print(f"\n{title}")
    print(char * len(title))


def section_health(session):
    _header("PIPELINE HEALTH")

    # Totals
    n_users = session.query(func.count(User.pk)).scalar()
    n_clips = session.query(func.count(Clip.pk)).scalar()
    n_disq = session.query(func.count(Clip.pk)).filter(Clip.clip_disqualified == 1).scalar()
    print(f"\nUsers:              {n_users:>8,}")
    print(f"Clips total:        {n_clips:>8,}")
    print(f"Clips disqualified: {n_disq:>8,}  ({_pct(n_disq, n_clips)})")

    # Music phase
    n_music_resolved = session.query(func.count(Clip.pk)).filter(Clip.has_music.is_not(None)).scalar()
    n_has_music = session.query(func.count(Clip.pk)).filter(Clip.has_music == 1).scalar()
    n_no_music = session.query(func.count(Clip.pk)).filter(Clip.has_music == 0).scalar()
    n_music_features = (
        session.query(func.count(Clip.pk))
        .join(Music, Clip.music_id == Music.id)
        .filter(Music.has_features == "yes")
        .scalar()
    )
    print(f"\nMusic resolved:     {n_music_resolved:>8,}  ({_pct(n_music_resolved, n_clips)} of clips)")
    print(f"  with music:       {n_has_music:>8,}  ({_pct(n_has_music, n_music_resolved)})")
    print(f"  no music:         {n_no_music:>8,}  ({_pct(n_no_music, n_music_resolved)})")
    print(f"  with features:    {n_music_features:>8,}  ({_pct(n_music_features, n_has_music)} of music clips)")

    # Speech phase
    n_speech_resolved = session.query(func.count(Clip.pk)).filter(Clip.has_speech.is_not(None)).scalar()
    n_has_speech = session.query(func.count(Clip.pk)).filter(Clip.has_speech == 1).scalar()
    n_no_speech = session.query(func.count(Clip.pk)).filter(Clip.has_speech == 0).scalar()
    print(f"\nSpeech resolved:    {n_speech_resolved:>8,}  ({_pct(n_speech_resolved, n_clips)} of clips)")
    print(f"  with speech:      {n_has_speech:>8,}  ({_pct(n_has_speech, n_speech_resolved)})")
    print(f"  silent:           {n_no_speech:>8,}  ({_pct(n_no_speech, n_speech_resolved)})")

    # Captions phase
    n_caption = session.query(func.count(Clip.pk)).filter(
        Clip.caption_text.is_not(None), Clip.caption_text != ""
    ).scalar()
    n_lang_detected = session.query(func.count(Clip.pk)).filter(
        Clip.caption_language.is_not(None), Clip.caption_language != ""
    ).scalar()
    n_non_en = session.query(func.count(Clip.pk)).filter(
        Clip.caption_language.is_not(None),
        Clip.caption_language != "",
        func.lower(Clip.caption_language).notlike("en%"),
    ).scalar()
    n_translated = session.query(func.count(Clip.pk)).filter(
        Clip.caption_language.is_not(None),
        func.lower(Clip.caption_language).notlike("en%"),
        Clip.caption_translation.is_not(None),
        Clip.caption_translation != "",
    ).scalar()
    print(f"\nCaptions with text: {n_caption:>8,}  ({_pct(n_caption, n_clips)} of clips)")
    print(f"  language detected:{n_lang_detected:>8,}  ({_pct(n_lang_detected, n_caption)})")
    print(f"  non-English:      {n_non_en:>8,}  ({_pct(n_non_en, n_lang_detected)} of detected)")
    print(f"  translated:       {n_translated:>8,}  ({_pct(n_translated, n_non_en)} of non-English)")

    # Downloads
    n_video_fail = session.query(func.count(Download.entity_pk)).filter(
        Download.file_type == "video", Download.success == False
    ).scalar()
    print(f"\nVideo download failures: {n_video_fail:>5,}")


def section_engagement(session):
    _header("ENGAGEMENT", "-")
    rows = session.query(
        Clip.like_count, Clip.play_count, Clip.comment_count, Clip.reshare_count
    ).all()
    if not rows:
        print("No clip data.")
        return

    fields = [
        (0, "like_count"),
        (1, "play_count"),
        (2, "comment_count"),
        (3, "reshare_count"),
    ]
    for idx, label in fields:
        vals = [r[idx] or 0 for r in rows]
        mean = statistics.mean(vals)
        median = statistics.median(vals)
        p90 = _p90(vals)
        print(f"  {label:<18}  mean={mean:>10,.0f}  median={median:>10,.0f}  p90={p90:>10,.0f}")


def main():
    session = get_session()
    try:
        section_health(session)
        section_engagement(session)
    finally:
        session.close()


if __name__ == "__main__":
    main()
