#!/usr/bin/env python
"""Dataset analysis report: pipeline health + content statistics."""

import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import func  # noqa: E402

from modules.database import Clip, Download, Music, User, get_session  # noqa: E402


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
    n_users = session.query(func.count(User.id)).scalar()
    n_clips = session.query(func.count(Clip.id)).scalar()
    n_disq = session.query(func.count(Clip.id)).filter(Clip.disqualified == 1).scalar()
    print(f"\nUsers:              {n_users:>8,}")
    print(f"Clips total:        {n_clips:>8,}")
    print(f"Clips disqualified: {n_disq:>8,}  ({_pct(n_disq, n_clips)})")

    # Music phase
    n_music_resolved = (
        session.query(func.count(Clip.id)).filter(Clip.has_music.is_not(None)).scalar()
    )
    n_has_music = (
        session.query(func.count(Clip.id)).filter(Clip.has_music == 1).scalar()
    )
    n_no_music = session.query(func.count(Clip.id)).filter(Clip.has_music == 0).scalar()
    n_music_features = (
        session.query(func.count(Clip.id))
        .join(Music, Clip.music_id == Music.id)
        .filter(Music.has_features == "yes")
        .scalar()
    )
    print(
        f"\nMusic resolved:     {n_music_resolved:>8,}  ({_pct(n_music_resolved, n_clips)} of clips)"
    )
    print(
        f"  with music:       {n_has_music:>8,}  ({_pct(n_has_music, n_music_resolved)})"
    )
    print(
        f"  no music:         {n_no_music:>8,}  ({_pct(n_no_music, n_music_resolved)})"
    )
    print(
        f"  with features:    {n_music_features:>8,}  ({_pct(n_music_features, n_has_music)} of music clips)"
    )

    # Speech phase
    n_speech_resolved = (
        session.query(func.count(Clip.id)).filter(Clip.has_speech.is_not(None)).scalar()
    )
    n_has_speech = (
        session.query(func.count(Clip.id)).filter(Clip.has_speech == 1).scalar()
    )
    n_no_speech = (
        session.query(func.count(Clip.id)).filter(Clip.has_speech == 0).scalar()
    )
    print(
        f"\nSpeech resolved:    {n_speech_resolved:>8,}  ({_pct(n_speech_resolved, n_clips)} of clips)"
    )
    print(
        f"  with speech:      {n_has_speech:>8,}  ({_pct(n_has_speech, n_speech_resolved)})"
    )
    print(
        f"  silent:           {n_no_speech:>8,}  ({_pct(n_no_speech, n_speech_resolved)})"
    )

    # Captions phase
    n_caption = (
        session.query(func.count(Clip.id))
        .filter(Clip.caption_text.is_not(None), Clip.caption_text != "")
        .scalar()
    )
    n_lang_detected = (
        session.query(func.count(Clip.id))
        .filter(Clip.caption_language.is_not(None), Clip.caption_language != "")
        .scalar()
    )
    n_non_en = (
        session.query(func.count(Clip.id))
        .filter(
            Clip.caption_language.is_not(None),
            Clip.caption_language != "",
            func.lower(Clip.caption_language).notlike("en%"),
        )
        .scalar()
    )
    n_translated = (
        session.query(func.count(Clip.id))
        .filter(
            Clip.caption_language.is_not(None),
            func.lower(Clip.caption_language).notlike("en%"),
            Clip.caption_translation.is_not(None),
            Clip.caption_translation != "",
        )
        .scalar()
    )
    print(
        f"\nCaptions with text: {n_caption:>8,}  ({_pct(n_caption, n_clips)} of clips)"
    )
    print(
        f"  language detected:{n_lang_detected:>8,}  ({_pct(n_lang_detected, n_caption)})"
    )
    print(
        f"  non-English:      {n_non_en:>8,}  ({_pct(n_non_en, n_lang_detected)} of detected)"
    )
    print(
        f"  translated:       {n_translated:>8,}  ({_pct(n_translated, n_non_en)} of non-English)"
    )

    # Downloads
    n_video_fail = (
        session.query(func.count(Download.entity_id))
        .filter(Download.file_type == "video", ~Download.success)
        .scalar()
    )
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
        print(
            f"  {label:<18}  mean={mean:>10,.0f}  median={median:>10,.0f}  p90={p90:>10,.0f}"
        )


def section_captions(session):
    _header("CAPTIONS", "-")
    n_total = session.query(func.count(Clip.id)).scalar()
    n_empty = (
        session.query(func.count(Clip.id))
        .filter((Clip.caption_text.is_(None)) | (Clip.caption_text == ""))
        .scalar()
    )
    print(f"\nNo caption: {n_empty:,} / {n_total:,} ({_pct(n_empty, n_total)})")

    lengths = [
        len(r[0])
        for r in session.query(Clip.caption_text)
        .filter(Clip.caption_text.is_not(None), Clip.caption_text != "")
        .all()
    ]
    if lengths:
        print(
            f"Avg caption length: {statistics.mean(lengths):.0f} chars"
            f"  (median {statistics.median(lengths):.0f})"
        )

    lang_rows = (
        session.query(Clip.caption_language, func.count(Clip.id))
        .filter(Clip.caption_language.is_not(None), Clip.caption_language != "")
        .group_by(Clip.caption_language)
        .order_by(func.count(Clip.id).desc())
        .limit(10)
        .all()
    )
    total_detected = sum(r[1] for r in lang_rows)
    print("\nCaption language distribution (top 10):")
    for lang, cnt in lang_rows:
        print(f"  {lang:<6}  {cnt:>6,}  ({_pct(cnt, total_detected)})")


def section_music(session):
    _header("MUSIC", "-")
    n_with = session.query(func.count(Clip.id)).filter(Clip.has_music == 1).scalar()
    n_without = session.query(func.count(Clip.id)).filter(Clip.has_music == 0).scalar()
    total_resolved = n_with + n_without
    print(
        f"\nWith music: {n_with:,} / {total_resolved:,} ({_pct(n_with, total_resolved)})"
    )
    print(
        f"No music:   {n_without:,} / {total_resolved:,} ({_pct(n_without, total_resolved)})"
    )

    top_tracks = (
        session.query(Music.artist, Music.track, func.count(Clip.id))
        .join(Clip, Clip.music_id == Music.id)
        .group_by(Music.id)
        .order_by(func.count(Clip.id).desc())
        .limit(10)
        .all()
    )
    print("\nTop 10 tracks by clip count:")
    for artist, track, cnt in top_tracks:
        label = f"{artist} – {track}" if artist else track
        print(f"  {cnt:>4}x  {label[:60]}")

    feature_rows = (
        session.query(
            Music.tempo,
            Music.valence,
            Music.danceability,
            Music.energy,
            Music.acousticness,
        )
        .join(Clip, Clip.music_id == Music.id)
        .filter(Music.has_features == "yes")
        .all()
    )
    feature_names = ["tempo", "valence", "danceability", "energy", "acousticness"]
    if feature_rows:
        print(
            f"\nAudio features (mean ± std, across {len(feature_rows)} clips with features):"
        )
        for i, feat in enumerate(feature_names):
            vals = [r[i] for r in feature_rows if r[i] is not None]
            if vals:
                mean = statistics.mean(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                print(f"  {feat:<15}  {mean:>7.3f} ± {std:.3f}")


def section_speech(session):
    _header("SPEECH", "-")
    n_with = session.query(func.count(Clip.id)).filter(Clip.has_speech == 1).scalar()
    n_without = session.query(func.count(Clip.id)).filter(Clip.has_speech == 0).scalar()
    total_resolved = n_with + n_without
    print(
        f"\nWith speech: {n_with:,} / {total_resolved:,} ({_pct(n_with, total_resolved)})"
    )
    print(
        f"Silent:      {n_without:,} / {total_resolved:,} ({_pct(n_without, total_resolved)})"
    )

    lang_rows = (
        session.query(Clip.speech_language, func.count(Clip.id))
        .filter(
            Clip.has_speech == 1,
            Clip.speech_language.is_not(None),
            Clip.speech_language != "",
        )
        .group_by(Clip.speech_language)
        .order_by(func.count(Clip.id).desc())
        .limit(10)
        .all()
    )
    total_lang = sum(r[1] for r in lang_rows)
    print("\nSpeech language distribution (top 10):")
    for lang, cnt in lang_rows:
        print(f"  {lang:<6}  {cnt:>6,}  ({_pct(cnt, total_lang)})")

    quality_rows = (
        session.query(Clip.speech_confidence, Clip.speech_avg_logprob)
        .filter(Clip.has_speech == 1, Clip.speech_confidence.is_not(None))
        .all()
    )
    if quality_rows:
        conf_vals = [r[0] for r in quality_rows if r[0] is not None]
        logprob_vals = [r[1] for r in quality_rows if r[1] is not None]
        if conf_vals:
            print(f"\nMean speech confidence:  {statistics.mean(conf_vals):.3f}")
        if logprob_vals:
            print(f"Mean speech avg logprob: {statistics.mean(logprob_vals):.3f}")


def main():
    session = get_session()
    try:
        print("=" * 52)
        print("inst2vec — Dataset Analysis Report")
        print("=" * 52)
        section_health(session)
        _header("CONTENT STATISTICS")
        section_engagement(session)
        section_captions(session)
        section_music(session)
        section_speech(session)
        print()
    finally:
        session.close()


if __name__ == "__main__":
    main()
