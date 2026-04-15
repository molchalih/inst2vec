"""
Clip vs music enrichment stats (speech, music link, Spotify, ReccoBeats features).

Run from repo root or scripts/:
  python scripts/audio_music_stats.py
"""

import os
import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from modules.database import engine  # noqa: E402

# Must match music table columns used by ReccoBeats features in modules/audio_processor.py
FEATURE_COLS = (
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "valence",
)


def main():
    feature_predicates = " AND ".join(f"m.{c} IS NOT NULL" for c in FEATURE_COLS)
    music_all_features_where = " AND ".join(f"{c} IS NOT NULL" for c in FEATURE_COLS)

    sql = text(
        f"""
        SELECT
            (SELECT COUNT(*) FROM clips) AS clips_total,
            (SELECT COUNT(*) FROM clips
             WHERE speech_transcription IS NOT NULL
               AND LENGTH(TRIM(speech_transcription)) > 0) AS clips_with_speech,
            (SELECT COUNT(*) FROM clips WHERE music_id IS NOT NULL) AS clips_with_music_link,
            (SELECT COUNT(*) FROM clips c
             INNER JOIN music m ON m.id = c.music_id
             WHERE m.spotify_id IS NOT NULL
               AND LENGTH(TRIM(m.spotify_id)) > 0
               AND LOWER(TRIM(m.spotify_id)) != 'none')
                AS clips_music_has_spotify,
            (SELECT COUNT(*) FROM clips c
             INNER JOIN music m ON m.id = c.music_id
             WHERE m.reccobeats_id IS NOT NULL
               AND LENGTH(TRIM(m.reccobeats_id)) > 0
               AND LOWER(TRIM(m.reccobeats_id)) != 'none')
                AS clips_music_has_reccobeats,
            (SELECT COUNT(*) FROM clips c
             INNER JOIN music m ON m.id = c.music_id
             WHERE {feature_predicates}) AS clips_music_has_all_features,

            (SELECT COUNT(*) FROM music) AS music_total,
            (SELECT COUNT(*) FROM music
             WHERE spotify_id IS NOT NULL
               AND LENGTH(TRIM(spotify_id)) > 0
               AND LOWER(TRIM(spotify_id)) != 'none')
                AS music_with_spotify,
            (SELECT COUNT(*) FROM music
             WHERE reccobeats_id IS NOT NULL
               AND LENGTH(TRIM(reccobeats_id)) > 0
               AND LOWER(TRIM(reccobeats_id)) != 'none')
                AS music_with_reccobeats,
            (SELECT COUNT(*) FROM music WHERE {music_all_features_where})
                AS music_with_all_features,
            (SELECT COUNT(*) FROM music WHERE has_features = 'yes')
                AS music_has_features_yes,
            (SELECT COUNT(*) FROM music WHERE has_features = 'none')
                AS music_has_features_none,
            (SELECT COUNT(*) FROM clips
             WHERE music_id IS NULL
               AND (
                   speech_transcription IS NULL
                   OR LENGTH(TRIM(speech_transcription)) = 0
               )) AS clips_neither_music_nor_speech,
            (SELECT COUNT(*) FROM music m
             WHERE NOT EXISTS (
                 SELECT 1 FROM clips c WHERE c.music_id = m.id
             )) AS music_rows_unlinked_from_clips
        """
    )

    with engine.connect() as conn:
        row = conn.execute(sql).mappings().one()
        audio_type_none = None
        try:
            audio_type_none = conn.execute(
                text("SELECT COUNT(*) FROM clips WHERE audio_type = 'none'")
            ).scalar_one()
        except Exception:
            pass

    def pct(part, whole):
        if not whole:
            return "n/a"
        return f"{100.0 * part / whole:.1f}%"

    ct = row["clips_total"]
    mt = row["music_total"]

    print("=== clips ===")
    print(f"  total:                    {ct}")
    print(f"  with speech (non-empty):  {row['clips_with_speech']}  ({pct(row['clips_with_speech'], ct)})")
    print(f"  with music_id:            {row['clips_with_music_link']}  ({pct(row['clips_with_music_link'], ct)})")
    print(
        f"  music has spotify_id:     {row['clips_music_has_spotify']}  ({pct(row['clips_music_has_spotify'], ct)})"
    )
    print(
        f"  music has reccobeats_id:  {row['clips_music_has_reccobeats']}  ({pct(row['clips_music_has_reccobeats'], ct)})"
    )
    print(
        f"  music has full features:  {row['clips_music_has_all_features']}  ({pct(row['clips_music_has_all_features'], ct)})"
    )
    print(
        f"  neither music nor speech: {row['clips_neither_music_nor_speech']}  "
        f"({pct(row['clips_neither_music_nor_speech'], ct)})  "
        f"(no music_id and empty/null speech_transcription)"
    )
    if audio_type_none is not None:
        print(
            f"  audio_type='none' (resolved empty): {audio_type_none}  ({pct(audio_type_none, ct)})"
        )

    print("\n=== music (unique tracks) ===")
    print(f"  total rows:               {mt}")
    print(f"  with spotify_id:          {row['music_with_spotify']}  ({pct(row['music_with_spotify'], mt)})")
    print(f"  with reccobeats_id:       {row['music_with_reccobeats']}  ({pct(row['music_with_reccobeats'], mt)})")
    print(
        f"  with all feature fields:  {row['music_with_all_features']}  ({pct(row['music_with_all_features'], mt)})"
    )
    print(
        f"  has_features='yes':       {row['music_has_features_yes']}  ({pct(row['music_has_features_yes'], mt)})"
    )
    print(
        f"  has_features='none':      {row['music_has_features_none']}  ({pct(row['music_has_features_none'], mt)})"
    )
    print(
        f"  never linked from a clip: {row['music_rows_unlinked_from_clips']}  "
        f"({pct(row['music_rows_unlinked_from_clips'], mt)})"
    )


if __name__ == "__main__":
    main()
