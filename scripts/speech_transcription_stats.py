"""
Print min / max / avg character length for clips.speech_transcription.

Run from repo root or from scripts/:
  python scripts/speech_transcription_stats.py
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


def main():
    sql = text(
        """
        SELECT
            COUNT(*) AS clips_total,
            SUM(CASE WHEN speech_transcription IS NULL THEN 1 ELSE 0 END) AS null_count,
            SUM(CASE WHEN speech_transcription IS NOT NULL THEN 1 ELSE 0 END) AS non_null_count,
            MIN(CASE
                WHEN speech_transcription IS NOT NULL
                THEN LENGTH(speech_transcription)
            END) AS char_len_min,
            MAX(CASE
                WHEN speech_transcription IS NOT NULL
                THEN LENGTH(speech_transcription)
            END) AS char_len_max,
            AVG(CASE
                WHEN speech_transcription IS NOT NULL
                THEN LENGTH(speech_transcription)
            END) AS char_len_avg,
            MIN(CASE
                WHEN speech_transcription IS NOT NULL
                     AND LENGTH(TRIM(speech_transcription)) > 0
                THEN LENGTH(speech_transcription)
            END) AS char_len_min_nonempty,
            MAX(CASE
                WHEN speech_transcription IS NOT NULL
                     AND LENGTH(TRIM(speech_transcription)) > 0
                THEN LENGTH(speech_transcription)
            END) AS char_len_max_nonempty,
            AVG(CASE
                WHEN speech_transcription IS NOT NULL
                     AND LENGTH(TRIM(speech_transcription)) > 0
                THEN LENGTH(speech_transcription)
            END) AS char_len_avg_nonempty,
            SUM(CASE
                WHEN speech_transcription IS NOT NULL
                     AND LENGTH(TRIM(speech_transcription)) > 0
                THEN 1
                ELSE 0
            END) AS nonempty_count
        FROM clips
        """
    )

    with engine.connect() as conn:
        row = conn.execute(sql).mappings().one()

    print("speech_transcription (clips)")
    print(f"  clips_total:        {row['clips_total']}")
    print(f"  null:               {row['null_count']}")
    print(f"  non_null:           {row['non_null_count']}")
    print(f"  non_empty (trim):   {row['nonempty_count']}")
    print("  --- lengths in characters (all non-null, includes empty string) ---")
    print(f"  min:  {row['char_len_min']}")
    print(f"  max:  {row['char_len_max']}")
    avg = row["char_len_avg"]
    print(f"  avg:  {float(avg) if avg is not None else None}")
    print("  --- lengths in characters (non-empty after trim) ---")
    print(f"  min:  {row['char_len_min_nonempty']}")
    print(f"  max:  {row['char_len_max_nonempty']}")
    avg_n = row["char_len_avg_nonempty"]
    print(f"  avg:  {float(avg_n) if avg_n is not None else None}")


if __name__ == "__main__":
    main()
