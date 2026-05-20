"""Seed the current `music` table from a pre-refactor `inst2vec.db`.

Source schema (legacy): id, artist, track, spotify_id, reccobeats_id, all
feature columns + has_features. Missing IDs are stored as the literal
string "none" rather than NULL.

Target schema (current): adds recognition_status, is_reccobeats_resolved,
is_audio_features_extracted. After seeding, the four feature sub-stages
in `modules/music/features.py` skip seeded rows by predicate (no code
changes needed in the pipeline):

  Stage 1 (Spotify)        skips if recognition_status != "pending"
  Stage 2 (RB id)          skips if is_reccobeats_resolved IS NOT NULL
  Stage 3 (catalog)        skips if is_audio_features_extracted IS NOT NULL
  Stage 4 (upload fallback) skips if is_audio_features_extracted IS NOT NULL

ACR (`modules/music/classify.py::_get_or_create_music`) already reuses
any Music row matching the same (artist, track) by unique constraint, so
clips fingerprinted to a seeded track inherit it automatically.

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db \\
      uv run python scripts/migrate_seed_music_from_old_db.py
    SOURCE_DB_PATH=data/old/inst2vec.db \\
      DATABASE_URL=postgresql://... \\
      uv run python scripts/migrate_seed_music_from_old_db.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from core.database import Music
from modules.music.state import FEATURE_FIELDS, UPLOAD_FIELDS

_DEFAULT_SOURCE = Path("data/old/inst2vec.db")
_NULL_SENTINELS = {"none", "null"}


def _clean_id(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in _NULL_SENTINELS:
        return None
    return s


def _has_upload_features(row: dict) -> bool:
    return all(row.get(f) is not None for f in UPLOAD_FIELDS)


def _read_source(path: Path) -> list[dict]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT artist, track, spotify_id, reccobeats_id, "
            + ", ".join(f'"{f}"' for f in FEATURE_FIELDS)
            + " FROM music"
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def seed_from_old_db(source_path: Path, session: Session) -> dict[str, int]:
    """Seed `Music` rows from a legacy DB. Returns per-bucket counts."""
    stats = {"seeded": 0, "skipped_existing": 0, "missing_features": 0}
    for src in _read_source(Path(source_path)):
        artist = (src["artist"] or "").strip()
        track = (src["track"] or "").strip()
        if not artist and not track:
            continue

        existing = session.query(Music).filter_by(artist=artist, track=track).first()
        if existing and existing.is_audio_features_extracted is True:
            stats["skipped_existing"] += 1
            continue

        spotify_id = _clean_id(src["spotify_id"])
        reccobeats_id = _clean_id(src["reccobeats_id"])
        full = _has_upload_features(src)
        if not full:
            stats["missing_features"] += 1

        recognition_status = "matched" if spotify_id else "no_match"
        if reccobeats_id is not None:
            is_reccobeats_resolved: bool | None = True
        elif spotify_id is not None:
            is_reccobeats_resolved = False  # Spotify matched, RB catalog miss
        else:
            is_reccobeats_resolved = None  # Stage 2 never runs without spotify_id

        target = existing or Music(artist=artist, track=track)
        target.spotify_id = spotify_id
        target.reccobeats_id = reccobeats_id
        target.recognition_status = recognition_status
        target.is_reccobeats_resolved = is_reccobeats_resolved
        target.is_audio_features_extracted = True if full else None
        for f in FEATURE_FIELDS:
            setattr(target, f, src[f])
        if existing is None:
            session.add(target)
        stats["seeded"] += 1

    session.commit()
    return stats


def main() -> int:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        return 1
    source = Path(os.environ.get("SOURCE_DB_PATH", str(_DEFAULT_SOURCE)))
    if not source.exists():
        print(f"ERROR: source DB not found: {source}", file=sys.stderr)
        return 1

    safe_url = make_url(url).render_as_string(hide_password=True)
    print(f"Seeding {safe_url} from {source} ...")
    engine = create_engine(url)
    session = sessionmaker(bind=engine)()
    try:
        stats = seed_from_old_db(source, session)
    finally:
        session.close()
    print(
        f"Done. seeded={stats['seeded']} "
        f"skipped_existing={stats['skipped_existing']} "
        f"missing_features={stats['missing_features']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
