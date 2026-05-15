"""Production migration: rename clips.has_music → is_music_recognized,
add music.is_audio_features_extracted, drop music.has_features.

Backfill rules:
    music.is_audio_features_extracted =
        TRUE  iff has_features = 'yes'
        FALSE iff has_features = 'none'
        NULL  otherwise

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db uv run python scripts/migrate_music_state.py
    DATABASE_URL=postgresql://... uv run python scripts/migrate_music_state.py
"""

from __future__ import annotations

import os
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def migrate_database(database_url_or_engine: str | Engine) -> None:
    if isinstance(database_url_or_engine, Engine):
        engine = database_url_or_engine
    else:
        engine = create_engine(database_url_or_engine)

    with engine.begin() as conn:
        dialect = engine.dialect.name
        if dialect == "sqlite":
            _sqlite_migrate(conn)
        elif dialect == "postgresql":
            _postgres_migrate(conn)
        else:
            raise RuntimeError(f"Unsupported dialect: {dialect}")


def _sqlite_migrate(conn) -> None:
    clip_cols = {
        r[1] for r in conn.execute(text("PRAGMA table_info(clips)")).fetchall()
    }
    music_cols = {
        r[1] for r in conn.execute(text("PRAGMA table_info(music)")).fetchall()
    }

    fully_migrated = (
        "has_music" not in clip_cols
        and "is_music_recognized" in clip_cols
        and "has_features" not in music_cols
        and "is_audio_features_extracted" in music_cols
    )
    if fully_migrated:
        print("  SKIP: already migrated")
        return

    if "has_music" in clip_cols and "is_music_recognized" not in clip_cols:
        conn.execute(
            text("ALTER TABLE clips RENAME COLUMN has_music TO is_music_recognized")
        )
        print("  OK: renamed clips.has_music → is_music_recognized")

    if "is_audio_features_extracted" not in music_cols:
        conn.execute(
            text("ALTER TABLE music ADD COLUMN is_audio_features_extracted BOOLEAN")
        )
        print("  OK: added music.is_audio_features_extracted")

    if "has_features" in music_cols:
        conn.execute(
            text(
                "UPDATE music SET is_audio_features_extracted = "
                "CASE "
                "  WHEN has_features = 'yes' THEN 1 "
                "  WHEN has_features = 'none' THEN 0 "
                "  ELSE NULL "
                "END"
            )
        )
        print("  OK: backfilled is_audio_features_extracted from has_features")

        _drop_has_features_sqlite(conn)
        print("  OK: dropped music.has_features")


def _drop_has_features_sqlite(conn) -> None:
    """SQLite < 3.35 needs a table rebuild to drop a column. We rebuild
    even on 3.35+ to keep behavior uniform across SQLite versions in CI."""
    conn.execute(text("PRAGMA foreign_keys = OFF"))
    conn.execute(text("PRAGMA legacy_alter_table = ON"))

    fk_rows = conn.execute(text("PRAGMA foreign_key_list(music)")).fetchall()
    fk_by_id: dict[int, list[tuple[int, str, str, str]]] = {}
    for row in fk_rows:
        fk_id, seq, fk_table, fk_from, fk_to = row[0], row[1], row[2], row[3], row[4]
        fk_by_id.setdefault(fk_id, []).append((seq, fk_from, fk_table, fk_to))

    conn.execute(text("ALTER TABLE music RENAME TO music_old"))

    pragma = conn.execute(text("PRAGMA table_info(music_old)")).fetchall()
    col_defs: list[str] = []
    kept_cols: list[str] = []
    for row in pragma:
        name, typ, notnull, dflt, pk = row[1], row[2], row[3], row[4], row[5]
        if name == "has_features":
            continue
        kept_cols.append(name)
        pk_str = " PRIMARY KEY" if pk else ""
        nn_str = " NOT NULL" if notnull else ""
        df_str = f" DEFAULT {dflt}" if dflt is not None else ""
        col_defs.append(f"{name} {typ}{pk_str}{nn_str}{df_str}")

    fk_defs: list[str] = []
    for fk_id in sorted(fk_by_id.keys()):
        cols_in_fk = sorted(fk_by_id[fk_id], key=lambda x: x[0])
        from_cols = [col[1] for col in cols_in_fk]
        fk_table = cols_in_fk[0][2]
        to_cols = [col[3] for col in cols_in_fk]
        fk_defs.append(
            f"FOREIGN KEY ({', '.join(from_cols)}) "
            f"REFERENCES {fk_table}({', '.join(to_cols)})"
        )

    all_defs = col_defs + fk_defs
    conn.execute(
        text(
            "CREATE TABLE music (\n  "
            + ",\n  ".join(all_defs)
            + ",\n  UNIQUE (artist, track)\n)"
        )
    )
    conn.execute(
        text(
            f"INSERT INTO music ({', '.join(kept_cols)}) "
            f"SELECT {', '.join(kept_cols)} FROM music_old"
        )
    )
    conn.execute(text("DROP TABLE music_old"))

    conn.execute(text("PRAGMA legacy_alter_table = OFF"))
    conn.execute(text("PRAGMA foreign_keys = ON"))

    violations = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
    if violations:
        raise RuntimeError(f"FK integrity violated after migration: {violations}")


def _postgres_migrate(conn) -> None:
    clip_cols = {
        r[0]
        for r in conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='clips'"
            )
        ).fetchall()
    }
    music_cols = {
        r[0]
        for r in conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='music'"
            )
        ).fetchall()
    }

    if "has_music" in clip_cols and "is_music_recognized" not in clip_cols:
        conn.execute(
            text("ALTER TABLE clips RENAME COLUMN has_music TO is_music_recognized")
        )

    if "is_audio_features_extracted" not in music_cols:
        conn.execute(
            text("ALTER TABLE music ADD COLUMN is_audio_features_extracted BOOLEAN")
        )

    if "has_features" in music_cols:
        conn.execute(
            text(
                "UPDATE music SET is_audio_features_extracted = "
                "CASE "
                "  WHEN has_features = 'yes' THEN TRUE "
                "  WHEN has_features = 'none' THEN FALSE "
                "  ELSE NULL "
                "END"
            )
        )
        conn.execute(text("ALTER TABLE music DROP COLUMN has_features"))
    print("  OK: postgres migration complete")


if __name__ == "__main__":
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    migrate_database(database_url)
