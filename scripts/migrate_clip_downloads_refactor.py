"""Production migration: drop clips.eligibility + downloads table, add clips.is_downloaded.

Backfill rule:
    clips.is_downloaded = TRUE  iff  clips.is_selected = 1
                                AND  ∃ downloads row (entity_id=clip.id, file_type='video', success=1)
    All other selected clips keep is_downloaded = NULL (re-attempted on next pipeline run).

Usage:
    DATABASE_URL=sqlite:///data/inst2vec.db uv run python scripts/migrate_clip_downloads_refactor.py
    DATABASE_URL=postgresql://... uv run python scripts/migrate_clip_downloads_refactor.py
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
    table_names = {
        r[0]
        for r in conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
    }

    has_eligibility = "eligibility" in clip_cols
    has_is_downloaded = "is_downloaded" in clip_cols
    has_downloads_table = "downloads" in table_names

    if not has_eligibility and has_is_downloaded and not has_downloads_table:
        print("  SKIP: already migrated")
        return

    if not has_is_downloaded:
        conn.execute(text("ALTER TABLE clips ADD COLUMN is_downloaded BOOLEAN"))
        print("  OK: added clips.is_downloaded")

    if has_downloads_table:
        conn.execute(
            text(
                "UPDATE clips SET is_downloaded = 1 "
                "WHERE is_selected = 1 "
                "AND id IN ("
                "  SELECT entity_id FROM downloads "
                "  WHERE file_type = 'video' AND success = 1"
                ")"
            )
        )
        print("  OK: backfilled is_downloaded from legacy downloads table")

    if has_eligibility:
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        conn.execute(text("PRAGMA legacy_alter_table = ON"))

        # Capture FK constraints BEFORE renaming, since PRAGMA FK list uses current table names
        fk_rows = conn.execute(text("PRAGMA foreign_key_list(clips)")).fetchall()
        # fk_rows: (id, seq, table, from, to, on_update, on_delete, match)
        fk_by_id: dict[int, list[tuple[int, str, str, str]]] = {}
        for row in fk_rows:
            fk_id, seq, fk_table, fk_from, fk_to = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
            )
            if fk_id not in fk_by_id:
                fk_by_id[fk_id] = []
            fk_by_id[fk_id].append((seq, fk_from, fk_table, fk_to))

        conn.execute(text("ALTER TABLE clips RENAME TO clips_old"))

        pragma = conn.execute(text("PRAGMA table_info(clips_old)")).fetchall()
        col_defs: list[str] = []
        kept_cols: list[str] = []
        for row in pragma:
            name, typ, notnull, dflt, pk = row[1], row[2], row[3], row[4], row[5]
            if name == "eligibility":
                continue
            kept_cols.append(name)
            pk_str = " PRIMARY KEY" if pk else ""
            nn_str = " NOT NULL" if notnull else ""
            df_str = f" DEFAULT {dflt}" if dflt is not None else ""
            col_defs.append(f"{name} {typ}{pk_str}{nn_str}{df_str}")

        # Add FK constraints
        fk_defs: list[str] = []
        for fk_id in sorted(fk_by_id.keys()):
            cols_in_fk = sorted(fk_by_id[fk_id], key=lambda x: x[0])
            from_cols = [col[1] for col in cols_in_fk]
            # All columns in a composite FK should reference the same table
            fk_table = cols_in_fk[0][2]
            to_cols = [col[3] for col in cols_in_fk]
            fk_def = f"FOREIGN KEY ({', '.join(from_cols)}) REFERENCES {fk_table}({', '.join(to_cols)})"
            fk_defs.append(fk_def)

        all_defs = col_defs + fk_defs
        conn.execute(text(f"CREATE TABLE clips ({', '.join(all_defs)})"))
        conn.execute(
            text(
                f"INSERT INTO clips ({', '.join(kept_cols)}) "
                f"SELECT {', '.join(kept_cols)} FROM clips_old"
            )
        )
        conn.execute(text("DROP TABLE clips_old"))

        conn.execute(text("PRAGMA legacy_alter_table = OFF"))
        conn.execute(text("PRAGMA foreign_keys = ON"))

        violations = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
        if violations:
            raise RuntimeError(f"FK integrity violated after migration: {violations}")

        print("  OK: dropped clips.eligibility")

    if has_downloads_table:
        conn.execute(text("DROP TABLE downloads"))
        print("  OK: dropped downloads table")


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
    tables = {
        r[0]
        for r in conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public'"
            )
        ).fetchall()
    }

    if "is_downloaded" not in clip_cols:
        conn.execute(text("ALTER TABLE clips ADD COLUMN is_downloaded BOOLEAN"))

    if "downloads" in tables:
        conn.execute(
            text(
                "UPDATE clips SET is_downloaded = TRUE "
                "WHERE is_selected = TRUE "
                "AND id IN ("
                "  SELECT entity_id FROM downloads "
                "  WHERE file_type = 'video' AND success = TRUE"
                ")"
            )
        )
        conn.execute(text("DROP TABLE downloads"))

    if "eligibility" in clip_cols:
        conn.execute(text("ALTER TABLE clips DROP COLUMN eligibility"))
    print(
        "  OK: clips.eligibility dropped, clips.is_downloaded added, downloads dropped"
    )


if __name__ == "__main__":
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)
    migrate_database(database_url)
