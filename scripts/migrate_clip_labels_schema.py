"""One-shot SQLite migration: rebuild ``clip_labels`` with composite PK.

Pre-multimodal-labels installations have ``clip_labels`` keyed on
``clip_id`` alone; multimodal-labels requires ``(clip_id, label_case)``
and a nullable ``source_hash`` column. SQLite ALTER TABLE cannot change a
primary key in place, so we rebuild the table:

    CREATE TABLE clip_labels_new …
    INSERT INTO clip_labels_new SELECT *, 'video', NULL FROM clip_labels
    DROP TABLE clip_labels
    ALTER TABLE clip_labels_new RENAME TO clip_labels

Idempotent: detects the new schema (``label_case`` column present) and
short-circuits. Safe to re-run. Reads ``DATABASE_URL`` from the env (or
``.env``) and only supports ``sqlite:///`` URLs.

Run:
    uv run python scripts/migrate_clip_labels_schema.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path


def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _sqlite_path(url: str) -> str:
    if not url.startswith("sqlite:///"):
        raise SystemExit(f"only sqlite urls supported, got: {url}")
    return url.replace("sqlite:///", "", 1)


def migrate(db_path: str) -> str:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(clip_labels)").fetchall()}
        if not cols:
            return "noop: clip_labels table does not exist"
        if "label_case" in cols:
            return "noop: clip_labels already on multimodal schema"
        before = conn.execute("SELECT COUNT(*) FROM clip_labels").fetchone()[0]
        conn.executescript(
            """
            BEGIN;
            CREATE TABLE clip_labels_new (
                clip_id BIGINT NOT NULL,
                label_case VARCHAR NOT NULL DEFAULT 'video',
                status VARCHAR NOT NULL,
                validation VARCHAR,
                payload JSON,
                warnings JSON,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_hash VARCHAR,
                PRIMARY KEY (clip_id, label_case),
                FOREIGN KEY (clip_id) REFERENCES clips (id)
            );
            INSERT INTO clip_labels_new
                (clip_id, label_case, status, validation, payload, warnings,
                 error, attempts, updated_at, source_hash)
            SELECT clip_id, 'video', status, validation, payload, warnings,
                   error, attempts, updated_at, NULL
            FROM clip_labels;
            DROP TABLE clip_labels;
            ALTER TABLE clip_labels_new RENAME TO clip_labels;
            COMMIT;
            """
        )
        after = conn.execute("SELECT COUNT(*) FROM clip_labels").fetchone()[0]
        if after != before:
            raise SystemExit(
                f"row-count mismatch: {before} before, {after} after — aborting"
            )
        return f"migrated: clip_labels rebuilt with composite PK, {after} rows"


if __name__ == "__main__":
    _load_env()
    db_url = os.environ.get("DATABASE_URL", "sqlite:///data/inst2vec.db")
    print(migrate(_sqlite_path(db_url)), file=sys.stdout)
