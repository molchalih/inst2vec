from sqlalchemy import create_engine, inspect, text

from scripts.migrate_eligibility_columns import migrate_database


def test_sqlite_migration_renames_columns_and_maps_values(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, user_disqualified INTEGER)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE clips (id INTEGER PRIMARY KEY, user_id INTEGER, disqualified INTEGER)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE cluster_runs (id INTEGER PRIMARY KEY, disqualified INTEGER, in_current_grid INTEGER)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO users (id, user_disqualified) VALUES (1, NULL), (2, 0), (3, 1)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO clips (id, user_id, disqualified) VALUES (10, 1, NULL), (11, 1, 0), (12, 1, 1)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO cluster_runs (id, disqualified, in_current_grid) VALUES (100, NULL, 1), (101, 0, 1), (102, 1, 0)"
            )
        )

    migrate_database(eng)

    insp = inspect(eng)
    assert {c["name"] for c in insp.get_columns("users")} >= {"eligibility"}
    assert {c["name"] for c in insp.get_columns("clips")} >= {"eligibility"}
    assert {c["name"] for c in insp.get_columns("cluster_runs")} >= {"eligibility"}
    clip_columns = {c["name"]: c for c in insp.get_columns("clips")}
    assert clip_columns["eligibility"]["default"] in {"0", 0, "'0'"}
    assert clip_columns["eligibility"]["nullable"] is False

    with eng.connect() as conn:
        user_rows = conn.execute(
            text("SELECT id, eligibility FROM users ORDER BY id")
        ).fetchall()
        clip_rows = conn.execute(
            text("SELECT id, eligibility FROM clips ORDER BY id")
        ).fetchall()
        run_rows = conn.execute(
            text("SELECT id, eligibility FROM cluster_runs ORDER BY id")
        ).fetchall()

    assert user_rows == [(1, 0), (2, 1), (3, 2)]
    assert clip_rows == [(10, 0), (11, 1), (12, 2)]
    assert run_rows == [(100, 0), (101, 1), (102, 2)]
