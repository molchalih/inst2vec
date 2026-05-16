from sqlalchemy import create_engine, inspect, text

from scripts.migrate_cluster_runs_passes_validation import migrate_database


def _legacy_cluster_runs_ddl() -> str:
    """Mirror the pre-refactor schema of `cluster_runs` for SQLite."""
    return """
    CREATE TABLE cluster_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding_case TEXT NOT NULL,
        umap_n_components INTEGER NOT NULL,
        umap_n_neighbors INTEGER NOT NULL,
        umap_min_dist REAL NOT NULL,
        umap_metric TEXT NOT NULL,
        umap2d_n_neighbors INTEGER NOT NULL,
        umap2d_min_dist REAL NOT NULL,
        umap2d_metric TEXT NOT NULL,
        hdbscan_min_cluster_size INTEGER NOT NULL,
        hdbscan_min_samples INTEGER,
        hdbscan_cluster_selection_method TEXT NOT NULL,
        hdbscan_metric TEXT NOT NULL,
        random_state INTEGER NOT NULL,
        n_clusters INTEGER NOT NULL,
        noise_ratio REAL NOT NULL,
        min_size INTEGER NOT NULL,
        median_size INTEGER NOT NULL,
        max_size INTEGER NOT NULL,
        eligibility INTEGER NOT NULL DEFAULT 0,
        dbcv REAL,
        silhouette REAL,
        param_plateau_score REAL,
        in_current_grid INTEGER,
        dataset_hash TEXT,
        validation_config_hash TEXT,
        created_at TEXT
    )
    """


def test_sqlite_migration_drops_legacy_and_adds_passes_validation(tmp_path):
    db_path = tmp_path / "legacy.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

    with eng.begin() as conn:
        conn.execute(text(_legacy_cluster_runs_ddl()))
        conn.execute(
            text(
                "INSERT INTO cluster_runs ("
                "id, embedding_case, umap_n_components, umap_n_neighbors, umap_min_dist,"
                " umap_metric, umap2d_n_neighbors, umap2d_min_dist, umap2d_metric,"
                " hdbscan_min_cluster_size, hdbscan_min_samples,"
                " hdbscan_cluster_selection_method, hdbscan_metric, random_state,"
                " n_clusters, noise_ratio, min_size, median_size, max_size,"
                " eligibility, dbcv, silhouette, param_plateau_score,"
                " in_current_grid, dataset_hash, validation_config_hash,"
                " created_at"
                ") VALUES ("
                "1, 'video', 3, 5, 0.1, 'cosine', 5, 0.1, 'cosine',"
                " 5, NULL, 'eom', 'euclidean', 42,"
                " 3, 0.05, 5, 10, 15,"
                " 0, 0.5, 0.4, 0.48,"
                " 1, 'datahash', 'cfghash',"
                " '2026-01-01 00:00:00')"
            )
        )

    migrate_database(eng)

    insp = inspect(eng)
    cols = {c["name"] for c in insp.get_columns("cluster_runs")}
    assert "passes_validation" in cols
    for legacy in (
        "eligibility",
        "in_current_grid",
        "dataset_hash",
        "validation_config_hash",
    ):
        assert legacy not in cols, f"{legacy} should be dropped"

    with eng.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, embedding_case, dbcv, silhouette, param_plateau_score,"
                " passes_validation FROM cluster_runs"
            )
        ).fetchall()
    assert rows == [(1, "video", 0.5, 0.4, 0.48, None)]


def test_sqlite_migration_is_idempotent_on_new_schema(tmp_path):
    db_path = tmp_path / "fresh.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")

    from modules.database import Base

    Base.metadata.create_all(eng)
    # Should not raise / not rebuild the table.
    migrate_database(eng)

    insp = inspect(eng)
    cols = {c["name"] for c in insp.get_columns("cluster_runs")}
    assert "passes_validation" in cols
    assert "eligibility" not in cols


def test_sqlite_migration_no_op_when_table_missing(tmp_path):
    db_path = tmp_path / "empty.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")
    # No tables at all — must exit cleanly.
    migrate_database(eng)
    assert "cluster_runs" not in inspect(eng).get_table_names()
