import importlib
import sys
from pathlib import Path

from sqlalchemy import text


def test_quarto_default_engine_resolves_relative_sqlite_url(monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    data_dir.mkdir(exist_ok=True)

    db_path = data_dir / "test_quarto_relative_path.sqlite"
    db_path.unlink(missing_ok=True)

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///data/{db_path.name}")
    monkeypatch.chdir(repo_root / "docs")
    sys.modules.pop("docs.quarto_helpers", None)

    quarto_helpers = importlib.import_module("docs.quarto_helpers")

    try:
        quarto_helpers._get_default_engine.cache_clear()
        with quarto_helpers._get_default_engine().connect() as conn:
            assert conn.execute(text("select 1")).scalar() == 1
        assert db_path.exists()
    finally:
        quarto_helpers._get_default_engine().dispose()
        db_path.unlink(missing_ok=True)
