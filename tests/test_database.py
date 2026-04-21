import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_relative_sqlite_database_url_is_resolved_from_repo_root(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///data/inst2vec.db")

    if "modules.database" in sys.modules:
        del sys.modules["modules.database"]

    database = importlib.import_module("modules.database")

    resolved = database._resolve_database_url("sqlite:///data/inst2vec.db")
    expected = Path(__file__).resolve().parents[1] / "data" / "inst2vec.db"

    assert resolved == f"sqlite:///{expected}"
