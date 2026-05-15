import importlib
import sys
from pathlib import Path

import pytest
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


def test_get_engine_raises_before_init(monkeypatch):
    import modules.database as db_mod

    monkeypatch.setattr(db_mod, "_engine", None)
    with pytest.raises((AssertionError, RuntimeError)):
        db_mod.get_engine()


def test_init_db_sets_engine(tmp_path, monkeypatch):
    import modules.database as db_mod

    monkeypatch.setattr(db_mod, "_engine", None)
    url = f"sqlite:///{tmp_path}/test.db"
    identity_url = "sqlite:///:memory:"
    db_mod.init_db(url, identity_url)
    assert db_mod.get_engine() is not None


def test_user_has_follower_count_column():
    from modules.database import User

    assert hasattr(User, "follower_count")


def test_clip_has_video_duration_column():
    from modules.database import Clip

    assert hasattr(Clip, "video_duration")


def test_clip_has_taken_at_column():
    from modules.database import Clip

    assert hasattr(Clip, "taken_at")


def test_clip_has_is_downloaded_column():
    from modules.database import Clip

    assert "is_downloaded" in Clip.__table__.columns


def test_clip_used_in_analysis_returns_two_clauses():
    from modules.database import clip_used_in_analysis

    clauses = clip_used_in_analysis()
    assert len(clauses) == 2
    rendered = " | ".join(str(c) for c in clauses)
    assert "is_selected" in rendered
    assert "is_downloaded" in rendered
