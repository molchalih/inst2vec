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
    from core.database import engine as engine_mod

    monkeypatch.setattr(engine_mod, "_main_engine", None)
    with pytest.raises((AssertionError, RuntimeError)):
        engine_mod.get_engine()


def test_init_db_sets_engine(tmp_path, monkeypatch):
    from core.database import engine as engine_mod

    monkeypatch.setattr(engine_mod, "_main_engine", None)
    url = f"sqlite:///{tmp_path}/test.db"
    identity_url = "sqlite:///:memory:"
    engine_mod.init_db(url, identity_url)
    assert engine_mod.get_engine() is not None


def test_user_has_follower_count_column():
    from core.database import User

    assert hasattr(User, "follower_count")


def test_clip_has_video_duration_column():
    from core.database import Clip

    assert hasattr(Clip, "video_duration")


def test_clip_has_taken_at_column():
    from core.database import Clip

    assert hasattr(Clip, "taken_at")


def test_clip_has_is_downloaded_column():
    from core.database import Clip

    assert "is_downloaded" in Clip.__table__.columns


def test_clip_used_in_analysis_returns_two_clauses():
    from core.database import clip_used_in_analysis

    clauses = clip_used_in_analysis()
    assert len(clauses) == 2
    rendered = " | ".join(str(c) for c in clauses)
    assert "is_selected" in rendered
    assert "is_downloaded" in rendered


def test_clip_no_longer_has_eligibility_column():
    from core.database import Clip

    assert "eligibility" not in Clip.__table__.columns


def test_download_model_removed():
    import core.database as db

    assert not hasattr(db, "Download")


def test_clip_needs_speech_detection_filter():
    """Returns only clips that are selected, downloaded, and NULL is_speech_detected."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from core.database import (
        Base,
        Clip,
        User,
        clip_needs_speech_detection,
    )

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success"))
        s.add(
            Clip(
                id=10,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=None,
            )
        )
        s.add(
            Clip(
                id=11,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=True,
            )
        )
        s.add(
            Clip(
                id=12,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=False,
            )
        )
        s.add(
            Clip(
                id=13,
                user_id=1,
                is_selected=False,
                is_downloaded=True,
                is_speech_detected=None,
            )
        )
        s.add(
            Clip(
                id=14,
                user_id=1,
                is_selected=True,
                is_downloaded=False,
                is_speech_detected=None,
            )
        )
        s.commit()
        ids = [c.id for c in s.query(Clip).filter(*clip_needs_speech_detection()).all()]
        assert ids == [10]


def test_clip_has_detected_speech_filter():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from core.database import (
        Base,
        Clip,
        User,
        clip_has_detected_speech,
    )

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success"))
        s.add(
            Clip(
                id=10,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=True,
            )
        )
        s.add(
            Clip(
                id=11,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=None,
            )
        )
        s.add(
            Clip(
                id=12,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=False,
            )
        )
        s.commit()
        ids = [c.id for c in s.query(Clip).filter(*clip_has_detected_speech()).all()]
        assert ids == [10]


def test_clip_needs_speech_translation_filter():
    """Selected + downloaded + is_speech_detected=True + non-empty
    non-English transcription/language + missing translation."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from core.database import (
        Base,
        Clip,
        User,
        clip_needs_speech_translation,
    )

    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1, parse_status="success"))
        # Translatable: non-English, has transcription, no translation
        s.add(
            Clip(
                id=10,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=True,
                speech_transcription="bonjour",
                speech_language="fr",
                speech_translation=None,
            )
        )
        # Already translated — excluded
        s.add(
            Clip(
                id=11,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=True,
                speech_transcription="bonjour",
                speech_language="fr",
                speech_translation="hello",
            )
        )
        # English — excluded
        s.add(
            Clip(
                id=12,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=True,
                speech_transcription="hello",
                speech_language="en",
                speech_translation=None,
            )
        )
        # is_speech_detected False — excluded
        s.add(
            Clip(
                id=13,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=False,
                speech_transcription="bonjour",
                speech_language="fr",
                speech_translation=None,
            )
        )
        # Empty transcription — excluded
        s.add(
            Clip(
                id=14,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                is_speech_detected=True,
                speech_transcription="",
                speech_language="fr",
                speech_translation=None,
            )
        )
        s.commit()
        ids = [
            c.id for c in s.query(Clip).filter(*clip_needs_speech_translation()).all()
        ]
        assert ids == [10]


def test_user_embedding_has_nullable_source_hash():
    from sqlalchemy import inspect

    from core.database import UserEmbedding, get_engine

    cols = {c["name"]: c for c in inspect(get_engine()).get_columns("user_embeddings")}
    assert "source_hash" in cols, "UserEmbedding must expose source_hash"
    assert cols["source_hash"]["nullable"] is True
    assert UserEmbedding.source_hash.property.columns[0].nullable is True
