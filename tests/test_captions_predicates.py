"""Tests for caption-stage database predicates."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import (
    Base,
    Clip,
    User,
    has_clean_caption,
    has_raw_caption,
    needs_caption_cleaning,
    needs_caption_language_detection,
    needs_caption_translation,
)


def _seed(s: Session) -> None:
    s.add(User(id=1, parse_status="success"))
    s.add(
        Clip(
            id=1,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text="hello @bob",
        )
    )
    s.add(
        Clip(
            id=2,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text="hi",
            caption_clean="hi",
        )
    )
    s.add(
        Clip(
            id=3,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text="hola",
            caption_clean="hola",
            caption_language="es",
        )
    )
    s.add(
        Clip(
            id=4,
            user_id=1,
            is_selected=True,
            is_downloaded=True,
            caption_text="hi",
            caption_clean="hi",
            caption_language="en",
        )
    )
    s.add(Clip(id=5, user_id=1, is_selected=False, caption_text="x"))
    s.commit()


def _ids(rows):
    return sorted(c.id for c in rows)


def test_predicates_select_correct_clips():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        _seed(s)
        assert _ids(s.query(Clip).filter(*needs_caption_cleaning()).all()) == [1]
        assert _ids(
            s.query(Clip).filter(*needs_caption_language_detection()).all()
        ) == [2]
        assert _ids(s.query(Clip).filter(*needs_caption_translation()).all()) == [3]


def test_has_raw_and_clean_caption_filters():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        _seed(s)
        assert _ids(s.query(Clip).filter(*has_raw_caption()).all()) == [1, 2, 3, 4, 5]
        assert _ids(s.query(Clip).filter(*has_clean_caption()).all()) == [2, 3, 4]


def test_translation_predicate_skips_undetermined_language():
    """Rows sealed as 'und' (Lingua-undetected) are terminally classified
    and must not be selected for translation."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        _seed(s)
        s.add(
            Clip(
                id=11,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                caption_text="🔥🔥🔥",
                caption_clean="🔥🔥🔥",
                caption_language="und",
            )
        )
        s.commit()
        assert 11 not in _ids(s.query(Clip).filter(*needs_caption_translation()).all())
        assert 11 not in _ids(
            s.query(Clip).filter(*needs_caption_language_detection()).all()
        )


def test_translation_predicate_skips_english_prefix_variants():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        _seed(s)
        s.add(
            Clip(
                id=10,
                user_id=1,
                is_selected=True,
                is_downloaded=True,
                caption_text="hi",
                caption_clean="hi",
                caption_language="en-us",
            )
        )
        s.commit()
        assert 10 not in _ids(s.query(Clip).filter(*needs_caption_translation()).all())
