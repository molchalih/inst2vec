from __future__ import annotations

import pytest
from sqlalchemy import Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from core.lang import is_english, sql_is_english


@pytest.mark.parametrize(
    "code, expected",
    [
        ("en", True),
        ("EN", True),
        ("en-US", True),
        ("en_GB", True),
        ("eng", True),
        ("English", True),
        ("fr", False),
        ("de", False),
        ("", False),
        (None, False),
    ],
)
def test_is_english(code, expected):
    assert is_english(code) is expected


class _Base(DeclarativeBase):
    pass


class _Row(_Base):
    __tablename__ = "lang_rows"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lang: Mapped[str | None] = mapped_column(String, nullable=True)


def test_sql_is_english_matches_python_helper():
    engine = create_engine("sqlite:///:memory:")
    _Base.metadata.create_all(engine)
    with Session(engine) as s:
        s.add_all(
            [
                _Row(id=1, lang="en"),
                _Row(id=2, lang="EN"),
                _Row(id=3, lang="eng"),
                _Row(id=4, lang="en-US"),
                _Row(id=5, lang="fr"),
                _Row(id=6, lang=None),
            ]
        )
        s.commit()

        matched_ids = {
            r.id for r in s.query(_Row).filter(sql_is_english(_Row.lang)).all()
        }
        assert matched_ids == {1, 2, 3, 4}
