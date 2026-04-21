import os
import sys

from IPython.display import Markdown
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.database import Base, User


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def test_users_summary_to_markdown_renders_curated_metrics():
    eng = _make_engine()
    with Session(eng) as s:
        s.add_all(
            [
                User(
                    pk=1,
                    username="alpha",
                    full_name="Alpha",
                    profile_pic_url="https://example.com/a.jpg",
                    profile_pic_url_hd="https://example.com/a-hd.jpg",
                    following_count=100,
                    city_name="Berlin",
                    user_disqualified=0,
                    parse_status="success",
                ),
                User(
                    pk=2,
                    username="beta",
                    profile_pic_url="https://example.com/b.jpg",
                    following_count=300,
                    user_disqualified=1,
                    parse_status="success",
                ),
                User(
                    pk=3,
                    username="gamma",
                    user_disqualified=None,
                    parse_status=None,
                ),
            ]
        )
        s.commit()

    from generators.dataset_summary_users import users_summary_to_markdown

    out = users_summary_to_markdown(eng)

    assert out.startswith("| Metric | Value |")
    assert "| Total users | 3 |" in out
    assert "| Parsed users | 2 (66.7%) |" in out
    assert "| Unresolved users | 1 (33.3%) |" in out
    assert "| Users kept | 1 (33.3%) |" in out
    assert "| Users disqualified | 1 (33.3%) |" in out
    assert "| Users with full name | 1 (33.3%) |" in out
    assert "| Users with profile picture | 2 (66.7%) |" in out
    assert "| Users with HD profile picture | 1 (33.3%) |" in out
    assert "| Users with city | 1 (33.3%) |" in out
    assert "| Following count (median, mean, min-max) | 200, 200.0, 100-300 |" in out


def test_users_summary_to_markdown_uses_dash_for_missing_numeric_values():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(User(pk=1, username="alpha", parse_status="success", user_disqualified=0))
        s.commit()

    from generators.dataset_summary_users import users_summary_to_markdown

    out = users_summary_to_markdown(eng)

    assert "| Following count (median, mean, min-max) | - |" in out


def test_render_users_summary_returns_markdown_object():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(User(pk=1, username="alpha", parse_status="success", user_disqualified=0))
        s.commit()

    from docs.quarto_helpers import render_users_summary
    from generators.dataset_summary_users import users_summary_to_markdown

    rendered = render_users_summary(eng=eng)

    assert isinstance(rendered, Markdown)
    assert rendered.data == users_summary_to_markdown(eng)


def test_users_summary_legacy_db_without_parse_status_column():
    """Older on-disk DBs may lack users.parse_status; summary must not crash."""
    eng = create_engine("sqlite:///:memory:")
    with eng.connect() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE users (
                    pk BIGINT PRIMARY KEY,
                    username VARCHAR NOT NULL UNIQUE,
                    full_name VARCHAR,
                    profile_pic_url VARCHAR,
                    profile_pic_url_hd VARCHAR,
                    following_count INTEGER,
                    city_name VARCHAR,
                    user_disqualified INTEGER
                )
                """
            )
        )
        conn.execute(
            text(
                "INSERT INTO users (pk, username, full_name, user_disqualified) "
                "VALUES (1, 'a', 'A', 0)"
            )
        )
        conn.commit()

    from generators.dataset_summary_users import users_summary_to_markdown

    out = users_summary_to_markdown(eng)

    assert "| Parsed users | - |" in out
    assert "| Unresolved users | - |" in out
    assert "| Total users | 1 |" in out
