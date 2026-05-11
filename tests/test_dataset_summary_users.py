import os
import sys

from IPython.display import Markdown
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.database import Base, Clip, User


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _add_clip(
    session: Session,
    *,
    user_pk: int,
    pk: int,
    play_count: int | None,
    disqualified: int | None,
):
    session.add(
        Clip(user_id=user_pk, id=pk, play_count=play_count, disqualified=disqualified)
    )


def test_users_summary_to_markdown_renders_curated_metrics():
    eng = _make_engine()
    with Session(eng) as s:
        s.add_all(
            [
                User(
                    id=1,
                    username="alpha",
                    full_name="Alpha",
                    profile_pic_url="https://example.com/a.jpg",
                    profile_pic_url_hd="https://example.com/a-hd.jpg",
                    following_count=100,
                    city_name="Berlin",
                    user_disqualified=0,
                ),
                User(
                    id=2,
                    username="beta",
                    full_name="Beta",
                    profile_pic_url="https://example.com/b.jpg",
                    following_count=500,
                    city_name="Paris",
                    user_disqualified=1,
                ),
                User(
                    id=3,
                    username="gamma",
                    following_count=50,
                    user_disqualified=0,
                ),
                User(
                    id=4,
                    username="delta",
                    following_count=200,
                    user_disqualified=0,
                ),
            ]
        )
        _add_clip(s, user_pk=1, pk=100, play_count=10, disqualified=0)
        _add_clip(s, user_pk=1, pk=101, play_count=20, disqualified=0)
        _add_clip(s, user_pk=1, pk=102, play_count=None, disqualified=0)
        _add_clip(s, user_pk=2, pk=200, play_count=40, disqualified=0)
        _add_clip(s, user_pk=3, pk=300, play_count=None, disqualified=1)
        _add_clip(s, user_pk=4, pk=400, play_count=30, disqualified=0)
        _add_clip(s, user_pk=3, pk=401, play_count=21, disqualified=0)
        s.commit()

    from generators.dataset_summary_users import users_summary_to_markdown

    out = users_summary_to_markdown(eng)

    assert out.startswith("| Metric | Value |")
    assert r"| $N$ | 4 |" in out
    assert r"| $N_{\mathrm{kept}}$ | 3 (75.0%) |" in out
    assert r"| $\tilde{x}_{\mathrm{following}}$ | 100 |" in out
    assert r"| $\mu_\mathrm{following}$ | 116.7 |" in out
    assert r"| $[\min-max]_{\mathrm{following}}$ | 50-200 |" in out
    assert r"| $\tilde{x}_{\mathrm{views}}$ | 21 |" in out
    assert r"| $\mu_\mathrm{views}$ | 22.0 |" in out


def test_users_summary_to_markdown_scopes_users_to_kept_rows():
    eng = _make_engine()
    with Session(eng) as s:
        s.add_all(
            [
                User(
                    id=1,
                    username="kept_full",
                    full_name="Kept User",
                    profile_pic_url="https://example.com/kept.jpg",
                    profile_pic_url_hd="https://example.com/kept-hd.jpg",
                    city_name="Berlin",
                    following_count=100,
                    user_disqualified=0,
                ),
                User(
                    id=2,
                    username="disqualified_rich",
                    full_name="Disqualified User",
                    profile_pic_url="https://example.com/disqualified.jpg",
                    following_count=500,
                    city_name="Paris",
                    user_disqualified=1,
                ),
                User(
                    id=3,
                    username="kept_sparse",
                    following_count=50,
                    user_disqualified=0,
                ),
            ]
        )
        _add_clip(s, user_pk=1, pk=100, play_count=200, disqualified=0)
        _add_clip(s, user_pk=1, pk=101, play_count=100, disqualified=1)
        _add_clip(s, user_pk=3, pk=300, play_count=50, disqualified=0)
        s.commit()

    from generators.dataset_summary_users import users_summary_to_markdown

    out = users_summary_to_markdown(eng)

    assert r"| $N_{\mathrm{kept}}$ | 2 (66.7%) |" in out
    assert r"| $\tilde{x}_{\mathrm{following}}$ | 75 |" in out
    assert r"| $\mu_\mathrm{following}$ | 75.0 |" in out
    assert r"| $[\min-max]_{\mathrm{following}}$ | 50-100 |" in out
    assert r"| $\tilde{x}_{\mathrm{views}}$ | 125 |" in out
    assert r"| $\mu_\mathrm{views}$ | 125.0 |" in out


def test_users_summary_to_markdown_uses_dash_for_missing_numeric_values():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(User(id=1, username="alpha", user_disqualified=0))
        s.commit()

    from generators.dataset_summary_users import users_summary_to_markdown

    out = users_summary_to_markdown(eng)

    assert r"| $\tilde{x}_{\mathrm{following}}$ | - |" in out
    assert r"| $\mu_\mathrm{following}$ | - |" in out
    assert r"| $[\min-max]_{\mathrm{following}}$ | - |" in out
    assert r"| $\tilde{x}_{\mathrm{views}}$ | - |" in out
    assert r"| $\mu_\mathrm{views}$ | - |" in out


def test_render_users_summary_returns_markdown_object():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(User(id=1, username="alpha", user_disqualified=0))
        s.commit()

    from docs.quarto_helpers import render_users_summary
    from generators.dataset_summary_users import users_summary_to_markdown

    rendered = render_users_summary(eng=eng)

    assert isinstance(rendered, Markdown)
    assert rendered.data == users_summary_to_markdown(eng)


def test_users_summary_legacy_db_without_parse_status_column():
    """Older on-disk DBs may lack users.parse_status; summary must not crash."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE users (
                    id BIGINT PRIMARY KEY,
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
                "INSERT INTO users (id, username, full_name, user_disqualified) "
                "VALUES (1, 'a', 'A', 0)"
            )
        )

    from generators.dataset_summary_users import users_summary_to_markdown

    out = users_summary_to_markdown(eng)

    assert r"| $N$ | 1 |" in out
    assert r"| $N_{\mathrm{kept}}$ | 1 (100.0%) |" in out
    assert r"| $\tilde{x}_{\mathrm{following}}$ | - |" in out
    assert r"| $\mu_\mathrm{following}$ | - |" in out
    assert r"| $[\min-max]_{\mathrm{following}}$ | - |" in out
    assert r"| $\tilde{x}_{\mathrm{views}}$ | - |" in out
    assert r"| $\mu_\mathrm{views}$ | - |" in out


def test_users_summary_legacy_db_without_play_count_columns():
    """Legacy DB with legacy clips schema should still render safely."""
    eng = create_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE users (
                    id BIGINT PRIMARY KEY,
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
            text("CREATE TABLE clips (id BIGINT PRIMARY KEY, user_id BIGINT NOT NULL)")
        )
        conn.execute(
            text(
                "INSERT INTO users (id, username, full_name, user_disqualified) "
                "VALUES (1, 'a', 'A', 0)"
            )
        )
        conn.execute(text("INSERT INTO clips (id, user_id) VALUES (1, 1)"))

    from generators.dataset_summary_users import users_summary_to_markdown

    out = users_summary_to_markdown(eng)

    assert r"| $N$ | 1 |" in out
    assert r"| $N_{\mathrm{kept}}$ | 1 (100.0%) |" in out
    assert r"| $\tilde{x}_{\mathrm{following}}$ | - |" in out
    assert r"| $\mu_\mathrm{following}$ | - |" in out
    assert r"| $[\min-max]_{\mathrm{following}}$ | - |" in out
    assert r"| $\tilde{x}_{\mathrm{views}}$ | - |" in out
    assert r"| $\mu_\mathrm{views}$ | - |" in out
