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
    user_id: int,
    id: int,
    play_count: int | None,
    is_selected: bool,
    is_downloaded: bool | None = None,
):
    session.add(
        Clip(
            user_id=user_id,
            id=id,
            play_count=play_count,
            is_selected=is_selected,
            is_downloaded=is_downloaded,
        )
    )


def test_users_summary_to_markdown_renders_curated_metrics():
    eng = _make_engine()
    with Session(eng) as s:
        s.add_all(
            [
                User(
                    id=1,
                    following_count=100,
                    is_eligible=True,
                ),
                User(
                    id=2,
                    following_count=500,
                    is_eligible=False,
                ),
                User(
                    id=3,
                    following_count=50,
                    is_eligible=True,
                ),
                User(
                    id=4,
                    following_count=200,
                    is_eligible=True,
                ),
            ]
        )
        _add_clip(
            s, user_id=1, id=100, play_count=10, is_selected=True, is_downloaded=True
        )
        _add_clip(
            s, user_id=1, id=101, play_count=20, is_selected=True, is_downloaded=True
        )
        _add_clip(
            s, user_id=1, id=102, play_count=None, is_selected=True, is_downloaded=True
        )
        _add_clip(
            s, user_id=2, id=200, play_count=40, is_selected=True, is_downloaded=True
        )
        _add_clip(
            s, user_id=3, id=300, play_count=None, is_selected=False, is_downloaded=None
        )
        _add_clip(
            s, user_id=4, id=400, play_count=30, is_selected=True, is_downloaded=True
        )
        _add_clip(
            s, user_id=3, id=401, play_count=21, is_selected=True, is_downloaded=True
        )
        s.commit()

    from modules.visualization.tables.dataset_summary_users import (
        users_summary_to_markdown,
    )

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
                    following_count=100,
                    is_eligible=True,
                ),
                User(
                    id=2,
                    following_count=500,
                    is_eligible=False,
                ),
                User(
                    id=3,
                    following_count=50,
                    is_eligible=True,
                ),
            ]
        )
        _add_clip(
            s, user_id=1, id=100, play_count=200, is_selected=True, is_downloaded=True
        )
        _add_clip(
            s, user_id=1, id=101, play_count=100, is_selected=False, is_downloaded=None
        )
        _add_clip(
            s, user_id=3, id=300, play_count=50, is_selected=True, is_downloaded=True
        )
        s.commit()

    from modules.visualization.tables.dataset_summary_users import (
        users_summary_to_markdown,
    )

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
        s.add(User(id=1, is_eligible=True))
        s.commit()

    from modules.visualization.tables.dataset_summary_users import (
        users_summary_to_markdown,
    )

    out = users_summary_to_markdown(eng)

    assert r"| $\tilde{x}_{\mathrm{following}}$ | - |" in out
    assert r"| $\mu_\mathrm{following}$ | - |" in out
    assert r"| $[\min-max]_{\mathrm{following}}$ | - |" in out
    assert r"| $\tilde{x}_{\mathrm{views}}$ | - |" in out
    assert r"| $\mu_\mathrm{views}$ | - |" in out


def test_render_users_summary_returns_markdown_object():
    eng = _make_engine()
    with Session(eng) as s:
        s.add(User(id=1, is_eligible=True))
        s.commit()

    from docs.quarto_helpers import render_users_summary
    from modules.visualization.tables.dataset_summary_users import (
        users_summary_to_markdown,
    )

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
                    id INTEGER PRIMARY KEY,
                    following_count INTEGER,
                    is_eligible BOOLEAN
                )
                """
            )
        )
        conn.execute(text("INSERT INTO users (id, is_eligible) VALUES (1, 1)"))

    from modules.visualization.tables.dataset_summary_users import (
        users_summary_to_markdown,
    )

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
                    id INTEGER PRIMARY KEY,
                    following_count INTEGER,
                    is_eligible BOOLEAN
                )
                """
            )
        )
        conn.execute(
            text("CREATE TABLE clips (id BIGINT PRIMARY KEY, user_id BIGINT NOT NULL)")
        )
        conn.execute(text("INSERT INTO users (id, is_eligible) VALUES (1, 1)"))
        conn.execute(text("INSERT INTO clips (id, user_id) VALUES (1, 1)"))

    from modules.visualization.tables.dataset_summary_users import (
        users_summary_to_markdown,
    )

    out = users_summary_to_markdown(eng)

    assert r"| $N$ | 1 |" in out
    assert r"| $N_{\mathrm{kept}}$ | 1 (100.0%) |" in out
    assert r"| $\tilde{x}_{\mathrm{following}}$ | - |" in out
    assert r"| $\mu_\mathrm{following}$ | - |" in out
    assert r"| $[\min-max]_{\mathrm{following}}$ | - |" in out
    assert r"| $\tilde{x}_{\mathrm{views}}$ | - |" in out
    assert r"| $\mu_\mathrm{views}$ | - |" in out
