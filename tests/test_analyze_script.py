"""Smoke test for scripts/analyze.py: report runs to completion without
exceptions on a populated DB. Guards against AttributeError on missing
Clip columns and against missing init_db() before get_session()."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def populated_db(tmp_path, monkeypatch):
    db_url = f"sqlite:///{tmp_path}/inst2vec.db"
    identity_url = f"sqlite:///{tmp_path}/identity.db"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("IDENTITY_DB_URL", identity_url)

    from core.database import Clip, User, get_session, init_db

    init_db(db_url, identity_url)
    session = get_session()
    session.add(User(id=1, is_selected=True, is_eligible=True))
    session.add(User(id=2, is_selected=True, is_eligible=False))
    session.add(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
    session.add(Clip(id=11, user_id=2, is_selected=False, is_downloaded=False))
    session.commit()
    session.close()
    return db_url, identity_url


def test_analyze_main_runs_without_errors(populated_db):
    """scripts/analyze.py must not raise AttributeError on Clip.disqualified
    or AssertionError on uninitialized engine."""
    db_url, identity_url = populated_db
    repo_root = Path(__file__).parent.parent
    env = os.environ.copy()
    env["DATABASE_URL"] = db_url
    env["IDENTITY_DB_URL"] = identity_url
    result = subprocess.run(
        [sys.executable, "scripts/analyze.py"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"analyze.py failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "PIPELINE HEALTH" in result.stdout
    assert "Clips disqualified" in result.stdout
