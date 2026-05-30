"""Serving DB config + engine wiring.

The serving database is a separate, read-optimised store the offload
script writes and the atlas API reads. It has its own ``ServingBase``
metadata, its own engine global, and an explicit ``init_serving_db`` so
ordinary pipeline runs that never touch serving stay cheap.
"""

from __future__ import annotations

import os

from core.config import Secrets


def test_secrets_exposes_serving_database_url():
    s = Secrets(
        database_url="sqlite:///:memory:",
        identity_db_url="sqlite:///:memory:",
        hiker_api_key="k",
        huggingface_token="t",
    )
    # A sensible SQLite default is supplied when the env var is unset.
    assert s.serving_database_url


def test_load_runtime_config_reads_serving_env(monkeypatch):
    from core.config import load_runtime_config

    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("IDENTITY_DB_URL", "sqlite:///:memory:")
    monkeypatch.setenv("HIKER_API_KEY", "k")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "t")
    monkeypatch.setenv("SERVING_DATABASE_URL", "sqlite:///custom_serving.db")
    _, secrets = load_runtime_config()
    assert secrets.serving_database_url == "sqlite:///custom_serving.db"


def test_init_serving_db_creates_tables_and_session(tmp_path):
    from sqlalchemy import inspect

    from core.database import (
        ServingBase,
        get_serving_engine,
        get_serving_session,
        init_serving_db,
    )

    url = f"sqlite:///{tmp_path / 'serving.db'}"
    init_serving_db(url)
    engine = get_serving_engine()
    names = set(inspect(engine).get_table_names())
    assert names >= set(ServingBase.metadata.tables)
    with get_serving_session() as s:
        # A trivial query proves the session binds to the serving engine.
        assert s.bind is engine


def test_init_serving_db_wraps_bare_path(tmp_path):
    from core.database import get_serving_engine, init_serving_db

    bare = str(tmp_path / "bare_serving.db")
    init_serving_db(bare)
    assert str(get_serving_engine().url).startswith("sqlite:///")
    # Re-init the in-memory default so later session-scoped tests are unaffected.
    os.environ.setdefault("SERVING_DATABASE_URL", "sqlite:///:memory:")
