"""Alembic environment for the three inst2vec databases.

One env.py serves three logical databases selected by ``-n <name>``:

    alembic -n main     upgrade head
    alembic -n identity upgrade head
    alembic -n serving  upgrade head

Each name maps to (1) its own ``version_locations`` (from alembic.ini),
(2) its own SQLAlchemy metadata, and (3) the env var holding its URL — the
same sources ``core.config`` reads, so migrations target SQLite (dev) or
Postgres (prod) unchanged. Bare file paths are wrapped with ``sqlite:///`` to
match the engine layer.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the project importable when alembic is invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.database.engine import _wrap_sqlite_path
from core.database.identity import IdentityBase
from core.database.models import Base
from core.database.serving_models import ServingBase

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The logical DB is the ini section alembic selects from ``-n``; default to
# ``main`` so a bare ``alembic`` invocation still resolves.
_DB_NAME = config.config_ini_section or "main"

_METADATA = {
    "main": Base.metadata,
    "identity": IdentityBase.metadata,
    "serving": ServingBase.metadata,
}
_URL_ENV = {
    "main": "DATABASE_URL",
    "identity": "IDENTITY_DB_URL",
    "serving": "SERVING_DATABASE_URL",
}

if _DB_NAME not in _METADATA:
    raise RuntimeError(
        f"unknown alembic database '{_DB_NAME}'; expected one of {sorted(_METADATA)}"
    )

target_metadata = _METADATA[_DB_NAME]


def _resolve_url() -> str:
    # An explicit ``-x url=...`` (or sqlalchemy.url in the ini) wins; else read
    # the per-database env var, matching core.config.
    cli = context.get_x_argument(as_dictionary=True).get("url")
    if cli:
        return _wrap_sqlite_path(cli)
    configured = config.get_main_option("sqlalchemy.url")
    if configured:
        return _wrap_sqlite_path(configured)
    env_var = _URL_ENV[_DB_NAME]
    url = os.environ.get(env_var)
    if not url:
        raise RuntimeError(
            f"{env_var} must be set to run alembic for the '{_DB_NAME}' database"
        )
    return _wrap_sqlite_path(url)


def run_migrations_offline() -> None:
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        render_as_batch=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _resolve_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
