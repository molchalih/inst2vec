"""Uvicorn entrypoint for the swipe-anchor backend.

Run with the ``serving`` dependency group::

    uv run --group serving python -m swipe_anchor.backend

Configuration via environment (mirrors ``services.atlas_api``):
    APP_DATABASE_URL        app-store DB url (default: sqlite:///data/swipe_anchor.db)
    SWIPE_ANCHOR_TOKEN      bearer token; empty disables auth (local MVP)
    SWIPE_ANCHOR_CORS       allowed CORS origin; empty disables CORS
    SWIPE_ANCHOR_HOST/PORT  bind address (default 0.0.0.0:8100)
    SWIPE_ANCHOR_LOG        activity log file (default: data/swipe_anchor.log)
    SWIPE_ANCHOR_FRONTEND_DIST  if set, serve the built frontend from this dir at
                            the same origin (single-tunnel deploy, no CORS)
    SWIPE_ANCHOR_MEDIA_DIR  if set, serve creator reels/posters from this dir at
                            /media (e.g. the pipeline's data/source)
    TG_TOK                  Telegram bot token; empty -> Mini App wrapper OFF
                            (no /tg/* routes). Same token the bot uses.
    SA_TG_INTERNAL_TOKEN    shared secret for bot->backend /tg/register calls
                            (X-Internal-Token header)

Manage access codes with ``python -m swipe_anchor.backend.codes`` (add / list).
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy.orm import Session

from swipe_anchor.backend.app import build_app
from swipe_anchor.config import Settings
from swipe_anchor.db import create_app_engine, make_session_factory, session_scope


def _configure_logging() -> None:
    """Activity log to stdout (visible under screen) + a durable file.

    The per-choice timing/identity lines come from ``swipe_anchor.activity``.
    """
    log_path = Path(os.environ.get("SWIPE_ANCHOR_LOG", "data/swipe_anchor.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    root = logging.getLogger("swipe_anchor")
    root.setLevel(logging.INFO)
    if not root.handlers:
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        file = logging.FileHandler(log_path)
        file.setFormatter(fmt)
        root.addHandler(stream)
        root.addHandler(file)


def _build_from_env():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # python-dotenv is optional for local runs
        pass

    # Treat an empty value as unset so a blank APP_DATABASE_URL falls back to the
    # default file path (not the silent in-memory ``sqlite:///``).
    url = os.environ.get("APP_DATABASE_URL") or "sqlite:///data/swipe_anchor.db"
    engine = create_app_engine(url)
    factory = make_session_factory(engine)

    @contextmanager
    def session_factory() -> Iterator[Session]:
        with session_scope(factory) as s:
            yield s

    settings = Settings.from_env()
    app = build_app(
        session_factory=session_factory,
        token=os.environ.get("SWIPE_ANCHOR_TOKEN", ""),
        cors_origin=os.environ.get("SWIPE_ANCHOR_CORS", ""),
        frontend_dist=os.environ.get("SWIPE_ANCHOR_FRONTEND_DIST") or None,
        media_dir=os.environ.get("SWIPE_ANCHOR_MEDIA_DIR") or None,
        settings=settings,
        # Telegram Mini App wrapper (feature-flagged: empty TG_TOK -> endpoints
        # 404, default deploys unchanged).
        tg_bot_token=os.environ.get("TG_TOK", ""),
        tg_internal_token=os.environ.get("SA_TG_INTERNAL_TOKEN", ""),
    )
    return app, session_factory, settings


def main() -> int:
    import asyncio

    import uvicorn

    from swipe_anchor.backend.sweep import run_sweep_loop

    _configure_logging()
    app, session_factory, settings = _build_from_env()

    @app.on_event("startup")
    async def _start_sweep() -> None:
        app.state.sweep_task = asyncio.create_task(run_sweep_loop(session_factory, settings))

    @app.on_event("shutdown")
    async def _stop_sweep() -> None:
        task = getattr(app.state, "sweep_task", None)
        if task:
            task.cancel()

    uvicorn.run(
        app,
        host=os.environ.get("SWIPE_ANCHOR_HOST", "0.0.0.0"),
        port=int(os.environ.get("SWIPE_ANCHOR_PORT", "8100")),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
