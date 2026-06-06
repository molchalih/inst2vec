"""The app can run a single sweep tick via the exposed coroutine (design §3.5)."""

import asyncio

from swipe_anchor.backend.sweep import sweep_once
from swipe_anchor.config import Settings
from swipe_anchor.db import create_app_engine, make_session_factory, session_scope


def test_sweep_once_runs_against_a_factory() -> None:
    engine = create_app_engine("sqlite:///:memory:")
    factory = make_session_factory(engine)

    def session_factory():
        return session_scope(factory)

    # Should not raise on an empty store.
    asyncio.run(sweep_once(session_factory, Settings()))
