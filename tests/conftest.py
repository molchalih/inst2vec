import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("IDENTITY_DB_URL", "sqlite:///:memory:")


@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    """Initialize main + identity DB for the whole test session."""
    from modules.database import init_db

    init_db("sqlite:///:memory:", "sqlite:///:memory:")
