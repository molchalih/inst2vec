"""Smoke test for the stage_state table and StageState model."""

from datetime import datetime

import pytest

from modules.database import (
    Base,
    StageState,
    get_engine,
    get_session,
)


@pytest.fixture
def fresh_stage_state():
    Base.metadata.create_all(get_engine())
    session = get_session()
    session.query(StageState).delete()
    session.commit()
    yield session
    session.close()


def test_stage_state_columns_present():
    cols = {c.name for c in StageState.__table__.columns}
    assert cols == {
        "stage_name",
        "scope_key",
        "data_hash",
        "config_hash",
        "dependency_hash",
        "updated_at",
    }


def test_stage_state_primary_key_is_stage_name_scope_key():
    pk = {c.name for c in StageState.__table__.primary_key.columns}
    assert pk == {"stage_name", "scope_key"}


def test_stage_state_merge_and_get(fresh_stage_state):
    session = fresh_stage_state
    session.merge(
        StageState(
            stage_name="t1",
            scope_key="alpha",
            data_hash="d1",
            config_hash="c1",
            dependency_hash="x1",
        )
    )
    session.commit()

    row = session.get(StageState, ("t1", "alpha"))
    assert row is not None
    assert row.data_hash == "d1"
    assert row.config_hash == "c1"
    assert row.dependency_hash == "x1"
    assert isinstance(row.updated_at, datetime)


def test_stage_state_merge_overwrites_in_place(fresh_stage_state):
    session = fresh_stage_state
    session.merge(
        StageState(
            stage_name="t1",
            scope_key="alpha",
            data_hash="d1",
            config_hash="c1",
            dependency_hash="x1",
        )
    )
    session.commit()
    session.merge(
        StageState(
            stage_name="t1",
            scope_key="alpha",
            data_hash="d2",
            config_hash="c2",
            dependency_hash="x2",
        )
    )
    session.commit()

    row = session.get(StageState, ("t1", "alpha"))
    assert row is not None
    assert row.data_hash == "d2"
    assert row.config_hash == "c2"
    assert row.dependency_hash == "x2"
    assert (
        session.query(StageState).filter_by(stage_name="t1", scope_key="alpha").count()
        == 1
    )
