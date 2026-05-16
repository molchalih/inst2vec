"""Unit tests for the fingerprint helper."""

import pytest

from modules.database import Base, StageState, get_engine, get_session
from modules.fingerprint import (
    Fingerprint,
    describe_diff,
    hash_rows,
    hash_text,
    is_stale,
    mark_complete,
)

# ── hash_rows / hash_text ────────────────────────────────────────────────────


def test_hash_rows_empty_is_stable():
    assert hash_rows([]) == hash_rows([])
    # well-defined and non-empty
    assert isinstance(hash_rows([]), str)
    assert len(hash_rows([])) == 64


def test_hash_rows_deterministic_for_same_input():
    rows = [(1, "a"), (2, "b"), (3, "c")]
    assert hash_rows(rows) == hash_rows(rows)


def test_hash_rows_order_sensitive():
    a = hash_rows([(1, "a"), (2, "b")])
    b = hash_rows([(2, "b"), (1, "a")])
    assert a != b


def test_hash_rows_value_change_changes_digest():
    a = hash_rows([(1, "a")])
    b = hash_rows([(1, "b")])
    assert a != b


def test_hash_rows_no_collision_on_split():
    # 0x1E record separator prevents (1,2),(3) from colliding with (1),(2,3)
    a = hash_rows([(1, 2), (3,)])
    b = hash_rows([(1,), (2, 3)])
    assert a != b


def test_hash_text_deterministic():
    assert hash_text("hello") == hash_text("hello")
    assert hash_text("hello") != hash_text("world")


def test_hash_text_empty_is_stable():
    assert hash_text("") == hash_text("")
    assert len(hash_text("")) == 64


# ── is_stale / mark_complete / describe_diff ─────────────────────────────────


@pytest.fixture
def fresh_state():
    Base.metadata.create_all(get_engine())
    session = get_session()
    session.query(StageState).delete()
    session.commit()
    yield session
    session.close()


def _fp(d: str = "d", c: str = "c", x: str = "x") -> Fingerprint:
    return Fingerprint(data=d, config=c, dependency=x)


def test_is_stale_true_when_no_row(fresh_state):
    assert is_stale(fresh_state, "stg", "scp", _fp()) is True


def test_is_stale_false_when_all_match(fresh_state):
    mark_complete(fresh_state, "stg", "scp", _fp())
    fresh_state.commit()
    assert is_stale(fresh_state, "stg", "scp", _fp()) is False


def test_is_stale_true_when_data_differs(fresh_state):
    mark_complete(fresh_state, "stg", "scp", _fp("d1"))
    fresh_state.commit()
    assert is_stale(fresh_state, "stg", "scp", _fp("d2")) is True


def test_is_stale_true_when_config_differs(fresh_state):
    mark_complete(fresh_state, "stg", "scp", _fp(c="c1"))
    fresh_state.commit()
    assert is_stale(fresh_state, "stg", "scp", _fp(c="c2")) is True


def test_is_stale_true_when_dependency_differs(fresh_state):
    mark_complete(fresh_state, "stg", "scp", _fp(x="x1"))
    fresh_state.commit()
    assert is_stale(fresh_state, "stg", "scp", _fp(x="x2")) is True


def test_mark_complete_does_not_commit(fresh_state):
    # Without our own commit, the row stays dirty in the session.
    mark_complete(fresh_state, "stg", "scp", _fp())
    assert (
        any(isinstance(o, StageState) for o in fresh_state.new)
        or any(isinstance(o, StageState) for o in fresh_state.dirty)
        or any(isinstance(o, StageState) for o in fresh_state.identity_map.values())
    )


def test_describe_diff_no_prior_state(fresh_state):
    assert describe_diff(fresh_state, "stg", "scp", _fp()) == "no prior state"


def test_describe_diff_empty_when_all_match(fresh_state):
    mark_complete(fresh_state, "stg", "scp", _fp())
    fresh_state.commit()
    assert describe_diff(fresh_state, "stg", "scp", _fp()) == ""


def test_describe_diff_lists_changed_fields(fresh_state):
    mark_complete(fresh_state, "stg", "scp", Fingerprint("d1", "c1", "x1"))
    fresh_state.commit()
    diff = describe_diff(fresh_state, "stg", "scp", Fingerprint("d2", "c1", "x2"))
    assert diff == "data+dependency"


# ── stage_dependency_hash ────────────────────────────────────────────────────


def test_stage_dependency_hash_returns_empty_text_hash_when_absent(fresh_state):
    from modules.fingerprint import hash_text, stage_dependency_hash

    assert stage_dependency_hash(fresh_state, "missing_stage", "scp") == hash_text("")


def test_stage_dependency_hash_combines_three_fields(fresh_state):
    from modules.fingerprint import hash_text, stage_dependency_hash

    mark_complete(fresh_state, "upstream", "case-a", _fp("d1", "c1", "x1"))
    fresh_state.commit()

    expected = hash_text("d1" + "c1" + "x1")
    assert stage_dependency_hash(fresh_state, "upstream", "case-a") == expected


def test_stage_dependency_hash_changes_when_any_field_changes(fresh_state):
    from modules.fingerprint import stage_dependency_hash

    mark_complete(fresh_state, "upstream", "case-a", _fp("d1", "c1", "x1"))
    fresh_state.commit()
    first = stage_dependency_hash(fresh_state, "upstream", "case-a")

    mark_complete(fresh_state, "upstream", "case-a", _fp("d1", "c1", "x2"))
    fresh_state.commit()
    second = stage_dependency_hash(fresh_state, "upstream", "case-a")

    assert first != second
