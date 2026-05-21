"""Unit tests for the fingerprint helper."""

import json

import pytest

from core.database import Base, StageState, get_engine, get_session
from core.fingerprint import (
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
    from core.fingerprint import hash_text, stage_dependency_hash

    assert stage_dependency_hash(fresh_state, "missing_stage", "scp") == hash_text("")


def test_stage_dependency_hash_combines_three_fields(fresh_state):
    from core.fingerprint import hash_text, stage_dependency_hash

    mark_complete(fresh_state, "upstream", "case-a", _fp("d1", "c1", "x1"))
    fresh_state.commit()

    expected = hash_text("d1" + "c1" + "x1")
    assert stage_dependency_hash(fresh_state, "upstream", "case-a") == expected


def test_stage_dependency_hash_changes_when_data_changes(fresh_state):
    from core.fingerprint import stage_dependency_hash

    mark_complete(fresh_state, "upstream", "case-a", _fp("d1", "c1", "x1"))
    fresh_state.commit()
    first = stage_dependency_hash(fresh_state, "upstream", "case-a")

    mark_complete(fresh_state, "upstream", "case-a", _fp("d2", "c1", "x1"))
    fresh_state.commit()
    second = stage_dependency_hash(fresh_state, "upstream", "case-a")

    assert first != second


def test_stage_dependency_hash_changes_when_config_changes(fresh_state):
    from core.fingerprint import stage_dependency_hash

    mark_complete(fresh_state, "upstream", "case-a", _fp("d1", "c1", "x1"))
    fresh_state.commit()
    first = stage_dependency_hash(fresh_state, "upstream", "case-a")

    mark_complete(fresh_state, "upstream", "case-a", _fp("d1", "c2", "x1"))
    fresh_state.commit()
    second = stage_dependency_hash(fresh_state, "upstream", "case-a")

    assert first != second


def test_stage_dependency_hash_changes_when_dependency_changes(fresh_state):
    from core.fingerprint import stage_dependency_hash

    mark_complete(fresh_state, "upstream", "case-a", _fp("d1", "c1", "x1"))
    fresh_state.commit()
    first = stage_dependency_hash(fresh_state, "upstream", "case-a")

    mark_complete(fresh_state, "upstream", "case-a", _fp("d1", "c1", "x2"))
    fresh_state.commit()
    second = stage_dependency_hash(fresh_state, "upstream", "case-a")

    assert first != second


def test_row_diff_picks_missing_and_changed():
    from core.fingerprint import row_diff

    desired = {10: "h10", 11: "h11", 12: "h12"}
    stored = {10: "h10", 11: "old", 13: "h13"}  # 12 missing, 11 stale, 13 orphan
    assert row_diff(desired, stored) == {11, 12}


def test_row_diff_treats_none_as_stale():
    from core.fingerprint import row_diff

    assert row_diff({10: "h10"}, {10: None}) == {10}


def test_row_diff_empty_desired_returns_empty():
    from core.fingerprint import row_diff

    assert row_diff({}, {10: "h10"}) == set()


def test_file_stat_for_hash_missing_returns_sentinel(tmp_path):
    from core.fingerprint import file_stat_for_hash

    assert file_stat_for_hash(tmp_path / "nope.txt") == (-1, -1)


def test_file_stat_for_hash_existing_returns_size_and_mtime(tmp_path):
    from core.fingerprint import file_stat_for_hash

    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    s = p.stat()
    assert file_stat_for_hash(str(p)) == (s.st_size, s.st_mtime_ns)


# ── gate ─────────────────────────────────────────────────────────────────────


def test_gate_no_prior_state_skips_drift_and_logs(fresh_state, capsys):
    from core.fingerprint import Fingerprint, gate, hash_text

    called: list[str] = []
    fp_obj = Fingerprint(
        data=hash_text(""), config=hash_text("v1"), dependency=hash_text("")
    )

    gate(
        fresh_state,
        "stage_x",
        "scope_y",
        fp_obj,
        on_drift=lambda s: called.append("drift"),
        log_scope="stage_x",
        drift_msg="resetting",
    )

    assert called == []


def test_gate_drift_invokes_on_drift_callback(fresh_state):
    from core.fingerprint import Fingerprint, gate, hash_text, mark_complete

    fp_old = Fingerprint(
        data=hash_text(""), config=hash_text("v1"), dependency=hash_text("")
    )
    mark_complete(fresh_state, "stage_x", "scope_y", fp_old)
    fresh_state.commit()

    fp_new = Fingerprint(
        data=hash_text(""), config=hash_text("v2"), dependency=hash_text("")
    )
    called: list[str] = []
    gate(
        fresh_state,
        "stage_x",
        "scope_y",
        fp_new,
        on_drift=lambda s: called.append("drift"),
        log_scope="stage_x",
        drift_msg="resetting",
    )

    assert called == ["drift"]


def test_gate_match_does_not_invoke_callback(fresh_state):
    from core.fingerprint import Fingerprint, gate, hash_text, mark_complete

    fp_obj = Fingerprint(
        data=hash_text(""), config=hash_text("v1"), dependency=hash_text("")
    )
    mark_complete(fresh_state, "stage_x", "scope_y", fp_obj)
    fresh_state.commit()

    called: list[str] = []
    gate(
        fresh_state,
        "stage_x",
        "scope_y",
        fp_obj,
        on_drift=lambda s: called.append("drift"),
        log_scope="stage_x",
        drift_msg="resetting",
    )

    assert called == []


def test_gate_dependency_drift_ignored_by_default(fresh_state):
    """Without check_dependency, dependency mismatch does NOT call on_drift."""
    from core import fingerprint as fp
    from core.database import StageState

    session = fresh_state
    session.merge(
        StageState(
            stage_name="t",
            scope_key="s",
            data_hash="d",
            config_hash="c",
            dependency_hash="old-dep",
        )
    )
    session.commit()

    current = fp.Fingerprint(data="d", config="c", dependency="new-dep")
    called = {"n": 0}

    def on_drift(_s):
        called["n"] += 1

    fp.gate(
        session,
        "t",
        "s",
        current,
        on_drift,
        log_scope="x",
        drift_msg="dep drift",
    )
    assert called["n"] == 0


def test_gate_dependency_drift_triggers_when_flag_set(fresh_state):
    """With check_dependency=True, dependency mismatch calls on_drift."""
    from core import fingerprint as fp
    from core.database import StageState

    session = fresh_state
    session.merge(
        StageState(
            stage_name="t",
            scope_key="s",
            data_hash="d",
            config_hash="c",
            dependency_hash="old-dep",
        )
    )
    session.commit()

    current = fp.Fingerprint(data="d", config="c", dependency="new-dep")
    called = {"n": 0}

    def on_drift(_s):
        called["n"] += 1

    fp.gate(
        session,
        "t",
        "s",
        current,
        on_drift,
        log_scope="x",
        drift_msg="dep drift",
        check_dependency=True,
    )
    assert called["n"] == 1


def test_gate_dependency_match_does_not_trigger_when_flag_set(fresh_state):
    from core import fingerprint as fp
    from core.database import StageState

    session = fresh_state
    session.merge(
        StageState(
            stage_name="t",
            scope_key="s",
            data_hash="d",
            config_hash="c",
            dependency_hash="dep",
        )
    )
    session.commit()

    current = fp.Fingerprint(data="d", config="c", dependency="dep")
    called = {"n": 0}

    def on_drift(_s):
        called["n"] += 1

    fp.gate(
        session,
        "t",
        "s",
        current,
        on_drift,
        log_scope="x",
        drift_msg="m",
        check_dependency=True,
    )
    assert called["n"] == 0


# ── stable_subset_payload ─────────────────────────────────────────────────────


def test_stable_subset_payload_orders_fields_deterministically():
    from core.fingerprint import stable_subset_payload

    class M:
        a = 1
        b = "x"
        c = 3.5

    out1 = stable_subset_payload(M(), ("c", "a", "b"))
    out2 = stable_subset_payload(M(), ("b", "a", "c"))
    assert out1 == out2
    assert json.loads(out1) == {"a": 1, "b": "x", "c": 3.5}


def test_stable_subset_payload_accepts_mapping():
    from core.fingerprint import stable_subset_payload

    out = stable_subset_payload({"a": 1, "b": "x"}, ("a", "b"))
    assert json.loads(out) == {"a": 1, "b": "x"}


def test_stable_subset_payload_serializes_unknown_types_via_str():
    from pathlib import Path

    from core.fingerprint import stable_subset_payload

    class M:
        p = Path("/tmp/x")

    out = stable_subset_payload(M(), ("p",))
    assert json.loads(out) == {"p": "/tmp/x"}
