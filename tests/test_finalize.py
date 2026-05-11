import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from modules.database import Base, Clip, User


def _make_engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def _make_user(session, id: int, clips_play_counts: list[int]) -> User:
    user = User(
        id=id, username=f"user{id}", full_name=f"User {id}", parse_status="success"
    )
    session.add(user)
    session.flush()
    for i, plays in enumerate(clips_play_counts):
        clip = Clip(
            id=id * 1000 + i,
            user_id=id,
            play_count=plays,
        )
        session.add(clip)
    session.commit()
    return user


def test_pass_a_pre_gates_users_with_too_few_raw_clips(monkeypatch):
    """Users with < TARGET_CLIPS_PER_USER raw clips are disqualified before stats."""
    eng = _make_engine()
    with Session(eng) as s:
        _make_user(
            s, id=1, clips_play_counts=[100000]
        )  # 1 clip — below default TARGET=4
        _make_user(
            s, id=2, clips_play_counts=[100000] * 5
        )  # 5 clips — survives pre-gate

    monkeypatch.setattr("modules.finalize.get_session", lambda: Session(eng))
    monkeypatch.setenv("FINALIZE_TARGET_CLIPS_PER_USER", "4")
    monkeypatch.setenv("FINALIZE_GLOBAL_MIN_PLAYS_PERCENTILE", "0")
    monkeypatch.setenv("FINALIZE_GLOBAL_MIN_PLAYS", "0")
    monkeypatch.setenv("FINALIZE_CREATOR_ROBUST_Z_THRESHOLD", "-99")

    import modules.finalize as fin_mod

    monkeypatch.setattr(fin_mod, "TARGET_CLIPS_PER_USER", 4)

    from modules.finalize import finalize_user_dataset

    finalize_user_dataset("A")

    with Session(eng) as s:
        u1 = s.get(User, 1)
        u2 = s.get(User, 2)
        assert u1 is not None
        assert u2 is not None
        assert u1.user_disqualified == 1  # pre-gated
        assert u2.user_disqualified == 0  # survived


def test_pass_a_pre_gated_users_excluded_from_percentile_floor(monkeypatch):
    """Dead accounts (few clips, low plays) must not drag down the percentile floor.

    Without pre-gate, the 5th percentile would be computed from all 6 clips (1 + 5).
    With pre-gate, user1 is disqualified immediately, so percentile is from 5 clips only.
    The difference: when we have [1, 100k, 100k, 100k, 100k, 100k], the 5th percentile is
    around the lowest value (1). But if we pre-gate user1 out first and compute from
    [100k, 100k, 100k, 100k, 100k], the 5th percentile is 100k.
    """
    eng = _make_engine()
    with Session(eng) as s:
        _make_user(s, id=1, clips_play_counts=[1])  # 1 clip, pre-gated out
        _make_user(s, id=2, clips_play_counts=[100000] * 5)  # 5 clips, survives

    monkeypatch.setattr("modules.finalize.get_session", lambda: Session(eng))
    monkeypatch.setenv("FINALIZE_TARGET_CLIPS_PER_USER", "4")
    monkeypatch.setenv("FINALIZE_GLOBAL_MIN_PLAYS_PERCENTILE", "5")
    monkeypatch.setenv("FINALIZE_GLOBAL_MIN_PLAYS", "0")
    monkeypatch.setenv("FINALIZE_CREATOR_ROBUST_Z_THRESHOLD", "-99")
    monkeypatch.setenv("FINALIZE_PASS_A_RECOMPUTE_FROM_SCRATCH", "1")

    import modules.finalize as fin_mod

    monkeypatch.setattr(fin_mod, "TARGET_CLIPS_PER_USER", 4)
    monkeypatch.setattr(fin_mod, "PASS_A_RECOMPUTE_FROM_SCRATCH", True)
    monkeypatch.setattr(fin_mod, "GLOBAL_MIN_PLAYS_PERCENTILE", 5.0)

    from modules.finalize import finalize_user_dataset

    finalize_user_dataset("A")

    with Session(eng) as s:
        u1 = s.get(User, 1)
        u2 = s.get(User, 2)
        assert u1 is not None
        assert u2 is not None
        # User1 must be pre-gated (disqualified immediately, not due to stats)
        assert u1.user_disqualified == 1
        # User2 survives with all clips intact because floor is computed only from user2
        surviving_clips = [c for c in u2.clips if c.disqualified == 0]
        assert (
            len(surviving_clips) == 5
        )  # all 5 clips survive (dead user excluded from floor)


def test_pass_a_re_gates_after_stat_disq(monkeypatch):
    """A user who passes pre-gate but loses clips to stat disq gets re-gated if below TARGET."""
    eng = _make_engine()
    with Session(eng) as s:
        # 5 clips total — passes pre-gate (5 >= 4)
        # 4 clips have very low plays → stat disq removes them → 1 remains → re-gate fires
        _make_user(s, id=1, clips_play_counts=[1, 1, 1, 1, 1000000])

    monkeypatch.setattr("modules.finalize.get_session", lambda: Session(eng))
    monkeypatch.setenv("FINALIZE_TARGET_CLIPS_PER_USER", "4")
    monkeypatch.setenv("FINALIZE_GLOBAL_MIN_PLAYS_PERCENTILE", "0")
    monkeypatch.setenv(
        "FINALIZE_GLOBAL_MIN_PLAYS", "500000"
    )  # floor disqualifies 4 clips
    monkeypatch.setenv("FINALIZE_CREATOR_ROBUST_Z_THRESHOLD", "-99")
    monkeypatch.setenv("FINALIZE_PASS_A_RECOMPUTE_FROM_SCRATCH", "1")

    import modules.finalize as fin_mod

    monkeypatch.setattr(fin_mod, "TARGET_CLIPS_PER_USER", 4)
    monkeypatch.setattr(fin_mod, "GLOBAL_MIN_PLAYS", 500000)
    monkeypatch.setattr(fin_mod, "PASS_A_RECOMPUTE_FROM_SCRATCH", True)

    from modules.finalize import finalize_user_dataset

    finalize_user_dataset("A")

    with Session(eng) as s:
        u = s.get(User, 1)
        assert u is not None
        assert u.user_disqualified == 1  # re-gated: only 1 clip survives stat disq


def test_pass_b_does_not_pre_gate(monkeypatch):
    """Pass B does not apply the pre-gate; all parsed users go through the loop."""
    eng = _make_engine()
    with Session(eng) as s:
        # User with only 1 clip — would be pre-gated in Pass A, but not in Pass B
        u = _make_user(s, id=1, clips_play_counts=[100000])
        # Pre-set clip as eligible (disqualified=0) so Pass B considers it
        s.query(Clip).filter(Clip.user_id == 1).update({"disqualified": 0})
        s.commit()

    monkeypatch.setattr("modules.finalize.get_session", lambda: Session(eng))
    monkeypatch.setenv("FINALIZE_TARGET_CLIPS_PER_USER", "4")
    monkeypatch.setenv("FINALIZE_GLOBAL_MIN_PLAYS_PERCENTILE", "0")
    monkeypatch.setenv("FINALIZE_GLOBAL_MIN_PLAYS", "0")
    monkeypatch.setenv("FINALIZE_CREATOR_ROBUST_Z_THRESHOLD", "-99")

    import modules.finalize as fin_mod

    monkeypatch.setattr(fin_mod, "TARGET_CLIPS_PER_USER", 4)

    from modules.finalize import finalize_user_dataset

    finalize_user_dataset("B")

    with Session(eng) as s:
        u = s.get(User, 1)
        assert u is not None
        # Pass B does not pre-gate; user gets disqualified only by clip-count re-gate
        # (which is the same behavior as before — 1 clip < 4)
        assert u.user_disqualified == 1  # disqualified by clip count in Pass B loop
