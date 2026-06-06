"""Auto-gold + boundary generation are deterministic and well-formed (§3.6)."""

import random

from swipe_anchor.export.build_demo import make_boundary_comparisons, make_gold_items


def _creators():
    return [
        {"id": 1, "cluster": 0, "group": "Music"},
        {"id": 2, "cluster": 0, "group": "Music"},
        {"id": 3, "cluster": 0, "group": "Music"},
        {"id": 4, "cluster": 1, "group": "Art"},
        {"id": 5, "cluster": 1, "group": "Art"},
        {"id": 6, "cluster": 1, "group": "Art"},
    ]


def test_gold_known_odd_is_the_out_group_creator() -> None:
    golds = make_gold_items(_creators(), rng=random.Random(0), max_gold=3)
    assert golds, "expected at least one gold triple"
    groups = {1: "Music", 2: "Music", 3: "Music", 4: "Art", 5: "Art", 6: "Art"}
    for g in golds:
        members = {g["creator_a"], g["creator_b"], g["creator_c"]}
        assert g["known_odd"] in members
        odd_group = groups[g["known_odd"]]
        other = [c for c in members if c != g["known_odd"]]
        assert groups[other[0]] == groups[other[1]] != odd_group  # 2 same + 1 odd


def test_boundary_comparisons_span_two_clusters() -> None:
    bs = make_boundary_comparisons(_creators(), rng=random.Random(0), max_boundary=5)
    assert bs
    clusters = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
    for b in bs:
        cset = {clusters[b["creator_a"]], clusters[b["creator_b"]], clusters[b["creator_c"]]}
        assert len(cset) == 2  # crosses exactly one boundary
        assert b["kind"] == "boundary"


def test_generation_is_deterministic() -> None:
    a = make_gold_items(_creators(), rng=random.Random(7), max_gold=3)
    b = make_gold_items(_creators(), rng=random.Random(7), max_gold=3)
    assert a == b


def test_reset_content_tables_clears_reliability_events_first() -> None:
    # [P2] Re-seeding a store that already collected answers must not hit an FK
    # error: reliability_events FK comparisons and must be deleted before them.
    import uuid

    from sqlalchemy.orm import Session

    from swipe_anchor.db import create_app_engine
    from swipe_anchor.db.models import (
        Annotator,
        Assignment,
        Comparison,
        ReliabilityEvent,
        Response,
    )
    from swipe_anchor.export.build_demo import reset_content_tables

    engine = create_app_engine("sqlite:///:memory:")
    with Session(engine) as s:
        s.add(Annotator(annotator_id="ann"))
        s.add(Comparison(comparison_id="c1", creator_a=1, creator_b=2, creator_c=3))
        aid = str(uuid.uuid4())
        s.add(Assignment(assignment_id=aid, comparison_id="c1", annotator_id="ann", status="answered"))
        s.add(Response(response_id=str(uuid.uuid4()), assignment_id=aid, comparison_id="c1", annotator_id="ann", odd_creator_id=3))
        s.add(ReliabilityEvent(annotator_id="ann", comparison_id="c1", gold_correct=False))
        s.flush()

        reset_content_tables(s)  # must NOT raise IntegrityError

        assert s.query(Comparison).count() == 0
        assert s.query(ReliabilityEvent).count() == 0
        assert s.get(Annotator, "ann") is not None  # annotators preserved
