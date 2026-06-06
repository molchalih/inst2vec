"""Stage-3 hand-off artifact (plan §1.3).

The app's reason for existing is the export the pipeline consumes: a read-only
``triplets.jsonl`` + optional learned ``embedding.npy``. These tests pin the file
layout and the per-triplet provenance fields.
"""

import json
import random
from datetime import UTC, datetime

import numpy as np
import pytest
from sqlalchemy.orm import Session

from swipe_anchor.backend.service import next_batch, record_response
from swipe_anchor.config import Settings
from swipe_anchor.db import create_app_engine
from swipe_anchor.db.models import Assignment, Comparison
from swipe_anchor.export.anchor_export import export_anchor


@pytest.fixture
def session() -> Session:
    engine = create_app_engine("sqlite:///:memory:")
    with Session(engine) as s:
        yield s


def _answer(s: Session, cid: str, a: int, b: int, c: int, odd: int) -> None:
    # min_overlap=1 + confidence_threshold=0.0 forces immediate confident retire so
    # consensus triplets (response_id=None) are materialized for the export.
    s.add(
        Comparison(
            comparison_id=cid,
            creator_a=a,
            creator_b=b,
            creator_c=c,
            target_k=5,
            seed_group="Artist",
            expected_modality="caption_terms",
        )
    )
    s.commit()
    next_batch(s, "ann-1", n=1, rng=random.Random(0))
    s.commit()
    asg = s.query(Assignment).filter_by(comparison_id=cid).one()
    record_response(
        s, asg.assignment_id, odd_id=odd, confidence=1.0,
        settings=Settings(min_overlap=1, confidence_threshold=0.0),
    )
    s.commit()


def test_export_writes_triplets_jsonl_with_provenance(
    session: Session, tmp_path
) -> None:
    _answer(session, "c1", 10, 20, 30, odd=30)
    when = datetime(2026, 6, 5, tzinfo=UTC)

    manifest = export_anchor(session, tmp_path, build_timestamp=when)

    lines = (tmp_path / "triplets.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert {
        (r["anchor_user_id"], r["positive_user_id"], r["negative_user_id"])
        for r in rows
    } == {(10, 20, 30), (20, 10, 30)}
    r0 = rows[0]
    assert r0["seed_group"] == "Artist"
    assert r0["expected_modality"] == "caption_terms"
    # Weight is the consensus posterior_max.
    # One voter at default reliability 0.5 → competence = 1/3 + 0.5*(2/3) = 2/3
    # → single-vote posterior_max = exactly 2/3.
    assert r0["weight"] == pytest.approx(2 / 3, abs=1e-6)
    assert manifest["counts"]["triplets"] == 2


def test_meta_json_records_counts_and_timestamp(session: Session, tmp_path) -> None:
    _answer(session, "c1", 10, 20, 30, odd=30)
    when = datetime(2026, 6, 5, tzinfo=UTC)

    export_anchor(session, tmp_path, build_timestamp=when)

    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["counts"]["triplets"] == 2
    assert meta["schema_version"] >= 1
    assert meta["build_timestamp"] == "2026-06-05T00:00:00+00:00"
    assert "export_hash" in meta


def test_empty_store_yields_empty_export(session: Session, tmp_path) -> None:
    when = datetime(2026, 6, 5, tzinfo=UTC)
    manifest = export_anchor(session, tmp_path, build_timestamp=when)

    assert (tmp_path / "triplets.jsonl").read_text() == ""
    assert manifest["counts"]["triplets"] == 0


def test_geometry_written_when_provided(session: Session, tmp_path) -> None:
    _answer(session, "c1", 10, 20, 30, odd=30)
    when = datetime(2026, 6, 5, tzinfo=UTC)
    geometry = {10: np.zeros(8), 20: np.ones(8), 30: np.full(8, 2.0)}

    export_anchor(session, tmp_path, build_timestamp=when, geometry=geometry)

    emb = np.load(tmp_path / "embedding.npy")
    index = json.loads((tmp_path / "embedding_index.json").read_text())
    assert emb.shape == (3, 8)
    assert len(index) == 3
    # row i maps to a user id present in the geometry
    assert set(int(v) for v in index.values()) == {10, 20, 30}
