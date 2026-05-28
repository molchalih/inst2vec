from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from core.database import Base, ClusterLabel
from modules.labels.cases import REGISTRY


def _engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


def test_cluster_label_round_trip() -> None:
    eng = _engine()
    with Session(eng) as s:
        s.add(
            ClusterLabel(
                embedding_case="video",
                cluster_id=3,
                status="success",
                validation="ok",
                payload={"cluster_label": "soft domestic"},
                warnings=[],
                error=None,
                attempts=1,
                sampled_clip_ids=[10, 11],
            )
        )
        s.commit()
        row = s.get(ClusterLabel, ("video", 3))
        assert row is not None
        assert row.payload["cluster_label"] == "soft domestic"
        assert row.sampled_clip_ids == [10, 11]


def test_cluster_label_composite_pk_distinct_per_case() -> None:
    eng = _engine()
    with Session(eng) as s:
        s.add(
            ClusterLabel(
                embedding_case="video", cluster_id=0, status="pending", attempts=0
            )
        )
        s.add(
            ClusterLabel(
                embedding_case="audio", cluster_id=0, status="pending", attempts=0
            )
        )
        s.commit()
        rows = (
            s.execute(select(ClusterLabel).order_by(ClusterLabel.embedding_case))
            .scalars()
            .all()
        )
        assert [r.embedding_case for r in rows] == ["audio", "video"]


def test_per_case_required_key_sets_have_a_repertoire_key() -> None:
    """The cluster validator looks up the unique ``dominant_*_repertoire``
    key from each ``LabelCaseSpec``; every default case must expose exactly
    one such key so the lookup is unambiguous."""
    for case, spec in REGISTRY.items():
        repertoire = [
            k
            for k in spec.cluster_required_keys
            if k.startswith("dominant_") and k.endswith("_repertoire")
        ]
        assert len(repertoire) == 1, (case, repertoire)


def test_per_case_required_key_sets_share_common_keys() -> None:
    common = {
        "cluster_label",
        "cluster_summary",
        "dominant_aesthetic_logic",
        "taste_signalling",
        "visibility_orientation",
        "internal_variations",
        "boundary_notes",
        "tool_tags",
    }
    for case, spec in REGISTRY.items():
        assert common.issubset(spec.cluster_required_keys), case
