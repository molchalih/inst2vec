from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.database import Base, Clip, ClipLabel, User


def _seed_with_label(eng) -> None:
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(Clip(id=100, user_id=1, is_selected=True, is_downloaded=True))
        s.add(
            ClipLabel(
                clip_id=100,
                status="success",
                validation="ok",
                warnings=[],
                payload={
                    "observable_visual_tags": [
                        {"tag": "warm kitchen", "evidence": "lamp"}
                    ],
                    "aesthetic_tags": [
                        {
                            "tag": "soft vignette",
                            "grounded_in": ["warm kitchen"],
                            "confidence": "medium",
                        },
                    ],
                    "community_signalling_tags": [
                        {
                            "tag": "homecore",
                            "grounded_in": ["soft vignette"],
                            "confidence": "low",
                        },
                    ],
                    "one_sentence_visual_reading": "tight kitchen vignette with warm palette",
                },
                attempts=1,
            )
        )
        s.commit()


def test_per_user_json_contains_clip_labels() -> None:
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    _seed_with_label(eng)

    from modules.visualization.export import _render_user_clips_block

    with Session(eng) as s:
        block = _render_user_clips_block(s, user_id=1, case="video")

    assert isinstance(block, list)
    assert len(block) == 1
    entry = block[0]
    assert entry["clip_id"] == 100
    assert entry["validation"] == "ok"
    assert entry["warnings"] == []
    assert entry["sentence"].startswith("tight kitchen")
    tags = entry["tags"]
    assert "observable" in tags and "aesthetic" in tags and "community" in tags
    assert tags["observable"][0]["tag"] == "warm kitchen"
    assert tags["aesthetic"][0]["confidence"] == "medium"


def test_per_user_json_omits_failed_and_warn_passes() -> None:
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    with Session(eng) as s:
        s.add(User(id=1, is_selected=True))
        s.add(Clip(id=10, user_id=1, is_selected=True, is_downloaded=True))
        s.add(Clip(id=11, user_id=1, is_selected=True, is_downloaded=True))
        s.add(ClipLabel(clip_id=10, status="failed", attempts=3))
        s.add(
            ClipLabel(
                clip_id=11,
                status="success",
                validation="warn",
                warnings=["S1"],
                payload={
                    "observable_visual_tags": [{"tag": "single", "evidence": "x"}],
                    "aesthetic_tags": [
                        {
                            "tag": "spare aesthetic",
                            "grounded_in": ["single"],
                            "confidence": "low",
                        },
                    ],
                    "community_signalling_tags": [
                        {
                            "tag": "minimal register",
                            "grounded_in": ["spare aesthetic"],
                            "confidence": "low",
                        },
                    ],
                    "one_sentence_visual_reading": "minimal scene",
                },
                attempts=1,
            )
        )
        s.commit()

    from modules.visualization.export import _render_user_clips_block

    with Session(eng) as s:
        block = _render_user_clips_block(s, user_id=1, case="video")

    assert {entry["clip_id"] for entry in block} == {11}
    assert block[0]["validation"] == "warn"
    assert "tag_count_out_of_range" in block[0]["warnings"]
