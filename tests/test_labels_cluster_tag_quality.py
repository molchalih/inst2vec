import json

from core.config import LabelsSettings
from modules.labels.validation import validate_cluster


def _labels() -> LabelsSettings:
    return LabelsSettings()


def _valid_cluster_payload() -> dict:
    return {
        "cluster_label": "muted earth tones",
        "cluster_summary": "x" * 140,
        "dominant_visual_repertoire": [
            {
                "tag": "soft naturalism",
                "description": "y" * 60,
                "recurrence": "dominant",
            },
            {
                "tag": "muted earth tones",
                "description": "y" * 60,
                "recurrence": "frequent",
            },
            {
                "tag": "handheld camera",
                "description": "y" * 60,
                "recurrence": "occasional",
            },
        ],
        "dominant_aesthetic_logic": [
            {
                "tag": "lo-fi polish",
                "grounded_in": ["soft naturalism"],
                "description": "z" * 60,
            },
            {
                "tag": "warm grading",
                "grounded_in": ["muted earth tones"],
                "description": "z" * 60,
            },
            {
                "tag": "loose framing",
                "grounded_in": ["handheld camera"],
                "description": "z" * 60,
            },
        ],
        "taste_signalling": {
            "label": "indie",
            "description": "d" * 60,
            "confidence": "high",
        },
        "visibility_orientation": {
            "label": "niche",
            "description": "d" * 60,
            "confidence": "low",
        },
        "internal_variations": [
            {"variation": "studio vs outdoor", "description": "v" * 60}
        ],
        "boundary_notes": "b" * 60,
        "tool_tags": ["capcut", "vsco"],
    }


def test_clean_payload_passes():
    payload, status, _warnings = validate_cluster(
        json.dumps(_valid_cluster_payload()), _labels(), case="video"
    )
    assert status in ("ok", "warn")
    assert payload is not None


def test_sc9_warns_on_connector_word():
    # Connector words are a SOFT warn (SC9), not a hard fail: the grammar's
    # word/char cap blocks real squashes, so a connector in a short descriptive
    # phrase ("plate with sauce") is flagged, not rejected.
    p = _valid_cluster_payload()
    p["dominant_aesthetic_logic"][0]["tag"] = "plate with sauce"
    payload, status, warnings = validate_cluster(json.dumps(p), _labels(), case="video")
    assert status == "warn"
    assert "SC9" in warnings
    assert payload is not None


def test_hc5_rejects_oversize_tag():
    # >cluster_tag_max_words (5) is still a hard fail (structural backstop).
    p = _valid_cluster_payload()
    p["tool_tags"] = ["one two three four five six seven"]
    _payload, status, warnings = validate_cluster(
        json.dumps(p), _labels(), case="video"
    )
    assert status == "failed"
    assert warnings == ["HC5"]


def test_hc6_rejects_non_ascii_emoji():
    p = _valid_cluster_payload()
    p["tool_tags"] = ["vibe \U0001f525"]
    _payload, status, warnings = validate_cluster(
        json.dumps(p), _labels(), case="video"
    )
    assert status == "failed"
    assert warnings == ["HC6"]


def test_hc6_takes_priority_over_hc5():
    # Tag is BOTH non-ASCII (é) AND an HC5 oversize squash (>5 words). HC6 is
    # checked first, so it must win.
    p = _valid_cluster_payload()
    p["tool_tags"] = ["café alpha bravo charlie delta echo"]
    _payload, status, warnings = validate_cluster(
        json.dumps(p), _labels(), case="video"
    )
    assert status == "failed"
    assert warnings == ["HC6"]


def test_sc8_warns_when_summary_outside_target_band():
    p = _valid_cluster_payload()
    p["cluster_summary"] = "short summary"  # < 120 chars
    _payload, status, warnings = validate_cluster(
        json.dumps(p), _labels(), case="video"
    )
    assert status == "warn"
    assert "SC8" in warnings
