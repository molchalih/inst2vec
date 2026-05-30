import json

from core.config import LabelsSettings
from modules.labels.validation import (
    _balance_brackets,
    _parse,
    validate,
    validate_cluster,
)


def _labels(**overrides) -> LabelsSettings:
    base = dict(
        case_prompts={"video": "x"},
        cluster_case_prompts={"video": "cluster prompt body"},
    )
    base.update(overrides)
    return LabelsSettings(**base)


def _clean_payload() -> dict:
    return {
        "observable_visual_tags": [
            {"tag": "warm kitchen interior", "evidence": "tungsten lamp"},
            {"tag": "shallow depth of field", "evidence": "blurred bg"},
            {"tag": "handheld camera", "evidence": "subtle frame drift"},
        ],
        "aesthetic_tags": [
            {
                "tag": "soft domestic vignette",
                "grounded_in": ["warm kitchen interior", "shallow depth of field"],
                "confidence": "medium",
            },
            {
                "tag": "intimate handheld realism",
                "grounded_in": ["handheld camera"],
                "confidence": "high",
            },
            {
                "tag": "warm-toned framing",
                "grounded_in": ["warm kitchen interior"],
                "confidence": "low",
            },
        ],
        "community_signalling_tags": [
            {
                "tag": "slow-living domestic taste",
                "grounded_in": ["soft domestic vignette"],
                "confidence": "low",
            },
            {
                "tag": "homecore visual register",
                "grounded_in": ["warm-toned framing", "warm kitchen interior"],
                "confidence": "medium",
            },
            {
                "tag": "personal-blog reels register",
                "grounded_in": ["intimate handheld realism"],
                "confidence": "low",
            },
        ],
        "one_sentence_visual_reading": "tight handheld kitchen vignette with warm domestic palette and shallow depth of field",
    }


def test_clean_payload_returns_ok() -> None:
    payload, status, warnings = validate(
        json.dumps(_clean_payload()), _labels(), case="video"
    )
    assert status == "ok"
    assert warnings == []
    assert payload is not None
    assert payload["one_sentence_visual_reading"].startswith("tight handheld")


def test_h1_non_json() -> None:
    payload, status, warnings = validate("not json", _labels(), case="video")
    assert status == "failed"
    assert warnings == ["H1"]
    assert payload is None


def test_h2_missing_key() -> None:
    bad = _clean_payload()
    del bad["one_sentence_visual_reading"]
    payload, status, warnings = validate(json.dumps(bad), _labels(), case="video")
    assert status == "failed"
    assert warnings == ["H2"]
    assert payload is None


def test_h2_extra_key() -> None:
    bad = _clean_payload()
    bad["extra"] = []
    _payload, status, warnings = validate(json.dumps(bad), _labels(), case="video")
    assert status == "failed"
    assert warnings == ["H2"]


def test_h3_non_list_tag_block() -> None:
    bad = _clean_payload()
    bad["observable_visual_tags"] = "not a list"
    _payload, status, warnings = validate(json.dumps(bad), _labels(), case="video")
    assert status == "failed"
    assert warnings == ["H3"]


def test_h3_observable_missing_evidence() -> None:
    bad = _clean_payload()
    bad["observable_visual_tags"][0] = {"tag": "missing evidence"}
    _payload, status, warnings = validate(json.dumps(bad), _labels(), case="video")
    assert status == "failed"
    assert warnings == ["H3"]


def test_h3_aesthetic_missing_grounded_in() -> None:
    bad = _clean_payload()
    bad["aesthetic_tags"][0] = {"tag": "x", "confidence": "high"}
    _payload, status, warnings = validate(json.dumps(bad), _labels(), case="video")
    assert status == "failed"
    assert warnings == ["H3"]


def test_s1_too_few_observable_tags() -> None:
    bad = _clean_payload()
    bad["observable_visual_tags"] = bad["observable_visual_tags"][:2]
    payload, status, warnings = validate(
        json.dumps(bad), _labels(min_tags_per_kind=3), case="video"
    )
    assert status == "warn"
    assert "S1" in warnings
    assert payload is not None


def test_s1_too_many_aesthetic_tags() -> None:
    bad = _clean_payload()
    bad["aesthetic_tags"] = bad["aesthetic_tags"] * 5
    _payload, status, warnings = validate(
        json.dumps(bad), _labels(max_tags_per_kind=4), case="video"
    )
    assert status == "warn"
    assert "S1" in warnings


def test_s2_tag_too_short() -> None:
    bad = _clean_payload()
    bad["observable_visual_tags"][0]["tag"] = "x"
    _payload, status, warnings = validate(
        json.dumps(bad), _labels(min_tag_chars=3), case="video"
    )
    assert status == "warn"
    assert "S2" in warnings


def test_s2_tag_too_long() -> None:
    bad = _clean_payload()
    bad["observable_visual_tags"][0]["tag"] = "x" * 200
    _payload, status, warnings = validate(
        json.dumps(bad), _labels(max_tag_chars=60), case="video"
    )
    assert status == "warn"
    assert "S2" in warnings


def test_s3_duplicate_tags_within_kind() -> None:
    bad = _clean_payload()
    bad["observable_visual_tags"][1]["tag"] = bad["observable_visual_tags"][0][
        "tag"
    ].upper()
    _payload, status, warnings = validate(json.dumps(bad), _labels(), case="video")
    assert status == "warn"
    assert "S3" in warnings


def test_s4_hashtag_like_tag() -> None:
    bad = _clean_payload()
    bad["observable_visual_tags"][0]["tag"] = "#kitchen"
    _payload, status, warnings = validate(json.dumps(bad), _labels(), case="video")
    assert status == "warn"
    assert "S4" in warnings


def test_s4_single_token_tag() -> None:
    bad = _clean_payload()
    bad["observable_visual_tags"][0]["tag"] = "kitchen"
    _payload, status, warnings = validate(json.dumps(bad), _labels(), case="video")
    assert status == "warn"
    assert "S4" in warnings


def test_s6_bad_confidence() -> None:
    bad = _clean_payload()
    bad["aesthetic_tags"][0]["confidence"] = "definitely"
    _payload, status, warnings = validate(json.dumps(bad), _labels(), case="video")
    assert status == "warn"
    assert "S6" in warnings


def test_s7_ungrounded_aesthetic() -> None:
    bad = _clean_payload()
    bad["aesthetic_tags"][0]["grounded_in"] = ["nothing observed"]
    _payload, status, warnings = validate(json.dumps(bad), _labels(), case="video")
    assert status == "warn"
    assert "S7" in warnings


def test_s7_community_grounded_in_observable_is_allowed() -> None:
    ok = _clean_payload()
    # community grounded_in references an observable tag — allowed.
    ok["community_signalling_tags"][0]["grounded_in"] = ["warm kitchen interior"]
    _payload, status, warnings = validate(json.dumps(ok), _labels(), case="video")
    assert status == "ok"
    assert warnings == []


def test_s8_sentence_too_short() -> None:
    bad = _clean_payload()
    bad["one_sentence_visual_reading"] = "short"
    _payload, status, warnings = validate(
        json.dumps(bad), _labels(min_sentence_chars=20), case="video"
    )
    assert status == "warn"
    assert "S8" in warnings


def test_multiple_soft_fails_are_collected() -> None:
    bad = _clean_payload()
    bad["observable_visual_tags"] = bad["observable_visual_tags"][:1]
    bad["one_sentence_visual_reading"] = "short"
    _payload, status, warnings = validate(
        json.dumps(bad),
        _labels(min_tags_per_kind=3, min_sentence_chars=20),
        case="video",
    )
    assert status == "warn"
    assert "S1" in warnings
    assert "S8" in warnings


def test_parse_recovers_fenced_block_with_trailing_comma() -> None:
    # A markdown-fenced object with a trailing comma: strict json.loads
    # rejects it and the bracket-balancer leaves it unchanged, so recovery
    # depends entirely on the json-repair fallback being importable.
    raw = '```json\n{"a": 1, "b": 2,}\n```'
    assert json.loads is not None
    try:
        json.loads(raw)
        raise AssertionError("expected strict json.loads to reject the fenced block")
    except (ValueError, TypeError):
        pass
    # Bracket-balancer must NOT be the thing that fixes it.
    assert _balance_brackets(raw) == raw
    assert _parse(raw) == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# Cluster validator tests
# ---------------------------------------------------------------------------


def _clean_cluster_payload() -> dict:
    return {
        "cluster_label": "soft domestic vignette",
        "cluster_summary": "tight handheld kitchen scenes with warm tungsten tones and shallow bokeh focus; domestic intimacy expressed through close framing and muted personal register",
        "dominant_visual_repertoire": [
            {
                "tag": "warm kitchen interiors",
                "description": "tungsten-lit domestic rooms repeating across clips of a sufficient length",
                "recurrence": "dominant",
            },
            {
                "tag": "shallow bokeh",
                "description": "blurred backgrounds isolate hands and faces consistently",
                "recurrence": "frequent",
            },
            {
                "tag": "handheld camera",
                "description": "subtle frame drift across the represented clips",
                "recurrence": "frequent",
            },
        ],
        "dominant_aesthetic_logic": [
            {
                "tag": "intimate domestic realism",
                "grounded_in": ["warm kitchen interiors", "handheld camera"],
                "description": "the recurring close warm framing reads as intimate rather than staged",
            },
            {
                "tag": "softly tactile palette",
                "grounded_in": ["warm kitchen interiors", "shallow bokeh"],
                "description": "warmth and blur combine into a tactile near-haptic surface treatment",
            },
            {
                "tag": "muted personal register",
                "grounded_in": ["handheld camera"],
                "description": "handheld imperfection signals a low-affect personal voice",
            },
        ],
        "taste_signalling": {
            "label": "homecore visual register",
            "description": "the repertoire aligns with a slow-living aesthetic affinity expressed through domestic intimacy",
            "confidence": "medium",
        },
        "visibility_orientation": {
            "label": "low-stakes ordinariness",
            "description": "the clips stage attention toward ordinariness and intimacy rather than spectacle or polish",
            "confidence": "low",
        },
        "internal_variations": [
            {
                "variation": "bathroom-lit grooming clips",
                "description": "minor sub-strand of cool-lit grooming clips inside a broader warm kitchen-led repertoire",
            },
        ],
        "boundary_notes": "differs from adjacent food-styling clusters by lacking top-down plating shots and overhead lighting setups",
        "tool_tags": ["homecore", "warm-palette", "handheld-domestic"],
    }


def _cluster_labels(**overrides) -> LabelsSettings:
    base = dict(
        case_prompts={"video": "x"},
        cluster_case_prompts={"video": "y"},
    )
    base.update(overrides)
    return LabelsSettings(**base)


def test_cluster_clean_payload_returns_ok() -> None:
    payload, status, warnings = validate_cluster(
        json.dumps(_clean_cluster_payload()), _cluster_labels(), case="video"
    )
    assert status == "ok"
    assert warnings == []
    assert payload is not None
    assert payload["cluster_label"].startswith("soft domestic")


def test_cluster_hc1_not_json() -> None:
    payload, status, warnings = validate_cluster(
        "not json", _cluster_labels(), case="video"
    )
    assert payload is None and status == "failed" and warnings == ["HC1"]


def test_cluster_hc2_missing_key() -> None:
    bad = _clean_cluster_payload()
    bad.pop("tool_tags")
    payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(), case="video"
    )
    assert payload is None and status == "failed" and warnings == ["HC2"]


def test_cluster_hc3_wrong_type() -> None:
    bad = _clean_cluster_payload()
    bad["cluster_summary"] = 42
    payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(), case="video"
    )
    assert payload is None and status == "failed" and warnings == ["HC3"]


def test_cluster_sc1_repertoire_count_out_of_range() -> None:
    bad = _clean_cluster_payload()
    bad["dominant_visual_repertoire"] = bad["dominant_visual_repertoire"][:1]
    _payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(cluster_min_tags=3), case="video"
    )
    assert status == "warn" and "SC1" in warnings


def test_cluster_sc2_tag_length_out_of_range() -> None:
    bad = _clean_cluster_payload()
    bad["dominant_visual_repertoire"][0]["tag"] = "ab"
    _payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(), case="video"
    )
    assert status == "warn" and "SC2" in warnings


def test_cluster_sc2_tag_over_cluster_cap_but_under_max_tag_chars() -> None:
    # vLLM does not enforce the schema's maxLength, so a tag longer than
    # cluster_tag_max_chars (but within the looser legacy max_tag_chars) must
    # still warn — otherwise over-long tags pass silently.
    bad = _clean_cluster_payload()
    bad["dominant_visual_repertoire"][0]["tag"] = "x" * 40
    _payload, status, warnings = validate_cluster(
        json.dumps(bad),
        _cluster_labels(cluster_tag_max_chars=28, max_tag_chars=60),
        case="video",
    )
    assert status == "warn" and "SC2" in warnings


def test_cluster_sc2_covers_tool_tags() -> None:
    bad = _clean_cluster_payload()
    bad["tool_tags"] = ["x" * 40]
    _payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(cluster_tag_max_chars=28), case="video"
    )
    assert status == "warn" and "SC2" in warnings


def test_cluster_sc3_duplicate_tag() -> None:
    bad = _clean_cluster_payload()
    bad["dominant_visual_repertoire"][1]["tag"] = bad["dominant_visual_repertoire"][0][
        "tag"
    ]
    _payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(), case="video"
    )
    assert status == "warn" and "SC3" in warnings


def test_cluster_sc4_ungrounded_aesthetic() -> None:
    bad = _clean_cluster_payload()
    bad["dominant_aesthetic_logic"][0]["grounded_in"] = ["nope"]
    _payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(), case="video"
    )
    assert status == "warn" and "SC4" in warnings


def test_cluster_sc5_invalid_confidence() -> None:
    bad = _clean_cluster_payload()
    bad["taste_signalling"]["confidence"] = "very high"
    payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(), case="video"
    )
    assert status == "warn" and "SC5" in warnings
    assert payload is not None
    assert payload["taste_signalling"]["confidence"] == "very high"


def test_cluster_sc6_sentence_length() -> None:
    bad = _clean_cluster_payload()
    bad["cluster_summary"] = "x"
    _payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(cluster_min_sentence_chars=20), case="video"
    )
    assert status == "warn" and "SC6" in warnings


def test_cluster_sc7_empty_tool_tags() -> None:
    bad = _clean_cluster_payload()
    bad["tool_tags"] = []
    _payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(), case="video"
    )
    assert status == "warn" and "SC7" in warnings


def test_cluster_sc7_hashtag_like_tool_tag() -> None:
    bad = _clean_cluster_payload()
    bad["tool_tags"] = ["#homecore", "ok-tag", "another-ok"]
    _payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(), case="video"
    )
    assert status == "warn" and "SC7" in warnings


def test_cluster_multiple_warnings_are_sorted() -> None:
    bad = _clean_cluster_payload()
    bad["dominant_visual_repertoire"] = bad["dominant_visual_repertoire"][:1]
    bad["tool_tags"] = []
    _payload, status, warnings = validate_cluster(
        json.dumps(bad), _cluster_labels(cluster_min_tags=3), case="video"
    )
    assert status == "warn"
    assert warnings == sorted(warnings)
    assert {"SC1", "SC7"}.issubset(set(warnings))


# ---------------------------------------------------------------------------
# Per-case validator coverage (spoken / auditory / sandwich / textual)
# ---------------------------------------------------------------------------


def _retarget_clip_payload(
    visual: dict, observable_key: str, sentence_key: str
) -> dict:
    out = {
        observable_key: visual["observable_visual_tags"],
        "aesthetic_tags": visual["aesthetic_tags"],
        "community_signalling_tags": visual["community_signalling_tags"],
        sentence_key: visual["one_sentence_visual_reading"],
    }
    return out


def test_validate_spoken_case_accepts_audio_keys() -> None:
    payload = _retarget_clip_payload(
        _clean_payload(),
        "observable_audio_tags",
        "one_sentence_audio_reading",
    )
    parsed, status, warnings = validate(json.dumps(payload), _labels(), case="spoken")
    assert status == "ok"
    assert warnings == []
    assert parsed is not None


def test_validate_spoken_case_rejects_visual_keys() -> None:
    # The visual-keyed payload must not validate as a spoken payload (H2:
    # the required-key set diverges).
    _payload, status, warnings = validate(
        json.dumps(_clean_payload()), _labels(), case="spoken"
    )
    assert status == "failed"
    assert warnings == ["H2"]


def test_validate_auditory_case_accepts_music_keys() -> None:
    payload = _retarget_clip_payload(
        _clean_payload(),
        "observable_music_tags",
        "one_sentence_music_reading",
    )
    parsed, status, _warnings = validate(
        json.dumps(payload), _labels(), case="auditory"
    )
    assert status == "ok"
    assert parsed is not None


def test_validate_sandwich_case_accepts_multimodal_keys() -> None:
    payload = _retarget_clip_payload(
        _clean_payload(),
        "observable_multimodal_tags",
        "one_sentence_multimodal_reading",
    )
    parsed, status, _warnings = validate(
        json.dumps(payload), _labels(), case="sandwich"
    )
    assert status == "ok"
    assert parsed is not None


def _retarget_cluster_payload(visual: dict, repertoire_key: str) -> dict:
    out = dict(visual)
    out[repertoire_key] = out.pop("dominant_visual_repertoire")
    return out


def test_validate_cluster_spoken_accepts_audio_repertoire() -> None:
    payload = _retarget_cluster_payload(
        _clean_cluster_payload(), "dominant_audio_repertoire"
    )
    parsed, status, _warnings = validate_cluster(
        json.dumps(payload), _cluster_labels(), case="spoken"
    )
    assert status == "ok"
    assert parsed is not None


def test_validate_cluster_auditory_accepts_music_repertoire() -> None:
    payload = _retarget_cluster_payload(
        _clean_cluster_payload(), "dominant_music_repertoire"
    )
    parsed, status, _warnings = validate_cluster(
        json.dumps(payload), _cluster_labels(), case="auditory"
    )
    assert status == "ok"
    assert parsed is not None


def test_validate_cluster_sandwich_accepts_multimodal_repertoire() -> None:
    payload = _retarget_cluster_payload(
        _clean_cluster_payload(), "dominant_multimodal_repertoire"
    )
    parsed, status, _warnings = validate_cluster(
        json.dumps(payload), _cluster_labels(), case="sandwich"
    )
    assert status == "ok"
    assert parsed is not None


def test_validate_cluster_spoken_rejects_visual_repertoire_keys() -> None:
    _parsed, status, warnings = validate_cluster(
        json.dumps(_clean_cluster_payload()), _cluster_labels(), case="spoken"
    )
    assert status == "failed"
    assert warnings == ["HC2"]
