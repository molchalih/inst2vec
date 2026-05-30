"""Per-case prompt sub-tables on ``LabelsSettings``.

Stage-1 prompts move out of the flat ``labels.prompt`` / ``labels.cluster_prompt``
strings and into ``labels.case_prompts`` / ``labels.cluster_case_prompts``
sub-tables. The legacy flat keys are accepted with a ``DeprecationWarning``
fallback (SPEC §5.6).
"""

from __future__ import annotations

import warnings

import pytest

from core.config import LabelsSettings, _load_settings


def test_legacy_flat_prompt_falls_back_to_video() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = LabelsSettings(prompt="LEGACY", cluster_prompt="LEGACY_CLUSTER")
    assert s.case_prompts == {"video": "LEGACY"}
    assert s.cluster_case_prompts == {"video": "LEGACY_CLUSTER"}
    # Loading legacy keys should emit DeprecationWarning(s) — at least one.
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "expected DeprecationWarning when legacy labels.prompt is used"
    )


def test_new_case_prompts_table_loads() -> None:
    s = LabelsSettings(
        case_prompts={"video": "A", "audio": "B"},
        cluster_case_prompts={"video": "C", "audio": "D"},
    )
    assert s.case_prompts == {"video": "A", "audio": "B"}
    assert s.cluster_case_prompts == {"video": "C", "audio": "D"}


def test_both_old_and_new_warns_and_prefers_new() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = LabelsSettings(
            prompt="LEGACY",
            cluster_prompt="LEGACY_CLUSTER",
            case_prompts={"video": "NEW"},
            cluster_case_prompts={"video": "NEW_CLUSTER"},
        )
    # New wins.
    assert s.case_prompts == {"video": "NEW"}
    assert s.cluster_case_prompts == {"video": "NEW_CLUSTER"}
    # Deprecation warning explicitly fires when both are present.
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep, "expected DeprecationWarning when legacy + new keys coexist"


def test_legacy_warning_fires_once_per_load() -> None:
    """SPEC §5.6: the deprecation warning fires once per ``LabelsSettings``
    instance, not per attribute access."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LabelsSettings(prompt="A", cluster_prompt="B")
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    # Either a single combined warning or two distinct ones (clip + cluster) —
    # but never one per attribute access.
    assert 1 <= len(dep) <= 2


def test_default_config_toml_loads_all_default_case_prompts() -> None:
    settings = _load_settings()
    cps = settings.labels.case_prompts
    cl = settings.labels.cluster_case_prompts
    # Cases that run the per-clip pass (video) or carry a clip prompt body
    # (auditory) must have a non-empty clip prompt; spoken / textual skip the
    # clip pass so their clip prompt resolves to an empty string by design.
    for case in ("video", "sandwich", "auditory"):
        assert case in cps, f"case_prompts.{case} missing from config.toml"
        assert cps[case].strip(), f"case_prompts.{case} is empty"
    for case in ("spoken", "textual"):
        assert case in cps, f"case_prompts.{case} missing from config.toml"
    # Every active case (incl. the text-only spoken / textual) must have a
    # non-empty cluster prompt; the cluster pass runs for all of them.
    for case in ("video", "sandwich", "auditory", "spoken", "textual"):
        assert case in cl, f"cluster_case_prompts.{case} missing"
        assert cl[case].strip(), f"cluster_case_prompts.{case} is empty"


def test_video_case_prompt_matches_legacy_prompt_text() -> None:
    """The shipped ``case_prompts.video`` text must be the byte-identical
    body of the previous flat ``labels.prompt`` (SPEC §5.6 — no semantic
    drift for the video flow as part of the migration)."""
    settings = _load_settings()
    text = settings.labels.case_prompts["video"]
    # Sentinel substrings from the original visual-only prompt body.
    assert "observable_visual_tags" in text
    assert "one_sentence_visual_reading" in text


@pytest.mark.parametrize("case", ("sandwich", "auditory"))
def test_clip_case_prompts_target_their_modality(case: str) -> None:
    """Cases that carry a clip-pass prompt body name their modality's
    per-clip keys (observable_<modality>_tags / one_sentence_<modality>_reading)
    and the cluster prompt names dominant_<modality>_repertoire."""
    settings = _load_settings()
    text = settings.labels.case_prompts[case]
    cluster_text = settings.labels.cluster_case_prompts[case]
    modality = {"sandwich": "multimodal", "auditory": "music"}[case]
    assert f"observable_{modality}_tags" in text
    assert f"one_sentence_{modality}_reading" in text
    assert f"dominant_{modality}_repertoire" in cluster_text


@pytest.mark.parametrize(
    ("case", "repertoire"),
    (
        ("spoken", "dominant_audio_repertoire"),
        ("textual", "dominant_textual_repertoire"),
    ),
)
def test_clip_skipped_cluster_prompts_name_repertoire(
    case: str, repertoire: str
) -> None:
    """spoken / textual skip the clip pass, so only the cluster prompt is
    asserted — it must name the modality's repertoire key."""
    settings = _load_settings()
    cluster_text = settings.labels.cluster_case_prompts[case]
    assert repertoire in cluster_text
