from core.config import LabelsSettings
from modules.labels.state import (
    STAGE_CLUSTER_LABELS,
    STAGE_LABELS,
    clip_labels_config_payload,
    clip_scope_for,
    cluster_labels_config_payload,
    cluster_scope_for,
)


def _labels(**overrides) -> LabelsSettings:
    base = dict(
        case_prompts={
            "video": "You are doing visual analysis.\n",
            "audio": "audio prompt",
        },
        cluster_case_prompts={"video": "cluster prompt body", "audio": "audio cluster"},
    )
    base.update(overrides)
    return LabelsSettings(**base)


def test_payload_stable_for_unrelated_changes() -> None:
    a = _labels(parallelism=1)
    b = _labels(parallelism=4)
    # parallelism is NOT in the fingerprint payload (runtime knob).
    assert clip_labels_config_payload(a, case="video") == clip_labels_config_payload(
        b, case="video"
    )


def test_payload_drifts_on_prompt_change() -> None:
    a = _labels(case_prompts={"video": "prompt A"})
    b = _labels(case_prompts={"video": "prompt B"})
    assert clip_labels_config_payload(a, case="video") != clip_labels_config_payload(
        b, case="video"
    )


def test_payload_drifts_on_model_or_generation_param() -> None:
    base = _labels()
    for k, v in [
        ("model_path", "./models/other"),
        ("frame_count", 16),
        ("max_new_tokens", 200),
        ("generation_seed", 1),
        ("min_tags_per_kind", 1),
        ("max_tag_chars", 80),
    ]:
        drifted = _labels(**{k: v})
        assert clip_labels_config_payload(
            base, case="video"
        ) != clip_labels_config_payload(drifted, case="video"), k


def test_payload_is_stable_string() -> None:
    # The payload no longer round-trips through ``json.loads`` (the per-case
    # prompt is appended as a suffix), but it must still be a stable string
    # that round-trips byte-for-byte across construction.
    a = clip_labels_config_payload(_labels(), case="video")
    b = clip_labels_config_payload(_labels(), case="video")
    assert a == b and isinstance(a, str) and a


def test_stage_and_scope_constants() -> None:
    assert STAGE_LABELS == "labels"


def test_clip_scope_for_returns_case() -> None:
    assert clip_scope_for("video") == "video"
    assert clip_scope_for("audio") == "audio"


def test_clip_labels_config_payload_changes_per_case_prompt() -> None:
    a = _labels(case_prompts={"video": "A", "audio": "X"})
    b = _labels(case_prompts={"video": "B", "audio": "X"})
    # The clip-pass payload is case-keyed: drifting case_prompts.video must
    # drift the video payload but not the audio payload.
    assert clip_labels_config_payload(a, case="video") != clip_labels_config_payload(
        b, case="video"
    )
    assert clip_labels_config_payload(a, case="audio") == clip_labels_config_payload(
        b, case="audio"
    )


def test_other_case_prompt_change_does_not_affect_this_cases_payload() -> None:
    a = _labels(case_prompts={"video": "X", "audio": "A"})
    b = _labels(case_prompts={"video": "X", "audio": "B"})
    assert clip_labels_config_payload(a, case="video") == clip_labels_config_payload(
        b, case="video"
    )


def test_max_new_tokens_for_returns_override_or_global_default() -> None:
    s = _labels(
        max_new_tokens=1200,
        clip_max_new_tokens_overrides={"sandwich": 2048},
    )
    # The overridden case sees its per-case cap; every other case falls back to
    # the global ``max_new_tokens``.
    assert s.max_new_tokens_for("sandwich") == 2048
    assert s.max_new_tokens_for("video") == 1200
    assert s.max_new_tokens_for("spoken") == 1200


def test_max_new_tokens_override_drifts_only_the_overridden_case() -> None:
    # Adding a per-case override is surgical: it must drift ONLY the overridden
    # case's fingerprint so the re-label is confined to that case. The other
    # four cases' payloads must stay byte-identical (no wipe / re-label).
    base = _labels()
    over = _labels(clip_max_new_tokens_overrides={"sandwich": 2048})
    assert clip_labels_config_payload(
        base, case="sandwich"
    ) != clip_labels_config_payload(over, case="sandwich")
    for case in ("video", "audio", "spoken", "textual"):
        assert clip_labels_config_payload(
            base, case=case
        ) == clip_labels_config_payload(over, case=case), case


def test_max_new_tokens_override_value_drifts_payload() -> None:
    # Changing the override value (not just adding one) re-drifts the payload so
    # a later cap bump still forces the affected case to re-label.
    a = _labels(clip_max_new_tokens_overrides={"sandwich": 2048})
    b = _labels(clip_max_new_tokens_overrides={"sandwich": 1800})
    assert clip_labels_config_payload(a, case="sandwich") != clip_labels_config_payload(
        b, case="sandwich"
    )


def test_labels_settings_includes_cluster_defaults() -> None:
    s = _labels()
    assert s.cluster_max_new_tokens == 1400
    assert s.cluster_sample_token_budget == 7500
    assert s.cluster_max_clips_per_user == 2
    assert s.cluster_max_attempts == 3


def test_cluster_stage_and_scope_helpers() -> None:
    assert STAGE_CLUSTER_LABELS == "cluster_labels"
    assert cluster_scope_for("video") == "video"


def test_cluster_payload_excludes_clip_only_knobs() -> None:
    a = _labels(case_prompts={"video": "A"}, cluster_case_prompts={"video": "X"})
    b = _labels(case_prompts={"video": "B"}, cluster_case_prompts={"video": "X"})
    # Changing the clip prompt MUST NOT drift the cluster fingerprint.
    assert cluster_labels_config_payload(
        a, case="video"
    ) == cluster_labels_config_payload(b, case="video")


def test_cluster_labels_config_payload_changes_per_case_prompt() -> None:
    a = _labels(cluster_case_prompts={"video": "X", "audio": "AX"})
    b = _labels(cluster_case_prompts={"video": "Y", "audio": "AX"})
    assert cluster_labels_config_payload(
        a, case="video"
    ) != cluster_labels_config_payload(b, case="video")
    assert cluster_labels_config_payload(
        a, case="audio"
    ) == cluster_labels_config_payload(b, case="audio")


def test_cluster_payload_drifts_on_cluster_param() -> None:
    base = _labels()
    for k, v in [
        ("cluster_max_new_tokens", 999),
        ("cluster_sample_token_budget", 1000),
        ("cluster_max_clips_per_cluster", 5),
        ("cluster_max_clips_per_user", 1),
        ("cluster_min_tags", 1),
        ("cluster_max_tags", 4),
        ("cluster_min_sentence_chars", 1),
        ("cluster_max_sentence_chars", 50),
        ("generation_seed", 99),
        ("model_path", "./models/other"),
    ]:
        drifted = _labels(**{k: v})
        assert cluster_labels_config_payload(
            base, case="video"
        ) != cluster_labels_config_payload(drifted, case="video"), k


def test_cluster_payload_drifts_on_cluster_generation_param() -> None:
    base = _labels()
    for k, v in [
        ("cluster_model_path", "./models/other-30b"),
        ("cluster_tag_max_chars", 99),
        ("cluster_tag_max_words", 7),
        ("cluster_summary_max_chars", 999),
        ("cluster_summary_target_min", 1),
        ("cluster_summary_target_max", 999),
    ]:
        drifted = _labels(**{k: v})
        assert cluster_labels_config_payload(
            base, case="video"
        ) != cluster_labels_config_payload(drifted, case="video"), k


def test_cluster_payload_stable_for_runtime_or_unrelated_changes() -> None:
    base = _labels(cluster_max_attempts=3, parallelism=1)
    drifted = _labels(cluster_max_attempts=10, parallelism=4)
    # `cluster_max_attempts` is retry policy, `parallelism` is runtime —
    # neither affects the synthesised output, so neither is in the payload.
    assert cluster_labels_config_payload(
        base, case="video"
    ) == cluster_labels_config_payload(drifted, case="video")
