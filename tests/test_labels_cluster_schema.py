import pytest

from core.config import LabelsSettings
from modules.labels.cases import REGISTRY
from modules.labels.schema import cluster_schema


def _labels() -> LabelsSettings:
    return LabelsSettings()


def test_new_cluster_config_defaults_present():
    labels = LabelsSettings()
    assert labels.cluster_model_path.endswith("Qwen3-30B-A3B-GPTQ-Int4")
    assert labels.cluster_tag_max_chars == 28
    assert labels.cluster_tag_max_words == 5
    assert labels.cluster_summary_target_min == 120
    assert labels.cluster_summary_target_max == 160
    assert labels.cluster_summary_max_chars == 200


def test_schema_has_exact_required_keys_for_video_case():
    spec = REGISTRY["video"]
    schema = cluster_schema(spec, _labels())
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(spec.cluster_required_keys)
    assert "dominant_visual_repertoire" in schema["properties"]


def test_tag_schema_caps_length_and_words():
    schema = cluster_schema(REGISTRY["video"], _labels())
    tag = schema["properties"]["tool_tags"]["items"]
    assert tag["maxLength"] == 28
    assert "{0,4}" in tag["pattern"]


def test_summary_capped_at_200_and_arrays_bounded():
    schema = cluster_schema(REGISTRY["video"], _labels())
    assert schema["properties"]["cluster_summary"]["maxLength"] == 200
    rep = schema["properties"]["dominant_visual_repertoire"]
    assert rep["minItems"] == 3 and rep["maxItems"] == 12
    assert schema["properties"]["cluster_label"]["maxLength"] == 40
    conf = schema["properties"]["taste_signalling"]
    assert conf["properties"]["confidence"]["enum"] == ["high", "medium", "low"]


@pytest.mark.parametrize("case_name", sorted(REGISTRY))
def test_every_case_required_keys_are_all_defined_properties(case_name):
    spec = REGISTRY[case_name]
    schema = cluster_schema(spec, _labels())
    required = set(schema["required"])
    properties = set(schema["properties"])
    assert required <= properties, (
        f"{case_name}: required keys missing from properties: {required - properties}"
    )
    assert spec.repertoire_key in schema["properties"]
