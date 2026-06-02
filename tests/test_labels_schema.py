from core.config import LabelsSettings
from modules.labels.cases import REGISTRY
from modules.labels.schema import clip_schema


def test_clip_schema_keys_match_case_required_keys():
    labels = LabelsSettings()
    for case, spec in REGISTRY.items():
        schema = clip_schema(spec, labels)
        assert set(schema["properties"]) == set(spec.clip_required_keys), case
        assert schema["required"] == sorted(spec.clip_required_keys), case
        assert schema["additionalProperties"] is False


def test_clip_schema_observable_and_grounded_shapes():
    labels = LabelsSettings()
    spec = REGISTRY["video"]
    schema = clip_schema(spec, labels)
    obs = schema["properties"][spec.observable_key]
    assert obs["type"] == "array"
    assert set(obs["items"]["required"]) == {"tag", "evidence"}
    aes = schema["properties"]["aesthetic_tags"]
    assert set(aes["items"]["required"]) == {"tag", "grounded_in", "confidence"}
    assert aes["items"]["properties"]["confidence"]["enum"] == ["high", "medium", "low"]
    assert schema["properties"][spec.sentence_key]["type"] == "string"
