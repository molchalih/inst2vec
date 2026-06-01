import pytest

from core.config import LabelsSettings, _load_settings
from modules.labels.prompts import prompt_for, prompt_for_cluster


def _labels(**overrides) -> LabelsSettings:
    base = dict(
        case_prompts={
            "video": "  clip prompt  ",
            "audio": "  audio clip prompt  ",
        },
        cluster_case_prompts={
            "video": "  cluster prompt  ",
            "audio": "  audio cluster prompt  ",
        },
    )
    base.update(overrides)
    return LabelsSettings(**base)


def test_clip_prompt_strips_whitespace() -> None:
    assert prompt_for(_labels(), case="video") == "clip prompt"


def test_cluster_prompt_strips_whitespace() -> None:
    assert prompt_for_cluster(_labels(), case="video") == "cluster prompt"


def test_prompt_for_returns_case_specific_text() -> None:
    s = _labels()
    assert prompt_for(s, case="video") == "clip prompt"
    assert prompt_for(s, case="audio") == "audio clip prompt"


def test_prompt_for_missing_case_raises() -> None:
    s = _labels(case_prompts={"video": "only video"})
    with pytest.raises(ValueError, match=r"missing labels\.case_prompts\.audio"):
        prompt_for(s, case="audio")


def test_prompt_for_cluster_returns_case_specific_text() -> None:
    s = _labels()
    assert prompt_for_cluster(s, case="video") == "cluster prompt"
    assert prompt_for_cluster(s, case="audio") == "audio cluster prompt"


def test_prompt_for_cluster_missing_case_raises() -> None:
    s = _labels(cluster_case_prompts={"video": "only video"})
    with pytest.raises(
        ValueError, match=r"missing labels\.cluster_case_prompts\.audio"
    ):
        prompt_for_cluster(s, case="audio")


def test_all_cases_have_clip_prompts():
    settings = _load_settings()
    for case in ("video", "spoken", "textual", "auditory", "sandwich"):
        assert prompt_for(settings.labels, case=case).strip()
