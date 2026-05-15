"""Unit tests for modules/captions/state.py."""

from modules.captions.state import (
    SCOPE_CLEAN,
    SCOPE_DETECT,
    SCOPE_TRANSLATE,
    clean_caption_text,
)


def test_scope_constants_are_distinct_strings():
    assert isinstance(SCOPE_CLEAN, str) and SCOPE_CLEAN
    assert isinstance(SCOPE_DETECT, str) and SCOPE_DETECT
    assert isinstance(SCOPE_TRANSLATE, str) and SCOPE_TRANSLATE
    assert len({SCOPE_CLEAN, SCOPE_DETECT, SCOPE_TRANSLATE}) == 3


def test_clean_caption_text_removes_mentions_and_collapses_whitespace():
    assert clean_caption_text("hello @bob   world\n") == "hello world"


def test_clean_caption_text_handles_only_mentions():
    assert clean_caption_text("@a @b @c") == ""


def test_clean_caption_text_empty_inputs():
    assert clean_caption_text("") == ""
    assert clean_caption_text("   \n  ") == ""
