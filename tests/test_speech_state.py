"""Unit tests for modules/speech/state.py helpers and constants."""

from modules.speech.state import (
    HALLUCINATION_MARKERS,
    SCOPE_CLASSIFY,
    SCOPE_CLEAN,
    SCOPE_TRANSLATE,
    has_hallucination_marker,
    has_meaningful_speech_text,
    is_repeated_output,
)


def test_scope_constants_are_distinct_strings():
    assert isinstance(SCOPE_CLASSIFY, str) and SCOPE_CLASSIFY
    assert isinstance(SCOPE_TRANSLATE, str) and SCOPE_TRANSLATE
    assert isinstance(SCOPE_CLEAN, str) and SCOPE_CLEAN
    assert len({SCOPE_CLASSIFY, SCOPE_TRANSLATE, SCOPE_CLEAN}) == 3


def test_hallucination_markers_includes_known_marker():
    assert "DimaTorzok" in HALLUCINATION_MARKERS


def test_has_hallucination_marker_detects_known():
    assert has_hallucination_marker("subtitles by DimaTorzok") is True


def test_has_hallucination_marker_clean_text():
    assert has_hallucination_marker("hello world") is False


def test_has_hallucination_marker_empty_or_none():
    assert has_hallucination_marker("") is False
    assert has_hallucination_marker(None) is False


def test_has_meaningful_speech_text_strips_punctuation_and_counts_letters():
    # 5 letters → not meaningful when threshold is 8
    assert has_meaningful_speech_text("...!! hi ..", min_meaningful_chars=8) is False
    # 8 letters across mixed punctuation → meaningful
    assert has_meaningful_speech_text("hi! hi! yo yo", min_meaningful_chars=8) is True


def test_has_meaningful_speech_text_cyrillic():
    assert has_meaningful_speech_text("Привет!", min_meaningful_chars=6) is True


def test_has_meaningful_speech_text_handles_none():
    assert has_meaningful_speech_text(None, min_meaningful_chars=1) is False


def test_is_repeated_output_detects_single_token_loop():
    assert is_repeated_output("thanks thanks thanks thanks thanks") is True


def test_is_repeated_output_clean_text():
    assert is_repeated_output("the quick brown fox jumps over") is False


def test_is_repeated_output_empty_or_none():
    assert is_repeated_output("") is False
    assert is_repeated_output(None) is False


# ── is_too_short ─────────────────────────────────────────────────────────────


def test_is_too_short_handles_none_and_empty():
    from modules.speech.state import is_too_short

    assert is_too_short(None, min_chars=5) is True
    assert is_too_short("", min_chars=5) is True


def test_is_too_short_strips_whitespace_before_measuring():
    from modules.speech.state import is_too_short

    assert is_too_short("   you   ", min_chars=5) is True
    assert is_too_short("hello", min_chars=5) is False
    assert is_too_short("hello!", min_chars=5) is False


# ── has_low_letter_ratio ─────────────────────────────────────────────────────


def test_has_low_letter_ratio_flags_punctuation_only():
    from modules.speech.state import has_low_letter_ratio

    assert has_low_letter_ratio("...", min_ratio=0.3) is True
    assert has_low_letter_ratio("!!! ??? !!!", min_ratio=0.3) is True


def test_has_low_letter_ratio_flags_digit_heavy_noise():
    from modules.speech.state import has_low_letter_ratio

    # "1, 2, 3, 4, 5." → 0 letters, all noise
    assert has_low_letter_ratio("1, 2, 3, 4, 5.", min_ratio=0.3) is True


def test_has_low_letter_ratio_passes_real_speech():
    from modules.speech.state import has_low_letter_ratio

    assert has_low_letter_ratio("Hello, world!", min_ratio=0.3) is False
    assert (
        has_low_letter_ratio(
            "Rap is so powerful because people took it seriously.", min_ratio=0.3
        )
        is False
    )


def test_has_low_letter_ratio_empty_or_none():
    from modules.speech.state import has_low_letter_ratio

    assert has_low_letter_ratio(None, min_ratio=0.3) is True
    assert has_low_letter_ratio("", min_ratio=0.3) is True


def test_has_low_letter_ratio_handles_cyrillic():
    from modules.speech.state import has_low_letter_ratio

    assert has_low_letter_ratio("Привет, мир!", min_ratio=0.3) is False
