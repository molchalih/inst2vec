"""Unit tests for the file-backed hallucination loader."""

from __future__ import annotations

from pathlib import Path

from modules.speech.state import (
    HALLUCINATION_PHRASES,
    HALLUCINATIONS_DIR,
    _load_hallucination_phrases,
    has_hallucination_marker,
)


def test_hallucinations_dir_exists_and_contains_en_and_extras():
    assert HALLUCINATIONS_DIR.is_dir()
    assert (HALLUCINATIONS_DIR / "en.txt").is_file()
    assert (HALLUCINATIONS_DIR / "extras.txt").is_file()


def test_loader_skips_comment_and_blank_lines(tmp_path: Path):
    p = tmp_path / "h.txt"
    p.write_text(
        "# comment line\n\n   \nThanks for watching!\n I'll see you next time. \n",
        encoding="utf-8",
    )
    phrases = _load_hallucination_phrases([p], min_phrase_chars=5)
    assert "Thanks for watching!" in phrases
    assert "I'll see you next time." in phrases
    assert all(not phrase.startswith("#") for phrase in phrases)


def test_loader_drops_too_short_phrases(tmp_path: Path):
    p = tmp_path / "h.txt"
    p.write_text("Bye.\nNo.\nOK.\nThanks for watching!\n.\n", encoding="utf-8")
    phrases = _load_hallucination_phrases([p], min_phrase_chars=5)
    assert "Thanks for watching!" in phrases
    assert "." not in phrases
    assert "Bye." not in phrases
    assert "No." not in phrases


def test_loader_dedupes(tmp_path: Path):
    a = tmp_path / "a.txt"
    a.write_text("Thanks for watching!\n", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("Thanks for watching!\n", encoding="utf-8")
    phrases = _load_hallucination_phrases([a, b], min_phrase_chars=5)
    assert phrases.count("Thanks for watching!") == 1


def test_module_level_phrases_include_extras_marker():
    assert "DimaTorzok" in HALLUCINATION_PHRASES


def test_has_hallucination_marker_exact_match_normalized():
    # whitespace + case + trailing punctuation should still match exactly
    assert has_hallucination_marker("  thanks for watching!  ") is True


def test_has_hallucination_marker_long_substring():
    # Phrases past the substring-safe floor match inside a longer transcript.
    assert (
        has_hallucination_marker(
            "(intro music) Thanks for watching, and I'll see you next time. See ya!"
        )
        is True
    )


def test_has_hallucination_marker_keeps_legacy_marker():
    # extras.txt entries are substring-eligible regardless of length.
    assert has_hallucination_marker("subtitles by DimaTorzok") is True


def test_has_hallucination_marker_does_not_substring_match_short_common_phrase():
    # "Thank you." is a corpus entry but is short and ubiquitous in real
    # speech, so it must only fire on whole-text matches.
    assert (
        has_hallucination_marker(
            "Thank you. Now I will explain how I made this recipe."
        )
        is False
    )
    assert has_hallucination_marker("I love you. That's all I wanted to say.") is False
    # Exact whole-text match still works for the same phrase.
    assert has_hallucination_marker(" Thank you. ") is True


def test_has_hallucination_marker_clean_text():
    assert has_hallucination_marker("Rap is so powerful because") is False


def test_has_hallucination_marker_empty_or_none():
    assert has_hallucination_marker("") is False
    assert has_hallucination_marker(None) is False
