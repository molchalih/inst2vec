import importlib

import pytest

lint_docs = importlib.import_module("scripts.lint_docs")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(lint_docs, "REPO_ROOT", tmp_path)
    return tmp_path


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return rel


def test_check_versions_flags_stale(repo):
    rel = _write(repo, "a.md", "the version-6 contract\n")
    out = lint_docs.check_versions([rel], 7)
    assert len(out) == 1
    assert "version-6" in out[0]


def test_check_versions_passes_on_match(repo):
    rel = _write(repo, "a.md", "the version-7 contract\n")
    assert lint_docs.check_versions([rel], 7) == []


def test_check_versions_only_scans_docs(repo):
    rel = _write(repo, "mod.py", "VERSION = 'version-6'\n")
    assert lint_docs.check_versions([rel], 7) == []


def test_check_footprints_flags_token(repo):
    rel = _write(repo, "x.py", "# see docs/superpowers/specs/foo.md\n")
    out = lint_docs.check_footprints([rel])
    assert len(out) == 1
    assert "superpowers" in out[0]


def test_check_footprints_ignores_lockfile(repo):
    rel = _write(repo, "uv.lock", "name = 'anthropic'\n")
    assert lint_docs.check_footprints([rel]) == []


def test_check_footprints_skips_self(repo):
    rel = _write(repo, "scripts/lint_docs.py", "TOKENS = ('claude',)\n")
    assert lint_docs.check_footprints([rel]) == []


def test_check_links_flags_broken(repo):
    rel = _write(repo, "a.md", "see [x](missing.md)\n")
    out = lint_docs.check_links([rel])
    assert len(out) == 1
    assert "missing.md" in out[0]


def test_check_links_ok_for_tracked_target(repo):
    _write(repo, "real.md", "target\n")
    rel = _write(repo, "a.md", "see [x](real.md)\n")
    assert lint_docs.check_links([rel, "real.md"]) == []


def test_check_links_ignores_external(repo):
    rel = _write(repo, "a.md", "see [x](https://example.com) and [y](#anchor)\n")
    assert lint_docs.check_links([rel]) == []
