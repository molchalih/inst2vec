"""Lightweight documentation linter, run in CI.

Three checks over the git-tracked working tree:
  1. version consistency — every "version-<N>" string in tracked docs matches
     the canonical SCHEMA_VERSION.
  2. link integrity — relative Markdown links between tracked docs resolve.
  3. footprint guard — no tracked file names internal tooling.

Each violation prints as "path:line: message"; exits 1 if any are found.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The footprint guard necessarily contains the token list, so it must not scan
# itself or its test.
_FOOTPRINT_SELF = {"scripts/lint_docs.py", "tests/test_lint_docs.py"}

_FOOTPRINT_TOKENS = (
    "claude",
    "anthropic",
    "superpowers",
    ".claude/",
    "co-authored-by",
    "copilot",
    "ai-workflow",
)

_VERSION_RE = re.compile(r"version-(\d+)", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def schema_version() -> int:
    text = (REPO_ROOT / "core/contract.py").read_text()
    match = re.search(r"^SCHEMA_VERSION\s*=\s*(\d+)", text, re.MULTILINE)
    if not match:
        raise RuntimeError("SCHEMA_VERSION not found in core/contract.py")
    return int(match.group(1))


def check_versions(files: list[str], version: int) -> list[str]:
    violations: list[str] = []
    for rel in files:
        if not (rel.endswith(".md") or rel == ".env.example"):
            continue
        path = REPO_ROOT / rel
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for match in _VERSION_RE.finditer(line):
                if int(match.group(1)) != version:
                    violations.append(
                        f"{rel}:{lineno}: stale contract version "
                        f"'version-{match.group(1)}' (SCHEMA_VERSION is {version})"
                    )
    return violations


def check_links(files: list[str]) -> list[str]:
    tracked = set(files)
    violations: list[str] = []
    for rel in files:
        if not rel.endswith(".md"):
            continue
        path = REPO_ROOT / rel
        base = path.parent
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            for match in _MD_LINK_RE.finditer(line):
                target = match.group(1).split("#", 1)[0].strip()
                if not target or target.startswith(
                    ("http://", "https://", "mailto:", "#")
                ):
                    continue
                resolved = (base / target).resolve()
                try:
                    rel_resolved = resolved.relative_to(REPO_ROOT).as_posix()
                except ValueError:
                    continue
                if rel_resolved not in tracked and not resolved.exists():
                    violations.append(f"{rel}:{lineno}: broken doc link '{target}'")
    return violations


def check_footprints(files: list[str]) -> list[str]:
    violations: list[str] = []
    for rel in files:
        if rel in _FOOTPRINT_SELF or rel.endswith((".lock", ".lockb")):
            continue
        path = REPO_ROOT / rel
        try:
            content = path.read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            low = line.lower()
            for token in _FOOTPRINT_TOKENS:
                if token in low:
                    violations.append(f"{rel}:{lineno}: tooling footprint '{token}'")
    return violations


def main() -> int:
    files = tracked_files()
    violations = (
        check_versions(files, schema_version())
        + check_links(files)
        + check_footprints(files)
    )
    for violation in violations:
        print(violation)
    if violations:
        print(f"\n{len(violations)} documentation lint violation(s).")
        return 1
    print("docs lint: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
