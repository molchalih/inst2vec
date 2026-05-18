"""Canonical English-language detection.

Both forms use the same rule: case-insensitive prefix match on "en".
Matches "en", "EN", "eng", "en-US", "English", etc.

Use ``is_english(code)`` from Python code; use ``sql_is_english(col)``
inside SQLAlchemy filter clauses. Negate either with ``not`` / ``~`` as
needed.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func


def is_english(code: str | None) -> bool:
    """True iff ``code`` is present and is an English-family tag."""
    return bool(code) and code.lower().startswith("en")


def sql_is_english(col: Any):
    """SQLAlchemy clause: True iff ``col`` is an English-family tag.

    Equivalent of ``is_english`` for use in ``.filter()``.
    """
    return func.lower(col).like("en%")
