"""Pin the visualization schema version.

When this fails, the frontend's ``frontend/src/data/schemas/version.ts``
must be bumped in lockstep — the two values are the single contract
between the exporter and the Zod schemas.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.contract import SCHEMA_VERSION


def test_schema_version_is_pinned() -> None:
    assert SCHEMA_VERSION == 7


def test_frontend_schema_version_matches_backend() -> None:
    ts = (
        Path(__file__).resolve().parent.parent
        / "frontend"
        / "src"
        / "data"
        / "schemas"
        / "version.ts"
    ).read_text()
    m = re.search(r"SCHEMA_VERSION\s*=\s*(\d+)", ts)
    assert m is not None, "could not find SCHEMA_VERSION in version.ts"
    assert int(m.group(1)) == SCHEMA_VERSION
