"""Payload → bytes, identical to ``modules.visualization.export._write_json``.

Byte parity with the on-disk JSON tree depends on reusing the exact same
``json.dumps`` options the exporter uses. The API returns these raw bytes via
a ``Response`` (NOT ``JSONResponse``, which would re-serialise with different
separators), so endpoint bodies equal the static files byte-for-byte.
"""

from __future__ import annotations

import json


def to_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
