"""Closed Literal types shared by core.log and core.console (no other imports)."""

from typing import Literal

Verb = Literal[
    "INIT",
    "LOAD",
    "SCAN",
    "SKIP",
    "GET",
    "PUT",
    "EXTRACT",
    "CLEAN",
    "WRITE",
    "DELETE",
    "SEAL",
]
Result = Literal["ok", "ERR", "WARN"]
