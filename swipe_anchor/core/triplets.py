"""Odd-one-out -> triplet derivation (plan §1.2, §1.3).

A crowd odd-one-out answer over an unordered triple ``(A, B, C)`` where ``C`` is
crossed out asserts that the two *non-crossed* creators are mutually nearer to
each other than to the crossed one. That single answer yields **two** triplets:

    (anchor=A, positive=B, negative=C)
    (anchor=B, positive=A, negative=C)

each encoding ``cos(anchor, positive) > cos(anchor, negative)``. Keeping this as a
pure, dependency-free function makes it identically testable on the backend and
mirror-able in the frontend ``core/`` layer.
"""

from __future__ import annotations

from typing import NamedTuple


class Triplet(NamedTuple):
    """An ordinal constraint ``cos(anchor, positive) > cos(anchor, negative)``."""

    anchor: int
    positive: int
    negative: int


def derive_triplets(creators: tuple[int, int, int], odd_id: int) -> list[Triplet]:
    """Return the two triplets implied by crossing ``odd_id`` out of ``creators``.

    Raises ``ValueError`` if the triple does not contain three distinct creators
    or if ``odd_id`` is not one of them.
    """
    if len(set(creators)) != 3:
        raise ValueError(f"triple must hold three distinct creators, got {creators!r}")
    if odd_id not in creators:
        raise ValueError(f"odd_id {odd_id!r} is not in triple {creators!r}")

    kept = [c for c in creators if c != odd_id]
    a, b = kept
    return [
        Triplet(anchor=a, positive=b, negative=odd_id),
        Triplet(anchor=b, positive=a, negative=odd_id),
    ]
