from __future__ import annotations

from enum import StrEnum


class Eligibility(StrEnum):
    PENDING = "pending"
    ELIGIBLE = "eligible"
    DISQUALIFIED = "disqualified"


_DB_BY_ENUM = {
    Eligibility.PENDING: 0,
    Eligibility.ELIGIBLE: 1,
    Eligibility.DISQUALIFIED: 2,
}
_ENUM_BY_DB = {value: key for key, value in _DB_BY_ENUM.items()}


def eligibility_db(value: Eligibility) -> int:
    return _DB_BY_ENUM[value]


def eligibility_from_db(value: int | None) -> Eligibility:
    return _ENUM_BY_DB.get(0 if value is None else int(value), Eligibility.PENDING)


def is_pending(column):
    return column == eligibility_db(Eligibility.PENDING)


def is_eligible(column):
    return column == eligibility_db(Eligibility.ELIGIBLE)


def is_disqualified(column):
    return column == eligibility_db(Eligibility.DISQUALIFIED)
