"""Topo-sort label cases by ``consumes_label_cases``.

The clip pass iterates cases in the order this function returns so a
case that consumes another's ``ClipLabel.payload`` always runs after
its dependency, regardless of ``CASE_REGISTRY`` insertion order or the
ordering of ``default_cases(settings)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from modules.labels.cases import LabelCaseSpec


class CycleError(ValueError):
    pass


def case_run_order(
    cases: Iterable[str], *, registry: Mapping[str, LabelCaseSpec]
) -> list[str]:
    """Return ``cases`` topo-sorted by ``consumes_label_cases``.

    Ties broken by registry insertion order so the result is stable.
    Raises ``KeyError`` if any case is unknown, ``CycleError`` on a
    dependency cycle.
    """
    desired = list(cases)
    for name in desired:
        if name not in registry:
            raise KeyError(name)

    # visit state per case: 0 unseen, 1 on-stack, 2 done
    state: dict[str, int] = {}
    out: list[str] = []

    def visit(name: str, stack: list[str]) -> None:
        s = state.get(name, 0)
        if s == 2:
            return
        if s == 1:
            raise CycleError(" → ".join([*stack, name]))
        state[name] = 1
        spec = registry[name]
        for dep in spec.consumes_label_cases:
            if dep in desired:
                visit(dep, [*stack, name])
        state[name] = 2
        out.append(name)

    for name in desired:
        visit(name, [])
    return out
