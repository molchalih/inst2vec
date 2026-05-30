from __future__ import annotations

import pytest

from modules.labels.cases import REGISTRY
from modules.labels.ordering import CycleError, case_run_order


def test_default_registry_orders_video_before_sandwich_and_gemini() -> None:
    order = case_run_order(REGISTRY.keys(), registry=REGISTRY)
    assert order.index("video") < order.index("sandwich")
    assert order.index("video") < order.index("gemini")


def test_subset_orders_are_dependency_consistent() -> None:
    order = case_run_order(["sandwich", "video"], registry=REGISTRY)
    assert order == ["video", "sandwich"]


def test_unknown_case_raises() -> None:
    with pytest.raises(KeyError):
        case_run_order(["video", "nope"], registry=REGISTRY)


def test_cycle_raises() -> None:
    from dataclasses import replace

    bad = {
        "a": replace(REGISTRY["spoken"], name="a", consumes_label_cases=("b",)),
        "b": replace(REGISTRY["spoken"], name="b", consumes_label_cases=("a",)),
    }
    with pytest.raises(CycleError):
        case_run_order(["a", "b"], registry=bad)
