from __future__ import annotations

from modules.embeddings import cases as cases_mod
from modules.embeddings.cases import build_provider_router, remote_served_cases
from modules.embeddings.providers import ProviderRouter


class _Rec:
    def __init__(self, tag):
        self.tag = tag
        self.seen = []

    def embed(self, payload):
        self.seen.append(payload["case"])
        return [[1.0, 2.0]]


def test_router_dispatches_by_case():
    a, b = _Rec("a"), _Rec("b")
    r = ProviderRouter({"video": lambda: a, "maest": lambda: b})
    r.embed({"case": "video"})
    r.embed({"case": "maest"})
    r.embed({"case": "video"})
    assert a.seen == ["video", "video"] and b.seen == ["maest"]


def test_router_builds_each_backend_once():
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return _Rec("x")

    r = ProviderRouter({"video": factory})
    r.embed({"case": "video"})
    r.embed({"case": "video"})
    assert calls["n"] == 1  # memoized after first use


def test_build_router_shares_one_qwen_instance_lazily(monkeypatch):
    built = {"n": 0}

    def fake_qwen(settings, secrets, *, with_frames):
        built["n"] += 1
        return _Rec("qwen")

    monkeypatch.setattr(cases_mod, "qwen_provider", fake_qwen)
    router = build_provider_router(object(), None, ["video", "sandwich", "audio"])
    # Lazy: nothing built until first embed.
    assert built["n"] == 0
    router.embed({"case": "video"})
    router.embed({"case": "sandwich"})
    router.embed({"case": "audio"})
    # All three Qwen-backbone cases share ONE instance.
    assert built["n"] == 1


def test_remote_served_cases_excludes_local_only():
    served = remote_served_cases()
    assert "video" in served and "sandwich" in served and "audio" in served
    assert "maest" not in served and "gemini" not in served
