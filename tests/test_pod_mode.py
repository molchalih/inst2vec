"""--pod dispatches to the worker against the given host and never inits DBs."""

from __future__ import annotations

import sys

import main as main_mod


def test_pod_mode_invokes_worker(monkeypatch):
    called = {}

    def fake_run_pod(host: str, video_root: str) -> None:
        called["host"] = host
        called["video_root"] = video_root

    # cli() imports run_pod lazily from the embeddings package (so the pod
    # image, which copies modules/ but not main.py, can import it), so patch
    # the source module rather than a name on main.
    from modules.embeddings import pod as pod_mod

    monkeypatch.setattr(pod_mod, "run_pod", fake_run_pod)
    monkeypatch.setattr(
        sys, "argv", ["main.py", "--pod", "--host", "orch:8765", "--video-root", "/w/v"]
    )
    main_mod.cli()
    assert called == {"host": "orch:8765", "video_root": "/w/v"}


def test_no_pod_runs_pipeline(monkeypatch):
    ran = {"pipeline": False}
    monkeypatch.setattr(
        main_mod, "run_pipeline", lambda: ran.__setitem__("pipeline", True)
    )
    monkeypatch.setattr(sys, "argv", ["main.py"])
    main_mod.cli()
    assert ran["pipeline"] is True
