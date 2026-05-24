import time

from modules.embeddings.worker import DRAINED, UNREACHABLE, run_worker


class _UnreachableSource:
    """Always reports the coordinator as unreachable."""

    def lease(self, *, served_only):
        return UNREACHABLE


class _DrainAfterUnreachable:
    def __init__(self):
        self.calls = 0

    def lease(self, *, served_only):
        self.calls += 1
        return DRAINED if self.calls > 2 else UNREACHABLE


def test_worker_exits_after_unreachable_budget():
    src = _UnreachableSource()
    t0 = time.monotonic()
    run_worker(
        src,
        provider=object(),
        video_root="/x",
        inflight=1,
        served_only=True,
        poll_idle_s=0.01,
        unreachable_exit_s=0.05,
    )
    assert time.monotonic() - t0 < 5  # returned, did not hang


def test_unreachable_then_drain_does_not_exit_early():
    src = _DrainAfterUnreachable()
    run_worker(
        src,
        provider=object(),
        video_root="/x",
        inflight=1,
        served_only=True,
        poll_idle_s=0.01,
        unreachable_exit_s=100,
    )
    assert src.calls > 2  # kept polling through transient unreachability, then drained
