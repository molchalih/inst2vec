"""Concurrent dispatch invariants for the embeddings runner.

We don't load the real model; we inject a fake provider that records
which clips it was asked to embed.
"""

import threading
import time


def test_inflight_one_processes_serially():
    """With inflight=1 the runner should process one clip at a time."""
    from modules.embeddings.runner import _dispatch_embedding_jobs

    seen_concurrency: list[int] = []
    in_flight = {"n": 0}
    lock = threading.Lock()

    def fake_embed(job):
        with lock:
            in_flight["n"] += 1
            seen_concurrency.append(in_flight["n"])
        time.sleep(0.01)
        with lock:
            in_flight["n"] -= 1
        return job["clip_id"], b"vec"

    jobs = [{"clip_id": i} for i in range(8)]
    results = list(_dispatch_embedding_jobs(jobs, fake_embed, inflight=1))

    assert {r[0] for r in results} == set(range(8))
    assert max(seen_concurrency) == 1


def test_inflight_four_runs_up_to_four_in_parallel():
    from modules.embeddings.runner import _dispatch_embedding_jobs

    seen_concurrency: list[int] = []
    in_flight = {"n": 0}
    lock = threading.Lock()

    def fake_embed(job):
        with lock:
            in_flight["n"] += 1
            seen_concurrency.append(in_flight["n"])
        time.sleep(0.05)
        with lock:
            in_flight["n"] -= 1
        return job["clip_id"], b"vec"

    jobs = [{"clip_id": i} for i in range(8)]
    results = list(_dispatch_embedding_jobs(jobs, fake_embed, inflight=4))

    assert {r[0] for r in results} == set(range(8))
    assert max(seen_concurrency) >= 2  # at least some real parallelism
    assert max(seen_concurrency) <= 4  # but bounded


def test_failure_in_one_job_does_not_kill_others():
    from modules.embeddings.runner import _dispatch_embedding_jobs

    def fake_embed(job):
        if job["clip_id"] == 3:
            raise RuntimeError("boom")
        return job["clip_id"], b"v"

    jobs = [{"clip_id": i} for i in range(5)]
    results = list(_dispatch_embedding_jobs(jobs, fake_embed, inflight=2))

    succeeded = [r for r in results if r[1] is not None]
    failed = [r for r in results if r[1] is None]
    assert len(succeeded) == 4
    assert len(failed) == 1
    assert failed[0][0] == 3
