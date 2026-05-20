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


def test_elapsed_excludes_pool_queue_wait_under_saturation():
    """Under saturation (len(jobs) > inflight) the reported elapsed time
    must reflect only ``embed_fn`` execution, not pool-queue wait."""
    from modules.embeddings.runner import _dispatch_embedding_jobs

    def fake_embed(job):
        time.sleep(0.05)
        return job["clip_id"], b"v"

    # 4 jobs, 2 workers → batches of 2; without the fix, the second batch
    # would report ~0.10s (wait + work) instead of ~0.05s (work only).
    jobs = [{"clip_id": i} for i in range(4)]
    results = list(_dispatch_embedding_jobs(jobs, fake_embed, inflight=2))

    elapsed = [r[2] for r in results]
    assert all(e is not None for e in elapsed)
    # Allow some jitter but ensure no entry is anywhere close to 2× work time.
    assert max(elapsed) < 0.09, elapsed


def test_embed_with_token_fallback_injects_case_and_clip_id():
    """Runner must enrich every payload with `case` and `clip_id` so the
    remote embedder service (which requires both) accepts it, even for
    cases like `audio` whose payload_builder has no video path."""
    from types import SimpleNamespace

    from modules.embeddings.cases import CASE_REGISTRY
    from modules.embeddings.runner import _embed_with_token_fallback

    captured: list[dict] = []

    class _Provider:
        def embed(self, payload):
            captured.append(dict(payload))
            return [[1.0, 2.0]]

    clip = SimpleNamespace(id=42)

    _embed_with_token_fallback(
        _Provider(), CASE_REGISTRY["audio"], clip, "hello", None, None, None, None
    )
    _embed_with_token_fallback(
        _Provider(), CASE_REGISTRY["video"], clip, None, "/v/42.mp4", None, 1.0, 32
    )

    assert captured[0]["case"] == "audio"
    assert captured[0]["clip_id"] == 42
    assert captured[1]["case"] == "video"
    assert captured[1]["clip_id"] == 42
