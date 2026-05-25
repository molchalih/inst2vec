"""Shared concurrency primitives reused across pipeline stages.

`retry_with_backoff` is the single retry primitive used by the download and
profile-parsing stages so neither stage imports from the other.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable


def retry_with_backoff[T](
    fn: Callable[[], T],
    *,
    max_attempts: int,
    retry_delay: float,
    retry_jitter: float,
) -> T:
    """Call ``fn`` until it succeeds or attempts run out.

    On each failed attempt (any raised exception) sleep
    ``retry_delay + random.uniform(0, retry_jitter)`` seconds, except after the
    final attempt. Returns ``fn()``'s value on success; re-raises the last
    exception once ``max_attempts`` are exhausted.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(retry_delay + random.uniform(0, retry_jitter))
    assert last_exc is not None
    raise last_exc
