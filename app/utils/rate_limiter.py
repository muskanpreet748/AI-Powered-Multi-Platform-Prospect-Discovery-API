"""
Per-client rate limiting for our own endpoint.

Each discovery request triggers paid LLM and search API calls, so one client
must not be able to fire hundreds of requests per minute. This is a simple
"sliding window": we remember the timestamps of each client's recent requests
and refuse new ones once the limit for the last 60 seconds is reached.
"""

import time
from collections import deque


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0, max_clients: int = 10_000):
        self.max_requests = max_requests
        self.window = window_seconds
        self.max_clients = max_clients
        self._hits: dict[str, deque[float]] = {}

    def allow(self, client_id: str) -> bool:
        now = time.monotonic()
        if len(self._hits) > self.max_clients:
            self._drop_idle_clients(now)

        hits = self._hits.setdefault(client_id, deque())
        while hits and now - hits[0] > self.window:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True

    def _drop_idle_clients(self, now: float) -> None:
        idle = [cid for cid, hits in self._hits.items() if not hits or now - hits[-1] > self.window]
        for cid in idle:
            del self._hits[cid]
