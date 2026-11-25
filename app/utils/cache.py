"""
A tiny in-memory cache with expiry (TTL = "time to live").

Used to avoid paying twice for the same LLM call or search query. For a
multi-server deployment this would be replaced by Redis, but the interface
(get / set) would stay the same.
"""

import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    def __init__(self, ttl_seconds: int, max_items: int = 1000):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if time.monotonic() > expires_at:
            del self._data[key]
            return None
        self._data.move_to_end(key)  # mark as recently used
        return value

    def set(self, key: str, value: Any) -> None:
        self._data[key] = (time.monotonic() + self.ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:  # evict least recently used
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()
