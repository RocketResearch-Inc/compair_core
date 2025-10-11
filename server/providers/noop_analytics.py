"""Analytics provider placeholder that records no events."""
from __future__ import annotations


class NoopAnalytics:
    def track(self, *_: object, **__: object) -> None:
        return None
