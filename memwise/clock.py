from __future__ import annotations

from datetime import datetime, timedelta, timezone


class Clock:
    """A clock you can move forward.

    Memory behaviour depends on time (expiry, staleness, decay), so every module
    asks this object for "now" instead of calling datetime.now() directly. The
    demo uses a fixed start date and jumps ahead; the chat uses real time plus
    whatever offset you add with /advance.
    """

    def __init__(self, fixed_start: datetime | None = None):
        self._fixed = fixed_start
        self._offset = timedelta()

    def now(self) -> datetime:
        base = self._fixed if self._fixed is not None else datetime.now(timezone.utc)
        return base + self._offset

    def advance(self, days: float = 0) -> datetime:
        self._offset += timedelta(days=days)
        return self.now()


def days_between(earlier: datetime, later: datetime) -> float:
    return (later - earlier).total_seconds() / 86400
