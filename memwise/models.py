from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# Different kinds of memory behave differently:
#   fact        - "Lives in Lisbon". Stable-ish, can be superseded.
#   preference  - "Likes tea". Several can be true at once.
#   instruction - "Prefers short answers". Always loaded, always followed.
#   episode     - "Tried Redis and it ran out of memory". Fades unless it keeps being useful.
KINDS = ("fact", "preference", "instruction", "episode")


@dataclass
class Candidate:
    """Something the extractor thinks *might* be worth remembering.

    Nothing is stored yet. The reconciler decides what happens to it.
    """

    kind: str
    predicate: str           # e.g. "lives_in", "employer", "likes"
    value: str               # e.g. "Lisbon"
    text: str                # human-readable statement: "Lives in Lisbon"
    subject: str = "user"
    scope: str = "general"   # general | work | personal
    explicit: bool = False   # the user said "remember ..."
    negated: bool = False    # "I don't drink coffee anymore" -> retract an old memory
    forget: bool = False     # "forget that I live in Oslo" -> delete it, like the delete button
    source_kind: str = "stated"  # stated (user said it) | inferred (model guessed it)
    confidence: float = 0.8
    importance: float = 0.5
    sensitive: bool = False  # health, money, relationships, ...
    ttl_days: int | None = None  # situational facts get an expiry when they are written
    source_ref: str = ""     # which message it came from, e.g. "s1:4"


@dataclass
class Memory:
    """A stored claim about the user, with where it came from and when it was true."""

    id: str
    user_id: str
    kind: str
    subject: str
    predicate: str
    value: str
    text: str
    scope: str
    source_kind: str
    source_ref: str
    confidence: float
    importance: float
    sensitive: bool
    valid_from: datetime              # when we learned it
    last_confirmed: datetime          # last time the user said it again (NOT last time we used it)
    valid_to: datetime | None = None  # set when it stops being true; the row is kept as history
    expires_at: datetime | None = None
    superseded_by: str | None = None
    last_used: datetime | None = None
    use_count: int = 0
    embedding: list[float] = field(default_factory=list, repr=False)

    def is_current(self, now: datetime) -> bool:
        if self.valid_to is not None:
            return False
        if self.expires_at is not None and self.expires_at <= now:
            return False
        return True
