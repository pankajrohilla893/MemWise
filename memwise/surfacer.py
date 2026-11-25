"""Step 2 of the read path: should the assistant use this memory, and how?

Retrieving a memory is not the same as using it. Each retrieved memory gets one
of four actions:

    USE      personalize the answer quietly ("here are peanut-free snacks")
    MENTION  say it out loud because it changes the answer ("last time Redis ran out of memory")
    VERIFY   it may be out of date; check before relying on it ("are you still at Acme?")
    HOLD     keep it out of the prompt entirely

HOLD matters most. A memory that isn't in the prompt can't leak, so the model
never has to be trusted to "not mention" something.

The one exception is safety memories (`MemoryConfig.safety_predicates`, allergies by
default). Holding back "allergic to peanuts" because a dinner question scored as
unrelated could hurt someone, so they are always USE, in every scope, however
unrelated, old or uncertain they look. They stay in force until the user says otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .clock import days_between
from .config import DEFAULT_CONFIG, MemoryConfig
from .retriever import Scored


@dataclass
class SurfaceDecision:
    scored: Scored
    action: str
    reason: str


def decide(s: Scored, scope: str, now: datetime,
           config: MemoryConfig = DEFAULT_CONFIG) -> SurfaceDecision:
    m = s.memory

    # Checked before scope: an allergy filed under "personal" still matters when a work chat
    # plans the team lunch, and models don't file allergies consistently.
    if m.predicate in config.safety_predicates:
        return SurfaceDecision(s, "USE", "safety: missing it could hurt the user, so it is never held back")

    # Context integrity: work memories stay out of personal chats and vice versa.
    if m.scope != "general" and scope != m.scope:
        return SurfaceDecision(s, "HOLD", f"belongs to the {m.scope} context; this chat is {scope}")

    if m.sensitive and s.relevance < config.sensitive_relevant_enough:
        return SurfaceDecision(s, "HOLD", "sensitive and only loosely related")

    if s.relevance < config.relevant_enough:
        return SurfaceDecision(s, "HOLD", "not relevant enough to this request")

    age = days_between(m.last_confirmed, now)
    if m.kind == "fact" and age > config.stale_after_days:
        return SurfaceDecision(s, "VERIFY", f"last confirmed {age:.0f} days ago; may have changed")

    if m.confidence < config.low_confidence:
        return SurfaceDecision(s, "VERIFY", f"low confidence ({m.confidence:.2f})")

    if m.kind == "episode":
        return SurfaceDecision(s, "MENTION", "a past experience that should change the answer")

    if m.sensitive:
        return SurfaceDecision(s, "USE", "relevant; apply it without reciting the sensitive detail")

    return SurfaceDecision(s, "USE", "relevant; personalize without announcing it")
