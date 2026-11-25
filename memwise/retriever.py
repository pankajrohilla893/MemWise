"""Step 1 of the read path: find the memories that matter for this request.

Two tiers:
  * the PROFILE - a handful of memories loaded into every prompt (name,
    standing instructions). Small on purpose.
  * everything else is RETRIEVED per request and ranked by a blend of signals,
    not similarity alone (weights live in `MemoryConfig.weights`):

        score = 0.60 * relevance    how close it is to the request (embedding cosine)
              + 0.15 * recency      decays with time since last confirmed or used
              + 0.15 * importance   set when the memory was written
              + 0.10 * confidence   stated > inferred; grows each time it's confirmed

Only *current* memories are searched: superseded, retracted and expired rows
are history, and must never reach the prompt as if they were still true.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from .clock import days_between
from .config import DEFAULT_CONFIG, MemoryConfig
from .embed import Embedder, HashingEmbedder, cosine
from .models import Memory
from .store import MemoryStore

_DEFAULT_EMBEDDER = HashingEmbedder()


@dataclass
class Scored:
    memory: Memory
    relevance: float
    recency: float
    score: float


def profile(store: MemoryStore, user_id: str, now: datetime,
            config: MemoryConfig = DEFAULT_CONFIG) -> list[Memory]:
    """Always-loaded memories. Never sensitive, never scoped to one context."""
    return [
        m for m in store.current(user_id, now)
        if (m.kind in config.profile_kinds or m.predicate in config.profile_predicates)
        and not m.sensitive and m.scope == "general"
    ]


def recency(m: Memory, now: datetime, half_life_days: float) -> float:
    last = max(t for t in (m.last_confirmed, m.last_used) if t is not None)
    return math.pow(0.5, days_between(last, now) / half_life_days)


def retrieve(store: MemoryStore, user_id: str, query: str, now: datetime,
             config: MemoryConfig = DEFAULT_CONFIG,
             embedder: Embedder = _DEFAULT_EMBEDDER,
             exclude: frozenset[str] | set[str] = frozenset()) -> list[Scored]:
    q = embedder.embed(query)
    w = config.weights
    results = []
    for m in store.current(user_id, now):
        if m.id in exclude:
            continue
        rel = cosine(q, m.embedding)
        if rel < config.min_relevance:
            continue
        rec = recency(m, now, config.recency_half_life_days)
        score = (w["relevance"] * rel + w["recency"] * rec
                 + w["importance"] * m.importance + w["confidence"] * m.confidence)
        results.append(Scored(m, rel, rec, score))
    results.sort(key=lambda s: s.score, reverse=True)
    return results[: config.top_k]
