"""Background upkeep, run on a schedule (nightly in production) or with /maintain.

Memory gets worse if nothing ever tidies it. This job:
  * EXPIRE - closes situational facts whose expiry date has passed
             ("traveling to Tokyo next week" stops being true by itself)
  * MERGE  - folds near-duplicate memories into one
  * PRUNE  - deletes episodes nobody has used or restated in a long time

A production version would also condense old episodes into facts with an LLM
("tried Redis twice, both times it ran out of memory" -> "Redis is a poor fit
for this team's session cache") and rebuild the profile summary.
"""

from __future__ import annotations

from datetime import datetime

from .clock import days_between
from .config import DEFAULT_CONFIG, MemoryConfig
from .embed import cosine
from .store import MemoryStore


def run(store: MemoryStore, user_id: str, now: datetime,
        config: MemoryConfig = DEFAULT_CONFIG) -> list[str]:
    report: list[str] = []
    open_rows = [m for m in store.all(user_id) if m.valid_to is None]

    for m in open_rows:
        if m.expires_at is not None and m.expires_at <= now:
            m.valid_to = m.expires_at
            store.save(m)
            store.log(user_id, now, "EXPIRE", m.id, m.text)
            report.append(f"EXPIRE  {m.id} {m.text!r} (expired {m.expires_at:%Y-%m-%d})")

    current = [m for m in open_rows if m.valid_to is None]

    # Merge near-duplicates (multi-valued predicates only; single-valued ones are
    # already kept to one current value by the reconciler).
    closed: set[str] = set()
    for i, a in enumerate(current):
        for b in current[i + 1:]:
            if a.id in closed or b.id in closed:
                continue
            if a.predicate != b.predicate or a.predicate in config.single_valued:
                continue
            if cosine(a.embedding, b.embedding) >= config.duplicate_similarity:
                keep, drop = (a, b) if a.last_confirmed >= b.last_confirmed else (b, a)
                keep.confidence = min(1.0, max(keep.confidence, drop.confidence) + 0.05)
                keep.use_count += drop.use_count
                drop.valid_to, drop.superseded_by = now, keep.id
                store.save(keep)
                store.save(drop)
                closed.add(drop.id)
                store.log(user_id, now, "MERGE", keep.id, f"{drop.id} merged into {keep.id}")
                report.append(f"MERGE   {drop.id} {drop.text!r} into {keep.id}")

    for m in current:
        if m.id in closed or m.kind != "episode":
            continue
        last_touch = max(t for t in (m.last_confirmed, m.last_used) if t is not None)
        idle = days_between(last_touch, now)
        if idle > config.prune_episodes_after_days:
            store.delete(m.id)
            store.log(user_id, now, "PRUNE", m.id, f"{m.text!r} unused for {idle:.0f} days")
            report.append(f"PRUNE   {m.id} {m.text!r} (unused for {idle:.0f} days)")

    return report
