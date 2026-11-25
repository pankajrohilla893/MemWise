"""The facade your product talks to.

    ingest(user_id, messages)   after a session (or right away for "remember ...")  -> write path
    recall(user_id, message)    before each model call                              -> read path
    build_context(recall)       turn the recall into text for your system prompt
    maintain_all()              on a schedule (nightly)                             -> upkeep
    forget(user_id, memory_id)  when the user asks                                  -> user control
"""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Union

from . import maintenance, reconciler, retriever, surfacer
from .clock import Clock, days_between
from .config import MemoryConfig
from .embed import Embedder, HashingEmbedder, cosine
from .extractor import Extractor, RuleExtractor
from .models import Candidate, Memory
from .reconciler import Decision
from .store import MemoryStore
from .surfacer import SurfaceDecision

# A message is plain text or a (source_ref, text) pair, both meaning something the user said,
# or a chat turn {"role": "user" | "assistant", "content": "...", "ref": "..."} (ref optional),
# the same dicts you send to your model. The source_ref (e.g. "session42:3") is stored with
# each memory so you can trace it back. Assistant turns are context only: memories come from
# what the user said.
Message = Union[str, tuple[str, str], dict[str, Any]]


@dataclass
class Recall:
    """What `recall()` found for one message: the profile plus a decision per memory."""

    query: str
    scope: str
    profile: list[Memory] = field(default_factory=list)
    items: list[SurfaceDecision] = field(default_factory=list)

    @property
    def used(self) -> list[SurfaceDecision]:
        """Everything except HOLD: the memories that go into the prompt."""
        return [d for d in self.items if d.action != "HOLD"]


class MemoryManager:
    """Long-term memory for one product. Safe to share between threads.

    Args:
        store:     where memories live. Defaults to an in-memory SQLite store;
                   pass `MemoryStore("memory.db")` to persist.
        extractor: turns messages into candidate memories. Defaults to the offline
                   `RuleExtractor`; use `LLMExtractor` with any model (or your own)
                   in production.
        embedder:  turns text into vectors. Defaults to the built-in `HashingEmbedder`.
        config:    every threshold and rule; see `MemoryConfig`.
        clock:     where "now" comes from. Override it in tests.
    """

    def __init__(self, store: MemoryStore | None = None, extractor: Extractor | None = None,
                 clock: Clock | None = None, *, embedder: Embedder | None = None,
                 config: MemoryConfig | None = None):
        self.store = store or MemoryStore()
        self.extractor = extractor or RuleExtractor()
        self.clock = clock or Clock()
        self.embedder = embedder or HashingEmbedder()
        self.config = config or MemoryConfig()
        # Reconciling reads then writes; the lock keeps two threads from
        # superseding the same memory at once.
        self._lock = threading.RLock()

    # ---- write path ------------------------------------------------------
    def ingest(self, user_id: str, messages: Iterable[Message]) -> list[Decision]:
        """Extract candidates from a conversation and reconcile each one, in order.

        `messages` can be plain strings or (source_ref, text) pairs (the user's messages), or
        the whole conversation as {"role", "content"} chat turns, assistant replies included.
        An extractor that accepts them (like `LLMExtractor`) also sees the assistant's turns,
        to understand replies like "window, always", and the user's stored memories, to
        understand "I left last week" and to reuse the same wording when something changes.
        """
        turns = _turns(messages)
        pairs = [(t["ref"], t["content"]) for t in turns if t["role"] == "user"]
        if not pairs:
            return []
        extras: dict[str, Any] = {}
        takes = _keyword_args(self.extractor.extract)
        if "conversation" in takes:
            extras["conversation"] = turns
        if "known" in takes and self.config.extraction_context_memories > 0:
            extras["known"] = self._known(user_id, pairs)
        candidates = self.extractor.extract(pairs, **extras)  # may call an LLM: outside the lock
        return self.add(user_id, candidates)

    def _known(self, user_id: str, pairs: list[tuple[str, str]]) -> list[Memory]:
        """Stored memories to show the extractor: most relevant first, never sensitive ones."""
        memories = [m for m in self.store.current(user_id, self.clock.now()) if not m.sensitive]
        limit = self.config.extraction_context_memories
        if len(memories) > limit:
            q = self.embedder.embed(" ".join(text for _, text in pairs)[:4000])
            memories.sort(key=lambda m: cosine(q, m.embedding), reverse=True)
            memories = memories[:limit]
        return memories

    def add(self, user_id: str, candidates: Candidate | Iterable[Candidate]) -> list[Decision]:
        """Reconcile candidates you built yourself (a settings form, an import, your own extractor).

        They go through the same write gate and ADD/CONFIRM/SUPERSEDE/RETRACT
        rules as extracted ones.
        """
        if isinstance(candidates, Candidate):
            candidates = [candidates]
        with self._lock:
            now = self.clock.now()
            decisions = []
            for c in candidates:
                d = reconciler.decide(c, self.store, user_id, now, self.config, self.embedder)
                decisions.append(reconciler.apply(d, self.store, user_id, now, self.embedder))
            return decisions

    # ---- read path -------------------------------------------------------
    def recall(self, user_id: str, query: str, scope: str = "general",
               record_usage: bool = True) -> Recall:
        """Find what's relevant to `query` and decide USE / MENTION / VERIFY / HOLD for each.

        `scope` is the context of this conversation ("general", "work", "personal",
        or your own). Memories scoped to a different context are held back.
        """
        with self._lock:
            now = self.clock.now()
            prof = retriever.profile(self.store, user_id, now, self.config)
            hits = retriever.retrieve(self.store, user_id, query, now, self.config,
                                      self.embedder, exclude={m.id for m in prof})
            result = Recall(query, scope, prof,
                            [surfacer.decide(s, scope, now, self.config) for s in hits])

            # Using a memory does NOT confirm it is still true; only the user
            # restating it moves last_confirmed.
            if record_usage:
                for m in prof + [d.scored.memory for d in result.used]:
                    m.last_used, m.use_count = now, m.use_count + 1
                    self.store.save(m)
            return result

    def build_context(self, recall: Recall, include_guidelines: bool = True) -> str:
        """Render the memory section for your system prompt. HOLD items are simply absent.

        Returns an empty string when there is nothing to say, so you can always
        append it: `system = MY_PROMPT + "\\n\\n" + mm.build_context(r)`.
        """
        now = self.clock.now()
        lines: list[str] = []

        if recall.profile:
            lines += ["## About the user (always loaded)"]
            for m in recall.profile:
                tag = " - follow this" if m.kind == "instruction" else ""
                lines.append(f"- {m.text}{tag}")

        if recall.used:
            lines += ["", "## Memories relevant to this message"] if lines else \
                     ["## Memories relevant to this message"]
            for d in recall.used:
                m = d.scored.memory
                since = f"known since {m.valid_from:%Y-%m-%d}"
                if d.action == "USE":
                    how = "use naturally; don't announce that you remember it"
                    if m.sensitive:
                        how = "sensitive: apply it quietly, don't repeat it back"
                elif d.action == "MENTION":
                    how = "worth mentioning explicitly; it should change your answer"
                else:  # VERIFY
                    age = days_between(m.last_confirmed, now)
                    how = (f"relevant, but may be out of date (last confirmed {age:.0f} days "
                           "ago): in your reply, briefly ask whether it's still true")
                lines.append(f"- {m.text} ({since}) -> {how}")

        if lines and include_guidelines:
            lines += [
                "",
                "## How to use memory",
                "- Memories are background, not a script. Answer the user's actual message.",
                "- If the user contradicts a memory, believe the user.",
                "- Never invent memories that are not listed here.",
            ]
        return "\n".join(lines)

    # ---- upkeep ----------------------------------------------------------
    def maintain(self, user_id: str) -> list[str]:
        """Expire, merge and prune one user's memories. Returns a line per change."""
        with self._lock:
            return maintenance.run(self.store, user_id, self.clock.now(), self.config)

    def maintain_all(self) -> dict[str, list[str]]:
        """Run maintenance for every user. Call this from a nightly job."""
        return {u: report for u in self.store.user_ids() if (report := self.maintain(u))}

    def reembed(self, user_id: str | None = None) -> int:
        """Recompute stored embeddings with the current embedder (after switching models)."""
        with self._lock:
            users = [user_id] if user_id else self.store.user_ids()
            count = 0
            embed_many = getattr(self.embedder, "embed_many", None)  # one request per batch
            for u in users:
                memories = self.store.all(u)
                texts = [m.text for m in memories]
                vectors = embed_many(texts) if embed_many else [self.embedder.embed(t) for t in texts]
                for m, vec in zip(memories, vectors):
                    m.embedding = vec
                    self.store.save(m)
                    count += 1
            return count

    # ---- user control ----------------------------------------------------
    def forget(self, user_id: str, memory_id: str) -> list[str]:
        """Hard-delete one memory (and its older versions), and scrub it from the log."""
        with self._lock:
            m = self.store.get(memory_id)
            if m is None or m.user_id != user_id:
                return []
            removed = self.store.delete(memory_id)
            self.store.redact_log(removed)
            self.store.log(user_id, self.clock.now(), "FORGET", memory_id,
                           f"user deleted their {m.predicate} memory (removed {', '.join(removed)})")
            return removed

    def forget_all(self, user_id: str) -> int:
        """Delete everything stored about a user, including their audit log ("delete my data")."""
        with self._lock:
            return self.store.delete_user(user_id)

    def current(self, user_id: str) -> list[Memory]:
        """What is true now: the memories a user would see under "what do you know about me?"."""
        return self.store.current(user_id, self.clock.now())

    def history(self, user_id: str) -> list[Memory]:
        """Everything ever stored, including replaced, retracted and expired memories."""
        return self.store.all(user_id)


def _turns(messages: Iterable[Message]) -> list[dict[str, str]]:
    """Normalize every accepted message shape into ordered {"role", "content", "ref"} turns."""
    turns = []
    for i, m in enumerate(messages, 1):
        if isinstance(m, dict):
            role, content, ref = m.get("role"), m.get("content"), m.get("ref")
            if isinstance(content, list):  # content parts, as some chat APIs use
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            if role not in ("user", "assistant") or not isinstance(content, str) or not content:
                continue  # system prompts, tool calls and empty turns aren't conversation
        elif isinstance(m, tuple):
            role, (ref, content) = "user", m
        else:
            role, content, ref = "user", m, None
        turn = {"role": role, "content": content}
        if role == "user":
            turn["ref"] = str(ref) if ref else str(i)
        turns.append(turn)
    return turns


def _keyword_args(fn: Any) -> set[str]:
    """Which optional keyword arguments an extractor's `extract` accepts."""
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()
