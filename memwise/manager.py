"""The facade your product talks to.

    ingest(user_id, messages)   after a session (or right away for "remember ...")  -> write path
    recall(user_id, message)    before each model call                              -> read path
    build_context(recall)       turn the recall into text for your system prompt
    maintain_all()              on a schedule (nightly)                             -> upkeep
    forget(user_id, memory_id)  when the user asks                                  -> user control

With `MemoryConfig.keep_messages`, what the user said is also kept word for word (secrets
redacted) and searched at recall, next to the memories.
"""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Union

from . import maintenance, reconciler, retriever, surfacer
from .clock import Clock, days_between
from .config import MemoryConfig
from .dates import with_dates
from .embed import Embedder, HashingEmbedder, cosine
from .extractor import Extractor, RuleExtractor
from .models import Candidate, Memory, SavedMessage
from .reconciler import Decision, Judge, Verdict
from .retriever import Rewriter, ScoredMessage
from .store import MemoryStore
from .surfacer import SurfaceDecision

_log = logging.getLogger("memwise")

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
    # Memories that are no longer true, with PAST or HOLD; empty unless recall(include_past=True).
    past: list[SurfaceDecision] = field(default_factory=list)
    # With MemoryConfig.keep_messages: the message each shown memory came from, by memory id,
    # and other kept messages that match the query, best first. Held-back ones are left out.
    sources: dict[str, SavedMessage] = field(default_factory=dict)
    messages: list[ScoredMessage] = field(default_factory=list)
    # With a rewriter: the queries searched as well as `query`.
    searched: list[str] = field(default_factory=list)

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
        judge:     optional. Asked where the rules can't tell whether a new memory replaces
                   or repeats a stored one ("Went vegetarian" vs "Loves steak"). `LLMJudge`
                   asks any model; see `reconciler` for when it is asked and how its answer
                   is checked. Off by default: then the rules decide everything.
        rewriter:  optional. Writes search queries for each request, so recall also finds
                   what answers it in other words ("Would she enjoy camping?" -> "I love
                   hiking"). `LLMQueryRewriter` asks any model. Off by default.
    """

    def __init__(self, store: MemoryStore | None = None, extractor: Extractor | None = None,
                 clock: Clock | None = None, *, embedder: Embedder | None = None,
                 config: MemoryConfig | None = None, judge: Judge | None = None,
                 rewriter: Rewriter | None = None):
        self.store = store or MemoryStore()
        self.extractor = extractor or RuleExtractor()
        self.clock = clock or Clock()
        self.embedder = embedder or HashingEmbedder()
        self.config = config or MemoryConfig()
        self.judge = judge
        self.rewriter = rewriter
        # Reconciling reads then writes; the lock keeps two threads from
        # superseding the same memory at once.
        self._lock = threading.RLock()

    # ---- write path ------------------------------------------------------
    def ingest(self, user_id: str, messages: Iterable[Message]) -> list[Decision]:
        """Extract candidates from a conversation and reconcile each one, in order.

        `messages` can be plain strings or (source_ref, text) pairs (the user's messages), or
        the whole conversation as {"role", "content"} chat turns, assistant replies included.
        An extractor that accepts them (like `LLMExtractor`) also sees the assistant's turns,
        to understand replies like "window, always", the user's stored memories, to
        understand "I left last week" and to reuse the same wording when something changes,
        and the date (from the clock), to write "yesterday" as a date.
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
        if "now" in takes:
            extras["now"] = self.clock.now()
        candidates = self.extractor.extract(pairs, **extras)  # may call an LLM: outside the lock
        decisions = self.add(user_id, candidates)
        if self.config.keep_messages:
            self._keep_messages(user_id, pairs, decisions)
        return decisions

    def _keep_messages(self, user_id: str, pairs: list[tuple[str, str]], decisions: list[Decision]) -> None:
        """Keep what the user said, word for word, linked to the memories taken from it."""
        by_ref: dict[str, list[Decision]] = {}
        for d in decisions:
            by_ref.setdefault(d.candidate.source_ref, []).append(d)
        keep: list[tuple[str, str, list[Decision]]] = []
        for ref, text in pairs:
            ds = by_ref.get(ref, [])
            if any(d.candidate.forget for d in ds):
                continue  # "forget that I live in Oslo" names what is to be forgotten
            if self.config.require_consent_for_sensitive and any(
                    d.op == "IGNORE" and d.candidate.sensitive and not d.candidate.explicit for d in ds):
                continue  # a sensitive detail the user didn't ask us to keep
            if self.config.block_secrets:
                text = reconciler.redact_secrets(text)
            if text.strip():
                keep.append((ref, text, ds))
        if not keep:
            return
        embed_many = getattr(self.embedder, "embed_many", None)  # outside the lock: may be a network call
        texts = [text for _, text, _ in keep]
        vectors = embed_many(texts) if embed_many else [self.embedder.embed(t) for t in texts]
        with self._lock:
            now = self.clock.now()
            for (ref, text, ds), vector in zip(keep, vectors):
                ids = list(dict.fromkeys(d.result.id for d in ds if d.result is not None))
                if any(self.store.get(i) is None for i in ids):
                    continue  # forgotten in the meantime
                scope = next((d.candidate.scope for d in ds if d.candidate.scope != "general"), "general")
                self.store.insert_message(SavedMessage(
                    id=self.store.next_id("s"), user_id=user_id, ref=ref, text=text, at=now, scope=scope,
                    sensitive=any(d.candidate.sensitive for d in ds), memory_ids=ids, embedding=vector))

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
        rules as extracted ones, and the judge, if there is one.
        """
        if isinstance(candidates, Candidate):
            candidates = [candidates]
        candidates = list(candidates)
        verdicts = self._judge(user_id, candidates)  # may call a model: outside the lock
        with self._lock:
            now = self.clock.now()
            decisions = []
            for i, c in enumerate(candidates):
                verdict, shown = verdicts.get(i, (None, frozenset()))
                d = reconciler.decide(c, self.store, user_id, now, self.config, self.embedder,
                                      verdict=verdict, shown=shown)
                decisions.append(reconciler.apply(d, self.store, user_id, now, self.embedder))
            return decisions

    def _judge(self, user_id: str, candidates: list[Candidate]) -> dict[int, tuple[Verdict, frozenset[str]]]:
        """Ask the judge about the candidates the rules can't settle, in one call.

        This runs before the lock is taken, so a slow model doesn't block other threads. The
        verdicts are checked again under the lock: a memory that changed in the meantime is
        left alone, and the rules decide instead.
        """
        if self.judge is None or not candidates:
            return {}
        requests = reconciler.judge_requests(candidates, self.store, user_id, self.clock.now(),
                                             self.config, self.embedder)
        if not requests:
            return {}
        verdicts = self.judge.judge([r for _, r in requests])
        if len(verdicts) != len(requests):
            _log.warning("memwise: the judge returned %d verdicts for %d requests; the rules decided",
                         len(verdicts), len(requests))
            return {}
        return {i: (v, frozenset(m.id for m in r.related))
                for (i, r), v in zip(requests, verdicts) if v is not None}

    # ---- read path -------------------------------------------------------
    def recall(self, user_id: str, query: str, scope: str = "general",
               record_usage: bool = True, *, include_past: bool = False) -> Recall:
        """Find what's relevant to `query` and decide USE / MENTION / VERIFY / HOLD for each.

        `scope` is the context of this conversation ("general", "work", "personal",
        or your own). Memories scoped to a different context are held back. Safety
        memories (allergies, by default) are always included, however unrelated they look.

        `include_past=True` also finds memories that are no longer true (replaced, retracted
        or expired), for questions about the past ("where did I live before?"). They are kept
        in `Recall.past`, and `build_context` lists them apart, as history.

        With a `rewriter`, recall also searches with the queries it writes (`Recall.searched`).
        """
        searched = self._rewrite(query)  # may call a model: outside the lock
        many = getattr(self.embedder, "embed_many", None)  # one request for all, if it batches
        q, *also = many([query, *searched]) if many else [self.embedder.embed(x) for x in [query, *searched]]
        with self._lock:
            now = self.clock.now()
            prof = retriever.profile(self.store, user_id, now, self.config)
            hits = retriever.retrieve(self.store, user_id, query, now, self.config, self.embedder,
                                      exclude={m.id for m in prof}, query_vector=q, also=also)
            past = ([surfacer.decide_past(s, scope, self.config)
                     for s in retriever.past(self.store, user_id, q, now, self.config, also)]
                    if include_past else [])
            result = Recall(query, scope, prof,
                            [surfacer.decide(s, scope, now, self.config) for s in hits], past,
                            searched=searched)
            if self.config.keep_messages:
                self._recall_messages(result, user_id, q, list(zip(searched, also)))

            # Using a memory does NOT confirm it is still true; only the user
            # restating it moves last_confirmed.
            if record_usage:
                for m in prof + [d.scored.memory for d in result.used]:
                    m.last_used, m.use_count = now, m.use_count + 1
                    self.store.save(m)
            return result

    def _rewrite(self, query: str) -> list[str]:
        if self.rewriter is None:
            return []
        try:
            return [x for x in self.rewriter.rewrite(query) if x.strip()]
        except Exception as e:  # a custom rewriter's failure must not break recall
            _log.warning("memwise: the rewriter failed, so recall searched with the request alone: %s", e)
            return []

    def _recall_messages(self, result: Recall, user_id: str, q: list[float],
                         also: list[tuple[str, list[float]]]) -> None:
        """Fill in the message each shown memory came from, and other kept messages that match."""
        saved = self.store.messages(user_id)
        first: dict[str, SavedMessage] = {}
        for s in saved:  # oldest first: a memory's source is the message it was first taken from
            for i in s.memory_ids:
                first.setdefault(i, s)
        shown = [d.scored.memory for d in result.used] + [d.scored.memory for d in result.past
                                                           if d.action == "PAST"]
        for m in shown:
            s = first.get(m.id)
            if s is not None and not surfacer.hold_message(s, cosine(q, s.embedding), result.scope, self.config):
                result.sources[m.id] = s
        taken = {s.id for s in result.sources.values()}
        for x in retriever.messages(self.store, user_id, result.query, q, self.config,
                                    exclude=taken, saved=saved, limit=len(saved), also=also):
            if len(result.messages) == self.config.message_top_k:
                break
            asked = cosine(q, x.message.embedding)  # sensitive ones must match the request itself
            if not surfacer.hold_message(x.message, asked, result.scope, self.config):
                result.messages.append(x)

    def build_context(self, recall: Recall, include_guidelines: bool = True, *,
                      about: str = "the user") -> str:
        """Render the memory section for your system prompt. HOLD items are simply absent.

        Returns an empty string when there is nothing to say, so you can always
        append it: `system = MY_PROMPT + "\\n\\n" + mm.build_context(r)`. `about` names
        whose memory it is, for memory about something other than a person
        (`about="the project"` for a repository).
        """
        now = self.clock.now()
        lines: list[str] = []

        def heading(title: str) -> None:
            lines.extend(["", title] if lines else [title])

        if recall.profile:
            heading(f"## About {about} (always loaded)")
            for m in recall.profile:
                tag = " - follow this" if m.kind == "instruction" else ""
                lines.append(f"- {m.text}{tag}")

        safety = [d for d in recall.used if d.scored.memory.predicate in self.config.safety_predicates]
        relevant = [d for d in recall.used if d.scored.memory.predicate not in self.config.safety_predicates]

        if safety:
            heading("## Always respect (safety)")
            for d in safety:
                m = d.scored.memory
                lines.append(f"- {m.text} (known since {m.valid_from:%Y-%m-%d}) -> never suggest "
                             "anything that conflicts with this; only bring it up when it matters")

        if relevant:
            heading("## Memories relevant to this message")
            for d in relevant:
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
                self._said(lines, recall, m)

        past = [d for d in recall.past if d.action == "PAST"]
        if past:
            heading("## No longer true (history, for questions about the past)")
            for d in past:
                lines.append(f"- {d.scored.memory.text} ({self._when_true(d.scored.memory)})")
                self._said(lines, recall, d.scored.memory)

        if recall.messages:
            heading(f"## What {about} said before (word for word; may be relevant)")
            for x in sorted(recall.messages, key=lambda x: x.message.at):
                lines.append(f"- {_day(x.message.at)}: \"{with_dates(x.message.text, x.message.at)}\"")

        if lines and include_guidelines:
            lines += [
                "",
                "## How to use memory",
                "- Memories are background, not a script. Answer the user's actual message.",
                "- If the user contradicts a memory, believe the user.",
                "- Never invent memories that are not listed here.",
            ]
            if past:
                lines.append("- Memories that are no longer true are history: never treat them as true now.")
            if recall.messages or recall.sources:
                lines.append("- What was said is dated. Words like \"last week\" are followed by the dates they "
                             "meant on that day, in [brackets].")
        return "\n".join(lines)

    @staticmethod
    def _said(lines: list[str], recall: Recall, m: Memory) -> None:
        s = recall.sources.get(m.id)
        if s is not None:
            lines.append(f"    said on {_day(s.at)}: \"{with_dates(s.text, s.at)}\"")

    def _when_true(self, m: Memory) -> str:
        """"from 2026-01-01 until 2026-03-01, replaced by: Lives in Lisbon" for a past memory."""
        ended = m.valid_to or m.expires_at
        text = f"from {m.valid_from:%Y-%m-%d}" + (f" until {ended:%Y-%m-%d}" if ended else "")
        newer = self.store.get(m.superseded_by) if m.superseded_by else None
        if newer is not None:
            return f"{text}, replaced by: {newer.text}"
        if m.expires_at is not None and ended == m.expires_at:
            return f"{text}, expired"  # maintenance closes expired memories at their expiry date
        return f"{text}, then no longer true"

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
                saved = self.store.messages(u)
                texts = [s.text for s in saved]
                vectors = embed_many(texts) if embed_many and texts else [self.embedder.embed(t) for t in texts]
                for s, vec in zip(saved, vectors):
                    s.embedding = vec
                    self.store.save_message(s)
            return count

    # ---- user control ----------------------------------------------------
    def forget(self, user_id: str, memory_id: str) -> list[str]:
        """Hard-delete one memory (and its older versions), and scrub it from the log.

        Kept messages (`MemoryConfig.keep_messages`) it came from, or that name it, are
        deleted too."""
        with self._lock:
            m = self.store.get(memory_id)
            if m is None or m.user_id != user_id:
                return []
            removed, messages = reconciler.forget(self.store, user_id, memory_id)
            self.store.log(user_id, self.clock.now(), "FORGET", memory_id,
                           f"user deleted their {m.predicate} memory (removed {', '.join(removed)}"
                           + (f", and {messages} kept messages)" if messages else ")"))
            return removed

    def forget_all(self, user_id: str) -> int:
        """Delete everything stored about a user, including their audit log and kept messages
        ("delete my data")."""
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


def _day(when) -> str:
    """"Thursday 2023-05-25": the weekday helps a model work out "last Sunday"."""
    return f"{when:%A %Y-%m-%d}"


def _keyword_args(fn: Any) -> set[str]:
    """Which optional keyword arguments an extractor's `extract` accepts."""
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()
