"""Step 2 of the write path: decide what each candidate does to the memory store.

This is the step most memory systems skip, and it's why their memory turns into
a pile of contradictory notes. Every candidate is compared with what is already
stored and gets exactly one operation:

    ADD        new information
    CONFIRM    we already knew this; refresh last_confirmed (it's still true)
    SUPERSEDE  contradicts a current memory; close the old one, add the new one
    RETRACT    the user says something stopped being true; close it, add nothing
    FORGET     the user asked us to forget it; delete it and its history, as forget() does
    IGNORE     not worth keeping (the write gate), with a reason

The decision is rule-based, using the predicate: some predicates can only have
one value at a time (you live in one city; see `MemoryConfig.single_valued`),
others can have many (you can like tea *and* hiking).

Rules can't see links between different predicates: "Went vegetarian" doesn't
match "Loves steak". For those, `MemoryManager(judge=...)` adds an optional judge
(`LLMJudge` asks any model). It is only asked where the rules can't tell, and its
answer is checked before it is used:

    asked when    the rules would ADD, CONFIRM a memory in new words, SUPERSEDE, or
                  find nothing to retract, AND similar memories are stored
    never asked   for forget requests, anything the write gate refuses (secrets...),
                  sensitive details or safety memories (allergies)
    a rule's SUPERSEDE stands (one city at a time): the judge may only add other
                  memories the change makes out of date ("Went vegan" closes "Likes cheese").
                  The exception is a near-duplicate with new numbers ("EU 42" -> "EU 43"),
                  which the rules replace but the judge may keep as two (ADD)
    never shown   sensitive or safety memories, so it can't close them
    it answers    ADD (keep the rule's answer), UPDATE (SUPERSEDE the memories it
                  names), NONE (CONFIRM one), DELETE (RETRACT, for "no longer true")
    checked       it may only name memories it was shown that are still current;
                  anything else falls back to the rules

A judge never deletes: replaced memories are closed and kept as history, like
every other SUPERSEDE and RETRACT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Protocol

from .config import DEFAULT_CONFIG, MemoryConfig
from .embed import Embedder, HashingEmbedder, cosine
from .models import Candidate, Memory
from .store import MemoryStore

_DEFAULT_EMBEDDER = HashingEmbedder()


@dataclass
class Decision:
    op: str
    candidate: Candidate
    target: Memory | None = None   # the existing memory this acts on
    result: Memory | None = None   # the memory written (filled in by apply())
    reason: str = ""
    # More memories this SUPERSEDE or RETRACT closes, when a judge said the new memory makes
    # several untrue ("Went vegan" closes "Loves steak" and "Likes cheese").
    also: list[Memory] = field(default_factory=list)


def _target(d: Decision) -> Memory:
    if d.target is None:
        raise ValueError(f"{d.op} needs the existing memory it acts on")
    return d.target


def _same(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_NEW_NUMBERS = "the same thing with new numbers"


def _changed_numbers(old: Memory, c: Candidate) -> str:
    """"42 -> 43" when both mention numbers and they differ, else "". One side without numbers
    ("Runs every morning" for "Runs 5k every morning") is a restatement, not a change."""
    before = sorted(set(_NUMBER.findall(f"{old.value} {old.text}")))
    after = sorted(set(_NUMBER.findall(f"{c.value} {c.text}")))
    if before and after and before != after:
        return f"{', '.join(before)} -> {', '.join(after)}"
    return ""


# ---- secrets: never stored, whatever the model proposes and even if the user asks -----------
# Memory is the wrong place for credentials, and a model told "remember my card for next time"
# may comply. These patterns look for the secret itself, not the topic: "Never commit API keys"
# and "Prefers a password manager" are fine to keep, "password is Sunflower!2024" is not.
_CARD = re.compile(r"(?<![\d-])[3-6](?:[ -]?\d){12,18}(?![\d-])")
_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|"
                    r"xox[abposr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{30,})")
_SECRET_STATED = re.compile(
    r"\b(?:pass(?:word|code|phrase)|cvv2?|cvc|security code|api key|secret key|access token|"
    r"one-time code)s?\s*(?:is|was|=|:)\s*\S{3,}"
    r"|\b(?:pin|cvv2?|cvc|security code)s?\s*(?:code|number)?\s*(?:is|was|=|:)?\s*#?\d{3,}", re.I)
_SECRET_PREDICATE = re.compile(
    r"(?:^|_)(?:password|passcode|passphrase|pin|cvv|cvc|security_code|api_?key|secret_key|"
    r"access_token|token|iban|ssn|account_number|card_number)(?:$|_)", re.I)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def secret_kind(c: Candidate) -> str | None:
    """What kind of secret this candidate contains ("a card number", ...), or None."""
    text = f"{c.value}\n{c.text}"
    for m in _CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            return "a card number"
    if _TOKEN.search(text):
        return "an API key or token"
    if _SECRET_STATED.search(text) or _SECRET_PREDICATE.search(c.predicate):
        return "a password, PIN or security code"
    if _IBAN.search(text):
        return "a bank account number"
    if _SSN.search(text):
        return "a government ID number"
    return None


_REDACTED = "[secret removed]"


def redact_secrets(text: str) -> str:
    """The same text with every secret `secret_kind` looks for replaced, for kept messages."""
    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group())
        return _REDACTED if 13 <= len(digits) <= 19 and _luhn(digits) else m.group()
    text = _CARD.sub(card, text)
    for pattern in (_TOKEN, _SECRET_STATED, _IBAN, _SSN):
        text = pattern.sub(_REDACTED, text)
    return text


def write_gate(c: Candidate, config: MemoryConfig = DEFAULT_CONFIG) -> str | None:
    """Return a reason to NOT store this candidate, or None to keep it."""
    if config.block_secrets:
        kind = secret_kind(c)
        if kind:
            return f"looks like {kind}; secrets are never stored, even when asked"
    if config.require_consent_for_sensitive and c.sensitive and not c.explicit:
        return "sensitive detail mentioned in passing; needs explicit consent ('remember that ...')"
    if c.source_kind == "inferred" and c.confidence < config.min_inferred_confidence:
        return f"low-confidence inference ({c.confidence:.2f})"
    if c.importance < config.min_importance and not c.explicit:
        return "low importance"
    return None


# ---- the optional judge -------------------------------------------------------------------

JUDGE_OPS = ("ADD", "UPDATE", "NONE", "DELETE")


@dataclass
class Verdict:
    """A judge's answer for one candidate. It is checked before anything is done with it."""

    op: str                                        # ADD | UPDATE | NONE | DELETE
    ids: list[str] = field(default_factory=list)   # the stored memories it acts on, closest first
    reason: str = ""


@dataclass
class JudgeRequest:
    """One candidate the rules can't settle, with the stored memories the judge may act on."""

    candidate: Candidate
    related: list[Memory]


class Judge(Protocol):
    """Anything with this method can be a judge; `LLMJudge` asks a model.

    It gets every request from one `add()`/`ingest()` at once, so it can answer them in one
    call, and returns one Verdict (or None: let the rules decide) per request, in order.
    """

    def judge(self, requests: list[JudgeRequest]) -> list[Verdict | None]: ...


def showable(m: Memory, config: MemoryConfig = DEFAULT_CONFIG) -> bool:
    """Whether a judge may see (and so act on) this memory. Never sensitive or safety memories:
    they aren't sent to a model they don't need to reach, and a judge can't close them."""
    return not m.sensitive and m.predicate not in config.safety_predicates


def needs_judge(c: Candidate, rule: Decision, config: MemoryConfig = DEFAULT_CONFIG) -> bool:
    """Whether the rules' answer for this candidate is one a judge may improve on.

    Never for forget requests, sensitive details or safety memories: a judge that wrongly
    answered "already known" for an allergy would lose it.
    """
    if c.forget or c.sensitive or c.predicate in config.safety_predicates:
        return False
    if c.negated:
        return rule.op == "IGNORE"   # for "no longer true", only "nothing stored to retract"
    if rule.op in ("ADD", "SUPERSEDE"):
        return True
    # Same value in new words ("Runs 10k every morning" for "Runs 5k every morning").
    return (rule.op == "CONFIRM" and rule.target is not None and showable(rule.target, config)
            and not _same(rule.target.text, c.text))


def judge_requests(candidates: Iterable[Candidate], store: MemoryStore, user_id: str, now: datetime,
                   config: MemoryConfig = DEFAULT_CONFIG,
                   embedder: Embedder = _DEFAULT_EMBEDDER) -> list[tuple[int, JudgeRequest]]:
    """The candidates a judge should see (by position), each with its most similar memories.

    Read-only, so it can run before taking the manager's lock (a model call is slow). The
    memories may change before the verdicts are applied; `decide` checks them again then.
    """
    out = []
    pool: list[Memory] | None = None
    for i, c in enumerate(candidates):
        rule = decide(c, store, user_id, now, config, embedder)
        if not needs_judge(c, rule, config):
            continue
        if pool is None:
            pool = [m for m in store.current(user_id, now) if showable(m, config)]
        vec = embedder.embed(c.text)
        scored = sorted(((cosine(vec, m.embedding), m) for m in pool if m.subject == c.subject),
                        key=lambda pair: pair[0], reverse=True)
        related = [m for s, m in scored if s >= config.judge_min_similarity][: config.judge_related]
        # The memory the rules matched is shown first, so the judge can confirm or update it.
        if rule.target is not None and showable(rule.target, config):
            related = [rule.target] + [m for m in related if m.id != rule.target.id]
            related = related[: max(1, config.judge_related)]
        if related:
            out.append((i, JudgeRequest(c, related)))
    return out


def _judged(verdict: Verdict, shown: Iterable[str], c: Candidate, store: MemoryStore,
            user_id: str, now: datetime, config: MemoryConfig) -> list[Memory]:
    """The memories a verdict names that it may act on: shown to the judge, and still current."""
    allowed = set(shown)
    out: list[Memory] = []
    for memory_id in verdict.ids:
        m = store.get(memory_id) if memory_id in allowed else None
        if (m is not None and m.user_id == user_id and m.subject == c.subject and m.is_current(now)
                and showable(m, config) and all(o.id != m.id for o in out)):
            out.append(m)
    return out


def _model_says(verdict: Verdict, fallback: str) -> str:
    return f"model: {verdict.reason or fallback}"


def _with_verdict(rule: Decision, verdict: Verdict, shown: Iterable[str], store: MemoryStore,
                  user_id: str, now: datetime, config: MemoryConfig) -> Decision:
    c = rule.candidate
    targets = _judged(verdict, shown, c, store, user_id, now, config)
    replaces = verdict.op in ("UPDATE", "DELETE")

    if c.negated:   # the rules found nothing to retract
        if replaces and targets:
            return Decision("RETRACT", c, target=targets[0], also=targets[1:],
                            reason=_model_says(verdict, "no longer true"))
        return rule

    if rule.op == "SUPERSEDE":   # the rules' replacement stands; the judge may close more
        # Except a replacement guessed from numbers alone: "Has a 7-year-old" and "Has a 9-year-old"
        # may be two children, and only the judge can tell.
        if verdict.op == "ADD" and rule.reason.startswith(_NEW_NUMBERS):
            return Decision("ADD", c, reason=_model_says(verdict, "both are true"))
        if replaces and targets:
            more = [t for t in targets if t.id != _target(rule).id]
            if more:
                return Decision("SUPERSEDE", c, target=rule.target, also=more,
                                reason=f"{rule.reason}; {_model_says(verdict, 'also out of date')}")
        return rule

    if rule.op == "CONFIRM":   # same value in new words: confirm it, or let the new wording replace it
        matched = _target(rule)
        if replaces and any(t.id == matched.id for t in targets):
            return Decision("SUPERSEDE", c, target=matched,
                            also=[t for t in targets if t.id != matched.id],
                            reason=_model_says(verdict, "replaces it"))
        return rule

    # The rules would ADD.
    if replaces and targets:
        return Decision("SUPERSEDE", c, target=targets[0], also=targets[1:],
                        reason=_model_says(verdict, "replaces it"))
    if verdict.op == "NONE" and targets:
        return Decision("CONFIRM", c, target=targets[0], reason=_model_says(verdict, "already known"))
    return rule


def decide(c: Candidate, store: MemoryStore, user_id: str, now: datetime,
           config: MemoryConfig = DEFAULT_CONFIG,
           embedder: Embedder = _DEFAULT_EMBEDDER, *,
           verdict: Verdict | None = None, shown: Iterable[str] = ()) -> Decision:
    """Pick one operation for a candidate. `verdict` is a judge's answer, and `shown` the ids
    of the memories it was shown; it is only used where the rules allow (see the module doc)."""
    rule = _by_rules(c, store, user_id, now, config, embedder)
    if verdict is not None and needs_judge(c, rule, config):
        return _with_verdict(rule, verdict, shown, store, user_id, now, config)
    return rule


def _by_rules(c: Candidate, store: MemoryStore, user_id: str, now: datetime,
              config: MemoryConfig, embedder: Embedder) -> Decision:
    # 0. "Forget that ..." -> delete it. This includes sensitive memories: deleting what the
    #    user asks us to delete is always allowed. Prefer the same predicate, then any value match.
    if c.forget:
        matches = [m for m in store.all(user_id) if m.subject == c.subject and _same(m.value, c.value)]
        matches.sort(key=lambda m: (m.predicate != c.predicate, not m.is_current(now)))
        if matches:
            return Decision("FORGET", c, target=matches[0], reason="user asked to forget this")
        return Decision("IGNORE", c, reason="asked to forget something that isn't stored")

    existing = store.current_for(user_id, c.subject, c.predicate, now)

    # 1. "X is no longer true" -> close the matching memory, if there is one.
    if c.negated:
        for old in existing:
            if _same(old.value, c.value):
                return Decision("RETRACT", c, target=old, reason="user said this is no longer true")
        # A model may file the "no longer" under another predicate than the original
        # ("diet: coffee" for a stored "likes: coffee"), so match the value across predicates.
        # Never for sensitive memories: "I don't eat peanuts anymore" must not delete an allergy.
        for old in store.current(user_id, now):
            if (old.subject == c.subject and not old.sensitive and old.predicate != c.predicate
                    and _same(old.value, c.value)):
                return Decision("RETRACT", c, target=old,
                                reason=f"user said this is no longer true (stored as {old.predicate})")
        return Decision("IGNORE", c, reason="nothing stored to retract")

    # 2. The write gate: is this worth remembering at all?
    reason = write_gate(c, config)
    if reason:
        return Decision("IGNORE", c, reason=reason)

    # A model may give a lasting fact an expiry ("works at Acme" for 365 days). Ignore it.
    if c.ttl_days and c.predicate in config.lasting_predicates:
        c.ttl_days = None

    # 3. One value at a time: same value confirms, a different value supersedes.
    if c.predicate in config.single_valued and existing:
        old = existing[0]
        if _same(old.value, c.value):
            return Decision("CONFIRM", c, target=old, reason="restated; still true")
        return Decision("SUPERSEDE", c, target=old,
                        reason=f"replaces '{old.value}' (one {c.predicate} at a time)")

    # 4. Many values allowed: only a near-duplicate counts as "already known". Embeddings blur
    #    numbers ("Boot size is EU 42" and "... EU 43" score above 0.9), so a near-duplicate with
    #    different numbers is the same thing changing: replace it. Episodes are separate events.
    new_vec = embedder.embed(c.text) if existing else []
    for old in existing:
        if (_same(old.value, c.value)
                or cosine(old.embedding, new_vec) >= config.duplicate_similarity):
            changed = _changed_numbers(old, c)
            if not changed:
                return Decision("CONFIRM", c, target=old, reason="already known")
            if c.kind != "episode" and old.kind != "episode":
                return Decision("SUPERSEDE", c, target=old, reason=f"{_NEW_NUMBERS} ({changed})")

    # 5. "Dislikes X" cancels "likes X".
    opposite = config.opposites.get(c.predicate)
    if opposite:
        for old in store.current_for(user_id, c.subject, opposite, now):
            if _same(old.value, c.value):
                return Decision("SUPERSEDE", c, target=old, reason=f"contradicts '{old.text}'")

    return Decision("ADD", c, reason="new information")


def forget(store: MemoryStore, user_id: str, memory_id: str) -> tuple[list[str], int]:
    """Hard-delete a memory with its older versions, scrub them from the log, and delete the
    kept messages they came from or that name them. Returns the ids removed and the number
    of messages deleted."""
    chain = {m.id: m for m in store.all(user_id)}
    removed = store.delete(memory_id)
    store.redact_log(removed)
    names = {v for v in (chain[i].value.strip().lower() for i in removed if i in chain) if len(v) >= 4}
    doomed = [s.id for s in store.messages(user_id)
              if set(s.memory_ids) & set(removed)
              or any(re.search(rf"\b{re.escape(v)}\b", s.text.lower()) for v in names)]
    return removed, store.delete_messages(doomed)


def apply(d: Decision, store: MemoryStore, user_id: str, now: datetime,
          embedder: Embedder = _DEFAULT_EMBEDDER) -> Decision:
    c = d.candidate

    if d.op == "IGNORE":
        # Don't let the audit log keep what the write gate refused, or what the user wants forgotten.
        what = f"[redacted {c.predicate}]" if c.sensitive or c.forget or secret_kind(c) else repr(c.text)
        store.log(user_id, now, "IGNORE", None, f"{what}: {d.reason}")
        return d

    if d.op == "CONFIRM":
        old = _target(d)
        old.last_confirmed = now
        old.confidence = min(1.0, old.confidence + 0.05)
        if c.explicit:
            old.importance = min(1.0, max(old.importance, c.importance + 0.2))
        if c.ttl_days:
            old.expires_at = now + timedelta(days=c.ttl_days)
        store.save(old)
        store.log(user_id, now, "CONFIRM", old.id, old.text)
        d.result = old
        return d

    if d.op == "FORGET":
        old = _target(d)
        removed, messages = forget(store, user_id, old.id)
        store.log(user_id, now, "FORGET", old.id,
                  f"user asked to forget their {old.predicate} memory (removed {', '.join(removed)}"
                  + (f", and {messages} kept messages)" if messages else ")"))
        return d

    if d.op == "RETRACT":
        for old in [_target(d), *d.also]:
            old.valid_to = now
            store.save(old)
            store.log(user_id, now, "RETRACT", old.id, f"{old.text!r} no longer true")
        return d

    # ADD and SUPERSEDE both write a new memory.
    new = Memory(
        id=store.next_id(),
        user_id=user_id,
        kind=c.kind,
        subject=c.subject,
        predicate=c.predicate,
        value=c.value,
        text=c.text,
        scope=c.scope,
        source_kind=c.source_kind,
        source_ref=c.source_ref,
        # Things the user explicitly asked us to remember are trusted and kept longer.
        confidence=max(c.confidence, 0.95) if c.explicit else c.confidence,
        importance=min(1.0, c.importance + 0.2) if c.explicit else c.importance,
        sensitive=c.sensitive,
        valid_from=now,
        last_confirmed=now,
        expires_at=now + timedelta(days=c.ttl_days) if c.ttl_days else None,
        embedding=embedder.embed(c.text),
    )
    store.insert(new)
    d.result = new

    if d.op == "SUPERSEDE":
        for old in [_target(d), *d.also]:
            old.valid_to = now          # closed, not deleted: history stays queryable
            old.superseded_by = new.id
            store.save(old)
            store.log(user_id, now, "SUPERSEDE", new.id, f"{old.id} {old.text!r} -> {new.text!r}")
    else:
        store.log(user_id, now, "ADD", new.id, new.text)
    return d
