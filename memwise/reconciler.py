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

Here the decision is rule-based, using the predicate: some predicates can only
have one value at a time (you live in one city; see `MemoryConfig.single_valued`),
others can have many (you can like tea *and* hiking). For free-text memories that
don't fit a predicate, a production system can add an LLM judge: "does B
contradict, refine, or repeat A?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

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


def _target(d: Decision) -> Memory:
    if d.target is None:
        raise ValueError(f"{d.op} needs the existing memory it acts on")
    return d.target


def _same(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


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


def decide(c: Candidate, store: MemoryStore, user_id: str, now: datetime,
           config: MemoryConfig = DEFAULT_CONFIG,
           embedder: Embedder = _DEFAULT_EMBEDDER) -> Decision:
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

    # 4. Many values allowed: only a near-duplicate counts as "already known".
    new_vec = embedder.embed(c.text) if existing else []
    for old in existing:
        if (_same(old.value, c.value)
                or cosine(old.embedding, new_vec) >= config.duplicate_similarity):
            return Decision("CONFIRM", c, target=old, reason="already known")

    # 5. "Dislikes X" cancels "likes X".
    opposite = config.opposites.get(c.predicate)
    if opposite:
        for old in store.current_for(user_id, c.subject, opposite, now):
            if _same(old.value, c.value):
                return Decision("SUPERSEDE", c, target=old, reason=f"contradicts '{old.text}'")

    return Decision("ADD", c, reason="new information")


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
        removed = store.delete(old.id)
        store.redact_log(removed)
        store.log(user_id, now, "FORGET", old.id,
                  f"user asked to forget their {old.predicate} memory (removed {', '.join(removed)})")
        return d

    if d.op == "RETRACT":
        old = _target(d)
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
        old = _target(d)
        old.valid_to = now          # closed, not deleted: history stays queryable
        old.superseded_by = new.id
        store.save(old)
        store.log(user_id, now, "SUPERSEDE", new.id, f"{old.id} {old.text!r} -> {new.text!r}")
    else:
        store.log(user_id, now, "ADD", new.id, new.text)
    return d
