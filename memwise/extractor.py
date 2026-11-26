"""Step 1 of the write path: pull *candidate* memories out of what the user said.

`RuleExtractor` uses regular expressions so the project runs offline and you
can read exactly why something was extracted. It only understands the phrasings
in PATTERNS below. `memwise.llm.LLMExtractor` does the same job with any LLM
and handles any phrasing; both return the same `Candidate` objects, so the rest
of the pipeline doesn't care which one you use.

Extraction is deliberately *not* the decision to store. It produces candidates;
the reconciler decides what is worth keeping and how it fits with what's known.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Protocol

from .models import Candidate


class Extractor(Protocol):
    def extract(self, messages: Iterable[tuple[str, str]]) -> list[Candidate]:
        """messages: (source_ref, text) pairs from one session, user turns only.

        An extractor can also accept either of these keyword arguments; the manager passes
        them only to extractors that do (see `LLMExtractor`):
          conversation: every turn in order, {"role", "content"} dicts with a "ref" on user
                        turns. Assistant turns are context only: memories come from the user.
          known:        the user's current, non-sensitive memories, for reference.
          now:          when the conversation happened (the manager's clock), to turn
                        "yesterday" or "next month" into a date.
        """


_END = r"(?=[,.!?;]|$)"  # a value runs until punctuation or end of sentence
_TRAVEL_TTL = {"tomorrow": 3, "this week": 7, "next week": 14, "next month": 45}


def _clean(value: str) -> str:
    value = value.strip().strip(".,!?;:")
    value = re.sub(r"\s+(?:anymore|any more|too|a lot|now|by the way)$", "", value, flags=re.I)
    return value.strip()


def _title(value: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in value.split())


Builder = Callable[[re.Match], "Candidate | None"]


def _travel(m: re.Match) -> Candidate:
    when = (m.group(2) or "").lower()
    place = _title(m.group(1))
    return Candidate("fact", "traveling_to", place,
                     f"Traveling to {place}" + (f" ({when})" if when else ""),
                     importance=0.5, ttl_days=_TRAVEL_TTL.get(when, 14))


def _likes(m: re.Match) -> Candidate | None:
    value = _clean(m.group(1)).lower()
    if re.search(r"\b(answers?|responses?|replies)\b", value):
        return None  # that's a response-style instruction, handled by its own pattern
    return Candidate("preference", "likes", value, f"Likes {value}", importance=0.4)


PATTERNS: list[tuple[str, Builder]] = [
    # --- identity & situation ------------------------------------------------
    (r"\bmy name is ([A-Za-z][\w-]*)",
     lambda m: Candidate("fact", "name", _title(m.group(1)), f"Name is {_title(m.group(1))}",
                         importance=0.9)),
    (r"\bi (?:live in|am based in|'m based in)\s+([A-Za-z][\w-]*)",
     lambda m: Candidate("fact", "lives_in", _title(m.group(1)), f"Lives in {_title(m.group(1))}",
                         importance=0.7)),
    (r"\bi(?:'ve| have)? (?:just |recently )?moved to\s+([A-Za-z][\w-]*)",
     lambda m: Candidate("fact", "lives_in", _title(m.group(1)), f"Lives in {_title(m.group(1))}",
                         importance=0.7)),
    (r"\bi work (?:at|for)\s+([A-Za-z][\w&.-]*)",
     lambda m: Candidate("fact", "employer", _title(_clean(m.group(1))),
                         f"Works at {_title(_clean(m.group(1)))}", scope="work", importance=0.7)),
    (r"\bi work (?:at|for) [\w&.-]+ as an? ([\w -]+?)" + _END,
     lambda m: Candidate("fact", "job_title", _clean(m.group(1)).lower(),
                         f"Works as a {_clean(m.group(1)).lower()}", scope="work", importance=0.6)),
    (r"\bi no longer work (?:at|for)\s+([A-Za-z][\w&.-]*)",
     lambda m: Candidate("fact", "employer", _title(_clean(m.group(1))),
                         f"Works at {_title(_clean(m.group(1)))}", scope="work", negated=True)),
    (r"\bi(?:'m| am) (?:currently )?working on (?:the |a |an )?([\w -]+?)" + _END,
     lambda m: Candidate("fact", "current_project", _clean(m.group(1)),
                         f"Working on {_clean(m.group(1))}", scope="work", importance=0.5,
                         ttl_days=45)),
    (r"\bi(?:'m| am) (?:traveling|travelling|flying|heading) to ([A-Za-z][\w-]*)"
     r"(?:[^.!?]*?\b(tomorrow|this week|next week|next month))?", _travel),

    # --- how the user wants to be helped -------------------------------------
    (r"\bi (?:prefer|like|want)\s+((?:short|brief|concise|detailed|long)[\w ,-]*?"
     r"(?:answers|responses|replies)[\w ,-]*?)" + _END,
     lambda m: Candidate("instruction", "response_style", _clean(m.group(1)).lower(),
                         f"Prefers {_clean(m.group(1)).lower()}", importance=0.8)),

    # --- preferences -------------------------------------------------------
    # Negations first, so "stopped liking X" is handled before "started liking Y".
    (r"\bi (?:don't|do not|no longer) (?:like|love|drink|eat|enjoy|use)\s+([\w '-]+?)"
     r"(?:\s+any ?more)?" + _END,
     lambda m: Candidate("preference", "likes", _clean(m.group(1)).lower(),
                         f"Likes {_clean(m.group(1)).lower()}", negated=True)),
    (r"\bi (?:really )?(?:love|like|enjoy|prefer)\s+([\w '-]+?)" + _END, _likes),
    (r"\bswitched to\s+([\w '-]+?)" + _END, _likes),
    (r"\bi (?:hate|dislike|can't stand)\s+([\w '-]+?)" + _END,
     lambda m: Candidate("preference", "dislikes", _clean(m.group(1)).lower(),
                         f"Dislikes {_clean(m.group(1)).lower()}", importance=0.4)),

    # --- sensitive ---------------------------------------------------------
    (r"\bi(?:'m| am) allergic to\s+([\w '-]+?)" + _END,
     lambda m: Candidate("fact", "allergy", _clean(m.group(1)).lower(),
                         f"Allergic to {_clean(m.group(1)).lower()}", importance=0.9,
                         sensitive=True)),
    (r"\bi (?:was|got|have been) diagnosed with\s+([\w '-]+?)(?:\s+(?:last|this) \w+)?" + _END,
     lambda m: Candidate("fact", "health", _clean(m.group(1)).lower(),
                         f"Has {_clean(m.group(1)).lower()}", importance=0.8, sensitive=True)),

    # --- episodes: things that happened and may matter later ----------------
    (r"\b(?:i|we) (?:tried|used|decided to use|chose)\s+(.+?)" + _END,
     lambda m: Candidate("episode", "experience", _clean(m.group(1)),
                         f"Tried {_clean(m.group(1))}", scope="work", importance=0.5)),
]

_COMPILED = [(re.compile(p, re.I), build) for p, build in PATTERNS]
_REMEMBER = re.compile(r"^(?:please\s+)?(?:remember|note|don't forget)\s+(?:that\s+)?(.+)", re.I)
_FORGET = re.compile(r"^(?:please\s+)?(?:forget|delete|erase)\s+(?:that\s+|about\s+)?(.+)", re.I)


def _sentences(text: str) -> list[str]:
    text = text.replace("’", "'")
    # Split on sentence ends and on "..., and I ..." so each clause is matched alone.
    parts = re.split(r"(?<=[.!?])\s+|\s+and\s+(?=i\b)|,\s*(?=i\b)", text, flags=re.I)
    return [p.strip() for p in parts if p and p.strip()]


class RuleExtractor:
    def extract(self, messages: Iterable[tuple[str, str]]) -> list[Candidate]:
        out: list[Candidate] = []
        for ref, text in messages:
            for sentence in _sentences(text):
                explicit = forget = False
                remember = _REMEMBER.match(sentence)
                if remember:
                    explicit, sentence = True, remember.group(1)
                elif m := _FORGET.match(sentence):
                    # "Please forget that I live in Oslo": find the claim, then mark it for deletion.
                    forget, sentence = True, m.group(1)
                found = self._match(sentence)
                if forget:
                    for c in found:  # the claim to delete, not a new memory
                        c.forget, c.negated = True, False
                elif explicit and not found:
                    # "Remember that the wifi password is on the fridge" - no pattern
                    # for it, but the user asked, so keep it as a plain note.
                    note = _clean(sentence)
                    found = [Candidate("fact", "note", note, note, importance=0.6)]
                for c in found:
                    c.explicit = explicit
                    c.source_ref = ref
                    out.append(c)
        return out

    @staticmethod
    def _match(sentence: str) -> list[Candidate]:
        found: list[Candidate] = []
        seen: set[tuple[str, str, bool]] = set()
        for regex, build in _COMPILED:
            for m in regex.finditer(sentence):
                c = build(m)
                if c is None:
                    continue
                key = (c.predicate, c.value.lower(), c.negated)
                if key not in seen:
                    seen.add(key)
                    found.append(c)
        # A negated statement wins over a positive match of the same claim
        # ("I don't like coffee" must not also produce "likes coffee").
        negated = {(c.predicate, c.value.lower()) for c in found if c.negated}
        return [c for c in found if c.negated or (c.predicate, c.value.lower()) not in negated]
