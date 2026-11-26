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
`past()` searches that history, only when asked, for questions about the past;
its results are kept apart and labelled as no longer true.

With `MemoryConfig.keep_messages`, `messages()` also searches what the user said, word
for word, by embedding similarity and by keywords.

With a `Rewriter`, each search also runs with the queries it wrote for the request ("Would she
enjoy camping?" -> "I love hiking"): a memory's relevance is its best match over the request and
those queries. Sensitive memories still need to match the request as asked (`Scored.asked`), so
a rewrite never brings up a sensitive memory the request itself didn't.

Safety memories (`MemoryConfig.safety_predicates`, allergies by default) are always
returned, first, however unrelated they look, and don't count towards `top_k`; the
surfacer then uses them in every scope.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Protocol, Sequence

from .clock import days_between
from .config import DEFAULT_CONFIG, MemoryConfig
from .embed import Embedder, HashingEmbedder, cosine
from .models import Memory, SavedMessage
from .store import MemoryStore

_DEFAULT_EMBEDDER = HashingEmbedder()


class Rewriter(Protocol):
    """Anything with this method can rewrite requests for search; `LLMQueryRewriter` asks a model.

    It returns a few search queries for the request (none is fine), not including the request.
    """

    def rewrite(self, query: str) -> list[str]: ...


@dataclass
class Scored:
    memory: Memory
    relevance: float               # embedding cosine: the best over the request and its rewrites
    recency: float
    score: float
    asked: float | None = None     # the cosine with the request as asked; None: same as relevance

    @property
    def asked_relevance(self) -> float:
        return self.relevance if self.asked is None else self.asked


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
             exclude: frozenset[str] | set[str] = frozenset(),
             query_vector: list[float] | None = None,
             also: Sequence[list[float]] = ()) -> list[Scored]:
    """The current memories for `query`, best first: safety memories, then up to `top_k`.
    `also`: vectors of other queries for the same request (a `Rewriter`'s)."""
    q = query_vector if query_vector is not None else embedder.embed(query)
    w = config.weights
    results: list[Scored] = []
    safety: list[Scored] = []   # always returned, however unrelated they look
    for m in store.current(user_id, now):
        if m.id in exclude:
            continue
        asked = cosine(q, m.embedding)
        rel = max([asked] + [cosine(v, m.embedding) for v in also])
        is_safety = m.predicate in config.safety_predicates
        if rel < config.min_relevance and not is_safety:
            continue
        rec = recency(m, now, config.recency_half_life_days)
        score = (w["relevance"] * rel + w["recency"] * rec
                 + w["importance"] * m.importance + w["confidence"] * m.confidence)
        (safety if is_safety else results).append(Scored(m, rel, rec, score, asked))
    results.sort(key=lambda s: s.score, reverse=True)
    safety.sort(key=lambda s: s.score, reverse=True)
    return safety + results[: config.top_k]


def past(store: MemoryStore, user_id: str, query_vector: list[float], now: datetime,
         config: MemoryConfig = DEFAULT_CONFIG, also: Sequence[list[float]] = ()) -> list[Scored]:
    """Memories that are no longer true (replaced, retracted, expired), most relevant first.

    Ranked by relevance alone: how recent or important an old fact was doesn't matter when
    the question is about it. Forgotten memories are deleted, so they never come back here,
    and duplicates that maintenance merged are left out: they're still true, as the copy kept.
    """
    results = []
    for m in store.all(user_id):
        if m.is_current(now) or _merged(store, m):
            continue
        asked = cosine(query_vector, m.embedding)
        rel = max([asked] + [cosine(v, m.embedding) for v in also])
        if rel >= config.min_relevance:
            results.append(Scored(m, rel, 0.0, rel, asked))
    results.sort(key=lambda s: s.score, reverse=True)
    return results[: config.past_top_k]


@dataclass
class ScoredMessage:
    message: SavedMessage
    relevance: float   # embedding cosine with the query (the best over the query and `also`)
    keywords: float    # BM25 over the user's kept messages: shared rare words (names, titles)
    score: float       # the rankings combined (reciprocal rank fusion)


def messages(store: MemoryStore, user_id: str, query: str, query_vector: list[float],
             config: MemoryConfig = DEFAULT_CONFIG, exclude: frozenset[str] | set[str] = frozenset(),
             limit: int | None = None, saved: list[SavedMessage] | None = None,
             also: Sequence[tuple[str, list[float]]] = ()) -> list[ScoredMessage]:
    """Kept messages for `query`, best first (`MemoryConfig.keep_messages`).

    Rankings combined by reciprocal rank fusion: embedding similarity finds messages about
    the same thing in other words, and keywords find the exact name or title ("Becoming
    Nicole") that embeddings blur. When the query names a date or a month ("on 13 October
    2023", "in May 2023"), messages from then are a third ranking, most relevant first. A
    message needs `min_relevance`, a shared keyword or the named date to count at all. Scope
    and sensitivity are left to the caller (see surfacer).

    `also`: other queries for the same request, as (text, vector) (a `Rewriter`'s). Each adds
    its own meaning and keyword rankings.
    """
    saved = [s for s in (store.messages(user_id) if saved is None else saved) if s.id not in exclude]
    if not saved:
        return []
    docs = [_words(s.text) for s in saved]
    n, avg = len(docs), sum(len(d) for d in docs) / len(docs) or 1.0
    queries = [(query, query_vector), *also]
    relevance = [[cosine(v, s.embedding) for s in saved] for _, v in queries]
    keywords = [_bm25(docs, set(_words(text)), n, avg) for text, _ in queries]
    scored = [ScoredMessage(s, max(r[i] for r in relevance), max(k[i] for k in keywords), 0.0)
              for i, s in enumerate(saved)]
    rankings = []
    for rel, kw in zip(relevance, keywords):
        rankings.append(sorted((i for i in range(n) if rel[i] >= config.min_relevance),
                               key=lambda i: rel[i], reverse=True))
        rankings.append(sorted((i for i in range(n) if kw[i] > 0), key=lambda i: kw[i], reverse=True))
    periods = dates_in(query)
    rankings.append(sorted((i for i in range(n) if any(start <= saved[i].at.date() <= end
                                                       for start, end in periods)),
                           key=lambda i: relevance[0][i], reverse=True))
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            scored[i].score += 1 / (60 + rank)
    found = [x for x in scored if x.score > 0]
    found.sort(key=lambda x: x.score, reverse=True)
    return found[: config.message_top_k if limit is None else limit]


def _bm25(docs: list[list[str]], q: set[str], n: int, avg: float) -> list[float]:
    """BM25 (k1 1.2, b 0.75) of each document for the query words."""
    df: dict[str, int] = {}
    for d in docs:
        for w in set(d) & q:
            df[w] = df.get(w, 0) + 1
    out = []
    for d in docs:
        kw = 0.0
        if df:
            counts: dict[str, int] = {}
            for w in d:
                if w in df:
                    counts[w] = counts.get(w, 0) + 1
            for w, tf in counts.items():
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                kw += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(d) / avg))
        out.append(kw)
    return out


_MONTHS = {m: i for i, names in enumerate(
    ["jan january", "feb february", "mar march", "apr april", "may", "jun june", "jul july",
     "aug august", "sep sept september", "oct october", "nov november", "dec december"], 1)
    for m in names.split()}
_MONTH = r"(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_DAY = r"(?P<day>[0-3]?\d)(?:st|nd|rd|th)?"
_YEAR = r"(?P<year>(?:19|20)\d\d)"
_DATES = [re.compile(p, re.I) for p in (
    rf"\b{_DAY}\s+(?:of\s+)?{_MONTH},?\s+{_YEAR}\b",   # 13 October 2023
    rf"\b{_MONTH}\s+{_DAY},?\s+{_YEAR}\b",              # October 13, 2023
    rf"\b{_MONTH},?\s+{_YEAR}\b",                        # October 2023
    r"\b(?P<year>(?:19|20)\d\d)-(?P<month>[01]\d)-(?P<day>[0-3]\d)\b",  # 2023-10-13
)]


def dates_in(text: str) -> list[tuple[date, date]]:
    """The days and months `text` names with a year, as (first day, last day) periods.
    "13 October 2023" is that day, "October 2023" the month. Dates without a year are left
    out: "on 13 October" could be any year."""
    periods: list[tuple[date, date]] = []
    taken: list[tuple[int, int]] = []
    for pattern in _DATES:
        for m in pattern.finditer(text):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue  # "October 2023" inside "13 October 2023"
            month = m.group("month")
            month_n = int(month) if month.isdigit() else _MONTHS[month.lower().rstrip(".")]
            year = int(m.group("year"))
            try:
                if "day" in pattern.groupindex and m.group("day"):
                    day = date(year, month_n, int(m.group("day")))
                    periods.append((day, day))
                else:
                    first = date(year, month_n, 1)
                    last = (date(year + month_n // 12, month_n % 12 + 1, 1) - timedelta(days=1))
                    periods.append((first, last))
            except ValueError:
                continue  # "31 February 2023"
            taken.append((m.start(), m.end()))
    return periods


# Words too common to say what a message is about.
_STOPWORDS = frozenset(
    "a an the and or but if of to in on at for from by with about as is are was were be been being "
    "am do does did have has had i me my mine you your he him his she her it its we us our they "
    "them their this that these those what when where who whom which why how would could should "
    "can will just so not no yes too very really there here then than also some any all".split())


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()) if w not in _STOPWORDS]


_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _merged(store: MemoryStore, m: Memory) -> bool:
    """Replaced by a memory saying the same thing (a merged duplicate), not by a change."""
    newer = store.get(m.superseded_by) if m.superseded_by else None
    return (newer is not None and newer.value.strip().lower() == m.value.strip().lower()
            and _NUMBER.findall(newer.text) == _NUMBER.findall(m.text))
