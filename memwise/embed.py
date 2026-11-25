"""Embeddings: the `Embedder` interface, plus a tiny built-in one.

`HashingEmbedder` is a dependency-free stand-in so the library works out of the box:

  * words are hashed into a fixed-size vector (bag of words), and
  * a small hand-written list of "concepts" links related words, so that
    "snack" and "peanuts" end up close even though they share no letters.

It is fine for demos and tests, not for production. Use a real embedding model:
`memwise.llm.OpenAICompatibleEmbedder` works with any `/embeddings` API, or pass
any object with an `embed(text) -> list[float]` method:

    class MyEmbedder:
        def embed(self, text: str) -> list[float]:
            return my_embedding_api(text)          # any provider, or a local model

    mm = MemoryManager(store, embedder=MyEmbedder(), config=MemoryConfig.for_embedding_model())

Vectors should be L2-normalized (cosine similarity is computed as a dot product).
If you switch embedders on an existing database, call `mm.reembed(user_id)`.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        """Return an L2-normalized vector for `text`."""


class HashingEmbedder:
    """The built-in, dependency-free embedder (see module docstring)."""

    def embed(self, text: str) -> list[float]:
        return embed(text)


DIM = 512

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "in", "on", "for", "with", "at", "by",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "my", "me", "i",
    "you", "your", "we", "our", "can", "could", "should", "would", "do", "does", "any", "some",
    "what", "how", "about", "as", "so", "if", "just", "really", "please", "help", "ideas",
}

CONCEPTS: dict[str, set[str]] = {
    "food": {"food", "eat", "eating", "snack", "snacks", "recipe", "recipes", "dinner", "lunch",
             "breakfast", "cook", "cooking", "restaurant", "meal", "meals", "peanut", "peanuts",
             "allergic", "allergy", "vegetarian", "vegan", "bake", "dessert"},
    "drink": {"coffee", "tea", "drink", "drinks", "caffeine", "latte", "espresso", "cafe"},
    "place": {"live", "lives", "living", "city", "move", "moved", "moving", "berlin", "lisbon",
              "home", "apartment", "neighborhood", "neighbourhood", "weather", "local", "near",
              "nearby", "place", "area"},
    "work": {"work", "works", "working", "job", "company", "employer", "team", "manager",
             "office", "engineer", "career", "startup", "colleague", "colleagues", "performance",
             "review", "promotion", "salary", "acme"},
    "travel": {"travel", "traveling", "travelling", "trip", "flight", "flights", "flying",
               "hotel", "visit", "vacation", "holiday", "tokyo", "airport", "packing"},
    "code": {"code", "python", "typescript", "javascript", "bug", "function", "programming",
             "script", "api", "debug", "redis", "cache", "caching", "database", "session",
             "sessions", "backend", "server", "memory"},
    "health": {"diagnosed", "doctor", "medication", "health", "allergic", "allergy", "therapy",
               "sick", "symptoms"},
    "relax": {"relax", "relaxing", "weekend", "stress", "stressed", "rest", "hobby", "unwind"},
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


# Concepts get their own dimensions at the start of the vector; words are hashed
# into the rest. Keeping them apart stops a word like "flight" from colliding
# with the "place" concept and making unrelated memories look relevant.
_CONCEPT_INDEX = {name: i for i, name in enumerate(CONCEPTS)}


def _word_bucket(word: str) -> int:
    span = DIM - len(CONCEPTS)
    return len(CONCEPTS) + int(hashlib.md5(word.encode()).hexdigest(), 16) % span


def embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for raw in _tokens(text):
        if raw in STOPWORDS:
            continue
        vec[_word_bucket(_stem(raw))] += 1.0
        for concept, words in CONCEPTS.items():
            if raw in words:
                vec[_CONCEPT_INDEX[concept]] += 1.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"Embeddings of different sizes ({len(a)} and {len(b)}): the database was "
                         "built with another embedder. Call MemoryManager.reembed() after switching.")
    return sum(x * y for x, y in zip(a, b))
