"""Every tunable rule in one place.

Pass your own `MemoryConfig` to `MemoryManager` to change how memory behaves,
without editing library code:

    from memwise import MemoryConfig, MemoryManager

    config = MemoryConfig(stale_after_days=90, top_k=10)
    config.single_valued.add("timezone")      # a user has one timezone at a time
    mm = MemoryManager(config=config)

The similarity thresholds (`min_relevance`, `relevant_enough`,
`sensitive_relevant_enough`, `duplicate_similarity`) depend on the embedder. The
defaults suit the built-in `HashingEmbedder`. With a real embedding model, start
from `MemoryConfig.for_embedding_model()` and check the values for your model
with `python examples/tune_embedder.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _default_single_valued() -> set[str]:
    return {"name", "lives_in", "employer", "job_title", "current_project", "traveling_to",
            "response_style", "diet"}


@dataclass
class MemoryConfig:
    # ---- write path (reconciler) ----------------------------------------
    # Predicates that can only have one current value; a new value supersedes the old one.
    single_valued: set[str] = field(default_factory=_default_single_valued)
    # Predicates that cancel each other for the same value ("likes X" vs "dislikes X").
    opposites: dict[str, str] = field(
        default_factory=lambda: {"likes": "dislikes", "dislikes": "likes"})
    # Above this embedding similarity, two memories count as the same thing.
    duplicate_similarity: float = 0.85
    # Sensitive details (health, money, ...) are stored only if the user asked explicitly.
    require_consent_for_sensitive: bool = True
    # Inferred (not stated) memories below this confidence are dropped.
    min_inferred_confidence: float = 0.6
    # Candidates below this importance are dropped unless the user asked explicitly.
    min_importance: float = 0.2
    # How many stored memories an LLM extractor is shown, most relevant first, so it can
    # understand "I left last week" and reuse wording. Sensitive memories are never shown. 0: none.
    extraction_context_memories: int = 40
    # Lasting facts never get an expiry date, even if the extractor proposes one: they change
    # when the user says so, and get re-checked when old (see stale_after_days).
    lasting_predicates: set[str] = field(default_factory=lambda: {
        "name", "lives_in", "employer", "job_title", "response_style", "diet", "allergy", "health"})

    # ---- read path (retriever) ------------------------------------------
    weights: dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.60, "recency": 0.15, "importance": 0.15, "confidence": 0.10})
    recency_half_life_days: float = 60
    min_relevance: float = 0.12   # below this a memory isn't even a candidate
    top_k: int = 6
    # Always-loaded memories: these predicates and kinds go into every prompt.
    # response_style is listed too, so answer-style rules load even if a model files them as a preference.
    profile_predicates: set[str] = field(default_factory=lambda: {"name", "response_style"})
    profile_kinds: set[str] = field(default_factory=lambda: {"instruction"})

    # ---- read path (surfacer) -------------------------------------------
    relevant_enough: float = 0.22            # relevance needed to be used at all
    sensitive_relevant_enough: float = 0.30  # sensitive memories need a stronger reason
    stale_after_days: float = 180            # facts not reconfirmed this long get verified
    low_confidence: float = 0.6

    # ---- upkeep (maintenance) -------------------------------------------
    prune_episodes_after_days: float = 120

    @classmethod
    def for_embedding_model(cls, **overrides: Any) -> MemoryConfig:
        """Starting thresholds for a real embedding model, instead of the built-in one.

        Real models give related text lower scores, and unrelated text higher ones, than
        `HashingEmbedder`. These values come from `examples/tune_embedder.py` on a typical
        hosted model (OpenAI's text-embedding-3-small), where related memories scored about
        0.1-0.5 against a question and unrelated ones mostly below 0.2. Other models differ,
        so run the script with yours, and pass any changes as keyword arguments.

        `duplicate_similarity` is deliberately high: "Allergic to peanuts" and "Allergic to
        tree nuts" score about 0.81, and treating them as one memory would lose an allergy.
        """
        values: dict[str, Any] = dict(min_relevance=0.08, relevant_enough=0.16,
                                      sensitive_relevant_enough=0.18, duplicate_similarity=0.90)
        values.update(overrides)
        return cls(**values)


DEFAULT_CONFIG = MemoryConfig()
