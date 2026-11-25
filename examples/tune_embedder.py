"""Check how your embedding model scores memories, and get thresholds to start from.

    export MEMWISE_BASE_URL="https://api.your-provider.com/v1"   # any /embeddings API
    export MEMWISE_EMBED_MODEL="your-embedding-model"
    export MEMWISE_API_KEY="..."                                 # only if your provider needs one
    python examples/tune_embedder.py

Every embedding model has its own sense of "similar". This script embeds a small
labelled set (questions, the memories that should help answer them, and pairs that
are or aren't the same fact), shows the scores, and suggests `MemoryConfig` values.
Add examples from your own product to RELEVANT, SAME and DIFFERENT for better numbers.
"""

from __future__ import annotations

import os
import sys

# Lets you run this from a clone without installing.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from memwise.embed import cosine
from memwise.llm import LLMError, OpenAICompatibleEmbedder

MEMORIES = [
    "Name is Ana", "Lives in Lisbon", "Is vegetarian", "Allergic to peanuts", "Works at Tidewave",
    "Works as a data engineer", "Work involves wrangling Spark pipelines", "Prefers short answers",
    "Traveling to Tokyo next week", "Tried Redis for the session cache and it ran out of memory",
    "Likes tea", "Has diabetes", "Loves hiking", "Has a daughter named Mia",
]
# question -> the memories that should help answer it (every other memory shouldn't)
RELEVANT = {
    "Any snack ideas for this weekend?": {"Allergic to peanuts", "Is vegetarian", "Has diabetes"},
    "Can you suggest a dinner spot near my place tonight?": {"Lives in Lisbon", "Is vegetarian",
                                                            "Allergic to peanuts"},
    "What should I pack for my trip?": {"Traveling to Tokyo next week"},
    "How should I cache user sessions?": {"Tried Redis for the session cache and it ran out of memory"},
    "How should I structure retries in my Spark job?": {"Work involves wrangling Spark pipelines",
                                                        "Works as a data engineer"},
    "What's the weather like where I live?": {"Lives in Lisbon"},
    "Help me write a cover letter for a new job": {"Works at Tidewave", "Works as a data engineer"},
    "Recommend a warm drink for the evening": {"Likes tea"},
    "Plan a fun Saturday for me": {"Loves hiking", "Has a daughter named Mia"},
    "What dessert should I bake?": {"Has diabetes", "Allergic to peanuts", "Is vegetarian"},
    "Recommend a good book to read": set(),
    "Explain how TCP handshakes work": set(),
    "What is 17 times 23?": set(),
}
# pairs that are the same memory in other words ...
SAME = [("Prefers short answers", "Likes concise replies"), ("Allergic to peanuts", "Has a peanut allergy"),
        ("Allergic to peanuts", "Severely allergic to peanuts"), ("Likes tea", "Enjoys drinking tea"),
        ("Loves hiking", "Enjoys going on hikes")]
# ... and pairs that must stay two memories (merging them would lose information)
DIFFERENT = [("Allergic to peanuts", "Allergic to tree nuts"), ("Allergic to peanuts", "Allergic to shellfish"),
             ("Likes tea", "Likes coffee"), ("Likes tea", "Likes green tea"), ("Loves hiking", "Loves swimming"),
             ("Likes Italian food", "Likes Thai food"), ("Has a daughter named Mia", "Has a son named Leo")]


def best_threshold(related: list[float], unrelated: list[float]) -> float:
    """The cut-off that best separates the two groups (highest true-positive minus false-positive rate)."""
    def quality(t: float) -> float:
        return (sum(s >= t for s in related) / len(related)
                - sum(s >= t for s in unrelated) / len(unrelated))
    return max(sorted(set(related)), key=quality)


def main() -> None:
    try:
        embedder = OpenAICompatibleEmbedder.from_env()
    except LLMError as e:
        sys.exit(str(e))

    texts = MEMORIES + list(RELEVANT) + [t for pair in SAME + DIFFERENT for t in pair]
    vec = dict(zip(texts, embedder.embed_many(texts)))
    print(f"{embedder.model}: {len(vec[texts[0]])} dimensions\n")

    related, unrelated = [], []
    for question, wanted in RELEVANT.items():
        scores = sorted(((cosine(vec[question], vec[m]), m) for m in MEMORIES), reverse=True)
        for s, m in scores:
            (related if m in wanted else unrelated).append(s)
        top = "  ".join(f"{'✓' if m in wanted else '·'} {s:.2f} {m[:26]}" for s, m in scores[:3])
        print(f"{question}\n    {top}")

    same = [cosine(vec[a], vec[b]) for a, b in SAME]
    different = [cosine(vec[a], vec[b]) for a, b in DIFFERENT]
    relevant_enough = round(best_threshold(related, unrelated), 2)
    duplicate = round(min(0.97, max(different) + 0.05), 2)
    used = sum(s >= relevant_enough for s in related)
    leaked = sum(s >= relevant_enough for s in unrelated)

    print(f"\nRelated memories scored    {min(related):.2f} to {max(related):.2f}")
    print(f"Unrelated memories scored  {min(unrelated):.2f} to {max(unrelated):.2f}")
    print(f"Same fact, other words:    {min(same):.2f} to {max(same):.2f}")
    print(f"Different facts:           {min(different):.2f} to {max(different):.2f}")
    print(f"\nWith relevant_enough={relevant_enough}: {used}/{len(related)} related memories are used, "
          f"{leaked}/{len(unrelated)} unrelated ones get in.")
    print(f"With duplicate_similarity={duplicate}: {sum(s >= duplicate for s in same)}/{len(same)} "
          "rewordings are merged, and no different facts are.")
    print(f"""
Suggested starting point:

    config = MemoryConfig.for_embedding_model(
        min_relevance={max(0.0, relevant_enough - 0.08):.2f},
        relevant_enough={relevant_enough:.2f},
        sensitive_relevant_enough={relevant_enough + 0.02:.2f},
        duplicate_similarity={duplicate:.2f},
    )

Raise relevant_enough if unrelated memories reach your prompts; lower it if useful ones are
missed. Keep duplicate_similarity above every "different facts" score.""")


if __name__ == "__main__":
    main()
