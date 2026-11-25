"""A guided walkthrough of the memory system over ~8 months of simulated time.

    python examples/demo.py           # offline, rule-based extraction (no API key needed)
    python examples/demo.py --llm     # extract memories with your model (set MEMWISE_BASE_URL and MEMWISE_MODEL;
                                      # also set MEMWISE_EMBED_MODEL to use a real embedding model)

Read this file top to bottom alongside its output: each scene shows one idea.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

# Lets you run this from a clone without installing: `python examples/demo.py`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from memwise import Clock, MemoryManager, MemoryStore
from memwise import display as show

USER = "sam"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="extract memories with your model")
    parser.add_argument("--db", default=":memory:", help="SQLite file to use (default: in memory)")
    args = parser.parse_args()

    extractor = embedder = config = None
    if args.llm:
        from memwise import (LLMError, LLMExtractor, MemoryConfig, OpenAICompatible,
                             OpenAICompatibleEmbedder)
        try:
            extractor = LLMExtractor(OpenAICompatible.from_env())
            if os.environ.get("MEMWISE_EMBED_MODEL"):
                embedder = OpenAICompatibleEmbedder.from_env()
                config = MemoryConfig.for_embedding_model()
        except LLMError as e:
            sys.exit(str(e))

    clock = Clock(fixed_start=datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc))
    mm = MemoryManager(MemoryStore(args.db), extractor=extractor, clock=clock,
                       embedder=embedder, config=config)

    def session(name: str, messages: list[str]) -> None:
        print(c_dim(f"  {clock.now():%Y-%m-%d} · {name}"))
        for i, text in enumerate(messages, 1):
            print(f"  {show.c('user:', 'cyan')} {text}")
        print()
        refs = [(f"{name}:{i}", text) for i, text in enumerate(messages, 1)]
        show.decisions(mm.ingest(USER, refs))

    def ask(query: str, scope: str = "general", context: bool = False) -> None:
        print()
        print(f"  {show.c('user:', 'cyan')} {query}")
        r = mm.recall(USER, query, scope=scope)
        show.recall(r)
        if context:
            print()
            print(show.c("  What the model receives:", "bold"))
            for line in mm.build_context(r).splitlines():
                print(show.c("  │ ", "dim") + line)

    # ------------------------------------------------------------------
    show.heading("1. WRITE: a first conversation. Not everything said is worth keeping.")
    session("s1", [
        "Hi! My name is Sam.",
        "I live in Berlin and I work at Acme as a backend engineer.",
        "I prefer short answers with code examples.",
        "I love coffee.",
        "lol the weather is terrible today",
        "I was diagnosed with diabetes last year.",
        "Remember that I'm allergic to peanuts.",
        "I'm traveling to Tokyo next week.",
        "We tried Redis for the session cache and it kept running out of memory.",
    ])
    print(show.c("\n  Small talk produced no candidate at all. The diagnosis was extracted but\n"
                 "  IGNORED by the write gate: it is sensitive and Sam didn't ask us to keep it.\n"
                 "  The allergy is just as sensitive, but Sam said 'remember', so it is kept.", "dim"))

    # ------------------------------------------------------------------
    clock.advance(days=3)
    show.heading("2. READ: three days later. Retrieve, then decide whether to use each memory.")
    ask("Any snack ideas for my flight?", scope="personal", context=True)
    ask("How should I cache user sessions in our API?", scope="work")
    ask("I'm stressed about work. Any tips to relax this weekend?", scope="personal")
    print(show.c("\n  Name and answer style are PROFILE memories: loaded every time.\n"
                 "  The allergy is USED quietly (sensitive, but clearly relevant).\n"
                 "  The Redis episode is MENTIONED: it should change the advice.\n"
                 "  Work memories are HELD in a personal chat, so they never reach the model.", "dim"))

    # ------------------------------------------------------------------
    clock.advance(days=37)
    show.heading("3. CHANGE: forty days in. Facts stop being true.")
    session("s2", [
        "I just moved to Lisbon!",
        "I don't drink coffee anymore, switched to tea.",
        "My name is Sam, by the way.",
    ])
    print()
    print(show.c("  Nightly maintenance:", "bold"))
    for line in mm.maintain(USER) or ["(nothing to do)"]:
        print(f"  {line}")
    print()
    print(show.c("  Everything ever stored (history is kept, only current rows are used):", "bold"))
    show.memory_table(mm.history(USER), clock.now(), show_history=True)

    ask("Where can I get a good drink near home?", scope="personal")
    print(show.c("\n  Berlin and coffee are history now, so they can't be retrieved as if true.", "dim"))

    # ------------------------------------------------------------------
    clock.advance(days=210)
    show.heading("4. STALENESS: eight months in. Old facts get checked, unused episodes fade.")
    ask("Can you help me prepare for my performance review at work?", scope="work", context=True)
    print()
    print(show.c("  Nightly maintenance:", "bold"))
    for line in mm.maintain(USER) or ["(nothing to do)"]:
        print(f"  {line}")
    print(show.c("\n  Acme hasn't been confirmed in 250 days, so the model is told to check\n"
                 "  ('are you still at Acme?') instead of assuming. Using a memory doesn't\n"
                 "  confirm it; only the user restating it does.", "dim"))

    # ------------------------------------------------------------------
    show.heading("5. CONTROL: the user asks to forget something.")
    allergy = next(m for m in mm.current(USER) if m.predicate == "allergy")
    print(f"  {show.c('user:', 'cyan')} Please forget my allergy.")
    removed = mm.forget(USER, allergy.id)
    print(f"  {show.op('FORGET')} deleted {', '.join(removed)} (hard delete, not hidden)")
    print()
    print(show.c("  What Sam's memory holds now:", "bold"))
    show.memory_table(mm.current(USER), clock.now())

    # ------------------------------------------------------------------
    show.heading("6. AUDIT: every change, with a reason. This is how you debug memory.")
    show.log_table(mm.store.logs(USER))
    print()


def c_dim(text: str) -> str:
    return show.c(text, "dim")


if __name__ == "__main__":
    main()
