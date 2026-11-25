# Contributing

Thanks for helping. Issues, questions and pull requests are all welcome.

## Set up

```bash
git clone https://github.com/pankajrohilla893/MemWise.git
cd MemWise
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run the tests

```bash
python -m unittest discover -s tests -v
```

The tests never call a real API. The model is replaced by a fake function, and the HTTP
client talks to a fake server on your machine.

## Guidelines

- **Keep the core dependency-free.** Anything that needs a third-party package (a model SDK, a
  database driver) goes behind an optional extra.
- **Stay provider-neutral.** Talk to models through the `LLM` function type, never a specific
  vendor's SDK.
- **One behaviour, one test.** Name each test after the promise it checks
  (`test_held_memories_never_reach_the_prompt`).
- **Explain the why.** Every module starts with a comment explaining the idea it implements.
  Keep that up to date when behaviour changes.
- **Privacy first.** Changes must not weaken consent for sensitive data, the HOLD guarantee
  (held memories never reach the prompt), or complete deletion.

## Good first contributions

- A Postgres + pgvector `MemoryStore`
- More labelled examples for `examples/tune_embedder.py`
- More scenarios in `evals/scenarios.py`, especially from real products
- More `RuleExtractor` patterns, or patterns for other languages
- An LLM judge for contradictions between free-text memories
