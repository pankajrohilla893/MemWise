# Changelog

## 0.3.0 (2026-11-26)

Tested as a customer-support bot for an online store (`evals/support_bot.py`, 10 scenarios).
Out of the box 7 passed; these changes fix the other 3 and close a gap that passed by luck:

- **Secrets are never stored.** The write gate refuses card numbers (Luhn-checked), stated
  passwords, PINs and security codes, API keys, bank account and ID numbers, even when the user
  asks, and keeps them out of the audit log. Before, only the model's judgement kept "remember my
  card for next time" out of memory. `MemoryConfig.block_secrets` (on by default)
- **Problems the user had are remembered.** The extraction prompt now keeps things the user went
  through, such as a damaged or late order, as episodes. They were skipped as one-off details, so
  a support bot forgot the cracked blender by the next order
- **Addresses:** a new `address` predicate, single-valued and lasting, in the general scope. A
  home address was filed as a "personal" note, so it was held back in general chats and a new
  address didn't replace the old one
- `evals/support_bot.py`, and `evals/scenarios.main()` can now run any list of scenarios

## 0.2.0 (2026-11-26)

- **Allergies are never held back.** A consented "allergic to peanuts" scored just below the bar
  for sensitive memories against "Give me a quick dinner recipe idea", was held back, and a model
  suggested peanut noodles. Memories under `MemoryConfig.safety_predicates` (default: `allergy`)
  now go into every prompt, in every scope, in their own "Always respect (safety)" section, with
  an instruction to respect them and only bring them up when they matter. They don't take
  `top_k` slots and are never sent to VERIFY. Pass `safety_predicates=set()` to turn this off
- Scenario evals: a new `allergies_are_never_held_back` scenario (24 in all), a check that the
  allergy reaches the prompt in `memory_shapes_the_answer`, and judge questions phrased so that
  "yes" means pass (the judge once reasoned "no peanuts" and then answered "yes")
- `examples/git_commits.py`: memory for a repository instead of a person. It reads git commits
  (only new ones on each run) and keeps decisions, rules and lessons, each saved at its commit's
  date and linked to its hash, so replacements keep their history ("Jest, then Vitest"). On a
  24-commit test repo with gpt-6-luna it passed 19/19 checks in 5 of 5 runs
- `build_context(recall, about="the project")` for memory that isn't about a person

## 0.1.0 (2026-11-26)

First release.

- `MemoryManager` with `ingest`, `add`, `recall`, `build_context`, `current`, `history`,
  `forget`, `forget_all`, `maintain`, `maintain_all` and `reembed`
- A write gate with consent for sensitive data; ADD / CONFIRM / SUPERSEDE / RETRACT / FORGET / IGNORE reconciliation
- Validity windows, expiry and staleness checks; history is kept when facts change
- Ranked retrieval, and the USE / MENTION / VERIFY / HOLD decision
- SQLite store with an audit log, thread-safe, with collision-free ids across processes
- `MemoryConfig` for every threshold; pluggable `Embedder` and `Extractor`
- Works with any LLM: `LLMExtractor` takes any `(system, messages) -> str` function, and
  `OpenAICompatible` is a zero-dependency client for any `/chat/completions` API
- Real embedding models: `OpenAICompatibleEmbedder` for any `/embeddings` API (cached,
  batched), `MemoryConfig.for_embedding_model()` thresholds, and `examples/tune_embedder.py`
  to measure your own model. Mixing vectors from two embedders raises a clear error
- `ingest()` takes the whole conversation (`{"role", "content"}` turns). `LLMExtractor` uses the
  assistant's turns and the user's stored memories as context; memories still only come from the
  user, sensitive memories are never shown, and repeated-back memories don't count as confirmed
- "Please forget that…" in a conversation deletes the memory (FORGET), the same as `forget()`
- Scenario evals (`evals/scenarios.py`): 23 multi-session conversations to test with your own model
- Examples: quickstart, guided demo, interactive chat
