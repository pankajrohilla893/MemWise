# Changelog

## 0.4.0 (2026-11-26)

An optional model that judges what the rules can't see, and fixes found by testing every
scenario again with fresh data.

- **An optional judge for what the rules can't see.** `MemoryManager(judge=LLMJudge(llm))` asks a
  model whether a new memory replaces or repeats stored ones with a different predicate ("Went
  vegetarian" vs "Loves steak") or the same value in new words ("Runs 10k every morning" vs
  "Runs 5k every morning"). It's only asked where the rules can't tell, one call per
  conversation, outside the lock. Its answer is checked: it may only close current memories it
  was shown, never sees sensitive or safety memories, and never deletes anything (replaced
  memories are kept as history). When a rule replaces a memory (a new diet), that stands, and
  the judge can close others it makes out of date. Off by default. New: `LLMJudge`, `Judge`, `JudgeRequest`,
  `Verdict`, `parse_verdicts`, `Decision.also`, `MemoryConfig.judge_related` and
  `judge_min_similarity`
- `evals/judge.py`: 8 scenarios for the judge, and `--judge` / `--no-judge` for every eval file
- The extraction prompt says what `diet` means (how they eat overall, not a food they like): a
  model once filed "drinks green tea" as a diet, so "went vegan" replaced it
- The vegan-recipe eval counts cooking oil, salt, herbs and spices as plant-based; the grading
  model once failed a chickpea curry because the oil wasn't named

Found by testing every scenario again with fresh data (new people, places, a different shop and
a different repository), none of which appears in the prompts or the evals:

- **A changed number is an update, not a repeat.** "Hiking boot size is EU 43" scored over 0.9
  against "... EU 42", so it was confirmed as already known and the new size was lost. A
  near-duplicate whose numbers differ now replaces the old memory (episodes excepted: two late
  orders are two events). With the judge on, it may keep both instead ("Has a 7-year-old" and
  "Has a 9-year-old" may be two children)
- **How the user wants to be contacted is general.** "Text me, I can't take calls at work" was
  filed as a work memory, so a general support chat held it back and offered no follow-up
- **"I've given up on X" retracts X.** It was sometimes stored as a new experience, leaving
  "Loves sourdough baking" current
- **The judge checks every memory a change affects.** It closed "pastries for breakfast" for a
  new keto diet but kept "Loves pasta"; it now treats liking a food the diet excludes as out of
  date
- `examples/git_commits.py`: `--judge`, and the prompt files background-job tools as `queue` and
  names the language in a first commit. "Replace Celery with Django-Q" once left both current
- **"No longer" is kept even when the reason is private.** For "my knees gave out, so no more
  running", a model sometimes returned nothing at all, leaving "Goes jogging" current. The prompt
  now asks for the negated memory without the reason
- "I'm based in Melbourne" is where the user lives (`lives_in`); it was sometimes filed as a note
- **The built-in clients survive a flaky network.** A dropped connection, HTTP 408, 429 or 5xx
  is retried twice (after 0.5s and 1s; `retries=` to change it), and every failure is raised as
  `LLMError`. A connection reset used to escape as a raw `ConnectionResetError`, so the judge
  couldn't fall back to the rules and ingest crashed
- README: documents `LLMExtractor(llm, system_prompt=..., scopes=...)`, which has been available
  since 0.1.0 but was only shown in `examples/git_commits.py`. A new "Use your own extraction
  prompt" section shows how to extend the built-in prompt for a product

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
- Works with any LLM: `LLMExtractor` takes any `(system, messages) -> str` function, plus your
  own extraction prompt (`system_prompt`) and scope names (`scopes`), and `OpenAICompatible` is
  a zero-dependency client for any `/chat/completions` API
- Real embedding models: `OpenAICompatibleEmbedder` for any `/embeddings` API (cached,
  batched), `MemoryConfig.for_embedding_model()` thresholds, and `examples/tune_embedder.py`
  to measure your own model. Mixing vectors from two embedders raises a clear error
- `ingest()` takes the whole conversation (`{"role", "content"}` turns). `LLMExtractor` uses the
  assistant's turns and the user's stored memories as context; memories still only come from the
  user, sensitive memories are never shown, and repeated-back memories don't count as confirmed
- "Please forget that…" in a conversation deletes the memory (FORGET), the same as `forget()`
- Scenario evals (`evals/scenarios.py`): 23 multi-session conversations to test with your own model
- Examples: quickstart, guided demo, interactive chat
