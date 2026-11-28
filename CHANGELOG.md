# Changelog

## 0.7.0 (2026-11-28)

Replies can be streamed, calls no longer hang on networks with broken IPv6, and memwise has a
live demo.

- **Live demo.** `demo/app.py` is a web app with the same model in two chats side by side, on
  its own and with memwise, and panels showing every memory decision and why it was made. A
  two-minute tour follows one person over eight months: an allergy, a move to another city, a
  work question kept apart. It runs on Hugging Face Spaces at
  https://pankajrohilla899-memwisedemo.hf.space and is deployed from GitHub whenever `demo/`
  changes. It's an example, not part of the package
- **Streamed replies.** `OpenAICompatible.stream(system, messages)` yields the reply in pieces
  as the model writes it, so a chat can show the first words in under a second instead of the
  whole reply after a few. Connecting is retried like a normal call; a failure mid-reply raises
  `LLMError`. A server that doesn't stream gives the whole reply as one piece. The live demo
  uses it
- **No more hangs on networks with broken IPv6.** Some networks give a host IPv6 addresses but
  drop IPv6 traffic. Python connects to a host's addresses one at a time and waits the whole
  timeout on each, so there every call from `OpenAICompatible` and `OpenAICompatibleEmbedder`
  waited out `timeout` (two minutes by default) before trying IPv4, and gave up if that took
  too long. The clients now connect the way browsers do ("happy eyeballs", RFC 8305): a host's
  addresses are tried in turn, IPv6 and IPv4 alternating, and the next one starts when the
  last fails or hasn't connected in 0.25 s. The first connection made is used, and that
  family goes first for the host next time, so a network without working IPv6 costs 0.25 s
  once. Standard library only, and proxy settings still apply. They no longer go through a
  global opener installed with `urllib.request.install_opener`

## 0.6.0 (2026-11-26)

memwise can keep what the assistant said, and is measured on a second benchmark, LongMemEval.

- **Keep what the assistant said, too.** `MemoryConfig.keep_assistant_messages` (off by default)
  keeps the assistant's replies word for word, in pieces of at most `assistant_piece_chars`
  (1,200), so "which hostel did you recommend?" can be answered. No memory is taken from them:
  they're searched on their own and returned in `Recall.assistant_messages` (up to
  `assistant_message_top_k`, 8), and `build_context` lists them as what the assistant told the
  user. A reply follows the user message it answers (not kept when that message isn't; its
  scope and sensitivity). The `messages` table gets a `role` column; older files get it when
  opened. On a 60-question LongMemEval sample (10 of each type), memwise went from 75.0 to 88.3:
  questions about what the assistant said from 1 right to 9, with about 5,200 tokens of context
  a question instead of 3,300. One question that can't be answered (what a bus would save) was
  answered from an estimate the assistant had once given (so was it with the whole history). Recall is slower with many replies
  kept: p50 1.6 s against 0.7 s (3.3 s on histories with long replies), the embedding call included
- **LongMemEval:** `evals/longmemeval.py` runs the LongMemEval benchmark (cleaned S: 500
  questions, each with its own history of about 50 chats), with a whole-history baseline, and
  `evals/report.py` shows its question types. On a 60-question sample, memwise scores 88.3 with
  about 5,200 tokens of context a question; the whole history 93.3 with about 124,000. The
  results are in the README and on the results page, in a LongMemEval tab next to LoCoMo

## 0.5.1 (2026-11-26)

A fix: a new project no longer closes the projects before it.

- **Several projects can be current at once.** `current_project` is no longer single-valued, so
  a new project doesn't close the ones before it. Before, each project replaced the last: in the
  LoCoMo benchmark, a musician's studio, his plan to grow his brand worldwide, his tour and his
  album each closed the one before, and asked "Does Calvin want to expand his brand?" the model
  read "no longer true" and said no. 72 project memories were closed this way. A project now ends when
  the user says it's finished or dropped (the built-in prompt asks for a negated memory, and
  `RuleExtractor` understands "I've finished working on X" and "I'm no longer working on X"),
  when its expiry passes, or when the judge says a new one replaces it. To keep one project at a
  time, `config.single_valued.add("current_project")`. On LoCoMo the overall score is unchanged
  within noise (89.4 against 89.7 on all 1,540 questions, with a rebuilt memory); questions about
  projects went from 85.7% to 87.3%, and memories of any kind closed by a newer one fell from 78 to 5
- **LoCoMo:** the build format is now "4", so the benchmark builds its memory again with the
  new rule instead of reusing an old build

## 0.5.0 (2026-11-26)

Memory that keeps what the user said word for word, works out relative dates, can rewrite a
request before searching, and a LoCoMo benchmark with results: 89.7 on 1,540 questions with 78%
fewer tokens than sending the whole conversation.

- **The extractor knows when the conversation happened.** `MemoryManager` passes its clock's
  time to extractors that take `now`, and `LLMExtractor` shows it as `<today>`. The built-in
  prompt writes dates into events, trips and plans: "yesterday" becomes "on 24 May 2023",
  "last Sunday" becomes "on 21 May 2023". Before, "Ran a charity race last Sunday" lost its date,
  so months later nobody could say when it happened. Lasting facts, routines and preferences get
  no date
- **Recall can include the past.** `recall(..., include_past=True)` also finds replaced,
  retracted and expired memories for questions about the past ("where did I live before?").
  They go in `Recall.past`, and `build_context` lists them apart, under "No longer true", with
  when each was true and what replaced it. They follow the same scope and sensitivity rules.
  It's off by default, so nothing changes unless you ask. `MemoryConfig.past_top_k` (default 3)
- **Kept messages: what the user said, word for word.** `MemoryConfig(keep_messages=True)` keeps
  each of the user's messages next to the memories. `recall()` returns the message each shown
  memory came from (`Recall.sources`) and other kept messages that match (`Recall.messages`, up
  to `message_top_k`), found by meaning, by keywords and by the date a question names ("on 13
  October 2023", "in May 2023"), and `build_context` shows both, dated.
  Extraction drops details nobody knew would matter ("the book I read last year"); this keeps
  them. Secrets are redacted first. Messages follow the consent, scope and sensitivity rules of
  the memories taken from them, and `forget()` deletes the messages a memory came from and those
  that name it. Off by default. New: `SavedMessage`, `ScoredMessage`
- **Relative dates are worked out in code.** New `memwise.dates`: "last weekend" said on Monday
  17 July 2023 becomes "last weekend [the weekend before Mon 17 Jul 2023: Sat 15 Jul - Sun 16
  Jul]". Models often counted these a week off. `LLMExtractor` sees the conversation dated this
  way, and `build_context` shows kept messages dated from the day each was said. Only English
  phrases with one clear meaning are dated ("yesterday", "last Friday", "two weeks ago", "next
  month"); vague ones ("recently", "on Sunday") are left alone
- **An optional rewriter for requests that don't sound like their answer.**
  `MemoryManager(rewriter=LLMQueryRewriter(llm))` asks a model for up to three search queries
  per request ("Would she enjoy a weekend outdoors?" -> "I love hiking"), and recall searches
  with those as well (`Recall.searched`). Sensitive memories still have to match the request
  itself. One cached call per new request, outside the lock; if it fails, recall searches with
  the request alone. Off by default. New: `LLMQueryRewriter`, `Rewriter`, `parse_queries`,
  `Scored.asked`
- `evals/locomo.py` runs the [LoCoMo](https://github.com/snap-research/locomo) benchmark: months
  of chats between two friends, then questions about them. Download the data first; it isn't
  redistributed, and `evals/data/` is ignored by git. Each friend gets their own memory, and a
  model grades every answer. Grading is lenient by default, like published results, and
  `--grader strict` wants every part right.

  Options:
  - `--baseline`: the whole conversation in the prompt, for comparison
  - `--prompt friends`: keeps life events, with dates
  - `--store-sensitive`: stores sensitive details without asking
  - `--per-category N`: a sample of N questions per category
  - `--show-wrong N`: prints wrong answers with the memories they had
  - `--past`: recalls past memories as well
  - `--keep-messages`, `--message-top-k N`: keeps each friend's lines, and recalls N per friend
  - `--rewrite`: rewrites each question into search queries first
  - `--memory NAME`: reopens a memory built before, whatever the extraction prompt is now

  Each run saves a `.summary.json`, and `evals/report.py` turns summaries into one HTML results
  page. The page shows the score and mean context per question next to full context, by
  question type, with a tab per benchmark, and what memwise saves in tokens and money
  (`--price`, `--note`, `--reference` for a published result)
- **LoCoMo results in the README** and on `docs/benchmarks.html`: 89.7 on all 1,540 questions
  with about 4,900 tokens per question. On the same 382 questions, memwise scores 85.3 with 78%
  fewer tokens than the whole conversation in the prompt (88.2), with the same answer model

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
