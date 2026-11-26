# How memwise works

This page explains the design. For installation and usage, see the [README](../README.md).

## The pipeline

```
 WRITE PATH: ingest(), after a conversation
   messages ─► extractor ─► candidates ─► reconciler ─────────────► store
               (rules or    "might be     ADD / CONFIRM / SUPERSEDE /   SQLite rows with
                an LLM)      worth it"    RETRACT / FORGET / IGNORE     validity windows + audit log

 READ PATH: recall() + build_context(), before every model call
   message ─► retriever ──────────► surfacer ─────────────────► context block ─► your LLM
              profile + ranked      USE / MENTION / VERIFY / HOLD   HOLD items are simply absent
              current memories

 UPKEEP: maintain_all(), nightly
   EXPIRE situational facts · MERGE near-duplicates · PRUNE unused episodes

 USER CONTROL: forget() / forget_all()
   hard-delete the memory, its older versions, and its text in the audit log
```

## File map

| File | Responsibility |
|---|---|
| [`models.py`](../memwise/models.py) | `Candidate` (a proposal), `Memory` (a stored claim) and `SavedMessage` (a kept message) |
| [`extractor.py`](../memwise/extractor.py) | The `Extractor` interface and the offline `RuleExtractor` |
| [`llm.py`](../memwise/llm.py) | `LLMExtractor`, `LLMJudge` and `LLMQueryRewriter` (any model), the `OpenAICompatible` and `OpenAICompatibleEmbedder` clients, the prompts and parsers |
| [`dates.py`](../memwise/dates.py) | Relative dates ("last weekend", "two weeks ago") worked out in code and written next to the words |
| [`reconciler.py`](../memwise/reconciler.py) | The write gate, the ADD / CONFIRM / SUPERSEDE / RETRACT / FORGET / IGNORE decision, and the checks on an optional judge |
| [`store.py`](../memwise/store.py) | SQLite storage, audit log, locking |
| [`embed.py`](../memwise/embed.py) | The `Embedder` interface, the built-in `HashingEmbedder`, cosine similarity |
| [`retriever.py`](../memwise/retriever.py) | The always-loaded profile, ranked retrieval, kept-message search, and the `Rewriter` interface |
| [`surfacer.py`](../memwise/surfacer.py) | The USE / MENTION / VERIFY / HOLD decision |
| [`maintenance.py`](../memwise/maintenance.py) | Expire, merge, prune |
| [`manager.py`](../memwise/manager.py) | `MemoryManager`: the API your product calls |
| [`config.py`](../memwise/config.py) | `MemoryConfig`: every threshold and rule |

## A memory is a claim, not a note

```python
Memory(
    predicate="lives_in", value="Lisbon", text="Lives in Lisbon",
    kind="fact",              # fact | preference | instruction | episode
    scope="general",          # general | work | personal | your own
    source_kind="stated",     # stated by the user, or inferred by a model
    source_ref="s2:1",        # the message it came from
    confidence=0.8, importance=0.7, sensitive=False,
    valid_from=...,           # when we learned it
    valid_to=None,            # set when it stops being true; the row is kept
    expires_at=None,          # for situational facts ("traveling next week")
    superseded_by=None,       # the memory that replaced this one
    last_confirmed=...,       # last time the USER said it again
    last_used=..., use_count=...,  # last time WE put it in a prompt
)
```

The four kinds of memory behave differently:
- **Facts** can be replaced.
- **Preferences** can coexist: you can like tea *and* hiking.
- **Instructions** are loaded into every prompt.
- **Episodes** fade unless they keep being useful.

`last_confirmed` and `last_used` are deliberately separate fields. Using a memory 100 times
doesn't prove it's still true. Only the user restating it does.

## Write path

### 1. Extraction proposes

An extractor turns the conversation into `Candidate` objects. It never writes to the store.
Memories only come from the user's turns; the assistant's turns are context ("The second one,
please." only makes sense after the question). `LLMExtractor` is also shown the user's most
relevant stored memories (never sensitive ones), so it can resolve "I left last week" and reuse
the stored predicate and value when something changes. It is also told the date of the
conversation (from the manager's clock), so "yesterday" or "last Sunday" is written into the
memory as a date, and the memory still says when it happened months later. A stored memory that comes back without
pointing at any user message is dropped as an echo, so it isn't counted as re-confirmed. `RuleExtractor` uses regular expressions.
`LLMExtractor` makes one call per conversation to whatever model you connect, and reads the
JSON it returns. Both return the same type, so everything after this step is identical.

### 2. The write gate decides what's worth keeping

`write_gate()` in `reconciler.py` drops a candidate when:
- it contains a **secret**: a card number (checked with the Luhn algorithm), a stated password,
  PIN or security code, an API key, a bank account or ID number. These are refused even when the
  user asks, and redacted from the audit log;
- it's **sensitive** and the user didn't explicitly ask to remember it (consent);
- it's an **inferred** guess with low confidence;
- it has **low importance** and wasn't explicit.

### 3. Reconciling decides how it fits

`decide()` compares the candidate with what's stored and picks exactly one operation:

| Check, in order | Operation |
|---|---|
| The user asks to forget it ("please forget that I live in Oslo") | **FORGET**: delete it and its history, and scrub the log, as `forget()` does |
| The user says it's no longer true ("I don't drink coffee anymore") | **RETRACT** the matching memory (kept as history). If the model filed it under another predicate, it's matched by value, except for sensitive memories |
| The write gate rejects it | **IGNORE**, with a reason |
| A single-valued predicate (one city at a time) with the same value | **CONFIRM**: refresh `last_confirmed` |
| A single-valued predicate with a different value | **SUPERSEDE**: close the old memory, add the new one |
| A near-duplicate of an existing memory, with different numbers ("Boot size is EU 42" → "EU 43"; not for episodes) | **SUPERSEDE**: embeddings score these as near-identical, but the number is the news |
| A near-duplicate of an existing memory | **CONFIRM** |
| Contradicts its opposite ("dislikes X" vs "likes X") | **SUPERSEDE** |
| Anything else | **ADD** |

`apply()` carries out the decision. A replaced memory is **closed** (`valid_to`, `superseded_by`)
instead of being overwritten, so you can always answer "where did they used to live?" and
explain why a memory changed. Every operation is written to the audit log.

Projects (`current_project`) are not single-valued: people work on several at once (an album, a
tour, growing a brand), and a new one doesn't end the others. A project ends when the user says
it's finished or dropped (**RETRACT**), when its expiry passes, or when the judge says a new one
replaces it.

### 4. Optional: a judge for what the rules can't see

The rules only compare memories that share a predicate, so "Went vegetarian" (`diet`) is added
next to "Loves steak" (`likes`), and "Runs 10k every morning" only confirms "Runs 5k every
morning" because the value, running, is the same. `MemoryManager(judge=LLMJudge(llm))` asks a
model about exactly those cases, the way some memory systems let a model decide every update.
Here the model still only proposes:

| Step | What happens |
|---|---|
| Who is asked about | Candidates the rules would ADD, CONFIRM in different words, SUPERSEDE, or find nothing to retract for, when similar memories are stored (`judge_min_similarity`, up to `judge_related`) |
| Who never is | Forget requests, secrets and anything else the write gate refuses, sensitive candidates, safety predicates |
| What it sees | The candidate and its most similar current memories, with ids. Never sensitive or safety memories |
| What it answers | ADD, UPDATE (ids it replaces), NONE (the id it repeats), or DELETE (ids that stopped, for "no longer true") |
| What memwise does | UPDATE → SUPERSEDE every named memory · NONE → CONFIRM · DELETE → RETRACT · ADD → the rules' answer |
| What is checked | Each id must be one it was shown, for this user, and still current. A rule's SUPERSEDE (one city, one diet) stands, and UPDATE can only close more memories next to it. For a re-worded CONFIRM, UPDATE must name the memory the rules matched. Anything else falls back to the rules |

The call is made before the manager takes its lock, once for the whole conversation, and the
checks run again under the lock, so a memory deleted meanwhile is left alone. If the call fails
or the reply can't be read, a warning is logged and the rules decide.

**Why it's careful about NONE.** A wrong UPDATE is recoverable: the old memory is kept as
history. A wrong NONE drops the new memory. That's why allergies, which a wrong NONE could lose,
never go to the judge, and why its prompt says to choose ADD when unsure.

## Read path

### 1. Profile and retrieval

The **profile** is small and loaded every time: the user's name and standing instructions
("prefers short answers"; any `response_style` memory counts, whatever kind the model gave it). It is never sensitive and never scoped to one context.

Everything else is **retrieved** per message. Only current memories are searched, and they are
ranked by a blend of signals rather than similarity alone:

```
score = 0.60 × relevance + 0.15 × recency + 0.15 × importance + 0.10 × confidence
```

- **relevance** is the cosine similarity between the message and the memory. Its scale depends
  on the embedding model, which is why the thresholds that use it live in `MemoryConfig`
  (`MemoryConfig.for_embedding_model()` for real models).
- **recency** halves every 60 days since the memory was last confirmed or used.
- **importance** is boosted when the user said "remember".

**Past memories, only when asked.** `recall(..., include_past=True)` also searches what is no
longer true: replaced, retracted and expired memories. It ranks them by relevance alone and
keeps up to `MemoryConfig.past_top_k` (3) of them in `Recall.past`. This is for questions about
the past, such as "where did I live before?" or "when did I start the course?". They get
**PAST**, or **HOLD** under the same scope and sensitivity rules as current memories.
`build_context()` lists them in a separate "No longer true" section, with when each was true
and what replaced it, and tells the model never to treat them as true now. Two kinds of memory
never come back here:
- Forgotten memories are deleted.
- Duplicates that maintenance merged are left out, because they are still true, as the copy
  that was kept.

**Kept messages, only when turned on.** With `MemoryConfig.keep_messages`, `ingest()` also keeps
each of the user's messages word for word, with its date, an embedding, and the ids of the
memories taken from it (or confirmed by it). Secrets are redacted first. When consent is
required, a message the write gate refused a sensitive memory from isn't kept, and neither is
"please forget that ...". A message gets the scope of the memories taken from it, and is marked
sensitive if one of them was. At recall:
- each shown memory gets the message it was first taken from, in `Recall.sources`;
- other kept messages are searched with two rankings, embedding similarity and BM25 keywords
  (names and titles that embeddings blur), plus a third when the query names a date or a month
  ("on 13 October 2023", "in May 2023"): messages from then, most relevant first. The rankings
  are combined by reciprocal rank fusion, and the best `message_top_k` that pass the scope and
  sensitivity rules go in `Recall.messages`.

This is what extraction can't give: the details nobody knew would be asked about. In LoCoMo,
"Where did Caroline move from?" is answered by a line about her grandmother's necklace "from my
home country, Sweden", which no extracted memory kept.

**Relative dates are worked out in code.** Models count days badly: asked what "last weekend"
meant in a chat on Monday 17 July 2023, they often answer a week off. `memwise.dates` finds
English phrases with one clear meaning ("yesterday", "last Friday", "the week before last",
"two weeks ago", "next month") and writes the dates after them, in brackets: "last weekend [the
weekend before Mon 17 Jul 2023: Sat 15 Jul - Sun 16 Jul]". Weeks and weekends are said from the
day, the way people say them. Weekends get their Saturday and Sunday; weeks don't ("last week
[the week before Sun 16 Jul 2023]"), because said on a Sunday, "last week" can mean the calendar
week before or the seven days just gone. Phrases that could mean more than one date
("recently", "on Sunday", "next weekend") are left alone. It's
applied twice: `LLMExtractor` sees the dated conversation (when it is given `now`), and
`build_context()` shows kept messages dated, from the day each was said. The stored message
stays as it was said.

**Rewriting the request, only when turned on.** A request often doesn't sound like what answers
it: "Would she enjoy a weekend outdoors?" against "Loves hiking in the mountains". With
`MemoryManager(rewriter=LLMQueryRewriter(llm))`, `recall()` first asks the model for up to three
search queries (outside the lock; cached; on failure, recall searches with the request alone).
Every search then runs with the request and those queries: a memory's relevance is its best
cosine over all of them, and for kept messages each query adds its own two rankings to the
fusion. Sensitive memories and messages still need to match the request itself
(`Scored.asked`): a query the model wrote can find one, but not bring it up.

### 2. Surfacing: retrieving a memory isn't the same as using it

| Check, in order | Action |
|---|---|
| A safety memory (an allergy, by default) | **USE**, in every scope, however unrelated it looks |
| Belongs to a different scope than this conversation | **HOLD** |
| Sensitive and only loosely related | **HOLD** |
| Not relevant enough | **HOLD** |
| A fact not confirmed in 180+ days | **VERIFY**: the reply asks whether it's still true |
| Low confidence | **VERIFY** |
| An episode (something that happened) | **MENTION**: it should change the answer |
| Otherwise | **USE**, quietly |

**Why allergies skip the checks.** Holding back is the safe default for most memories, but not
for an allergy. Similarity can't reliably tell a food question from an unrelated one: with a
typical embedding model, "Allergic to peanuts" scored 0.146 against "Help me write a cover
letter" but 0.124 against some food questions, and "Give me a quick dinner recipe idea" fell
just below the bar for sensitive memories. In testing, that held-back allergy led a model to
suggest peanut noodles. So `MemoryConfig.safety_predicates` memories are retrieved every time
(without taking `top_k` slots), used in every scope (models sometimes file an allergy as
"personal", but a work chat about the team lunch needs it too), and never sent to VERIFY. They
stay in force until the user retracts or forgets them.

### 3. The prompt

`build_context()` writes the profile and each used memory, with an instruction matching its
action: "apply quietly", "worth mentioning", "check with the user first". Safety memories get
their own section: "never suggest anything that conflicts with this; only bring it up when it
matters". **HOLD items are not written at all.** A memory that isn't in the prompt can't leak,
which is safer than telling the model not to mention it. Safety memories are the deliberate
exception: there, missing the memory is the bigger risk.

## Maintenance

`maintain()` runs three passes:
- **EXPIRE:** situational facts whose expiry date has passed. The expiry is set when the fact is
  extracted ("next week" → 14 days).
- **MERGE:** near-duplicate memories are folded into one.
- **PRUNE:** episodes nobody used or restated in 120 days are deleted.

## Forgetting

- `forget(user_id, memory_id)` hard-deletes the memory and the older versions it replaced, then
  replaces their text in the audit log with `[forgotten]`.
- `forget_all(user_id)` removes every memory and log entry for the user.
- With kept messages, `forget` also deletes the messages the memory came from and every kept
  message that names its value ("Berlin"), and `forget_all` deletes them all.

The write gate also redacts sensitive candidates it ignores, so the log never keeps what the
gate refused to store.

## Concurrency

- `MemoryStore` and `MemoryManager` hold locks, so one instance can be shared by all threads of
  a web app.
- Memory ids are allocated under SQLite's write lock (`BEGIN IMMEDIATE`), so several processes
  sharing one database file never hand out the same id.
- File databases use WAL mode, so reads don't block writes.
- Two processes ingesting for the *same user* at the same moment can still race logically (both
  may see the old value). If that matters, run ingestion in a single worker queue.

## Memory vs RAG

Both use embeddings to find text by meaning, but they solve different problems:

| | RAG over documents | Memory |
|---|---|---|
| Source | Documents someone wrote | Conversations with the user |
| Unit stored | A passage | One claim, with time, source and confidence |
| What gets stored | Everything is indexed | Selected by a write gate |
| When things change | Re-index the document | The new fact replaces the old one; history is kept |
| After retrieval | Put it in the prompt | Decide: use, mention, verify or hold back |

Searching raw chat transcripts with RAG returns "I live in Berlin" and "I just moved to Lisbon"
side by side, and similarity alone can't tell which is current. In `memwise`, that decision
is made when the fact is **written** (SUPERSEDE), so only current facts are ever retrieved.
Each memory keeps `source_ref`, so you can still keep a searchable archive of raw transcripts
next to it, and `MemoryConfig.keep_messages` does that for you: the user's own messages, under
the memory rules for consent, scope, secrets and deletion, searched next to the memories.

## Design principles

1. **The LLM proposes, your code decides.** Extraction can use any model, but storing,
   replacing and forgetting follow tested rules. The optional judge proposes too: its answers
   are checked before they're used.
2. **Close rows, never overwrite them.** History explains every change.
3. **Write time is when to resolve contradictions**, not search time.
4. **Leave memories out rather than telling the model to stay quiet.**
5. **Deleting has to reach every copy, logs included.**
