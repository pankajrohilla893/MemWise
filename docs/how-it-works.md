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
| [`models.py`](../memwise/models.py) | `Candidate` (a proposal) and `Memory` (a stored claim) |
| [`extractor.py`](../memwise/extractor.py) | The `Extractor` interface and the offline `RuleExtractor` |
| [`llm.py`](../memwise/llm.py) | `LLMExtractor` (any model), the `OpenAICompatible` and `OpenAICompatibleEmbedder` clients, the extraction prompt and parser |
| [`reconciler.py`](../memwise/reconciler.py) | The write gate and the ADD / CONFIRM / SUPERSEDE / RETRACT / FORGET / IGNORE decision |
| [`store.py`](../memwise/store.py) | SQLite storage, audit log, locking |
| [`embed.py`](../memwise/embed.py) | The `Embedder` interface, the built-in `HashingEmbedder`, cosine similarity |
| [`retriever.py`](../memwise/retriever.py) | The always-loaded profile and ranked retrieval |
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
the stored predicate and value when something changes. A stored memory that comes back without
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
| A near-duplicate of an existing memory | **CONFIRM** |
| Contradicts its opposite ("dislikes X" vs "likes X") | **SUPERSEDE** |
| Anything else | **ADD** |

`apply()` carries out the decision. A replaced memory is **closed** (`valid_to`, `superseded_by`)
instead of being overwritten, so you can always answer "where did they used to live?" and
explain why a memory changed. Every operation is written to the audit log.

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
next to it.

## Design principles

1. **The LLM proposes, your code decides.** Extraction can use any model, but storing,
   replacing and forgetting follow tested rules.
2. **Close rows, never overwrite them.** History explains every change.
3. **Write time is when to resolve contradictions**, not search time.
4. **Leave memories out rather than telling the model to stay quiet.**
5. **Deleting has to reach every copy, logs included.**
