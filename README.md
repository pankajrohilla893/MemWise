# memwise

**Long-term memory for AI products that decides what to keep, notices when facts stop being true, and checks whether to speak up.**

[![tests](https://github.com/pankajrohilla893/MemWise/actions/workflows/tests.yml/badge.svg)](https://github.com/pankajrohilla893/MemWise/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![dependencies](https://img.shields.io/badge/core%20dependencies-none-brightgreen)

Models keep getting smarter, but the products built on them still forget who you are. Most AI
memory is a pile of notes an LLM pulled out of old chats, sitting behind a search box. Nothing
decides what was worth keeping, nothing notices when a fact stops being true, and nothing checks
whether a memory should be used at all.

`memwise` is a small Python library that does those jobs:

| Job | What happens | Example |
|---|---|---|
| **Write** | A write gate drops small talk and anything sensitive the user didn't ask you to keep. Every new memory is compared with what's already stored. | "I was diagnosed with…" (said in passing) → **not stored**. "Remember I'm allergic to peanuts" → **stored** |
| **Maintain** | New facts **replace** old ones and the history is kept. Trips and projects expire on their own. Old facts are flagged for re-checking. | "I moved to Lisbon" → Berlin is closed, not overwritten |
| **Retrieve** | Memories are ranked by relevance, recency, importance and confidence, not similarity alone. Only *current* facts are searched, and past ones only when you ask. | Outdated facts never reach the prompt as if still true |
| **Surface** | Each memory gets one of **USE** (quietly), **MENTION**, **VERIFY** or **HOLD** (left out of the prompt). Allergies are never held back. | Work facts stay out of a personal chat. "Still at Acme?" after 8 months. No peanut recipes for a peanut allergy |

It also gives users real control: they can see what's stored, delete one memory, or delete
everything. Deleting also scrubs the audit log.

- **Works with any model provider, hosted or local.** Built-in clients for any `/chat/completions`
  and `/embeddings` API, or plug in your own functions. memwise never ties you to a provider.
- **No dependencies in the core.** Pure Python and SQLite. `pip install` and go.
- **Readable.** Every module starts with a comment explaining the idea it implements.

## Benchmark results

[LoCoMo](https://github.com/snap-research/locomo) is the long-conversation benchmark that memory
systems are usually measured on: 10 chats between two friends, each spanning months (19 to 32
sessions), and 1,540 questions about them.

**memwise answers 89.7% of the 1,540 questions correctly, with about 4,900 tokens of memory per
question.**

| Question type | What it needs | Questions | memwise |
|---|---|---|---|
| Single hop | one fact | 841 | 93.1 |
| Multi-hop | several facts together | 282 | 90.4 |
| Temporal | when something happened | 321 | 87.5 |
| Open-domain | an inference beyond what was said | 96 | 65.6 |
| **All** | | **1,540** | **89.7** |

### 78% fewer tokens than sending the whole conversation

Without a memory system, you send the whole conversation with every request. On the same 382
questions (up to 10 of each type per conversation), answered by the same model with the same
prompt:

| | Score | Tokens per question |
|---|---|---|
| The whole conversation in the prompt | 88.2 | 22,069 |
| **memwise** | **85.3** | **4,809** |

memwise keeps 97% of the score with 22% of the tokens:

- **Each call costs 78% less for its memory.** Models bill input by the token. At an example price
  of $1 per million input tokens, 1,000 questions cost **$4.81** of context with memwise and
  $22.07 with the whole conversation. At your model's price, the saving is the same share.
- **Building the memory is cheap.** One extraction call per chat session, about 3,500 tokens
  each: 1.9 million tokens for all 10 conversations. Counting that, memwise used 73% fewer
  tokens over the whole benchmark.
- **It doesn't grow with the history.** The whole conversation gets longer with every chat, until
  it no longer fits the model's context window. memwise sends about the same amount however long
  the history is, set by `top_k` and `message_top_k`.

The sample has more open-domain questions, the hardest type, than the full set (82 of 382,
against 96 of 1,540), so both scores are lower here than on all 1,540 questions.

### Compared with Mem0

[Mem0 reports](https://mem0.ai/research) 92.5 on LoCoMo with 6,956 tokens per query. memwise
scores 89.7, 2.8 points lower, with 4,899 tokens: **30% fewer**. Mem0 measured with its own
models and grader, so this isn't a controlled comparison: some of the gap may come from the
models rather than the memory.

### How it was measured

- Each friend gets their own memwise. Every session goes in, in order, with its date. memwise
  recalls 10 memories and 30 of the friend's own lines (kept messages) per friend, including
  memories that are no longer true, marked as history. Sensitive details are stored without
  asking, because the questions are about them (by default memwise asks first).
- gpt-6-luna extracts the memories, answers and grades. text-embedding-3-small makes the
  embeddings. A model grades each answer against LoCoMo's, leniently, as published LoCoMo results
  are graded. Adversarial questions (category 5) are left out, as in published results.
- One run. Model output varies a little, so a repeat run can differ by about a point.

The [results page](docs/benchmarks.html) has the charts, by question type. GitHub shows HTML
files as source, so download it and open it in a browser. To run the benchmark yourself, see
[Benchmark: LoCoMo](#benchmark-locomo).

---

## Install

From GitHub:

```bash
pip install "git+https://github.com/pankajrohilla893/MemWise.git"
```

To pin a release, add its tag: `pip install "git+https://github.com/pankajrohilla893/MemWise.git@v0.5.1"`.
There is nothing else to install. memwise and its built-in LLM client use only the Python
standard library, so it works with whatever model provider you already use.

Or clone it to run the examples and tests:

```bash
git clone https://github.com/pankajrohilla893/MemWise.git
cd MemWise
pip install -e .
```

Requires Python 3.10+.

## Quick start (runs offline, no API key)

```python
from memwise import MemoryManager, MemoryStore

mm = MemoryManager(MemoryStore("memory.db"))

# After a conversation: hand over what the user said.
mm.ingest("user-42", [
    "Hi, my name is Ana and I live in Berlin.",
    "I prefer short answers.",
    "Remember that I'm allergic to peanuts.",
])

# Later, facts change. Berlin is closed, Lisbon becomes current.
mm.ingest("user-42", ["I just moved to Lisbon!"])

# Before each model call: get the memory section of your system prompt.
recall = mm.recall("user-42", "Any snack ideas for this weekend?")
system_prompt = "You are a friendly assistant.\n\n" + mm.build_context(recall)
```

`system_prompt` now contains:

```text
You are a friendly assistant.

## About the user (always loaded)
- Name is Ana
- Prefers short answers - follow this

## Always respect (safety)
- Allergic to peanuts (known since 2026-11-25) -> never suggest anything that conflicts with this; only bring it up when it matters

## How to use memory
- Memories are background, not a script. Answer the user's actual message.
- If the user contradicts a memory, believe the user.
- Never invent memories that are not listed here.
```

"Lives in Berlin" is gone because it was replaced. "Lives in Lisbon" isn't relevant to snacks,
so it isn't included either. The allergy is a **safety memory**: it goes into every prompt,
because missing it could hurt someone (see [Privacy and safety defaults](#privacy-and-safety-defaults)).

Run the full example: `python examples/quickstart.py`.

---

## Using it in your product

Your app calls the library at four points:

```
 every user message ──► recall() + build_context() ──► your LLM call
 end of conversation ─► ingest()          (in a background task: it may call an LLM)
 nightly job ─────────► maintain_all()
 settings page ───────► current() · forget() · forget_all()
```

### 1. Set up once: connect your models

```python
import os
from memwise import (LLMExtractor, MemoryConfig, MemoryManager, MemoryStore,
                     OpenAICompatible, OpenAICompatibleEmbedder)

provider = dict(base_url="https://api.your-provider.com/v1",   # see the table below
                api_key=os.environ["LLM_API_KEY"])             # leave out for local servers

llm = OpenAICompatible(model="your-chat-model", **provider)
embedder = OpenAICompatibleEmbedder(model="your-embedding-model", **provider)

memory = MemoryManager(
    MemoryStore("memory.db"),
    extractor=LLMExtractor(llm),                 # understands any phrasing
    embedder=embedder,                           # finds memories by meaning
    config=MemoryConfig.for_embedding_model(),   # thresholds for real embedding scores
)
```

`OpenAICompatible` works with any provider or server that offers the `/chat/completions`
format, and `OpenAICompatibleEmbedder` with any that offers `/embeddings`. Most offer both, and
the two models can come from different providers. Some base URLs (check your provider's docs to
confirm, including whether it offers embeddings):

| Where the model runs | `base_url` |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` |
| Groq | `https://api.groq.com/openai/v1` |
| Mistral | `https://api.mistral.ai/v1` |
| OpenRouter (many models, one key) | `https://openrouter.ai/api/v1` |
| Ollama, on your machine | `http://localhost:11434/v1` |
| LM Studio, on your machine | `http://localhost:1234/v1` |

**Using a different SDK, or a provider with its own format?** An LLM is just a function that
takes a system prompt and a chat history and returns text. Write one and pass it in:

```python
def my_llm(system: str, messages: list[dict]) -> str:
    # messages: [{"role": "user" | "assistant", "content": "..."}, ...]
    # Call any SDK, local model or company gateway here, and return the reply text.
    return my_provider.generate(system=system, messages=messages)

memory = MemoryManager(MemoryStore("memory.db"), extractor=LLMExtractor(my_llm))
```

Without an LLM, the offline default (`RuleExtractor`) only understands a fixed set of English
phrasings. It's for tests and demos; use `LLMExtractor` in production.

### 2. Before every model call

```python
SYSTEM = "You are a helpful travel assistant."

def reply(user_id: str, history: list[dict], scope: str = "general") -> str:
    """history: this conversation so far, ending with the user's new message."""
    recall = memory.recall(user_id, history[-1]["content"], scope=scope)
    system = SYSTEM + "\n\n" + memory.build_context(recall)
    return llm(system, history)                   # or however you call your model today
```

`build_context()` returns plain text, so the reply can come from any model. It doesn't have to
be the one that extracts memories: a small, cheap model is usually enough for extraction.

### 3. When a conversation ends

Pass the whole conversation, the same `{"role", "content"}` list you send to your model:

```python
def on_conversation_end(user_id: str, session_id: str, history: list[dict]):
    memory.ingest(user_id, [{**turn, "ref": f"{session_id}:{i}"}
                            for i, turn in enumerate(history, 1)])
```

Memories only ever come from what the **user** said. The assistant's turns are context, so the
extractor understands short replies: "The second one, please." after "Celsius or Fahrenheit?".
`LLMExtractor` is also shown the user's stored memories (the most relevant 40, never sensitive
ones), so "I left last week" closes the stored employer, and "not into climbing anymore" retires
"rock climbing". Stored memories it merely repeats back don't count as confirmed.

You can also pass just the user's messages, as strings or `(source_ref, text)` pairs. Each
`source_ref` is stored with the memory it produced, so you can trace any memory back to its message.

Extracting once per conversation is cheaper and more accurate than extracting on every message.
If a user says "remember…", you can also call `ingest` right away for just that message
(see [`examples/chat.py`](examples/chat.py)).

### 4. Nightly

```python
memory.maintain_all()   # expire trips and projects, merge duplicates, prune unused episodes
```

### 5. Let users see and control their memory

Example with FastAPI. Any web framework works the same way.

```python
from fastapi import Depends, FastAPI

app = FastAPI()

@app.get("/me/memories")
def list_memories(user_id: str = Depends(current_user_id)):   # from your auth, never the request body
    return [{"id": m.id, "text": m.text, "since": m.valid_from} for m in memory.current(user_id)]

@app.delete("/me/memories/{memory_id}")
def delete_memory(memory_id: str, user_id: str = Depends(current_user_id)):
    return {"deleted": memory.forget(user_id, memory_id)}

@app.delete("/me/memories")
def delete_all(user_id: str = Depends(current_user_id)):
    return {"deleted": memory.forget_all(user_id)}
```

One `MemoryManager` can be shared by all request threads. Several worker processes can also
open the same SQLite file.

---

## API at a glance

| Call | When to use it |
|---|---|
| `MemoryManager(store, extractor=..., embedder=..., config=..., judge=..., rewriter=...)` | Once, at startup. `judge` and `rewriter` are optional |
| `LLMExtractor(llm, system_prompt=..., scopes=...)` | Lets any model extract memories. `llm` is a function `(system, messages) -> str`. `system_prompt` and `scopes` are optional: your own extraction instructions and scope names (see [Use your own extraction prompt](#use-your-own-extraction-prompt)) |
| `LLMJudge(llm)` | Optional. Lets a model say when a new memory replaces stored ones the rules can't link ("Went vegetarian" vs "Loves steak"); see [Let a model judge what the rules can't see](#let-a-model-judge-what-the-rules-cant-see) |
| `LLMQueryRewriter(llm)` | Optional. Lets a model write search queries for each request, so recall finds what answers it in other words ("Would she enjoy camping?" finds "I love hiking"); see [Let a model rewrite the request](#let-a-model-rewrite-the-request) |
| `OpenAICompatible(base_url=..., model=..., api_key=...)` | A ready-made `llm` for any `/chat/completions` API |
| `OpenAICompatibleEmbedder(base_url=..., model=..., api_key=...)` | A ready-made embedder for any `/embeddings` API |
| `MemoryConfig.for_embedding_model()` | Thresholds to start from with a real embedding model |
| `ingest(user_id, messages)` | After a conversation. `messages` = the chat history (`{"role", "content"}` turns), or the user's messages as strings or `(source_ref, text)` pairs |
| `add(user_id, candidates)` | Write memories you built yourself (a settings form, an import) |
| `recall(user_id, message, scope="general")` | Before each model call. Returns a `Recall`. `include_past=True` also finds memories that are no longer true, for questions about the past ("where did I live before?") |
| `build_context(recall)` | The memory section for your system prompt (`""` if there's nothing to add). `about="the project"` for memory that isn't about a person |
| `current(user_id)` | "What do you know about me?" |
| `history(user_id)` | Everything ever stored, including replaced and expired memories |
| `forget(user_id, memory_id)` | Delete one memory and its older versions, and scrub it from the log |
| `forget_all(user_id)` | Delete everything about a user |
| `maintain(user_id)` / `maintain_all()` | Upkeep. Run nightly |
| `reembed(user_id=None)` | After switching to a different embedder |
| `store.logs(user_id)` | Audit log: every ADD / SUPERSEDE / EXPIRE / FORGET, with its reason |

A `Recall` has `.profile` (always-loaded memories), `.items` (one decision per retrieved memory,
with `.action`, `.reason` and `.scored.memory`), and `.used` (everything except HOLD). With
`include_past=True`, `.past` has the no-longer-true memories, each PAST or HOLD.
`build_context` lists the PAST ones in a section of their own, "No longer true", with when
each was true and what replaced it. With `MemoryConfig.keep_messages`, `.sources` has the
message each shown memory came from, and `.messages` other kept messages that match (see
[Keep what the user said](#keep-what-the-user-said-word-for-word)). With a rewriter,
`.searched` has the queries it searched with besides the request.

Every decision returned by `ingest()` and `add()` has `.op` (ADD, CONFIRM, SUPERSEDE, RETRACT,
FORGET or IGNORE), `.reason`, `.target` (the stored memory it acts on), `.result` (the memory
written) and `.also` (more memories it closed, when a judge said a change makes several untrue).

---

## Customize

### Change the rules

```python
from memwise import MemoryConfig, MemoryManager

config = MemoryConfig(
    stale_after_days=90,          # ask "is this still true?" sooner
    top_k=10,                     # retrieve more memories per message
    prune_episodes_after_days=365,
)
config.single_valued.add("timezone")          # a user has one timezone at a time
config.profile_predicates.add("preferred_language")  # include in every prompt
config.safety_predicates.add("medication")    # never held back, like allergies

memory = MemoryManager(store, config=config)
```

Every threshold is documented in [`memwise/config.py`](memwise/config.py).

### Embeddings: how memories are found

The built-in `HashingEmbedder` keeps the library dependency-free and works offline, but it only
links words from a short hand-written list. In production, use a real embedding model:
`OpenAICompatibleEmbedder` (see [Set up once](#1-set-up-once-connect-your-models)), or any
object with an `embed(text) -> list[float]` method:

```python
class MyEmbedder:
    def embed(self, text: str) -> list[float]:
        return my_provider.embed(text)    # any SDK or local model; return an L2-normalized vector

memory = MemoryManager(store, embedder=MyEmbedder(), config=MemoryConfig.for_embedding_model())
```

Give it an `embed_many(texts)` method as well, and `reembed()` will send whole batches.

**Tune the thresholds for your model.** Every embedding model scores similarity on its own
scale. `MemoryConfig.for_embedding_model()` is a starting point measured on a typical hosted
model. To check yours:

```bash
MEMWISE_EMBED_MODEL="your-embedding-model" python examples/tune_embedder.py
```

It shows how your model scores related and unrelated memories, and suggests values for
`min_relevance`, `relevant_enough`, `sensitive_relevant_enough` and `duplicate_similarity`.
Keep `duplicate_similarity` high: "Allergic to peanuts" and "Allergic to tree nuts" can score
0.8, and merging them would lose an allergy.

**Switching embedders on an existing database?** Call `memory.reembed()` once. Until you do,
`recall()` raises an error instead of comparing vectors from two different models.

**If location matters to most requests,** load it into every prompt. "A dinner spot near me" is
hard to link to "Lives in Lisbon" by meaning alone: `config.profile_predicates.add("lives_in")`.

### Keep what the user said, word for word

Extraction keeps what matters and drops the rest. When your product needs the details too
("which book did I say I read last year?"), turn on kept messages:

```python
config = MemoryConfig.for_embedding_model(keep_messages=True, message_top_k=10)
```

Then `ingest()` also keeps each of the user's messages, and `recall()` returns two more things:

- `.sources`: for each memory it shows, the message it came from. `build_context` puts it under
  the memory, as `said on Thursday 2023-05-25: "..."`.
- `.messages`: up to `message_top_k` other kept messages that match, found by meaning, by
  shared keywords (a name or a title that embeddings blur), and by the date the message names
  ("what did I say on 13 October 2023?", "in May 2023"). `build_context` lists them, dated,
  under "What the user said before".

Kept messages are shown as they were said, with relative dates worked out: on Monday 17 July
2023, "we went camping last weekend" is shown as "last weekend [the weekend before Mon 17 Jul
2023: Sat 15 Jul - Sun 16 Jul]", so the model doesn't have to count days (it often gets them a
week off). `memwise.dates` does this in code, for English phrases with one clear meaning
("yesterday", "last Friday", "two weeks ago", "next month"), and leaves vague ones ("recently",
"on Sunday") alone. `LLMExtractor` gets the same dates, so memories are dated correctly too.

It stores more about the user, so it's off by default. Secrets are redacted before a message is
kept. When consent is required, a message that the write gate refused a sensitive memory from
isn't kept, and neither is "please forget that ...". Kept messages follow the scope and
sensitivity rules of the memories taken from them. `forget()` deletes the messages a memory
came from and those that name it, and `forget_all()` deletes them all. A model can miss a
sensitive detail, so turn this on where keeping chat history is acceptable.

### Use your own extraction prompt

The built-in prompt is written for a personal assistant. When your product cares about other
things, pass your own instructions to `LLMExtractor`. The easiest way is to add to the built-in
prompt:

```python
from memwise import LLMExtractor, MemoryConfig
from memwise.llm import EXTRACTION_SYSTEM

STORE_NOTES = """
This is the support assistant of an online kitchenware store. Also keep the customer's sizes
(predicates apron_size and glove_size) and their preferred delivery slot (delivery_slot).
"""

extractor = LLMExtractor(llm, system_prompt=EXTRACTION_SYSTEM + STORE_NOTES,
                         scopes=("general", "orders", "personal"))

config = MemoryConfig.for_embedding_model()
config.single_valued |= {"apron_size", "glove_size", "delivery_slot"}   # a new value replaces the old
```

- **`system_prompt`** replaces the extraction instructions. The model still has to reply in the
  same JSON shape, `{"memories": [...]}`, with `kind` set to fact, preference, instruction or
  episode, plus a `predicate` and a `value`. Items it can't use are skipped, and missing optional
  fields get defaults.
- **`scopes`** are the contexts your product uses. A memory with any other scope is filed as
  "general". Pass the same names to `recall(user_id, message, scope="orders")`.
- **New predicates** that hold one value at a time go in `config.single_valued`, as above.
  Otherwise a new size is added next to the old one instead of replacing it. Projects
  (`current_project`) are not on that list, because people work on several at once: one ends
  when the user says it's finished or dropped, when its expiry passes, or when the judge says a
  new one replaces it. `config.single_valued.add("current_project")` keeps one at a time.
- For a completely different job, write the whole prompt yourself.
  [`examples/git_commits.py`](examples/git_commits.py) does this to remember a repository instead
  of a person.

### Let a model judge what the rules can't see

The rules compare a new memory with stored ones that share its predicate: one city at a time,
"dislikes X" cancels "likes X". They can't link different predicates, so "Went vegetarian" is
added next to "Loves steak", and "Runs 10k every morning" only confirms "Runs 5k every morning".
An optional judge asks a model about those cases:

```python
from memwise import LLMExtractor, LLMJudge, MemoryManager

memory = MemoryManager(store, extractor=LLMExtractor(llm), embedder=embedder, config=config,
                       judge=LLMJudge(llm))
```

The model answers ADD, UPDATE (the new memory replaces the ones it names), NONE (already known)
or DELETE (for "no longer true"). memwise keeps the decision:

- **It's only asked where the rules can't tell:** when they would add a memory, confirm one that
  is worded differently, replace one, or find nothing to retract, and similar memories are
  stored. Forget requests, secrets, sensitive details and allergies never reach it. When a rule
  replaces a memory (a new city, a new diet), that stands: the judge may only close other
  memories the change makes out of date ("Is vegan" also closes "Likes cheese").
- **Its answer is checked.** It may only name memories it was shown that are still current. It
  never sees sensitive or safety memories, so it can't close them. Anything else falls back to
  the rules.
- **Nothing is deleted.** A replaced memory is closed and kept as history, like every other
  SUPERSEDE. One change can close several memories ("Went vegan" closes "Loves steak" and
  "Likes cheese"), listed in the decision's `.also`.
- **One extra model call per conversation, at most,** made outside the manager's lock. If it
  fails, a warning is logged and the rules decide.

Change what it's shown with `config.judge_related` (how many similar memories, default 5) and
`config.judge_min_similarity`, or pass your own instructions with
`LLMJudge(llm, system_prompt=...)`. Any object with a `judge(requests)` method works too.

### Let a model rewrite the request

Recall finds memories that mean the same thing as the request, or share its rare words. Some
requests don't sound like what answers them: "Would she enjoy a weekend outdoors?" is answered by
"Loves hiking in the mountains", and "What pets does he have?" by "We adopted a puppy". An
optional rewriter asks a model what to search for:

```python
from memwise import LLMExtractor, LLMQueryRewriter, MemoryManager

memory = MemoryManager(store, extractor=LLMExtractor(llm), embedder=embedder, config=config,
                       rewriter=LLMQueryRewriter(llm))
```

- **Recall searches with the request and up to three queries the model writes** (in
  `Recall.searched`). A memory's relevance is its best match over all of them, and with kept
  messages each query adds its own rankings.
- **Sensitive memories still have to match the request itself.** A query the model writes can
  find a memory, but it can't bring up a sensitive one the request alone wouldn't have.
- **One extra model call per new request,** made outside the manager's lock. Answers are cached,
  and if the call fails, a warning is logged and recall searches with the request alone. It adds
  the model's response time to every new request, so it suits questions about the past more than
  live chat.

Pass your own instructions with `LLMQueryRewriter(llm, system_prompt=...)`, or any object with a
`rewrite(request) -> list[str]` method.

### Write your own extractor

`LLMExtractor` covers most cases. For full control over the model call (structured output,
retries, batching), write your own: any object with `extract(messages) -> list[Candidate]`,
where `messages` is a list of `(source_ref, text)` pairs. Reuse the built-in prompt, JSON
schema and parser:

```python
from memwise.llm import EXTRACTION_SCHEMA, EXTRACTION_SYSTEM, parse_memories

class MyExtractor:
    def extract(self, messages):
        transcript = "\n".join(f"[{ref}] {text}" for ref, text in messages)
        reply = call_my_model(EXTRACTION_SYSTEM, transcript, json_schema=EXTRACTION_SCHEMA)
        return parse_memories(reply)      # forgiving: handles prose, code fences, loose types

memory = MemoryManager(store, extractor=MyExtractor())
```

The extractor only **proposes** memories. Storing, replacing and forgetting are always done by
the library's own rules, which are tested.

### Write memories directly

```python
from memwise import Candidate

memory.add(user_id, Candidate(kind="fact", predicate="lives_in", value="Lisbon",
                              text="Lives in Lisbon", explicit=True))
```

### Scopes: keep contexts apart

Memories have a `scope`: `"general"`, `"work"`, `"personal"`, or your own strings. Pass the
current context to `recall(..., scope="personal")`. Memories from another context are held
back and never reach the prompt.

---

## Examples

| File | What it shows |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | The whole API on one page |
| [`examples/demo.py`](examples/demo.py) | A guided story over 8 months of simulated time: write gate, superseding, expiry, staleness, forgetting, audit log |
| [`examples/chat.py`](examples/chat.py) | An interactive chat with memory: type `/context` to see the prompt, `/log` to see every decision, `/advance 200` to jump ahead in time |
| [`examples/tune_embedder.py`](examples/tune_embedder.py) | How your embedding model scores memories, and the thresholds to use with it |
| [`examples/git_commits.py`](examples/git_commits.py) | Memory for a repository instead of a person: reads git commits and keeps the project's decisions, rules and lessons, with history ("Jest, then Vitest") and a link to each commit. `--ask "How should I cache sessions?"` answers with it |

```bash
python examples/demo.py
python examples/chat.py --offline
```

To chat with a real model, point the examples at any `/chat/completions` API first:

```bash
export MEMWISE_BASE_URL="http://localhost:11434/v1"   # e.g. Ollama on your machine
export MEMWISE_MODEL="llama3.2"
export MEMWISE_API_KEY="..."                          # only if your provider needs one
export MEMWISE_EMBED_MODEL="..."                      # optional: a real embedding model
python examples/chat.py
```

The embedding model uses the same base URL and key unless you also set `MEMWISE_EMBED_BASE_URL`
and `MEMWISE_EMBED_API_KEY`.

### Not just people: memory for a repository

The same machinery works for things other than a person. `examples/git_commits.py` reads a
repo's commits and keeps what a new developer should know. Only the extraction prompt changes:

```bash
python examples/git_commits.py path/to/repo --ask "How should I cache user sessions?"
```

```text
What shopfront knows (15):
  episode     Redis ran out of memory under peak production load; sessions returned to PostgreSQL  [84054161ee, 2025-02-21]
  fact        Uses PostgreSQL as the main database; orders and payments need transactions  [57eccb9e5b, 2025-01-08]
  fact        Uses Vitest for tests; it runs faster and works with the ESM build  [7fdb4cdd1a, 2025-05-20]
  instruction Never commit .env files; secrets are loaded from AWS Secrets Manager at startup  [3d2f49a8be, 2025-04-02]
  ...
No longer true (kept as history):
  Uses Jest for unit tests  [2025-01-10 to 2025-05-20]
  Hosted on Fly.io  [2025-03-03 to 2025-06-16]

> How should I cache user sessions?
Use PostgreSQL for sessions. Redis ran out of memory at peak load, and the project decided not to
bring Redis back until a properly sized cluster is available.
```

Routine commits ("fix typo", "bump version") leave no memory, a commit telling "AI assistants"
what to remember is ignored, and each run reads only new commits. The memory is kept inside
`.git`, so it is never committed. It stores conclusions, not history: to find which commit
changed a file, use `git log`. Add `--judge` to let a model catch replacements filed under
different names, such as "Uses Django-Q instead of Celery" next to "Processes photos with Celery".

## Test it with your model

The unit tests in `tests/` check the library's rules with a fake model. To check the whole
system with *your* model, run the scenario evals: 24 realistic multi-session conversations
(facts changing, "I quit coffee", sensitive details said in passing, instructions hidden in
pasted text, other people's facts, hypotheticals, trips that expire, work vs personal context,
stale facts, other languages, "please forget that…", short replies to the assistant's
questions, "I left last week", allergies), each with pass/fail checks on what was stored, what
reached the prompt and, for some, the reply.

```bash
python evals/scenarios.py              # needs MEMWISE_BASE_URL, MEMWISE_MODEL (and MEMWISE_API_KEY)
python evals/scenarios.py --repeat 3   # three runs each, to find flaky behaviour
```

Models don't answer the same way every time, so run with `--repeat` before trusting a result.
Add scenarios from your own product to [`evals/scenarios.py`](evals/scenarios.py), or start a
file for your product like [`evals/support_bot.py`](evals/support_bot.py): 10 scenarios for a
store's support assistant (customers kept apart, card numbers refused, a damaged delivery that
shapes the next order, an address change, fraud in a pasted email, "delete my data").

```bash
python evals/support_bot.py --repeat 3
```

[`evals/judge.py`](evals/judge.py) checks the optional judge: changes the rules can't see ("Went
vegetarian", "now 10k runs", "moved to the platform team"), and things it must keep (tea *and*
coffee, a general preference next to a specific favourite, a preference after one bad
experience). Run the other evals with `--judge` to check it changes nothing else.

```bash
python evals/judge.py --repeat 3       # --no-judge runs the same scenarios with the rules alone
python evals/scenarios.py --judge
```

### Benchmark: LoCoMo

[`evals/locomo.py`](evals/locomo.py) runs memwise on [LoCoMo](https://github.com/snap-research/locomo),
the long-conversation benchmark that Mem0 and others quote. It has 10 chats between two friends,
each spanning months (19 to 32 sessions), and 1,540 questions about them, plus 446 adversarial
ones. Each friend gets their own memory, a model answers from what memwise recalls, and the
same model grades the answer.

**Result:** 89.7 out of 100 on all 1,540 questions, with about 4,900 tokens of memory per
question.

**What it saves:** sending the whole conversation instead takes 22,069 tokens per question. On
the same 382 questions, memwise uses **78% fewer tokens** (4,809) and scores 85.3, against 88.2
for the whole conversation. Models bill input by the token, so the memory part of each call costs
78% less: at an example $1 per million input tokens, 1,000 questions cost $4.81 instead of
$22.07. Against Mem0's published 92.5 at 6,956 tokens per query, memwise uses 30% fewer tokens
and scores 2.8 points lower (Mem0 used its own models and grader, so this isn't a controlled
comparison).

The complete results, with charts by question type, are on the
[results page](docs/benchmarks.html) (download it and open it in a browser). See also
[Benchmark results](#benchmark-results) above.

The data isn't included, so download it once (2.8 MB):

```bash
mkdir -p evals/data
curl -L -o evals/data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
python evals/locomo.py --conv conv-26      # one conversation; leave --conv out for all 10
python evals/locomo.py --prompt friends --store-sensitive --past --keep-messages --top-k 10 --message-top-k 30
```

Options:
- `--baseline` also answers with the whole conversation in the prompt, for comparison.
- `--prompt friends` adds lines for chats between friends to the built-in prompt.
- `--store-sensitive` stores sensitive details without asking.
- `--past` recalls memories that are no longer true as well, as history.
- `--keep-messages` keeps each friend's lines word for word (`MemoryConfig.keep_messages`), and
  `--message-top-k N` sets how many matching lines are recalled per friend (default 10).
- `--rewrite` has the model write search queries for each question (`LLMQueryRewriter`).
- `--per-category N` runs a sample: up to N questions of each type per conversation.
- `--memory NAME` reopens a memory built before (a folder in `evals/data/locomo-memory/`), to
  compare recall settings on the same memory after the extraction prompt changed.
- `--show-wrong N` prints wrong answers with the memories they had.

Every answer is saved in `evals/data/locomo-results/`, with a `.summary.json` of the scores.
[`evals/report.py`](evals/report.py) turns summaries into a results page like
[docs/benchmarks.html](docs/benchmarks.html): memwise's score and context size next to the whole
conversation in the prompt, by question type, and what it saves. `--price` sets the example price
per million input tokens, and `--reference` adds a result someone else published.

```bash
python evals/report.py evals/data/locomo-results/*.summary.json --out benchmarks.html
```

## How it works

Read [docs/how-it-works.md](docs/how-it-works.md) for the design: the data model, the write and
read paths, maintenance, deletion, and how memory differs from RAG.

## Privacy and safety defaults

- **Consent for sensitive data.** Health and other sensitive details are only stored if the user
  explicitly asks. They are never part of the always-loaded profile.
- **Secrets are never stored.** Card numbers, security codes, passwords, PINs, API keys, bank
  account and ID numbers are refused by the write gate even when the user says "remember my card",
  and are redacted from the audit log. This doesn't rely on the model: it's a check in memwise
  itself, tuned to keep lookalikes such as "Never commit API keys" or "Visa card ending 4242".
- **Held-back memories are never sent.** HOLD items aren't in the prompt at all, so the model
  can't leak them.
- **Allergies are never held back.** Similarity scores can't reliably tell "dinner ideas" from
  "write a cover letter" for "Allergic to peanuts", and in testing, a held-back allergy led a
  model to suggest peanut noodles. So allergies (`MemoryConfig.safety_predicates`) go into every
  prompt, in every scope, with an instruction to respect them and only bring them up when they
  matter. Add your own safety predicates, or pass an empty set to turn this off.
- **Deletion is complete.** `forget` hard-deletes the memory and its older versions, and scrubs
  it from the audit log. `forget_all` removes everything about the user. With kept messages,
  `forget` also deletes the messages the memory came from and those that name it.
- **Nothing is kept word for word unless you ask.** `MemoryConfig.keep_messages` is off by
  default. When it's on, secrets are redacted first, and a message is kept under the same
  consent, scope and sensitivity rules as the memories taken from it.
- **Protection against memory poisoning.** The extraction prompt treats conversation text as
  data, not instructions, so pasted text can't plant memories.
- **Your data stays with you.** Memories live in your own SQLite file. Nothing is sent anywhere
  unless you connect an LLM, and then only to the model you chose. With a local model
  (Ollama, LM Studio), nothing leaves your machine.

## Status and limitations

This is an early release (0.5). Things to know before relying on it:

- **`RuleExtractor`** only understands a fixed list of English phrasings. Use it for tests and
  demos, and an LLM extractor in production.
- **`HashingEmbedder`** is a stand-in. Use a real embedding model in production, and tune the
  thresholds for it with `examples/tune_embedder.py`.
- **SQLite only.** It works for one server, or several processes on one machine. For many servers,
  a Postgres store would be the next step (the schema maps over directly).
- **Search is a scan.** Recall compares the query with every memory of the user (and every kept
  message, with `keep_messages`) in Python. That's quick for hundreds or a few thousand per user;
  for long histories, a vector index (pgvector, sqlite-vec) would be the next step.
- **Extraction quality depends on your model.** Small models sometimes skip fields or return
  loose JSON. The parser forgives that, but try your model on real conversations before
  shipping. The built-in client is tested against a local fake server, not a live provider.

## Roadmap

- Postgres + pgvector store
- An LLM judge for contradictions between free-text memories (done in 0.4.0 as `LLMJudge`; see
  [Let a model judge what the rules can't see](#let-a-model-judge-what-the-rules-cant-see))
- Condensing old episodes into facts during maintenance
- An async API

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
