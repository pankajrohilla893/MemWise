"""LongMemEval benchmark: about 50 past chats with an assistant, then one question about them.

LongMemEval (Wu et al., 2025; github.com/xiaowu0162/LongMemEval) has 500 questions. Each comes
with its own history of 38-62 chat sessions between a user and an assistant (about 115k tokens),
and the few sessions that hold the answer are hidden among the others. The questions are of six
types: single-session-user, single-session-assistant, single-session-preference, multi-session,
knowledge-update and temporal-reasoning. 30 of them (ids ending in _abs) ask about something
that was never said, and the right answer is that it can't be known. Download the cleaned S file
once (277 MB, MIT licence; it isn't redistributed with memwise):

    mkdir -p evals/data/longmemeval
    curl -L -o evals/data/longmemeval/longmemeval_s_cleaned.json \\
        https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

Then, with the same MEMWISE_* settings as evals/locomo.py:

    python evals/longmemeval.py --per-type 10              # a pilot: 10 per type, spread across each type
    python evals/longmemeval.py --per-type 10 --baseline   # also answer with the whole history in the prompt
    python evals/longmemeval.py                            # all 500 (about 24,000 sessions to build)
    python evals/longmemeval.py --question e47becba        # one question (repeatable)

How memwise is run (the set-up that did best on LoCoMo):
- Each question gets its own memory, built from its sessions in date order, with the clock set
  to each session's date. Turns go in as they are: the user's lines are the user's, the
  assistant's are context (memwise remembers what the user says, not what it was told).
  User lines longer than 12,000 characters (a pasted document or log) are cut, because the
  embedding model takes at most about 8,000 tokens.
- memwise's own extraction prompt; sensitive details are stored without asking (--consent asks,
  memwise's default); what the user said is kept word for word (MemoryConfig.keep_messages).
  --assistant-top-k N also keeps the assistant's replies, in pieces, and recalls up to N of them
  (MemoryConfig.keep_assistant_messages): the single-session-assistant questions need them.
- The question is asked on its own date. The memory is recalled with the question: memories,
  memories that are no longer true (as history; --no-past leaves them out) and other kept lines
  that match are listed with their chat's date, and the model answers from that. It thinks
  first, then answers on a last "Answer:" line, which is graded. --baseline uses the same
  prompt with every session in it instead, the assistant's replies included.
- The same model grades each answer against the gold one, with the benchmark's rule for its
  type: off-by-one day counts are accepted for temporal questions; for knowledge updates, the
  latest value is the answer; preference questions are graded against a description of the
  reply the user wants; an _abs question is answered right by saying it can't be known.

Each memory is built once and kept in evals/data/longmemeval-memory/ (--rebuild redoes them).
Every answer is saved to evals/data/longmemeval-results/ with what it was given, next to a
.summary.json of the scores (the format evals/report.py reads). A whole history is about 120k
tokens, so few of those calls run at once (--full-workers), and questions the model API failed
on are asked again after a pause. --resume FILE.jsonl keeps an earlier run's answers and asks
only the ones that failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from locomo import Metered, SessionClock, f1, grade  # noqa: E402  (the LoCoMo harness's helpers)
from memwise import (LLMError, LLMExtractor, MemoryConfig, MemoryManager, MemoryStore,  # noqa: E402
                     OpenAICompatible, OpenAICompatibleEmbedder)
from memwise.dates import with_dates  # noqa: E402
from memwise.llm import EXTRACTION_SYSTEM  # noqa: E402

DATA = HERE / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
MEMORY_DIR = HERE / "data" / "longmemeval-memory"
RESULTS_DIR = HERE / "data" / "longmemeval-results"
TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning"]
USER = "user"
USER_CHARS = 12_000   # longer user lines are cut before they're ingested
SHOWN_CHARS = 1_500   # a kept line is shown up to this long
INGEST_FORMAT = "1"   # bump when Memory.build or memwise changes what is built

ANSWER_SYSTEM = """You are a chat assistant, and the user is asking you about your past chats with
them. Today is {today}.

{knowledge}

A relative date ("yesterday", "last week") is relative to the date of the chat it comes from: work
it out from that date, or use the dates that follow it in [brackets]. When something changed over
time, the latest is what's true now; a memory marked "no longer true" was true until the date
given, for questions about the past. For "how many" questions, list each one you find, then
count. For "how long" or "how many days" questions, work it out from the dates step by step.

First think briefly: which lines answer the question. Then give the final answer on a last line
that starts with "Answer:", as short as it can be (a name, number, date or short phrase). If
nothing above tells you the answer, say so on that line: "Answer: I don't know; you haven't told
me ...". If the user asks for a recommendation, tips or advice, the Answer line is that advice,
in one or two sentences built on what they've told you: name the things of theirs it uses (what
they have, use, plan, like or have already tried) and fit it to them, rather than general tips."""

GRADE_END = "Write one short sentence of reasoning, then a last line that is exactly 'Answer: CORRECT' or 'Answer: WRONG'."
GRADE_BASE = """You grade answers in a memory benchmark. You get a question, the gold answer and a
predicted answer. The prediction is CORRECT if it gives the gold answer, even in other words, inside
a longer reply, or with the steps that lead to it. It is WRONG if it gives a different answer, gives
only part of what the gold answer needs, or says it doesn't know."""
GRADERS = {
    "default": f"{GRADE_BASE}\n{GRADE_END}",
    "temporal-reasoning": f"""{GRADE_BASE} A count of days, weeks or months that is off by one is still
CORRECT (18 days for a gold answer of 19 days), and so is any other answer the gold one accepts.
{GRADE_END}""",
    "knowledge-update": f"""{GRADE_BASE} The user's details changed over time: a prediction that also
mentions older values is CORRECT as long as the one it gives as current is the gold answer.
{GRADE_END}""",
    "single-session-preference": f"""You grade answers in a memory benchmark. You get a request, a
description of the reply the user would want (as the gold answer) and a predicted reply. The
prediction is CORRECT if it fits that description: it uses what the user has said about themselves,
and gets it right. It needn't cover every point of the description. It is WRONG if it is generic,
ignores or gets the user's details wrong, or says it doesn't know.
{GRADE_END}""",
    "abstention": f"""You grade answers in a memory benchmark. You get a question that can't be answered
from the user's past chats, an explanation of why (as the gold answer) and a predicted answer. The
prediction is CORRECT if it says it can't answer: that it doesn't know, the user never said, or the
information isn't there (it may mention related things the user did say). It is WRONG if it gives
an answer as if it were known.
{GRADE_END}""",
}


def when(written: str) -> datetime:
    return datetime.strptime(written, "%Y/%m/%d (%a) %H:%M").replace(tzinfo=timezone.utc)


def _day(at: datetime) -> str:
    return f"{at:%A} {at.day} {at:%B %Y}"


def is_abstention(item: dict) -> bool:
    return item["question_id"].endswith("_abs")


def sessions(item: dict) -> list[tuple[str, datetime, list[dict]]]:
    """(id, date, turns) for each session, oldest first."""
    out = [(sid, when(d), turns) for sid, d, turns in
           zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"])]
    return sorted(out, key=lambda s: s[1])


def transcript(item: dict) -> str:
    parts = []
    for n, (_, at, turns) in enumerate(sessions(item), 1):
        parts.append(f"Chat {n}, {_day(at)} {at:%H:%M}:\n"
                     + "\n".join(f"{t['role']}: {t['content']}" for t in turns))
    return "\n\n".join(parts)


# =========================================================================== memory

class Memory:
    """The memory one question's history builds."""

    def __init__(self, path: Path, extractor: LLMExtractor, embedder: Any, config: MemoryConfig):
        self.clock = SessionClock()
        self.mm = MemoryManager(MemoryStore(str(path)), extractor=extractor, clock=self.clock,
                                embedder=embedder, config=config)

    def build(self, item: dict) -> Counter:
        ops: Counter = Counter()
        for sid, at, turns in sessions(item):
            self.clock.at = at
            chat = []
            for i, t in enumerate(turns):
                if t["role"] == "user":
                    chat.append({"role": "user", "content": t["content"][:USER_CHARS], "ref": f"{sid}:{i}"})
                else:
                    chat.append({"role": "assistant", "content": t["content"], "ref": f"{sid}:{i}"})
            ops.update(d.op for d in self.mm.ingest(USER, chat))
        return ops


def open_memories(items: list[dict], args, extract_llm: Metered, embedder: Any,
                  config: MemoryConfig) -> tuple[dict[str, Memory], dict, Path]:
    """Build (or reopen) each question's memory. Returns them, what building did, and where."""
    key = "_".join([extract_llm.model, getattr(embedder, "model", "builtin"),
                    "consent" if args.consent else "sensitive",
                    *(["messages"] if config.keep_messages else []),
                    *([f"assistant{config.assistant_piece_chars}"] if config.keep_assistant_messages else []),
                    hashlib.sha1((EXTRACTION_SYSTEM + INGEST_FORMAT + str(USER_CHARS)).encode()).hexdigest()[:8]])
    folder = MEMORY_DIR / re.sub(r"[^A-Za-z0-9.-]+", "-", key)
    folder.mkdir(parents=True, exist_ok=True)
    extractor = LLMExtractor(extract_llm, scopes=("general",))
    count = {"done": 0}

    def build(item):
        path = folder / f"{item['question_id']}.db"
        done = path.with_suffix(".done.json")
        if args.rebuild or not done.exists():
            for attempt in range(3):  # a slow or busy API times out now and then: build it again
                for p in (path, done, Path(f"{path}-wal"), Path(f"{path}-shm")):
                    p.unlink(missing_ok=True)
                mem = Memory(path, extractor, embedder, config)
                start = time.time()
                try:
                    ops = mem.build(item)
                    break
                except LLMError as e:
                    mem.mm.store.close()
                    if attempt == 2:
                        raise
                    print(f"  building {path.stem} again (the model API failed: {str(e)[:80]})", flush=True)
            done.write_text(json.dumps({"ops": ops, "seconds": round(time.time() - start, 1)}))
            count["done"] += 1
            print(f"  built {count['done']}: {item['question_id']} ({item['question_type']}) "
                  f"in {time.time() - start:.0f}s", flush=True)
        else:
            mem = Memory(path, extractor, embedder, config)
        return item["question_id"], mem, json.loads(done.read_text())

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        built = list(pool.map(build, items))
    return {k: m for k, m, _ in built}, {k: info for k, _, info in built}, folder


def known(mem: Memory, question: str, past: bool) -> tuple[list[str], int, set[str]]:
    """What memwise lets through for this question (as evals/locomo.py shows it), and the refs
    of the assistant's replies shown."""
    r = mem.mm.recall(USER, question, record_usage=False, include_past=past)
    lines = []

    def said(m) -> None:
        if m.id in r.sources:
            s = r.sources[m.id]
            lines.append(f'    said: "{shown(s.text, s.at)}"')

    for m in list(r.profile) + [d.scored.memory for d in r.used]:
        lines.append(f"- {m.text} (chat of {_day(m.valid_from)})")
        said(m)
    for d in r.past:
        if d.action == "PAST":
            m = d.scored.memory
            ended = m.valid_to or m.expires_at
            lines.append(f"- {m.text} (chat of {_day(m.valid_from)}; no longer true"
                         + (f" from {_day(ended)})" if ended else ")"))
            said(m)
    if r.messages:
        lines.append("Other things the user said that may be relevant:")
        for x in sorted(r.messages, key=lambda x: x.message.at):
            lines.append(f'- [chat of {_day(x.message.at)}] "{shown(x.message.text, x.message.at)}"')
    if r.assistant_messages:
        lines.append("Things you told the user before (your own replies) that may be relevant:")
        for x in sorted(r.assistant_messages, key=lambda x: x.message.at):
            lines.append(f'- [chat of {_day(x.message.at)}] "{shown(x.message.text, x.message.at)}"')
    replies = {x.message.ref.split("#")[0] for x in r.assistant_messages}
    return lines, len(r.items) + len(r.past) + len(r.messages) + len(r.assistant_messages), replies


def shown(text: str, at: datetime) -> str:
    if len(text) > SHOWN_CHARS:
        text = text[:SHOWN_CHARS] + " …"
    return with_dates(text, at)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\s*\[[^\]]*\]", "", text)).strip().lower()


# =========================================================================== answering

FULL_SLOTS = threading.BoundedSemaphore(3)  # set from --full-workers


class _NoLimit:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


NO_LIMIT = _NoLimit()

def ask(job, memories: dict[str, Memory], answer_llm: Metered, full_llm: Metered, grade_llm: Metered,
        past: bool) -> dict:
    item, mode = job
    qid, qtype = item["question_id"], item["question_type"]
    gold = str(item["answer"])
    asked = when(item["question_date"])
    row: dict[str, Any] = {"question_id": qid, "type": qtype, "abstention": is_abstention(item), "mode": mode,
                           "question": item["question"], "gold": gold, "question_date": item["question_date"]}
    try:
        if mode == "memwise":
            mem = memories[qid]
            mem.clock.at = asked
            start = time.perf_counter()
            lines, found, replies = known(mem, item["question"], past)
            row.update(recall_ms=round((time.perf_counter() - start) * 1000),
                       memories=sum(1 for ln in lines if ln.startswith("- ")), retrieved=found)
            knowledge = "What you remember of your chats with the user:\n" + ("\n".join(lines) or "(nothing)")
            row["knowledge"] = knowledge
            # how many of the user's lines that hold the answer made it into the context
            evidence = [t["content"] for _, _, turns in sessions(item) for t in turns
                        if t["role"] == "user" and t.get("has_answer")]
            k = _clean(knowledge)
            row["evidence"] = len(evidence)
            row["evidence_shown"] = sum(_clean(t)[:60] in k for t in evidence)
            # and the assistant's replies that hold it (a piece of the reply shown counts)
            told = [f"{sid}:{i}" for sid, _, turns in sessions(item) for i, t in enumerate(turns)
                    if t["role"] == "assistant" and t.get("has_answer")]
            row["assistant_evidence"] = len(told)
            row["assistant_evidence_shown"] = sum(ref in replies for ref in told)
            llm = answer_llm
        else:
            knowledge = "Your past chats with the user, oldest first:\n\n" + transcript(item)
            llm = full_llm
        today = f"{_day(asked)}, {asked:%H:%M}"
        with FULL_SLOTS if mode == "full" else NO_LIMIT:  # whole histories: a few at a time
            reply = llm(ANSWER_SYSTEM.format(today=today, knowledge=knowledge),
                        [{"role": "user", "content": item["question"]}]).strip()
        found_ans = re.findall(r"^\W*(?:final\s+)?answer\s*\**\s*:\s*\**\s*(.+)", reply, re.I | re.M)
        prediction = found_ans[-1].strip() if found_ans else reply
        rule = "abstention" if is_abstention(item) else qtype
        correct, reason = grade(grade_llm, GRADERS.get(rule, GRADERS["default"]), item["question"], gold, prediction)
        row.update(reply=reply, prediction=prediction, correct=correct, f1=round(f1(prediction, gold), 3),
                   grade_reason=reason, context_tokens=len(knowledge) // 4)
    except LLMError as e:
        row["error"] = str(e)
    except Exception as e:  # one broken question shouldn't lose the run
        row["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
    return row


# =========================================================================== report

NAMES = {"memwise": "memwise", "full": "whole history"}


def report(rows: list[dict], modes: list[str]) -> None:
    ok = [r for r in rows if "error" not in r]
    head = f"{'type':<27}{'n':>5}" + "".join(f"   {NAMES[m]:>14}" for m in modes)
    print(head)
    print("-" * len(head))

    def line(name: str, pick) -> None:
        cells, n = "", 0
        for m in modes:
            items = [r for r in ok if r["mode"] == m and pick(r)]
            n = max(n, len(items))
            cells += f"   {sum(r['correct'] for r in items) / len(items):>14.1%}" if items else f"   {'-':>14}"
        print(f"{name:<27}{n:>5}{cells}")

    for t in TYPES:
        line(t, lambda r, t=t: r["type"] == t)
    print("-" * len(head))
    line("overall", lambda r: True)
    line("  of which abstention", lambda r: r["abstention"])
    for m in modes:
        items = [r for r in ok if r["mode"] == m]
        if not items:
            continue
        tokens = sum(r["context_tokens"] for r in items) / len(items)
        extra = ""
        if m == "memwise":
            ms = sorted(r["recall_ms"] for r in items)
            ev = sum(r["evidence"] for r in items)
            told = sum(r.get("assistant_evidence", 0) for r in items)
            extra = (f", {sum(r['memories'] for r in items) / len(items):.1f} memories and lines, "
                     f"recall p50 {ms[len(ms) // 2]} ms; the user's answer lines shown: "
                     f"{sum(r['evidence_shown'] for r in items)} of {ev}"
                     + (f"; the assistant's answer replies shown: "
                        f"{sum(r.get('assistant_evidence_shown', 0) for r in items)} of {told}" if told else ""))
        print(f"context, {NAMES[m]}: ~{tokens:,.0f} tokens a question{extra}")
    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"{len(errors)} questions failed and are not scored, e.g.: {errors[0]['error'][:200]}")


def summarize(rows: list[dict], modes: list[str], items: list[dict], args, llm: Any, embedder: Any,
              config: MemoryConfig, out: Path) -> dict:
    """The run's scores in the form evals/report.py reads."""
    summary: dict[str, Any] = {
        "benchmark": "LongMemEval",
        "date": f"{datetime.now():%Y-%m-%d}",
        "label": args.label or "memwise (" + ", ".join(
            ["sensitive " + ("only when asked" if args.consent else "stored")]
            + ([] if args.no_past else ["past recall"])
            + ([f"messages kept, {config.message_top_k} recalled"] if config.keep_messages else [])
            + ([f"assistant replies kept, {config.assistant_message_top_k} recalled"]
               if config.keep_assistant_messages else [])) + ")",
        "settings": {"dataset": DATA.name, "sensitive": "only when asked" if args.consent else "stored",
                     "past": not args.no_past, "top_k": args.top_k, "keep_messages": config.keep_messages,
                     "message_top_k": config.message_top_k if config.keep_messages else None,
                     "assistant_top_k": config.assistant_message_top_k if config.keep_assistant_messages else None,
                     "per_type": args.per_type or None,
                     "chat_model": llm.model, "embed_model": getattr(embedder, "model", "built-in")},
        "conversations": [i["question_id"] for i in items],
        "total_conversations": 500,
        "questions": len(items),
        "results": str(out.relative_to(HERE.parent)),
        "modes": {},
    }
    for m in modes:
        ok = [r for r in rows if r["mode"] == m and "error" not in r]
        if not ok:
            continue
        entry: dict[str, Any] = {
            "n": len(ok),
            "errors": sum(1 for r in rows if r["mode"] == m and "error" in r),
            "score": round(100 * sum(r["correct"] for r in ok) / len(ok), 1),
            "f1": round(sum(r["f1"] for r in ok) / len(ok), 3),
            "mean_tokens": round(sum(r["context_tokens"] for r in ok) / len(ok)),
            "categories": {},
        }
        for t in TYPES + ["abstention"]:
            got = [r for r in ok if (r["abstention"] if t == "abstention" else r["type"] == t)]
            if got:
                entry["categories"][t] = {"n": len(got),
                                          "score": round(100 * sum(r["correct"] for r in got) / len(got), 1)}
        if m == "memwise":
            entry["mean_memories"] = round(sum(r["memories"] for r in ok) / len(ok), 1)
            ms = sorted(r["recall_ms"] for r in ok)
            entry["recall_ms_p50"] = ms[len(ms) // 2]
        summary["modes"][m] = entry
    return summary


def show_wrong(rows: list[dict], count: int) -> None:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["mode"] == "memwise" and r.get("correct") is False:
            by[r["type"]].append(r)
    for t in TYPES:
        if by[t]:
            print(f"\n== wrong, {t} ({len(by[t])})")
            for r in by[t][:count]:
                print(f"Q: {r['question']}\n   gold: {r['gold'][:200]}\n   got:  {r['prediction'][:200]}")
                for ln in r.get("knowledge", "").splitlines()[1:13]:
                    print(f"     {ln[:200]}")


def pick(data: list[dict], per_type: int) -> list[dict]:
    """With per_type, that many of each type, spread evenly across the type (so the sample isn't
    just the first ones written)."""
    if not per_type:
        return data
    by: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        by[item["question_type"]].append(item)
    out = []
    for t in TYPES:
        qs = by[t]
        k = min(per_type, len(qs))
        out += [qs[round(i * (len(qs) - 1) / max(1, k - 1))] for i in range(k)]
    return out


# =========================================================================== main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--question", action="append", default=[], help="a question id (repeatable)")
    parser.add_argument("--type", action="append", default=[], choices=TYPES,
                        help="only questions of this type (repeatable; with --per-type, the same ones)")
    parser.add_argument("--per-type", type=int, default=0,
                        help="this many questions of each type, spread across it (default: all 500)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N of the questions picked")
    parser.add_argument("--baseline", action="store_true",
                        help="also answer with the whole history in the prompt (~115k tokens a question)")
    parser.add_argument("--consent", action="store_true",
                        help="store sensitive details only when asked (memwise's default; off here, as on LoCoMo)")
    parser.add_argument("--no-past", action="store_true", help="don't recall memories that are no longer true")
    parser.add_argument("--top-k", type=int, default=10, help="memories recalled (default 10)")
    parser.add_argument("--message-top-k", type=int, default=30,
                        help="kept lines recalled (default 30; 0 doesn't keep the user's lines)")
    parser.add_argument("--assistant-top-k", type=int, default=0,
                        help="also keep the assistant's replies and recall up to this many pieces (default 0: off)")
    parser.add_argument("--rebuild", action="store_true", help="build the memories again instead of reusing them")
    parser.add_argument("--workers", type=int, default=8, help="model calls in parallel (default 8)")
    parser.add_argument("--full-workers", type=int, default=3,
                        help="with --baseline: whole-history calls in parallel (default 3; they're ~120k tokens)")
    parser.add_argument("--resume", type=Path, metavar="FILE.jsonl",
                        help="keep this earlier run's answers and ask only the questions that failed in it")
    parser.add_argument("--label", default="", help="a name for this set-up on the results page")
    parser.add_argument("--show-wrong", type=int, default=0, metavar="N",
                        help="print N wrong answers per type with what memwise gave")
    args = parser.parse_args()

    if not DATA.exists():
        sys.exit(f"{DATA} not found. Download it first; see the instructions at the top of this file.")
    try:
        llm = OpenAICompatible.from_env()
        full = OpenAICompatible.from_env(timeout=600.0)  # a whole history takes a while to read
        embedder = OpenAICompatibleEmbedder.from_env() if os.environ.get("MEMWISE_EMBED_MODEL") else None
    except LLMError as e:
        sys.exit(str(e))
    if embedder is None:
        print("note: MEMWISE_EMBED_MODEL isn't set, so the built-in embedder is used; scores will be low")

    config_args: dict[str, Any] = dict(top_k=args.top_k, require_consent_for_sensitive=args.consent,
                                       keep_messages=args.message_top_k > 0,
                                       message_top_k=max(1, args.message_top_k),
                                       keep_assistant_messages=args.assistant_top_k > 0,
                                       assistant_message_top_k=max(1, args.assistant_top_k))
    config = MemoryConfig.for_embedding_model(**config_args) if embedder else MemoryConfig(**config_args)
    data = json.loads(DATA.read_text())
    items = [i for i in data if i["question_id"] in args.question] if args.question else pick(data, args.per_type)
    if args.type:
        items = [i for i in items if i["question_type"] in args.type]
    if args.limit:
        items = items[:args.limit]
    if not items:
        sys.exit(f"no question matches {args.question}")
    modes = ["memwise"] + (["full"] if args.baseline else [])
    global FULL_SLOTS
    FULL_SLOTS = threading.BoundedSemaphore(max(1, args.full_workers))
    earlier: dict[tuple[str, str], dict] = {}
    if args.resume:
        earlier = {(r["question_id"], r["mode"]): r for r in map(json.loads, args.resume.open())
                   if "error" not in r}

    extract_llm, answer_llm, full_llm, grade_llm = (
        Metered(llm, "extraction"), Metered(llm, "answers"), Metered(full, "answers, whole history"),
        Metered(llm, "grading"))
    print(f"LongMemEval (S, cleaned)   chat model: {llm.model}   "
          f"embeddings: {embedder.model if embedder else 'built-in'}\n"
          f"questions: {len(items)} ({', '.join(f'{t} {n}' for t, n in Counter(i['question_type'] for i in items).items())}; "
          f"{sum(map(is_abstention, items))} abstention)\n"
          f"sensitive: {'only when asked' if args.consent else 'stored'}   past: {'off' if args.no_past else 'on'}   "
          f"top_k: {args.top_k}   messages: "
          f"{f'kept, {args.message_top_k} recalled' if config.keep_messages else 'off'}   assistant replies: "
          f"{f'kept, {args.assistant_top_k} recalled' if config.keep_assistant_messages else 'off'}   "
          f"baseline: {'on' if args.baseline else 'off'}\n", flush=True)

    start = time.time()
    print(f"building {sum(len(i['haystack_sessions']) for i in items):,} sessions' memories "
          f"(reused where already built)", flush=True)
    memories, built, folder = open_memories(items, args, extract_llm, embedder, config)
    ops: Counter = Counter()
    for info in built.values():
        ops.update(info["ops"])
    current = sum(len(m.mm.current(USER)) for m in memories.values()) / len(memories)
    print(f"memory ({folder.relative_to(HERE.parent)}): {current:.0f} current memories a question on "
          f"average; " + "  ".join(f"{op} {n}" for op, n in sorted(ops.items())))
    if extract_llm.calls:
        print(f"  built in {time.time() - start:.0f}s", flush=True)

    jobs = [(item, mode) for item in items for mode in modes]

    def run(job):
        return ask(job, memories, answer_llm, full_llm, grade_llm, not args.no_past)

    rows: list[dict] = [earlier.get((item["question_id"], mode), {"error": "not asked yet"}) for item, mode in jobs]
    todo = [i for i, r in enumerate(rows) if "error" in r]
    print(f"answering {len(todo)} ({', '.join(modes)})"
          + (f"; {len(jobs) - len(todo)} kept from {args.resume.name}" if args.resume else ""), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for attempt in range(4):  # a busy API (rate limits) fails now and then: wait, then ask those again
            if attempt:
                failed = [i for i in todo if "error" in rows[i]]
                if not failed:
                    break
                print(f"asking {len(failed)} again in {30 * attempt}s (the model API failed: "
                      f"{rows[failed[0]]['error'][:80]})", flush=True)
                time.sleep(30 * attempt)
                todo = failed
            for i, r in zip(todo, pool.map(run, [jobs[i] for i in todo])):
                rows[i] = r

    print()
    report(rows, modes)
    print("\nmodel use: " + "; ".join(m.summary() for m in (extract_llm, answer_llm, full_llm, grade_llm)
                                      if m.calls))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / (f"{datetime.now():%Y-%m-%dT%H-%M-%S}-{folder.name}"
                         + ("" if args.no_past else "-past") + f"-k{args.top_k}-m{args.message_top_k}"
                         + (f"-a{args.assistant_top_k}" if args.assistant_top_k else "")
                         + (f"-per{args.per_type}" if args.per_type else "") + ".jsonl")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarize(rows, modes, items, args, llm, embedder, config, out)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"answers: {out.relative_to(HERE.parent)}   ({time.time() - start:.0f}s in all)")
    print(f"summary: {out.with_suffix('.summary.json').relative_to(HERE.parent)}")
    if args.show_wrong:
        show_wrong(rows, args.show_wrong)


if __name__ == "__main__":
    main()
