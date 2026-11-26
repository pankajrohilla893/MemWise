"""LoCoMo benchmark: months of chats between two friends, then questions about them.

LoCoMo (Maharana et al., 2024; github.com/snap-research/locomo) has 10 conversations of 19-32
sessions each and 1,986 questions. Mem0 and others quote scores on it, so this is how memwise
compares. Download the data once (2.8 MB; it isn't redistributed with memwise):

    mkdir -p evals/data
    curl -L -o evals/data/locomo10.json \\
        https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

Then, with the same MEMWISE_* settings as evals/scenarios.py:

    python evals/locomo.py --conv conv-26        # one conversation (152 questions)
    python evals/locomo.py                       # all 10 (1,540 questions)
    python evals/locomo.py --per-category 5      # a sample: up to 5 per category per conversation
    python evals/locomo.py --baseline            # also answer with the whole conversation in the prompt
    python evals/locomo.py --store-sensitive     # store identity, health, relationships unasked
    python evals/locomo.py --prompt friends      # the built-in prompt plus lines for chats between friends
    python evals/locomo.py --adversarial         # add category 5 (questions about the wrong person)
    python evals/locomo.py --keep-messages       # also keep and search what each friend said, word for word
    python evals/locomo.py --rewrite             # a model writes search queries for each question first

How memwise is run:
- Each friend gets their own memory. Every session is ingested once from each side: that
  friend's lines are the user's, the other friend's are the assistant's (context only). A first
  line gives the chat's date, and the clock is set to it, so every memory is dated.
- One scope ("general"): a question has no work or personal context to keep apart.
- memwise tells the extractor each chat's date, so "yesterday" is stored as a date.
- Questions are asked a day after the last session. Both friends' memories are recalled with the
  question; whatever isn't held back is listed with its date, and the model answers from that.
  It thinks first, then answers on a last "Answer:" line, which is graded. --baseline uses the
  same prompt with the whole conversation instead.
  --past also recalls memories that are no longer true (replaced or expired), as history.
  --keep-messages (MemoryConfig.keep_messages) keeps each friend's lines word for word: every
  memory is shown with the line it came from, and other lines that match the question are added.
  Relative dates in those lines ("last week") are followed by the dates they meant (memwise.dates).
  --rewrite (MemoryManager(rewriter=LLMQueryRewriter(llm))) has the model write a few search
  queries for each question, and recall searches with those as well as the question.
- The same model grades each answer against the gold one (CORRECT or WRONG). By default it
  grades generously, as published LoCoMo results were graded (an answer on the same topic, or
  naming the same time period, counts); --grader strict wants every part right. A rough token
  F1 is shown too.

Category numbers follow the dataset: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop,
5 adversarial. Like most published results, the score leaves out 5 unless --adversarial.

The memory is built once per setting and kept in evals/data/locomo-memory/ (--rebuild redoes it;
--memory NAME reopens an earlier build). A build that the model API fails is started again.
Every answer is saved to evals/data/locomo-results/ with the memories it was given, next to a
.summary.json of the scores. evals/report.py turns summaries into a results page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from memwise import (Clock, LLMError, LLMExtractor, LLMJudge, LLMQueryRewriter, MemoryConfig, MemoryManager,
                     MemoryStore, OpenAICompatible, OpenAICompatibleEmbedder)
from memwise.dates import with_dates
from memwise.llm import EXTRACTION_SYSTEM

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "locomo10.json"
MEMORY_DIR = HERE / "data" / "locomo-memory"
RESULTS_DIR = HERE / "data" / "locomo-results"

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
NOT_MENTIONED = "Not mentioned: nothing in the chats says this about that person"
INGEST_FORMAT = "4"  # bump when Memory.build, LLMExtractor or the reconciler's rules change what is built

# --prompt friends: memwise's prompt is written for an assistant and its user, and skips one-off
# events. LoCoMo asks about exactly those, so this keeps them, with dates.
FRIENDS_NOTES = """
These chats are between two friends, not a user and an assistant: the user's lines are one
friend, the assistant's lines are the other. Also keep what happens in the user's life, as
episodes: events they went to, trips, things they made, bought, read or did, milestones and
plans. Keep the people and pets in their life as facts."""

# The same answer prompt for memwise and for the whole conversation (--baseline). The model
# thinks first, then answers on an "Answer:" line, which is what gets graded.
ANSWER_SYSTEM = """You answer questions about two friends, {a} and {b}, from what you know of their chats.

{knowledge}

A relative date ("yesterday", "last week", "last Sunday") is relative to the date of the chat it
comes from: work it out from that date, or use the dates that follow it in [brackets]. A memory marked "no longer true" was true until the date
given: use it for questions about the past. Questions about what someone would likely do, think or
be need a reasonable inference from what you know: never answer them "unclear" or "not mentioned",
but pick the likelier answer and say why in a few words ("Likely no; she wants to be a counselor").
Questions asking for several things (which books, what activities) want every one you can find.

First think briefly: which lines answer the question, and, for "when" questions, the date worked
out step by step. Then give the final answer on a last line that starts with "Answer:", in as few
words as you can (a name, a date, a short phrase or list). Only if nothing above relates to the
question at all, answer "Answer: Not mentioned"."""

# --grader strict: the answer has to be the gold one, every part of it.
STRICT_GRADER = """You grade answers in a memory benchmark. You get a question, the gold answer and a
predicted answer. The prediction is CORRECT if it gives the same answer as the gold one, even in
other words, with more detail, or inside a longer reply. A date is correct if it names the same
day, month or year as the gold answer, in any format, including a relative date that works out to
it. The prediction is WRONG if it gives a different answer, misses the key part of the gold
answer, or says it doesn't know (unless the gold answer says it isn't mentioned). Write one short
sentence of reasoning, then a last line that is exactly 'Answer: CORRECT' or 'Answer: WRONG'."""

# --grader lenient (the default): grade generously, the way published LoCoMo results (Mem0's) were
# graded, so scores can be compared with theirs.
LENIENT_GRADER = """You grade answers in a memory benchmark. You get a question, the gold answer and a
predicted answer. Be generous: the prediction is CORRECT if it is about the same thing as the gold
answer, even if it is longer, worded differently, or names only some of several items. A date or
time is correct if it refers to the same date or period as the gold answer, in any format,
including a relative one ("the week before 9 June" and "early June 2023" both fit "June 2023").
It is WRONG if it gives a different answer, is about something else, or says it doesn't know
(unless the gold answer says it isn't mentioned). Write one short sentence of reasoning, then a
last line that is exactly 'Answer: CORRECT' or 'Answer: WRONG'."""
GRADERS = {"lenient": LENIENT_GRADER, "strict": STRICT_GRADER}


class SessionClock(Clock):
    """A clock set to each session's date while it's ingested."""

    def __init__(self) -> None:
        super().__init__()
        self.at = datetime(2000, 1, 1, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.at


class Metered:
    """Wraps a chat model to count calls and, roughly (4 characters a token), tokens."""

    def __init__(self, llm: Any, name: str):
        self.llm, self.name = llm, name
        self.model = getattr(llm, "model", "?")
        self.calls = self.chars_in = self.chars_out = 0
        self._lock = threading.Lock()

    def __call__(self, system: str, messages: list[dict]) -> str:
        reply = self.llm(system, messages)
        with self._lock:
            self.calls += 1
            self.chars_in += len(system) + sum(len(m["content"]) for m in messages)
            self.chars_out += len(reply)
        return reply

    def summary(self) -> str:
        return (f"{self.name} {self.calls} calls, ~{self.chars_in // 4:,} tokens in, "
                f"~{self.chars_out // 4:,} out")


# =========================================================================== the data

def sessions(conv: dict) -> list[tuple[int, datetime, str, list[dict]]]:
    """(number, date, date as written, turns) for each session, in order."""
    c = conv["conversation"]
    numbers = sorted(int(m.group(1)) for k in c if (m := re.fullmatch(r"session_(\d+)", k)))
    out = []
    for n in numbers:
        if not c[f"session_{n}"]:
            continue
        written = c[f"session_{n}_date_time"]
        when = datetime.strptime(written, "%I:%M %p on %d %B, %Y").replace(tzinfo=timezone.utc)
        out.append((n, when, written, c[f"session_{n}"]))
    return out


def line(turn: dict) -> str:
    text = turn["text"]
    if turn.get("blip_caption"):
        text += f" [shares a photo: {turn['blip_caption']}]"
    return text


def transcript(conv: dict) -> str:
    parts = []
    for n, _, written, turns in sessions(conv):
        parts.append(f"Session {n}, {written}:\n" + "\n".join(f"{t['speaker']}: {line(t)}" for t in turns))
    return "\n\n".join(parts)


def speakers(conv: dict) -> tuple[str, str]:
    return conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]


# =========================================================================== memory

class Memory:
    """One friend's memwise, built from every session as seen from their side."""

    def __init__(self, path: Path, extractor: LLMExtractor, embedder: Any, config: MemoryConfig,
                 judge: LLMJudge | None):
        self.path = path
        self.clock = SessionClock()
        self.mm = MemoryManager(MemoryStore(str(path)), extractor=extractor, clock=self.clock,
                                embedder=embedder, config=config, judge=judge)

    def build(self, conv: dict, me: str, other: str) -> Counter:
        ops: Counter = Counter()
        for _, when, _, turns in sessions(conv):
            if not any(t["speaker"] == me for t in turns):
                continue
            self.clock.at = when
            chat = [{"role": "assistant",
                     "content": f"(A chat between {me} and {other}. The user is {me}; "
                                f"the assistant's lines are {other}.)"}]
            for t in turns:
                if t["speaker"] == me:
                    chat.append({"role": "user", "content": line(t), "ref": t["dia_id"]})
                else:
                    chat.append({"role": "assistant", "content": line(t)})
            ops.update(d.op for d in self.mm.ingest(me, chat))
        return ops


def open_memories(convs: list[dict], args, extract_llm: Metered, embedder: Any,
                  config: MemoryConfig) -> tuple[dict[tuple[str, str], Memory], dict, Path]:
    """Build (or reopen) each friend's memory. Returns them, what building did, and where.

    The folder name includes a hash of the extraction prompt, so changing the prompt (or
    memwise's built-in one) builds the memory again."""
    system = EXTRACTION_SYSTEM + FRIENDS_NOTES if args.prompt == "friends" else EXTRACTION_SYSTEM
    key = "_".join([extract_llm.model, getattr(embedder, "model", "builtin"), args.prompt,
                    "sensitive" if args.store_sensitive else "consent",
                    "judge" if args.judge else "rules",
                    *(["messages"] if args.keep_messages else []),
                    hashlib.sha1((system + INGEST_FORMAT).encode()).hexdigest()[:8]])
    folder = MEMORY_DIR / (args.memory or re.sub(r"[^A-Za-z0-9.-]+", "-", key))
    if args.memory and not folder.is_dir():
        sys.exit(f"{folder} not found; the memories built so far are in {MEMORY_DIR}")
    folder.mkdir(parents=True, exist_ok=True)
    extractor = LLMExtractor(extract_llm, system_prompt=system, scopes=("general",))
    judge = LLMJudge(extract_llm) if args.judge else None

    jobs = []
    for conv in convs:
        a, b = speakers(conv)
        for me, other in ((a, b), (b, a)):
            jobs.append((conv, me, other))

    def build(job):
        conv, me, other = job
        path = folder / f"{conv['sample_id']}-{me}.db"
        done = path.with_suffix(".done.json")
        if args.memory and not done.exists():
            raise SystemExit(f"{path.name} was never finished in {folder.name}; --memory only reopens")
        if args.rebuild or not done.exists():
            for attempt in range(3):  # a slow or busy API times out now and then: build it again
                for p in (path, done, Path(f"{path}-wal"), Path(f"{path}-shm")):
                    p.unlink(missing_ok=True)
                mem = Memory(path, extractor, embedder, config, judge)
                start = time.time()
                try:
                    ops = mem.build(conv, me, other)
                    break
                except LLMError as e:
                    mem.mm.store.close()
                    if attempt == 2:
                        raise
                    print(f"  building {path.stem} again (the model API failed: {str(e)[:80]})")
            done.write_text(json.dumps({"ops": ops, "seconds": round(time.time() - start, 1)}))
        else:
            mem = Memory(path, extractor, embedder, config, judge)
        return (conv["sample_id"], me), mem, json.loads(done.read_text())

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        built = list(pool.map(build, jobs))
    return {k: m for k, m, _ in built}, {k: info for k, _, info in built}, folder


def _day(when: datetime) -> str:
    return f"{when:%A} {when.day} {when:%B %Y}"


def known(mem: Memory, name: str, question: str, past: bool) -> tuple[list[str], int, list[str]]:
    """What memwise lets through for this question: the profile, every memory not held back
    and, with `past`, the no-longer-true memories it found, marked as such. With kept messages,
    each memory's own line is under it, and other matching lines follow. Also returns how many
    were found and, with --rewrite, the queries searched besides the question."""
    r = mem.mm.recall(name, question, record_usage=False, include_past=past)
    lines = []

    def said(m) -> None:
        if m.id in r.sources:
            s = r.sources[m.id]
            lines.append(f'    said: "{with_dates(s.text, s.at)}"')

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
        lines.append(f"Other things {name} said that may be relevant:")
        for x in sorted(r.messages, key=lambda x: x.message.at):
            lines.append(f'- [chat of {_day(x.message.at)}] "{with_dates(x.message.text, x.message.at)}"')
    return lines, len(r.items) + len(r.past) + len(r.messages), r.searched


# =========================================================================== answering and grading

def grade(llm: Metered, system: str, question: str, gold: str, prediction: str) -> tuple[bool, str]:
    reply = llm(system, [{"role": "user", "content":
                                f"Question: {question}\nGold answer: {gold}\nPredicted answer: {prediction}"}])
    found = re.findall(r"answer:\s*\**\s*(correct|wrong)", reply, re.I)
    verdict = found[-1].lower() if found else ("correct" if re.search(r"\bcorrect\b", reply, re.I)
                                               and not re.search(r"\bwrong\b", reply, re.I) else "wrong")
    return verdict == "correct", " ".join(reply.split())


def _tokens(text: str) -> list[str]:
    text = "".join(ch for ch in str(text).lower() if ch not in string.punctuation)
    return [w for w in text.split() if w not in {"a", "an", "the", "and"}]


def f1(prediction: str, gold: str) -> float:
    p, g = _tokens(prediction), _tokens(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def ask(job, memories, answer_llm: Metered, grade_llm: Metered, transcripts: dict,
        grader: str, past: bool) -> dict:
    conv, qa, mode = job
    a, b = speakers(conv)
    sample = conv["sample_id"]
    category = qa["category"]
    gold = NOT_MENTIONED if category == 5 else str(qa["answer"])
    row: dict[str, Any] = {"sample": sample, "category": category, "mode": mode,
                           "question": qa["question"], "gold": gold, "evidence": qa.get("evidence", [])}
    try:
        if mode == "memwise":
            start = time.perf_counter()
            parts, shown, retrieved = [], 0, 0
            for name in (a, b):
                lines, found, searched = known(memories[(sample, name)], name, qa["question"], past)
                parts.append(f"What you remember about {name}:\n" + ("\n".join(lines) or "(nothing)"))
                shown += sum(1 for ln in lines if ln.startswith("- "))  # memories and messages
                retrieved += found
            row.update(recall_ms=round((time.perf_counter() - start) * 1000), memories=shown,
                       retrieved=retrieved, searched=searched)
            knowledge = "\n\n".join(parts)
            row["knowledge"] = knowledge
        else:
            knowledge = "Their chats:\n\n" + transcripts[sample]
        reply = answer_llm(ANSWER_SYSTEM.format(a=a, b=b, knowledge=knowledge),
                           [{"role": "user", "content": qa["question"]}]).strip()
        found = re.findall(r"^\W*(?:final\s+)?answer\s*\**\s*:\s*\**\s*(.+)", reply, re.I | re.M)
        prediction = found[-1].strip() if found else reply
        correct, reason = grade(grade_llm, grader, qa["question"], gold, prediction)
        row.update(reply=reply, prediction=prediction, correct=correct, f1=round(f1(prediction, gold), 3),
                   grade_reason=reason, context_tokens=len(knowledge) // 4)
    except LLMError as e:
        row["error"] = str(e)
    except Exception as e:  # one broken question shouldn't lose the run
        row["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
    return row


# =========================================================================== report

def report(rows: list[dict], modes: list[str], categories: list[int]) -> None:
    by: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        if "error" not in r:
            by[(r["mode"], r["category"])].append(r)
    names = {"memwise": "memwise", "full": "whole conversation"}
    head = f"{'category':<13}{'n':>5}" + "".join(f"   {names[m]:>18} {'F1':>5}" for m in modes)
    print(head)
    print("-" * len(head))

    def cells(items_by_mode):
        out = ""
        for m in modes:
            items = items_by_mode[m]
            if items:
                acc = sum(r["correct"] for r in items) / len(items)
                out += f"   {acc:>17.1%} {sum(r['f1'] for r in items) / len(items):>5.2f}"
            else:
                out += f"   {'-':>18} {'-':>5}"
        return out

    for c in categories:
        n = max((len(by[(m, c)]) for m in modes), default=0)
        print(f"{CATEGORIES[c]:<13}{n:>5}" + cells({m: by[(m, c)] for m in modes}))
    scored = [c for c in categories if c != 5]
    overall = {m: [r for c in scored for r in by[(m, c)]] for m in modes}
    print("-" * len(head))
    print(f"{'overall 1-4':<13}{max(len(v) for v in overall.values()):>5}" + cells(overall))
    if 5 in categories:
        print("(adversarial is shown on its own and not in the overall score)")

    for m in modes:
        items = overall[m]
        if not items:
            continue
        tokens = sum(r["context_tokens"] for r in items) / len(items)
        extra = ""
        if m == "memwise":
            mems = sum(r["memories"] for r in items) / len(items)
            ms = sorted(r["recall_ms"] for r in items)
            extra = f", {mems:.1f} memories and messages, recall p50 {ms[len(ms) // 2]} ms (embedding call included)"
        print(f"context, {names[m]}: ~{tokens:,.0f} tokens a question{extra}")
    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"{len(errors)} questions failed and are not scored, e.g.: {errors[0]['error'][:200]}")


def summarize(rows: list[dict], modes: list[str], categories: list[int], args, convs: list[dict],
              total_convs: int, llm: Any, embedder: Any, out: Path) -> dict:
    """The run's scores in a form evals/report.py reads (and any other tool can)."""
    scored = [c for c in categories if c != 5]
    summary: dict[str, Any] = {
        "benchmark": "LoCoMo",
        "date": f"{datetime.now():%Y-%m-%d}",
        "label": args.label or label(args),
        "settings": {"prompt": args.prompt, "sensitive": "stored" if args.store_sensitive else "only when asked",
                     "past": args.past, "judge": args.judge, "top_k": args.top_k, "grader": args.grader,
                     "keep_messages": args.keep_messages,
                     "message_top_k": args.message_top_k if args.keep_messages else None,
                     "rewrite": args.rewrite,
                     "chat_model": llm.model, "embed_model": getattr(embedder, "model", "built-in")},
        "conversations": [c["sample_id"] for c in convs],
        "total_conversations": total_convs,
        "questions": sum(1 for r in rows if r["mode"] == modes[0] and r["category"] in scored),
        "results": str(out.relative_to(HERE.parent)),
        "modes": {},
    }
    for m in modes:
        ok = [r for r in rows if r["mode"] == m and "error" not in r]
        main = [r for r in ok if r["category"] in scored]
        if not main:
            continue
        entry: dict[str, Any] = {
            "n": len(main),
            "errors": sum(1 for r in rows if r["mode"] == m and "error" in r),
            "score": round(100 * sum(r["correct"] for r in main) / len(main), 1),
            "f1": round(sum(r["f1"] for r in main) / len(main), 3),
            "mean_tokens": round(sum(r["context_tokens"] for r in main) / len(main)),
            "categories": {},
        }
        for c in categories:
            items = [r for r in ok if r["category"] == c]
            if items:
                entry["categories"][CATEGORIES[c]] = {
                    "n": len(items), "score": round(100 * sum(r["correct"] for r in items) / len(items), 1)}
        if m == "memwise":
            entry["mean_memories"] = round(sum(r["memories"] for r in main) / len(main), 1)
            ms = sorted(r["recall_ms"] for r in main)
            entry["recall_ms_p50"] = ms[len(ms) // 2]
        summary["modes"][m] = entry
    return summary


def label(args) -> str:
    """A name for the set-up, for the results page."""
    parts = []
    if args.prompt != "builtin":
        parts.append(f"{args.prompt} prompt")
    if args.store_sensitive:
        parts.append("sensitive stored")
    if args.past:
        parts.append("past recall")
    if args.keep_messages:
        parts.append(f"messages kept, {args.message_top_k} recalled")
    if args.rewrite:
        parts.append("questions rewritten")
    if args.judge:
        parts.append("judge")
    if args.top_k != 10:
        parts.append(f"top_k {args.top_k}")
    return "memwise (" + (", ".join(parts) or "defaults") + ")"


def show_wrong(rows: list[dict], count: int) -> None:
    wrong = [r for r in rows if r["mode"] == "memwise" and r.get("correct") is False]
    by_cat: dict[int, list[dict]] = defaultdict(list)
    for r in wrong:
        by_cat[r["category"]].append(r)
    for c in sorted(by_cat):
        print(f"\n== wrong, {CATEGORIES[c]} ({len(by_cat[c])})")
        for r in by_cat[c][:count]:
            print(f"Q: {r['question']}\n   gold: {r['gold']}\n   got:  {r['prediction']}")
            for ln in r.get("knowledge", "").splitlines()[:14]:
                print(f"     {ln}")


# =========================================================================== main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--conv", action="append", default=[],
                        help="a conversation id such as conv-26 (repeatable; default: all 10)")
    parser.add_argument("--per-category", type=int, default=0,
                        help="at most this many questions per category per conversation (default: all)")
    parser.add_argument("--adversarial", action="store_true", help="also ask category 5")
    parser.add_argument("--baseline", action="store_true",
                        help="also answer with the whole conversation in the prompt (many more tokens)")
    parser.add_argument("--prompt", choices=("builtin", "friends"), default="builtin",
                        help="extraction prompt: memwise's own, or with lines for chats between friends")
    parser.add_argument("--store-sensitive", action="store_true",
                        help="store sensitive details without being asked (memwise asks by default)")
    parser.add_argument("--judge", action="store_true", help="build memory with the optional LLMJudge")
    parser.add_argument("--past", action="store_true",
                        help="also recall memories that are no longer true (recall(include_past=True))")
    parser.add_argument("--keep-messages", action="store_true",
                        help="keep and search each friend's lines word for word (MemoryConfig.keep_messages)")
    parser.add_argument("--message-top-k", type=int, default=10,
                        help="with --keep-messages: matching lines recalled per friend (default 10)")
    parser.add_argument("--rewrite", action="store_true",
                        help="have the model write search queries for each question (LLMQueryRewriter)")
    parser.add_argument("--grader", choices=tuple(GRADERS), default="lenient",
                        help="lenient (default): graded like published results; strict: every part must match")
    parser.add_argument("--top-k", type=int, default=10, help="memories recalled per friend (default 10)")
    parser.add_argument("--rebuild", action="store_true", help="build the memory again instead of reusing it")
    parser.add_argument("--memory", default="", metavar="NAME",
                        help="reopen this memory (a folder in evals/data/locomo-memory/) whatever the prompt is now, "
                             "to compare recall settings on the same memory")
    parser.add_argument("--workers", type=int, default=8, help="model calls in parallel (default 8)")
    parser.add_argument("--label", default="", help="a name for this set-up on the results page")
    parser.add_argument("--show-wrong", type=int, default=0, metavar="N",
                        help="print N wrong answers per category with the memories they had")
    args = parser.parse_args()
    if args.memory and args.rebuild:
        sys.exit("--memory reopens a memory as it was built, so it can't be used with --rebuild")

    if not DATA.exists():
        sys.exit(f"{DATA} not found. Download it first; see the instructions at the top of this file.")
    try:
        llm = OpenAICompatible.from_env()
        embedder = OpenAICompatibleEmbedder.from_env() if os.environ.get("MEMWISE_EMBED_MODEL") else None
    except LLMError as e:
        sys.exit(str(e))
    if embedder is None:
        print("note: MEMWISE_EMBED_MODEL isn't set, so the built-in embedder is used; scores will be low")

    config_args: dict[str, Any] = dict(top_k=args.top_k,
                                       require_consent_for_sensitive=not args.store_sensitive,
                                       keep_messages=args.keep_messages, message_top_k=args.message_top_k)
    config = MemoryConfig.for_embedding_model(**config_args) if embedder else MemoryConfig(**config_args)
    data = json.loads(DATA.read_text())
    convs = [c for c in data if not args.conv or c["sample_id"] in args.conv]
    if not convs:
        sys.exit(f"no conversation matches {args.conv}; ids are {', '.join(c['sample_id'] for c in data)}")
    categories = [1, 2, 3, 4] + ([5] if args.adversarial else [])
    modes = ["memwise"] + (["full"] if args.baseline else [])

    questions = []
    for conv in convs:
        taken: Counter = Counter()
        for qa in conv["qa"]:
            c = qa["category"]
            if c in categories and (not args.per_category or taken[c] < args.per_category):
                taken[c] += 1
                questions.append((conv, qa))

    extract_llm, rewrite_llm, answer_llm, grade_llm = (Metered(llm, n) for n in
                                                       ("extraction", "rewriting", "answers", "grading"))
    print(f"LoCoMo   chat model: {llm.model}   embeddings: {embedder.model if embedder else 'built-in'}\n"
          f"conversations: {len(convs)}   questions: {len(questions)}   prompt: {args.prompt}   "
          f"sensitive: {'stored' if args.store_sensitive else 'only when asked'}   "
          f"judge: {'on' if args.judge else 'off'}   past: {'on' if args.past else 'off'}   "
          f"top_k: {args.top_k}   grader: {args.grader}   messages: "
          f"{f'kept, {args.message_top_k} recalled' if args.keep_messages else 'off'}   "
          f"rewrite: {'on' if args.rewrite else 'off'}\n")

    start = time.time()
    memories, built, folder = open_memories(convs, args, extract_llm, embedder, config)
    last = max(when for conv in convs for _, when, _, _ in sessions(conv))
    rewriter = LLMQueryRewriter(rewrite_llm) if args.rewrite else None  # one cache for both friends
    for mem in memories.values():  # questions are asked a day after the last session
        mem.clock.at = last + timedelta(days=1)
        mem.mm.rewriter = rewriter
    print(f"memory ({folder.relative_to(HERE.parent)}):")
    for (sample, name), mem in memories.items():
        ops = built[(sample, name)]["ops"]
        current = len(mem.mm.current(name))
        print(f"  {sample} {name:<9} {current:>4} current memories   "
              + "  ".join(f"{op} {n}" for op, n in sorted(ops.items())))
    if extract_llm.calls:
        print(f"  built in {time.time() - start:.0f}s")

    transcripts = {c["sample_id"]: transcript(c) for c in convs} if args.baseline else {}

    jobs = [(conv, qa, mode) for conv, qa in questions for mode in modes]

    def run(job):
        return ask(job, memories, answer_llm, grade_llm, transcripts, GRADERS[args.grader], args.past)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run, jobs))
        for _ in range(2):  # a slow or busy API times out now and then: ask those again
            failed = [i for i, r in enumerate(rows) if "error" in r]
            if not failed:
                break
            print(f"asking {len(failed)} questions again (the model API failed: {rows[failed[0]]['error'][:80]})")
            for i, r in zip(failed, pool.map(run, [jobs[i] for i in failed])):
                rows[i] = r

    print()
    report(rows, modes, categories)
    print("\nmodel use: " + "; ".join(m.summary() for m in (extract_llm, rewrite_llm, answer_llm, grade_llm)
                                      if m.calls))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / (f"{datetime.now():%Y-%m-%dT%H-%M-%S}-{folder.name}-{args.grader}"
                         + ("-past" if args.past else "") + f"-k{args.top_k}"
                         + (f"-m{args.message_top_k}" if args.keep_messages else "")
                         + ("-rw" if args.rewrite else "") + ".jsonl")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarize(rows, modes, categories, args, convs, len(data), llm, embedder, out)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"answers: {out.relative_to(HERE.parent)}   ({time.time() - start:.0f}s in all)")
    print(f"summary: {out.with_suffix('.summary.json').relative_to(HERE.parent)}")
    if args.show_wrong:
        show_wrong(rows, args.show_wrong)


if __name__ == "__main__":
    main()
