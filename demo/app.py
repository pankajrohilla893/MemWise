"""memwise live demo: the same model side by side, without and with long-term memory.

Each message goes to both chats. The plain chat remembers only the current session, like a
product built straight on a model API. The memwise chat gets the same system prompt plus what
memwise recalls about the visitor. Start a new session, and skip ahead in time, to see the
difference. Panels show what memwise wrote, what it recalled for each message (USE, MENTION,
VERIFY or HOLD, with the reason) and the exact memory it added to the prompt.

Every visitor gets their own memory, in RAM, gone when they leave.

Run it from a clone:

    pip install gradio
    OPENAI_API_KEY=... python demo/app.py

With no model configured it still runs: memwise extracts memories with its offline rules and the
chats say there's no model to answer.

Settings (environment variables; on Hugging Face, the Space's secrets and variables):

    OPENAI_API_KEY           enough on its own: OpenAI, with gpt-6-luna and text-embedding-3-small
    MEMWISE_MODEL            another chat model
    MEMWISE_EMBED_MODEL      another embedding model
    MEMWISE_BASE_URL         another provider: any /chat/completions API (then MEMWISE_API_KEY
                             for its key, and set both models)
    DEMO_VISITOR_LIMIT       messages and new sessions per visitor (default 40)
    DEMO_DAILY_LIMIT         messages and new sessions per day, all visitors together (default 1500)
    DEMO_REPLY_OPTIONS       extra request fields for the chat replies, as JSON,
                             e.g. {"max_completion_tokens": 600}. With the default model it's
                             {"reasoning_effort": "none"}: chat replies start in under a second
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(HERE, "memwise")):
    # Running from a clone: use the package in the repo. On the Space it's copied in beside this file.
    sys.path.insert(0, os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # for page.py, when this file is imported rather than run

import gradio as gr  # noqa: E402
import page  # noqa: E402

from memwise import (Clock, LLMExtractor, LLMJudge, MemoryConfig, MemoryManager,  # noqa: E402
                     MemoryStore, OpenAICompatible, OpenAICompatibleEmbedder, RuleExtractor,
                     __version__)

OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6-luna"                    # what the benchmarks used
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
USER = "visitor"
PERSONA = ("You are a helpful assistant in a chat app. Keep replies under about 120 words "
           "unless the user asks for more.")
REMEMBER = re.compile(r"\b(remember|don't forget|note that)\b", re.I)
MAX_CHARS = 600
WHEN = {"Right away": 0, "A day later": 1, "A week later": 7, "A month later": 30,
        "3 months later": 90, "8 months later": 240}

# A short story that shows each idea in turn. ("say", text, scope) sends a message in a chat
# about that scope; ("new", when, "") ends the session and starts the next one that much later.
TOUR: list[tuple[str, str, str]] = [
    ("say", "Hi! I'm Arjun. I live in Bangalore and work as a doctor. I like answers as short bullet points.", "general"),
    ("say", "Remember that I'm allergic to peanuts.", "general"),
    ("say", "Ugh, it's raining again today.", "general"),
    ("say", "I was diagnosed with asthma last year, by the way.", "general"),
    ("new", "A week later", ""),
    ("say", "Any snack ideas for my trek on Saturday?", "general"),
    ("say", "Big news: I just moved to Mumbai!", "general"),
    ("new", "3 months later", ""),
    ("say", "What's a nice day trip from where I live?", "general"),
    ("say", "Where did I live before?", "general"),
    ("new", "8 months later", ""),
    ("say", "Any tips to get through a long shift at work tomorrow?", "work"),
]


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


def _reply_options(default: dict) -> dict:
    raw = os.environ.get("DEMO_REPLY_OPTIONS", "").strip()
    if not raw:
        return default
    try:
        options = json.loads(raw)
    except ValueError:
        print("DEMO_REPLY_OPTIONS isn't valid JSON; ignoring it.", file=sys.stderr)
        return {}
    return options if isinstance(options, dict) else {}


def _settings() -> tuple[str | None, str | None, str | None, str | None]:
    """(base_url, model, embedding model, api_key). With only OPENAI_API_KEY set, that's OpenAI
    and the models the benchmarks used; the MEMWISE_* variables override any of it."""
    get = os.environ.get
    openai_key = get("OPENAI_API_KEY")
    key = get("MEMWISE_API_KEY") or openai_key
    base_url = get("MEMWISE_BASE_URL") or (OPENAI_URL if openai_key else None)
    on_openai = base_url == OPENAI_URL
    model = get("MEMWISE_MODEL") or (DEFAULT_MODEL if on_openai else None)
    embed_model = get("MEMWISE_EMBED_MODEL") or (DEFAULT_EMBED_MODEL if on_openai else None)
    return base_url, model, embed_model, key


def _connect():
    """The pieces every visitor shares: (chat model, extractor, judge, embedder, config)."""
    base_url, model, embed_model, key = _settings()
    if not base_url or not model:
        print("No model configured (set OPENAI_API_KEY): running offline.", file=sys.stderr)
        return None, RuleExtractor(), None, None, None
    # Chat replies skip the model's reasoning: they start in under a second instead of 1.5.
    # Extraction and judging keep it, since what memwise stores depends on them.
    fast = {"reasoning_effort": "none"} if (base_url, model) == (OPENAI_URL, DEFAULT_MODEL) else {}
    chat = OpenAICompatible(base_url=base_url, model=model, api_key=key, timeout=60, retries=1,
                            options=_reply_options(fast))
    # Extraction and judging get no reply cap: their JSON can be long.
    work = OpenAICompatible(base_url=base_url, model=model, api_key=key, timeout=90, retries=1)
    embedder = config = None
    if embed_model:
        embedder = OpenAICompatibleEmbedder(base_url=base_url, model=embed_model, api_key=key,
                                            timeout=30, retries=1)
        config = MemoryConfig.for_embedding_model()
    return chat, LLMExtractor(work), LLMJudge(work), embedder, config


CHAT, EXTRACTOR, JUDGE, EMBEDDER, CONFIG = _connect()
VISITOR_LIMIT = _env_int("DEMO_VISITOR_LIMIT", 40)
POOL = ThreadPoolExecutor(max_workers=32)
if EMBEDDER is not None:
    # Embed the tour's messages now, so recall for them is a search in memory, with no wait for
    # the embedding model. The embedder keeps what it embedded, for every visitor.
    POOL.submit(EMBEDDER.embed_many, [text for kind, text, _ in TOUR if kind == "say"])


class Budget:
    """Model calls cost money: a cap on actions per day, across all visitors."""

    def __init__(self, per_day: int):
        self.per_day = per_day
        self.day = None
        self.used = 0
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            today = datetime.now(timezone.utc).date()
            if today != self.day:
                self.day, self.used = today, 0
            if self.used >= self.per_day:
                return False
            self.used += 1
            return True


BUDGET = Budget(_env_int("DEMO_DAILY_LIMIT", 1500))


class Visitor:
    """One visitor's two chats and their memory. Nothing is shared between visitors."""

    def __init__(self):
        self.clock = Clock()
        self.mm = MemoryManager(MemoryStore(":memory:"), extractor=EXTRACTOR, clock=self.clock,
                                embedder=EMBEDDER, config=CONFIG, judge=JUDGE)
        self.session = 1
        self.turn = 0
        self.scope = "general"
        self.plain: list[dict] = []        # this session's chat without memory
        self.remembered: list[dict] = []   # this session's chat with memwise
        self.transcript: list[dict] = []   # the memwise chat with refs, for ingest()
        self.pending: str | None = None    # the message being answered
        self.used = 0                      # actions taken, against VISITOR_LIMIT
        # (kind, title, rows), newest last. kind: write, recall or session. rows: (op, text, why).
        self.events: list[tuple[str, str, list[tuple[str, str, str]]]] = []
        self.memory_block = ""             # the memory memwise added to the last prompt
        self.earlier_chars = 0             # every earlier session's messages, for the token comparison
        self.tour = 0
        self.reading = False               # memwise is reading the session that just ended

    def event(self, kind: str, title: str, rows: list[tuple[str, str, str]]) -> None:
        self.events.append((kind, title, rows))
        del self.events[:-14]


# ---- what the handlers do ----------------------------------------------------------------

def _allowed(v: Visitor) -> bool:
    if v.reading:
        gr.Warning("One moment: memwise is still reading the last session.")
        return False
    if v.used >= VISITOR_LIMIT:
        gr.Warning(f"This demo allows {VISITOR_LIMIT} messages and sessions per visitor. "
                   "Press Start over to begin again.")
        return False
    if not BUDGET.take():
        gr.Warning("The demo has used up today's model budget. Please come back tomorrow.")
        return False
    v.used += 1
    return True


def _accept(v: Visitor, text: str) -> bool:
    """Take a message for answering; False (with a warning) if it can't be."""
    text = text.strip()
    if not text or v.pending is not None:
        return False
    if len(text) > MAX_CHARS:
        gr.Warning(f"Please keep messages under {MAX_CHARS} characters.")
        return False
    if not _allowed(v):
        return False
    v.pending = text
    return True


class Reply:
    """A reply being written by a worker thread: `text` grows until `done` is set."""

    def __init__(self):
        self.text = ""
        self.done = threading.Event()


def _write(system: str, messages: list[dict], reply: Reply) -> None:
    """Stream the model's reply to `messages` into `reply`."""
    try:
        if CHAT is None:
            reply.text = "*(No model is configured, so there's no reply. The memory panels still work.)*"
            return
        for piece in CHAT.stream(system, messages):
            reply.text += piece
        if not reply.text.strip():
            reply.text = "*(empty reply)*"
    except Exception as e:  # shown in the chat; the page stays usable
        reply.text += f"\n\n*(The model call failed: {e})*"
    finally:
        reply.done.set()


def _reply(system: str, messages: list[dict]) -> Reply:
    reply = Reply()
    POOL.submit(_write, system, messages, reply)
    return reply


def _snip(text: str, n: int = 48) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _write_rows(decisions) -> list[tuple[str, str, str]]:
    rows = []
    for d in decisions:
        mem = d.result or d.target
        label = f"{mem.id}: {d.candidate.text}" if mem else d.candidate.text
        notes = [d.reason]
        if d.candidate.sensitive:
            notes.append("sensitive")
        if d.candidate.ttl_days:
            notes.append(f"expires in {d.candidate.ttl_days} days")
        rows.append((d.op, label, " · ".join(n for n in notes if n)))
        for m in d.also:
            rows.append((d.op, f"{m.id}: {m.text}", "also no longer true"))
    return rows or [("", "Nothing worth remembering", "small talk and passing remarks aren't kept")]


def _recall_rows(r) -> list[tuple[str, str, str]]:
    rows = [("PROFILE", f"{m.id}: {m.text}", "always loaded") for m in r.profile]
    for d in r.items:
        m = d.scored.memory
        rows.append((d.action, f"{m.id}: {m.text}", f"{d.reason} · relevance {d.scored.relevance:.2f}"))
    for d in r.past:
        m = d.scored.memory
        rows.append((d.action, f"{m.id}: {m.text}", d.reason))
    return rows or [("", "Nothing stored matches yet", "nothing added to the prompt")]


def respond(v: Visitor):
    """Answer the pending message in both chats, showing the replies as they're written.

    The plain chat starts at once. The memwise chat starts as soon as recall is done (a few
    milliseconds once the message is embedded). A message that asks to remember something is
    saved while the replies are written."""
    if v is None or v.pending is None:
        yield views(v)
        return
    text = v.pending
    user = {"role": "user", "content": text}
    plain = _reply(PERSONA, v.plain + [user])
    try:
        v.turn += 1
        ref = f"s{v.session}:{v.turn}"
        saving = POOL.submit(v.mm.ingest, USER, [(ref, text)]) if REMEMBER.search(text) else None
        r = v.mm.recall(USER, text, scope=v.scope, include_past=True)
        v.memory_block = v.mm.build_context(r)
        system = PERSONA + ("\n\n" + v.memory_block if v.memory_block else "")
        remembered = _reply(system, v.remembered + [user])
        used = len(r.profile) + len(r.used)
        held = sum(1 for d in r.items if d.action == "HOLD")
        v.event("recall", f"“{_snip(text)}”: {_count(used, 'memory', 'memories')} in the prompt"
                + (f", {held} held back" if held else ""), _recall_rows(r))
        yield _writing(v, plain, remembered, panels=True)
        shown = None
        while shown is None or not (plain.done.is_set() and remembered.done.is_set()):
            time.sleep(0.04)
            if (plain.text, remembered.text) != shown:
                shown = (plain.text, remembered.text)
                yield _writing(v, plain, remembered)
        v.plain += [user, {"role": "assistant", "content": plain.text}]
        v.remembered += [user, {"role": "assistant", "content": remembered.text}]
        v.transcript += [{"role": "user", "content": text, "ref": ref},
                         {"role": "assistant", "content": remembered.text}]
        if saving is not None:
            v.event("write", f"“{_snip(text)}”: saved right away, because you asked",
                    _write_rows(saving.result()))
    except Exception as e:  # keep the page usable on network or provider errors
        gr.Warning(f"Something went wrong: {e}")
    finally:
        v.pending = None
    yield views(v)


def _writing(v: Visitor, plain: Reply, remembered: Reply, panels: bool = False):
    """The outputs while the replies are being written: the two chats (and, once, the panels)."""
    skip = gr.skip()
    return (skip, _chat(v.plain, v.pending, plain.text), _chat(v.remembered, v.pending, remembered.text),
            skip, skip, _events(v) if panels else skip, skip, _prompt(v) if panels else skip,
            skip, skip, skip)


def _end_session(v: Visitor, when: str) -> None:
    if any(t["role"] == "user" for t in v.transcript):
        rows = _write_rows(v.mm.ingest(USER, v.transcript))
    else:
        rows = [("", "Nothing was said this session", "")]
    v.event("write", f"Session {v.session} ended: memwise read it and decided what to keep", rows)
    v.earlier_chars += sum(len(t["content"]) for t in v.remembered)
    v.clock.advance(WHEN.get(when, 0))
    report = v.mm.maintain(USER)
    v.session, v.turn = v.session + 1, 0
    v.plain, v.remembered, v.transcript, v.memory_block = [], [], [], ""
    rows = []
    for line in report:
        op, _, rest = line.partition(" ")
        rows.append((op, rest.strip(), "maintenance"))
    v.event("session", f"Session {v.session} · {when.lower()} · {_day(v.clock.now())}",
            rows or [("", "Both chats start empty. Only memwise remembers.", "")])
    gr.Info(f"Session {v.session} started, {when.lower()}. Both chats are empty: ask something "
            "and see which one still knows you.")


# ---- event handlers: each returns every output (see OUTPUTS) ------------------------------

def start(v):
    return views(v or Visitor())


def send(text: str, scope: str, v):
    v = v or Visitor()
    v.scope = scope
    if _accept(v, text or ""):
        return views(v, box="")
    return views(v)


def new_session(when: str, v):
    v = v or Visitor()
    if v.pending is not None:
        gr.Warning("Wait for the reply first.")
        yield views(v)
        return
    yield from _next_session(v, when)


def tour_step(v):
    v = v or Visitor()
    if v.tour >= len(TOUR) or v.pending is not None:
        yield views(v)
        return
    kind, value, chat_scope = TOUR[v.tour]
    if kind == "say":
        v.scope = chat_scope
        if _accept(v, value):
            v.tour += 1
        yield views(v, scope=chat_scope)
        return
    yield from _next_session(v, value, tour=True)


def _next_session(v: Visitor, when: str, tour: bool = False):
    """End the session and start the next. The chats clear at once; the panels fill in when
    memwise has read the session (a few seconds: a model reads it and judges any conflicts)."""
    if not _allowed(v):
        yield views(v)
        return
    v.reading = True
    v.event("write", f"Session {v.session} ended: memwise is reading it…",
            [("", "Deciding what to keep. This takes a few seconds.", "")])
    skip = gr.skip()
    yield (skip, [], [], skip, skip, _events(v), skip, skip, skip,
           gr.Button(value="memwise is reading the session…", interactive=False), skip)
    try:
        v.events.pop()   # the note above
        _end_session(v, when)
        if tour:
            v.tour += 1
    except Exception as e:
        gr.Warning(f"Something went wrong: {e}")
    finally:
        v.reading = False
    yield views(v)


def forget(memory_id, v):
    v = v or Visitor()
    if memory_id:
        removed = v.mm.forget(USER, memory_id)
        v.event("write", "You deleted a memory",
                [("FORGET", ", ".join(removed), "deleted, older versions too, and scrubbed from the log")]
                if removed else [("", "That memory was already gone", "")])
    return views(v)


def reset(v):
    return views(Visitor(), box="", scope="general")


# ---- rendering ------------------------------------------------------------------------------

def _chat(history: list[dict], pending: str | None, partial: str = "") -> list[dict]:
    if pending is None:
        return history
    return history + [{"role": "user", "content": pending},
                      {"role": "assistant", "content": partial or "…"}]


def _day(d: datetime, month: str = "%B") -> str:
    return f"{d.day} {d.strftime(month)} {d.year}"


def _tokens(chars: int) -> int:
    return round(chars / 4)  # a rough rule for English text


def _status(v: Visitor) -> str:
    now = v.clock.now()
    if CHAT:
        model = f"<span>model <b>{html.escape(CHAT.model)}</b></span>"
    else:
        model = "<span class='mw-offline'>Offline: no model configured</span>"
    stored = _count(len(v.mm.current(USER)), "memory", "memories")
    return (f"<div class='mw-statusbar'><span class='mw-pill'>Session {v.session}</span>"
            f"<span><b>{now:%A} {_day(now)}</b></span><span>{stored} stored</span>"
            f"<span>{v.used} of {VISITOR_LIMIT} actions used</span>{model}</div>")


def _badge(op: str) -> str:
    if not op:
        return ""
    return f'<span class="mw-badge mw-{html.escape(op)}">{html.escape(op)}</span>'


def _events(v: Visitor) -> str:
    if not v.events:
        return ('<p class="mw-empty">Nothing yet. Send a message or take the tour: every decision '
                "memwise makes shows up here, with the reason.</p>")
    parts = []
    for kind, title, rows in reversed(v.events):
        body = "".join(f"<tr><td>{_badge(op)}</td><td>{html.escape(text)}"
                       f'<div class="mw-why">{html.escape(why)}</div></td></tr>'
                       for op, text, why in rows)
        parts.append(f'<div class="mw-block mw-block-{kind}"><div class="mw-title">{html.escape(title)}</div>'
                     f'<table class="mw-table">{body}</table></div>')
    return "".join(parts)


def _known(v: Visitor) -> str:
    current = v.mm.current(USER)
    ids = {m.id for m in current}
    rows = []
    for m in current:
        notes = []
        if m.scope != "general":
            notes.append(f"{m.scope} only")
        if m.sensitive:
            notes.append("sensitive")
        if m.expires_at:
            notes.append(f"expires {_day(m.expires_at, '%b')}")
        notes.append(f"confirmed {_day(m.last_confirmed, '%b')}")
        rows.append(f"<tr><td><code>{m.id}</code></td><td>{html.escape(m.text)}"
                    f'<div class="mw-why"><span class="mw-kind">{m.kind}</span>'
                    f'{html.escape(" · ".join(notes))}</div></td></tr>')
    if not rows:
        return ('<p class="mw-empty">Nothing yet. memwise writes when you say “remember …”, '
                "and when a session ends.</p>")
    out = f'<table class="mw-table">{"".join(rows)}</table>'
    past = [m for m in v.mm.history(USER) if m.id not in ids]
    if past:
        lines = []
        for m in past:
            if m.superseded_by:
                why = f"replaced by {m.superseded_by}"
            elif m.valid_to:
                why = "no longer true"
            else:
                why = "expired"
            if m.valid_to:
                why += f" on {_day(m.valid_to, '%b')}"
            lines.append(f'<tr><td><code>{m.id}</code></td><td><s>{html.escape(m.text)}</s>'
                         f'<div class="mw-why">{html.escape(why)}</div></td></tr>')
        out += (f'<details class="mw-history" open><summary>No longer true, kept as history '
                f'({len(past)})</summary><table class="mw-table">{"".join(lines)}</table></details>')
    return out


def _prompt(v: Visitor) -> str:
    if not v.memory_block:
        return "*Nothing yet. After your next message, this shows exactly what memwise added to the prompt.*"
    used = _tokens(len(v.memory_block))
    text = f"About **{used:,} tokens** of memory went into this message's prompt."
    if v.earlier_chars:
        text += (f" Sending every earlier session instead would be about "
                 f"**{_tokens(v.earlier_chars):,} tokens**, and that grows with every session, "
                 "while memwise's share stays about the same.")
    return f"{text}\n\n```text\n{v.memory_block}\n```"


def _tour_label(v: Visitor) -> str:
    if v.tour >= len(TOUR):
        return "Tour finished: keep chatting, or start over"
    kind, value, chat_scope = TOUR[v.tour]
    if kind == "say":
        what = f"say “{_snip(value, 46)}”"
        if chat_scope != "general":
            what += f" in a {chat_scope} chat"
    else:
        what = f"start a new session, {value.lower()}"
    lead = "Take the 2-minute tour" if v.tour == 0 else "Next tour step"
    return f"▶  {lead} ({v.tour + 1} of {len(TOUR)}): {what}"


def views(v: Visitor | None, box=None, scope=None):
    """Every output, in OUTPUTS order. `box` sets the message box and `scope` the scope
    buttons (None leaves them alone)."""
    if v is None:
        v = Visitor()
    choices = [(f"{m.id}: {m.text}", m.id) for m in v.mm.current(USER)]
    return (v, _chat(v.plain, v.pending), _chat(v.remembered, v.pending),
            gr.skip() if box is None else box, _status(v), _events(v), _known(v), _prompt(v),
            gr.Dropdown(choices=choices, value=None),
            gr.Button(value=_tour_label(v), interactive=v.tour < len(TOUR)),
            gr.skip() if scope is None else scope)


# ---- the page -------------------------------------------------------------------------------

with gr.Blocks(title="memwise: long-term memory for AI products", analytics_enabled=False) as demo:
    state = gr.State(None, time_to_live=2 * 3600)
    gr.HTML(page.HERO)
    with gr.Tabs(overflow_behavior="wrap"):  # on a phone, tab names wrap instead of hiding
        with gr.Tab("Try it live", id="try"):
            gr.HTML(page.STEPS)
            with gr.Row(equal_height=True):
                tour_btn = gr.Button(_tour_label(Visitor()), variant="primary", elem_id="mw-tour", scale=6)
                reset_btn = gr.Button("Start over", scale=1, min_width=120)
            with gr.Row(equal_height=True):
                with gr.Column(elem_classes="mw-chatcol"):
                    gr.HTML(page.SIDE_PLAIN)
                    plain_chat = gr.Chatbot(show_label=False, height=460, buttons=["copy"],
                                            placeholder=page.PLACEHOLDER_PLAIN, elem_classes="mw-chat")
                with gr.Column(elem_classes="mw-chatcol"):
                    gr.HTML(page.SIDE_MEMWISE)
                    memory_chat = gr.Chatbot(show_label=False, height=460, buttons=["copy"],
                                             placeholder=page.PLACEHOLDER_MEMWISE,
                                             elem_classes=["mw-chat", "mw-chat-mem"])
            with gr.Row(equal_height=True):
                box = gr.Textbox(placeholder="Type a message: it goes to both chats", show_label=False,
                                 scale=6, max_lines=4, container=False)
                send_btn = gr.Button("Send to both", variant="primary", scale=1, min_width=130)
            status = gr.HTML()
            with gr.Row(equal_height=True):
                when = gr.Dropdown(list(WHEN), value="A week later", label="Next session starts", scale=2)
                new_btn = gr.Button("End this session, start the next", scale=2)
                scope = gr.Radio(["general", "work", "personal"], value="general", scale=3,
                                 label="This chat is about",
                                 info="Work memories stay out of personal chats, and the other way round")
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML("<div class='mw-panel-title'><h3>What memwise did</h3>"
                            "<p>Every decision, newest first, with the reason.</p></div>")
                    gr.HTML(page.LEGEND)
                    events = gr.HTML(elem_classes="mw-panel")
                with gr.Column(scale=2):
                    gr.HTML("<div class='mw-panel-title'><h3>What memwise knows about you</h3>"
                            "<p>What's true now, and what it replaced.</p></div>")
                    known = gr.HTML(elem_classes="mw-panel")
                    with gr.Row(equal_height=True):
                        pick = gr.Dropdown([], label="Delete a memory", scale=3)
                        forget_btn = gr.Button("Delete", variant="stop", scale=1, min_width=90)
            with gr.Accordion("See the exact memory memwise added to the last prompt", open=False):
                prompt = gr.Markdown()
        with gr.Tab("Use it in any product", id="uses"):
            gr.HTML(page.USE_CASES)
            gr.HTML(page.CODE_TITLE)
            gr.Code(page.CODE, language="python", interactive=False, show_label=False, wrap_lines=True,
                    elem_classes="mw-code")
        with gr.Tab("How it works", id="how"):
            gr.HTML(page.HOW)
    gr.HTML(page.footer(__version__))

    OUTPUTS = [state, plain_chat, memory_chat, box, status, events, known, prompt, pick, tour_btn,
               scope]

    # Model calls cost money: none of this is offered as an API, only through the page.
    private = {"api_visibility": "private"}
    demo.load(start, [state], OUTPUTS, **private)
    for trigger in (box.submit, send_btn.click):
        trigger(send, [box, scope, state], OUTPUTS, show_progress="hidden", **private).then(
            respond, [state], OUTPUTS, show_progress="hidden", **private)
    tour_btn.click(tour_step, [state], OUTPUTS, show_progress="hidden", **private).then(
        respond, [state], OUTPUTS, show_progress="hidden", **private)
    new_btn.click(new_session, [when, state], OUTPUTS, show_progress="hidden", **private)
    forget_btn.click(forget, [pick, state], OUTPUTS, **private)
    reset_btn.click(reset, [state], OUTPUTS, **private)

demo.queue(default_concurrency_limit=16, max_size=64, api_open=False)

if __name__ == "__main__":
    demo.launch(css=page.CSS, theme=page.THEME, footer_links=["gradio", "settings"])
