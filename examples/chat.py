"""Chat with an assistant that has long-term memory, and watch the memory work.

    python examples/chat.py              # uses your model if MEMWISE_BASE_URL and MEMWISE_MODEL are set
    python examples/chat.py --offline    # no model calls: shows what the model *would* receive

Any model that speaks the /chat/completions format works, hosted or local:

    export MEMWISE_BASE_URL="http://localhost:11434/v1"   # e.g. Ollama on your machine
    export MEMWISE_MODEL="llama3.2"
    export MEMWISE_API_KEY="..."                          # only if your provider needs one
    export MEMWISE_EMBED_MODEL="..."                      # optional: a real embedding model

Memories are saved to memory.db, so they survive between runs.

Two write paths, like a real product:
  * "remember ..." is written immediately (the user asked; don't make them wait)
  * everything else is extracted when the session ends (/end or quitting),
    with the whole conversation as context. Cheaper, and more accurate.

Commands:
  /memories        what the assistant currently knows about you
  /history         every memory ever stored, including replaced and expired ones
  /log             audit log of every change
  /forget <id>     delete a memory (e.g. /forget m3)
  /scope <name>    switch context: general, work or personal
  /context         show the full prompt block from the last message
  /end             end the session: extract memories, run maintenance, start fresh
  /advance <days>  pretend time has passed (try it, then ask about something old)
  /maintain        run maintenance now (expire, merge, prune)
  /quit            end the session and exit
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Lets you run this from a clone without installing: `python examples/chat.py`
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from memwise import (Clock, LLMError, LLMExtractor, MemoryConfig, MemoryManager, MemoryStore,
                     OpenAICompatible, OpenAICompatibleEmbedder)
from memwise import display as show
from memwise.extractor import RuleExtractor

USER = "me"
PERSONA = "You are a helpful assistant with long-term memory about this user."
REMEMBER = re.compile(r"\b(remember|don't forget|note that)\b", re.I)


def connect(offline: bool):
    """Return (extractor, llm), falling back to offline pieces when no model is configured."""
    if offline:
        return RuleExtractor(), None
    try:
        llm = OpenAICompatible.from_env()
    except LLMError as e:
        print(show.c(f"{e} Running offline.", "yellow"))
        return RuleExtractor(), None
    return LLMExtractor(llm), llm


def embedding_setup(offline: bool):
    """Use a real embedding model if MEMWISE_EMBED_MODEL is set; otherwise the built-in one."""
    if offline or not os.environ.get("MEMWISE_EMBED_MODEL"):
        return None, None
    return OpenAICompatibleEmbedder.from_env(), MemoryConfig.for_embedding_model()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="don't call a model")
    parser.add_argument("--db", default="memory.db", help="SQLite file (default: memory.db)")
    args = parser.parse_args()

    extractor, llm = connect(args.offline)
    embedder, config = embedding_setup(args.offline or llm is None)
    clock = Clock()
    mm = MemoryManager(MemoryStore(args.db), extractor=extractor, clock=clock,
                       embedder=embedder, config=config)

    # Memories saved by an earlier run may come from a different embedder.
    stored = mm.history(USER)
    if stored and len(stored[0].embedding) != len(mm.embedder.embed("check")):
        print(show.c("  Stored memories came from another embedder; re-embedding them…", "dim"))
        mm.reembed()

    session_no = 1
    turn = 0
    scope = "general"
    pending: list[tuple[str, str]] = []   # user messages not yet ingested
    history: list[dict] = []              # this session's conversation, for the model
    transcript: list[dict] = []           # the same, with refs on user turns, for ingest()
    last_context = ""

    mode = f"model: {llm.model}" if llm else "offline"
    if embedder:
        mode += f", embeddings: {embedder.model}"
    print(show.c(f"Memory chat ({mode}). Type /help for commands, /quit to exit.", "bold"))

    def end_session() -> None:
        nonlocal pending, history, transcript, session_no, turn
        if pending:
            # The whole conversation goes in: the assistant's turns help the extractor understand
            # replies like "the second one", but memories only come from what the user said.
            print(show.c("\n  Ending session: extracting memories from the whole conversation…", "dim"))
            decisions = safe(lambda: mm.ingest(USER, transcript))
            if decisions is None and llm is None:
                # The model failed and we just switched offline: retry with the rule extractor.
                decisions = safe(lambda: mm.ingest(USER, transcript))
            show.decisions(decisions or [])
        report = mm.maintain(USER)
        for line in report:
            print(f"  {line}")
        pending, history, transcript = [], [], []
        session_no, turn = session_no + 1, 0

    def safe(fn):
        nonlocal llm
        try:
            return fn()
        except Exception as e:  # keep the REPL alive on API/network errors
            print(show.c(f"  error: {e}", "red"))
            if isinstance(e, LLMError) and e.status in (401, 403) and llm:
                print(show.c("  The API key was rejected; switching to offline mode.", "yellow"))
                llm, mm.extractor = None, RuleExtractor()
            return None

    while True:
        try:
            text = input(show.c(f"\n[{scope}] you> ", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            text = "/quit"
        if not text:
            continue

        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            if cmd in ("/quit", "/exit"):
                end_session()
                break
            elif cmd == "/help":
                print(__doc__.split("Commands:")[1])
            elif cmd == "/memories":
                show.memory_table(mm.current(USER), clock.now())
            elif cmd == "/history":
                show.memory_table(mm.history(USER), clock.now(), show_history=True)
            elif cmd == "/log":
                show.log_table(mm.store.logs(USER))
            elif cmd == "/forget":
                removed = mm.forget(USER, arg.strip())
                print(f"  deleted {', '.join(removed)}" if removed else "  no such memory")
            elif cmd == "/scope":
                if arg.strip() in ("general", "work", "personal"):
                    scope = arg.strip()
                else:
                    print("  scopes: general, work, personal")
            elif cmd == "/context":
                print(last_context or "  (send a message first)")
            elif cmd == "/end":
                end_session()
                print(show.c(f"  Session {session_no} started.", "dim"))
            elif cmd == "/advance":
                try:
                    days = float(arg)
                except ValueError:
                    print("  usage: /advance <days>")
                    continue
                print(f"  it is now {clock.advance(days):%Y-%m-%d}")
            elif cmd == "/maintain":
                for line in mm.maintain(USER) or ["(nothing to do)"]:
                    print(f"  {line}")
            else:
                print("  unknown command; try /help")
            continue

        turn += 1
        ref = f"s{session_no}:{turn}"

        # Fast path: the user asked us to remember, so write it now.
        if REMEMBER.search(text):
            print(show.c("  Explicit request: writing it now.", "dim"))
            show.decisions(safe(lambda: mm.ingest(USER, [(ref, text)])) or [])
        else:
            pending.append((ref, text))

        # Read path: recall, decide what to surface, build the prompt.
        r = mm.recall(USER, text, scope=scope)
        show.recall(r)
        last_context = PERSONA + "\n\n" + mm.build_context(r)

        history.append({"role": "user", "content": text})
        transcript.append({"role": "user", "content": text, "ref": ref})
        if llm:
            answer = safe(lambda: llm(last_context, history))
            if answer is None:
                history.pop()
                continue
        else:
            answer = "(offline: no model call. Type /context to see the prompt it would get.)"
        history.append({"role": "assistant", "content": answer})
        transcript.append({"role": "assistant", "content": answer})
        print(f"\n{show.c('assistant>', 'green')} {answer}")


if __name__ == "__main__":
    main()
