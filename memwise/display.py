"""Terminal output helpers shared by demo.py and chat.py."""

from __future__ import annotations

import os
import sys
from datetime import datetime

from .manager import Recall
from .models import Memory
from .reconciler import Decision

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_CODES = {
    "bold": "1", "dim": "2", "green": "32", "yellow": "33", "blue": "34",
    "magenta": "35", "cyan": "36", "red": "31",
}
OP_COLORS = {
    "ADD": "green", "CONFIRM": "blue", "SUPERSEDE": "magenta", "RETRACT": "yellow",
    "IGNORE": "dim", "USE": "green", "MENTION": "cyan", "VERIFY": "yellow", "HOLD": "dim",
    "EXPIRE": "yellow", "MERGE": "blue", "PRUNE": "red", "FORGET": "red",
}


def c(text: str, style: str) -> str:
    if not _COLOR or style not in _CODES:
        return text
    return f"\033[{_CODES[style]}m{text}\033[0m"


def fit(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def op(name: str) -> str:
    return c(f"{name:<9}", OP_COLORS.get(name, "bold"))


def heading(text: str) -> None:
    print()
    print(c("━" * 78, "dim"))
    print(c(text, "bold"))
    print(c("━" * 78, "dim"))


def decisions(ds: list[Decision]) -> None:
    if not ds:
        print(c("  (nothing worth remembering)", "dim"))
        return
    for d in ds:
        mem = d.result or d.target
        mem_id = mem.id if mem else "-"
        flags = []
        if d.candidate.explicit:
            flags.append("explicit")
        if d.candidate.sensitive:
            flags.append("sensitive")
        if d.candidate.ttl_days:
            flags.append(f"expires in {d.candidate.ttl_days}d")
        flag_text = c(f" [{', '.join(flags)}]", "dim") if flags else ""
        print(f"  {op(d.op)} {mem_id:<4} {d.candidate.text}{flag_text}")
        print(c(f"  {'':9} {'':4} why: {d.reason}", "dim"))


def recall(r: Recall) -> None:
    print(c(f"  scope: {r.scope}", "dim"))
    for m in r.profile:
        print(f"  {c('PROFILE  ', 'bold')} {m.id:<4} {m.text}")
    if not r.items:
        print(c("  (no other memories matched)", "dim"))
    for d in r.items:
        s = d.scored
        nums = c(f"rel {s.relevance:.2f}  score {s.score:.2f}", "dim")
        print(f"  {op(d.action)} {s.memory.id:<4} {fit(s.memory.text, 46):<46} {nums}")
        print(c(f"  {'':9} {'':4} why: {d.reason}", "dim"))


def memory_table(memories: list[Memory], now: datetime, show_history: bool = False) -> None:
    if not memories:
        print(c("  (no memories)", "dim"))
        return
    for m in memories:
        if m.is_current(now):
            state = c("current", "green")
        elif m.superseded_by:
            state = c(f"replaced by {m.superseded_by}", "magenta")
        elif m.expires_at and m.expires_at <= now:
            state = c("expired", "yellow")
        else:
            state = c("no longer true", "yellow")
        if not show_history and not m.is_current(now):
            continue
        window = f"{m.valid_from:%Y-%m-%d} → " + (f"{m.valid_to:%Y-%m-%d}" if m.valid_to else "now")
        extra = []
        if m.sensitive:
            extra.append("sensitive")
        if m.scope != "general":
            extra.append(m.scope)
        if m.expires_at and m.is_current(now):
            extra.append(f"expires {m.expires_at:%Y-%m-%d}")
        extra_text = c(f"  [{', '.join(extra)}]", "dim") if extra else ""
        print(f"  {m.id:<4} {fit(m.text, 40):<40} {c(window, 'dim')}  {state}{extra_text}")


def log_table(rows) -> None:
    for r in rows:
        print(f"  {c(r['at'][:10], 'dim')}  {op(r['op'])} {r['memory_id'] or '-':<4} {r['detail']}")
