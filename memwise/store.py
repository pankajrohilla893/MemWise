"""SQLite storage for memories, plus an audit log of every change.

Design choices worth copying:
  * One row per claim, with `valid_to` instead of UPDATE-in-place. When a fact
    stops being true we close its window and keep the row, so we still know
    "used to live in Berlin" and can explain why the memory changed.
  * `memory_log` records every operation (ADD, SUPERSEDE, EXPIRE, FORGET, ...).
    It is how you debug memory ("why does it think I like coffee?").
  * The schema maps directly onto Postgres + pgvector if you outgrow SQLite.

Concurrency: one `MemoryStore` can be shared by threads in a single process (for
example a FastAPI or Flask app); every method holds a lock. Several processes may
open the same file: SQLite serializes writes and ids are allocated under a write
lock, so they never collide.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from dataclasses import fields
from datetime import datetime

from .models import Memory

_DATETIME_FIELDS = {"valid_from", "last_confirmed", "valid_to", "expires_at", "last_used"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    text TEXT NOT NULL,
    scope TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    confidence REAL NOT NULL,
    importance REAL NOT NULL,
    sensitive INTEGER NOT NULL,
    valid_from TEXT NOT NULL,
    last_confirmed TEXT NOT NULL,
    valid_to TEXT,
    expires_at TEXT,
    superseded_by TEXT,
    last_used TEXT,
    use_count INTEGER NOT NULL DEFAULT 0,
    embedding TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories (user_id, subject, predicate);

CREATE TABLE IF NOT EXISTS memory_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    user_id TEXT NOT NULL,
    op TEXT NOT NULL,
    memory_id TEXT,
    detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_log_user ON memory_log (user_id);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_COLUMNS = [f.name for f in fields(Memory)]


def _locked(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class MemoryStore:
    """SQLite-backed memory storage. Use ":memory:" (the default) for tests."""

    def __init__(self, path: str = ":memory:"):
        self._lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- ids -------------------------------------------------------------
    @_locked
    def next_id(self) -> str:
        # BEGIN IMMEDIATE takes SQLite's write lock before reading the counter,
        # so two processes sharing one file can never hand out the same id.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT value FROM meta WHERE key = 'next_id'").fetchone()
            n = int(row["value"]) if row else 1
            self.db.execute(
                "INSERT INTO meta (key, value) VALUES ('next_id', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(n + 1),),
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        return f"m{n}"

    # ---- rows <-> objects ------------------------------------------------
    @staticmethod
    def _to_row(m: Memory) -> list:
        out = []
        for name in _COLUMNS:
            v = getattr(m, name)
            if name in _DATETIME_FIELDS:
                v = v.isoformat() if v else None
            elif name == "embedding":
                v = json.dumps(v)
            elif name == "sensitive":
                v = int(v)
            out.append(v)
        return out

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Memory:
        data = {}
        for name in _COLUMNS:
            v = row[name]
            if name in _DATETIME_FIELDS:
                v = datetime.fromisoformat(v) if v else None
            elif name == "embedding":
                v = json.loads(v)
            elif name == "sensitive":
                v = bool(v)
            data[name] = v
        return Memory(**data)

    # ---- writes ----------------------------------------------------------
    @_locked
    def insert(self, m: Memory) -> None:
        placeholders = ", ".join("?" for _ in _COLUMNS)
        self.db.execute(
            f"INSERT INTO memories ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            self._to_row(m),
        )
        self.db.commit()

    @_locked
    def save(self, m: Memory) -> None:
        assignments = ", ".join(f"{c} = ?" for c in _COLUMNS if c != "id")
        values = [v for c, v in zip(_COLUMNS, self._to_row(m)) if c != "id"]
        self.db.execute(f"UPDATE memories SET {assignments} WHERE id = ?", [*values, m.id])
        self.db.commit()

    @_locked
    def delete(self, memory_id: str) -> list[str]:
        """Hard delete. User-requested forgetting must really remove the data.

        Older versions that this memory replaced (the superseded_by chain) are
        the same claim, so they are deleted too. Returns every id removed.
        """
        if self.get(memory_id) is None:
            return []
        removed, frontier = [memory_id], [memory_id]
        while frontier:
            marks = ", ".join("?" for _ in frontier)
            rows = self.db.execute(
                f"SELECT id FROM memories WHERE superseded_by IN ({marks})", frontier
            ).fetchall()
            frontier = [r["id"] for r in rows if r["id"] not in removed]
            removed.extend(frontier)
        marks = ", ".join("?" for _ in removed)
        self.db.execute(f"DELETE FROM memories WHERE id IN ({marks})", removed)
        self.db.commit()
        return removed

    @_locked
    def delete_user(self, user_id: str) -> int:
        """Delete every memory and log entry for a user. Returns the number of memories removed."""
        cur = self.db.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        self.db.execute("DELETE FROM memory_log WHERE user_id = ?", (user_id,))
        self.db.commit()
        return cur.rowcount

    @_locked
    def redact_log(self, memory_ids: list[str]) -> None:
        """Scrub the content of past log entries about forgotten memories.

        Forgetting has to reach every copy: rows, embeddings, summaries built
        from them, and this log. The entries stay (so the history of *what
        happened* survives) but the text is gone.
        """
        if not memory_ids:
            return
        marks = ", ".join("?" for _ in memory_ids)
        self.db.execute(
            f"UPDATE memory_log SET detail = '[forgotten]' WHERE memory_id IN ({marks})", memory_ids
        )
        self.db.commit()

    @_locked
    def log(self, user_id: str, at: datetime, op: str, memory_id: str | None, detail: str) -> None:
        self.db.execute(
            "INSERT INTO memory_log (at, user_id, op, memory_id, detail) VALUES (?, ?, ?, ?, ?)",
            (at.isoformat(), user_id, op, memory_id, detail),
        )
        self.db.commit()

    # ---- reads -----------------------------------------------------------
    @_locked
    def get(self, memory_id: str) -> Memory | None:
        row = self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._from_row(row) if row else None

    @_locked
    def all(self, user_id: str) -> list[Memory]:
        """Every memory ever stored for this user, oldest first, including history."""
        rows = self.db.execute(
            "SELECT * FROM memories WHERE user_id = ? ORDER BY rowid", (user_id,)
        ).fetchall()
        return [self._from_row(r) for r in rows]

    def current(self, user_id: str, now: datetime) -> list[Memory]:
        return [m for m in self.all(user_id) if m.is_current(now)]

    @_locked
    def current_for(self, user_id: str, subject: str, predicate: str, now: datetime) -> list[Memory]:
        rows = self.db.execute(
            "SELECT * FROM memories WHERE user_id = ? AND subject = ? AND predicate = ? "
            "ORDER BY rowid",
            (user_id, subject, predicate),
        ).fetchall()
        return [m for m in map(self._from_row, rows) if m.is_current(now)]

    @_locked
    def user_ids(self) -> list[str]:
        """Every user with at least one stored memory (for nightly maintenance jobs)."""
        rows = self.db.execute("SELECT DISTINCT user_id FROM memories ORDER BY user_id").fetchall()
        return [r["user_id"] for r in rows]

    @_locked
    def logs(self, user_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM memory_log WHERE user_id = ? ORDER BY seq", (user_id,)
        ).fetchall()
