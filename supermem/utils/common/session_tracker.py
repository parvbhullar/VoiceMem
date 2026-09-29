"""Tracks each user's ingest turns + session boundaries, to trigger left/right-brain batch jobs.

Two trigger signals:
  - every 3 ingest turns (is_third_turn): right-brain short-term attribution (updates entity.description)
  - session_id change (session_changed): right-brain long-term attribution (updates slot.description) +
    left-brain subgraph decision (settles the retrieval tally accumulated in this session, see
    core.py::RunSubgraphCheckpoint / PrimeSubgraphFromQuery)

Also provides a generic "touched" ledger: what was touched in this turn/session, so batch jobs
only process those instead of a full scan. The namespace is a caller-chosen string
(e.g. "rb_entity_short"/"rb_slot_long"/"subgraph_pool"); namespaces don't interfere.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionTracker:
    """SQLite storage for turn/session state + touched refs. Thread-safe: a new connection per operation."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS session_state (
                user_id         TEXT PRIMARY KEY,
                last_session_id TEXT,
                turn_count      INTEGER NOT NULL DEFAULT 0,
                updated_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS touched_refs (
                user_id    TEXT NOT NULL,
                namespace  TEXT NOT NULL,
                ref        TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, namespace, ref)
            );
            """)

    def record_turn(self, user_id: str, session_id) -> dict:
        """Called once per Ingest().

        Returns
        -------
        dict: {"turn_count": int, "session_changed": bool, "is_third_turn": bool}
        session_changed is True only when the old and new session_id are both non-empty and differ --
        if the caller never passes session_id, this trigger never fires, so no false triggers.
        """
        new_sid = str(session_id) if session_id is not None else None
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM session_state WHERE user_id=?", (user_id,)
            ).fetchone()
            if row is None:
                c.execute(
                    """INSERT INTO session_state (user_id, last_session_id, turn_count, updated_at)
                       VALUES (?,?,1,?)""",
                    (user_id, new_sid, _utc_iso()),
                )
                return {"turn_count": 1, "session_changed": False, "is_third_turn": False}

            last_sid = row["last_session_id"]
            session_changed = last_sid is not None and new_sid is not None and last_sid != new_sid
            turn_count = row["turn_count"] + 1
            c.execute(
                "UPDATE session_state SET last_session_id=?, turn_count=?, updated_at=? WHERE user_id=?",
                (new_sid, turn_count, _utc_iso(), user_id),
            )
            return {
                "turn_count": turn_count,
                "session_changed": session_changed,
                "is_third_turn": turn_count % 3 == 0,
            }

    def get_current_session(self, user_id: str) -> str | None:
        """The session_id recorded by the latest record_turn() -- used to scope Algorithm 1's rho(H)
        formula to the session (in the paper, Q is the query set of the "current session", not
        lifetime history). Returns None if no turn was recorded or session_id was never passed."""
        with self._conn() as c:
            row = c.execute(
                "SELECT last_session_id FROM session_state WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["last_session_id"] if row else None

    # ── Touched refs ──────────────────────────────────────────────────────────

    def touch(self, user_id: str, namespace: str, ref: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO touched_refs (user_id, namespace, ref, created_at)
                   VALUES (?,?,?,?)""",
                (user_id, namespace, ref, _utc_iso()),
            )

    def count_touched(self, user_id: str, namespace: str) -> int:
        """Only peek at how many have accumulated, without clearing. For "run once N have built up" batch jobs."""
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM touched_refs WHERE user_id=? AND namespace=?",
                (user_id, namespace),
            ).fetchone()
        return int(row["n"]) if row else 0

    def pop_touched(self, user_id: str, namespace: str) -> list[str]:
        """Take and clear the touched refs accumulated in this namespace."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT ref FROM touched_refs WHERE user_id=? AND namespace=?",
                (user_id, namespace),
            ).fetchall()
            c.execute(
                "DELETE FROM touched_refs WHERE user_id=? AND namespace=?",
                (user_id, namespace),
            )
        return [r["ref"] for r in rows]
