"""Dynamic slot store for SuperMem.

Stores the new slots that emerge from the subgraph mechanism (SubgraphManager):
[(name, description, parent_slots), ...].
parent_slots is a list -- for a slot promoted by run_for_retrieved_pool (cross-slot_ref
retrieval during the priming stage), the participating entities may originally have been
spread across several different slot_refs; all of them are recorded, not just one.

The old sub_slot mechanism (a two-level parent_slot -> sub_slot structure, with
route_sub_slot query routing / assign_to_sub_slot write assignment) has been removed
entirely -- routing to new slots now relies on the slot tags in memory_tags themselves
(including base-7 and the dynamic slot names stored here), so a separate sub_slot level
is no longer needed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Dynamic slot store ────────────────────────────────────────────────────────

@dataclass
class DynamicSlot:
    name: str
    user_id: str
    description: str
    embedding: list[float] | None
    parent_slots: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_iso)


class DynamicSlotStore:
    """SQLite storage for automatically emerged new slots. Thread-safe: a new connection per operation."""

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
            CREATE TABLE IF NOT EXISTS dynamic_slots (
                name        TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding   TEXT,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (user_id, name)
            );
            """)
            # parent_slots_json: old databases lack this column, so ALTER TABLE adds it
            # (the CREATE TABLE for new databases doesn't include it either; this step
            # fills it in uniformly, so both cases end up with the column)
            try:
                c.execute("ALTER TABLE dynamic_slots ADD COLUMN parent_slots_json TEXT NOT NULL DEFAULT '[]'")
            except sqlite3.OperationalError:
                pass  # column already exists

    def get_dynamic_slots(self, user_id: str) -> list[DynamicSlot]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM dynamic_slots WHERE user_id=? ORDER BY name",
                (user_id,),
            ).fetchall()
        return [DynamicSlot(
            name=r["name"], user_id=r["user_id"], description=r["description"],
            embedding=json.loads(r["embedding"]) if r["embedding"] else None,
            parent_slots=json.loads(r["parent_slots_json"]) if r["parent_slots_json"] else [],
            created_at=r["created_at"],
        ) for r in rows]

    def create_dynamic_slot(
        self,
        user_id: str,
        name: str,
        description: str,
        embedding: list[float] | None = None,
        parent_slots: list[str] | None = None,
    ) -> DynamicSlot:
        parent_slots = parent_slots or []
        ds = DynamicSlot(
            name=name, user_id=user_id, description=description,
            embedding=embedding, parent_slots=parent_slots, created_at=_utc_iso(),
        )
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO dynamic_slots
                   (name, user_id, description, embedding, created_at, parent_slots_json)
                   VALUES (?,?,?,?,?,?)""",
                (name, user_id, description,
                 json.dumps(embedding) if embedding else None, ds.created_at,
                 json.dumps(parent_slots)),
            )
        return ds

    def get_parent_slots(self, user_id: str, name: str) -> list[str]:
        """All parent slots of a dynamic slot (the slot_refs where each entity promoted
        during subgraph judgement originally lived; there may be more than one).
        Returns [] for a non-dynamic slot or when no parent slots were recorded."""
        with self._conn() as c:
            row = c.execute(
                "SELECT parent_slots_json FROM dynamic_slots WHERE user_id=? AND name=?",
                (user_id, name),
            ).fetchone()
        if not row or not row["parent_slots_json"]:
            return []
        try:
            return json.loads(row["parent_slots_json"])
        except Exception:
            return []

    def get_children(self, user_id: str, parent_name: str) -> list[DynamicSlot]:
        """Return all dynamic slots whose parent_slots contain parent_name (direct children).
        Used for "drilling down" in hierarchical classification -- parent_name may be a
        base-7 slot or another dynamic slot (subgraphs can nest multiple levels); both
        are treated the same."""
        return [s for s in self.get_dynamic_slots(user_id) if parent_name in s.parent_slots]

    def exists(self, user_id: str, name: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM dynamic_slots WHERE user_id=? AND name=?",
                (user_id, name),
            ).fetchone()
        return row is not None
