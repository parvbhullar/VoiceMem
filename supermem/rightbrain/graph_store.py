"""Right-brain slot->entity->memory graph layer.

The same three-layer structure as the left brain, but the content is affective/subjective, not the left brain's factual categories:

  slot (5 initial categories: emotion / likes_dislikes / expression_style / thinking_pattern / coping_style)
    └── entity (a concrete affective node, e.g. "happy" under "emotion", "being interrupted" under "likes_dislikes")
          └── memory (id of a concrete memory hung under an entity, pointing to right_brain_memories.id)

A slot is a real graph node here (with its own id + description), not a string attribute like in the left brain.

Entities are deduplicated by semantic similarity (not exact string match): a newly extracted affective label is first compared
against the embeddings of existing entities in the same slot; if similar enough it is reused, otherwise a new one is created.
"""

from __future__ import annotations

import os

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from supermem.utils.common._graph_common import cosine as _cosine, new_id as _new_id, utc_iso as _utc_iso

#: How similar a new label must be to an existing entity in the same slot to count as the same one.
#:
#: It used to be 0.65 -- for E5 on short Chinese phrases that meant "merge everything": the measured baseline is already high,
#: "hates people smacking their lips while eating" vs "hates being interrupted" scored 0.934, "hates long meetings" vs it 0.922, and all got
#: merged into one entity. The result: after running for a while the right brain stopped growing new nodes, and nothing new the user said
#: showed up on the graph, so it looked like nothing was remembered.
#: Measured pairs that really should merge ("likes pour-over coffee" <-> "prefers pour-over coffee") scored 0.964,
#: so 0.95 cleanly separates the two cases.
DEFAULT_MATCH_THRESHOLD = float(os.environ.get("SUPERMEM_RB_ENTITY_MERGE", "0.95"))


# 5 initial affective slots; descriptions are left empty for now, to be filled in later.
# The initial entities of emotion reuse the original 8 emotion labels.
SEED_SLOTS: list[tuple[str, str, list[str]]] = [
    ("emotion", "", ["anxious", "sad", "wronged", "lonely", "conflicted", "calm", "happy", "tired"]),
    ("likes_dislikes", "", []),
    ("expression_style", "", []),
    ("thinking_pattern", "", []),
    ("coping_style", "", []),
]


@dataclass
class RBSlot:
    id: str
    user_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=_utc_iso)


@dataclass
class RBEntity:
    id: str
    user_id: str
    slot_id: str
    name: str
    description: str = ""
    embedding: list[float] | None = None
    source_entity_id: str | None = None  # left-brain cognitive_graph Entity.id; only filled for "relation node" entities
    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)


class RightBrainGraphStore:
    """SQLite storage for the right-brain slot->entity->memory three-layer graph. Thread-safe: a new connection per operation."""

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
            CREATE TABLE IF NOT EXISTS rb_slots (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                UNIQUE (user_id, name)
            );

            CREATE TABLE IF NOT EXISTS rb_entities (
                id                TEXT PRIMARY KEY,
                user_id           TEXT NOT NULL,
                slot_id           TEXT NOT NULL,
                name              TEXT NOT NULL,
                description       TEXT NOT NULL DEFAULT '',
                embedding         TEXT,
                source_entity_id  TEXT,
                created_at        TEXT NOT NULL,
                updated_at        TEXT NOT NULL,
                UNIQUE (user_id, slot_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_rbe_slot ON rb_entities(user_id, slot_id);

            CREATE TABLE IF NOT EXISTS rb_entity_memories (
                id          TEXT PRIMARY KEY,
                entity_id   TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                memory_id   TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                UNIQUE (entity_id, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_rbem_entity ON rb_entity_memories(entity_id);
            CREATE INDEX IF NOT EXISTS idx_rbem_memory ON rb_entity_memories(user_id, memory_id);
            """)
            # Migration: old databases (CREATE TABLE IF NOT EXISTS has no effect on existing tables) get the
            # source_entity_id column added -- every rb_graph.sqlite created before relation nodes shipped lacks it.
            cols = {row["name"] for row in c.execute("PRAGMA table_info(rb_entities)")}
            if "source_entity_id" not in cols:
                c.execute("ALTER TABLE rb_entities ADD COLUMN source_entity_id TEXT")
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_rbe_source "
                "ON rb_entities(user_id, slot_id, source_entity_id)"
            )

    # ── Seed initialisation ───────────────────────────────────────────────────

    def ensure_seed_slots(self, user_id: str) -> None:
        """Ensure this user has the 5 initial slots (+ the 8 initial entities under emotion). Idempotent, safe to call repeatedly."""
        for slot_name, slot_desc, seed_entities in SEED_SLOTS:
            slot = self.get_or_create_slot(user_id, slot_name, description=slot_desc)
            for ent_name in seed_entities:
                self.get_or_create_entity(user_id, slot.id, ent_name)

    # ── Slot ──────────────────────────────────────────────────────────────────

    def get_or_create_slot(self, user_id: str, name: str, *, description: str = "") -> RBSlot:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_slots WHERE user_id=? AND name=?", (user_id, name)
            ).fetchone()
            if row:
                return _row_to_slot(row)
            slot = RBSlot(id=_new_id(), user_id=user_id, name=name, description=description)
            c.execute(
                "INSERT INTO rb_slots (id, user_id, name, description, created_at) VALUES (?,?,?,?,?)",
                (slot.id, slot.user_id, slot.name, slot.description, slot.created_at),
            )
            return slot

    def get_slot_by_name(self, user_id: str, name: str) -> RBSlot | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_slots WHERE user_id=? AND name=?", (user_id, name)
            ).fetchone()
        return _row_to_slot(row) if row else None

    def list_slots(self, user_id: str) -> list[RBSlot]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM rb_slots WHERE user_id=? ORDER BY name", (user_id,)
            ).fetchall()
        return [_row_to_slot(r) for r in rows]

    def set_slot_description(self, slot_id: str, description: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE rb_slots SET description=? WHERE id=?", (description, slot_id))

    # ── Entity (exact name match, for seed / fixed-vocabulary cases) ─────────────

    def get_or_create_entity(
        self,
        user_id: str,
        slot_id: str,
        name: str,
        *,
        description: str = "",
        embedding: list[float] | None = None,
    ) -> RBEntity:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? AND name=?",
                (user_id, slot_id, name),
            ).fetchone()
            if row:
                return _row_to_entity(row)
            return self._insert_entity(c, user_id, slot_id, name, description, embedding)

    # ── Entity (semantic similarity match, for freely generated affective entities) ──

    def find_similar_entity(
        self,
        user_id: str,
        slot_id: str,
        embedding: list[float],
        *,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> RBEntity | None:
        """Find the most semantically similar entity in the same slot; return None if not similar enough."""
        best: RBEntity | None = None
        best_sim = -1.0
        for ent in self.get_entities_for_slot(user_id, slot_id):
            if ent.embedding is None:
                continue
            sim = _cosine(embedding, ent.embedding)
            if sim > best_sim:
                best_sim, best = sim, ent
        return best if best is not None and best_sim >= threshold else None

    def get_or_create_entity_semantic(
        self,
        user_id: str,
        slot_id: str,
        name: str,
        embedding: list[float],
        *,
        description: str = "",
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> tuple[RBEntity, bool]:
        """First look for an existing entity by semantic similarity and reuse it if found; otherwise create a new one.

        Returns
        -------
        (entity, created) -- created=True means a new entity was created this time.
        """
        existing = self.find_similar_entity(user_id, slot_id, embedding, threshold=threshold)
        if existing is not None:
            return existing, False
        with self._conn() as c:
            ent = self._insert_entity(c, user_id, slot_id, name, description, embedding)
        return ent, True

    def _insert_entity(
        self,
        c: sqlite3.Connection,
        user_id: str,
        slot_id: str,
        name: str,
        description: str,
        embedding: list[float] | None,
        source_entity_id: str | None = None,
    ) -> RBEntity:
        now = _utc_iso()
        ent = RBEntity(
            id=_new_id(), user_id=user_id, slot_id=slot_id, name=name,
            description=description, embedding=embedding,
            source_entity_id=source_entity_id, created_at=now, updated_at=now,
        )
        c.execute(
            """INSERT INTO rb_entities
               (id, user_id, slot_id, name, description, embedding, source_entity_id,
                created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ent.id, ent.user_id, ent.slot_id, ent.name, ent.description,
             json.dumps(embedding) if embedding is not None else None,
             ent.source_entity_id, ent.created_at, ent.updated_at),
        )
        return ent

    # ── Entity (linked exactly by left-brain entity.id, for "relation node" cases) ──
    # A left-brain entity (person/place/project...) always maps to the same right-brain relation node, via exact
    # source_entity_id match rather than semantic similarity -- semantic similarity is for deduplicating freely generated
    # abstract labels and is not needed here: renaming/merging a left-brain entity does not affect matching (the ID stays), the same
    # idea as today's fix of using the real entity.id for anchors instead of the name string.

    def get_entity_by_source_id(
        self, user_id: str, slot_id: str, source_entity_id: str,
    ) -> RBEntity | None:
        with self._conn() as c:
            row = c.execute(
                """SELECT * FROM rb_entities
                   WHERE user_id=? AND slot_id=? AND source_entity_id=?""",
                (user_id, slot_id, source_entity_id),
            ).fetchone()
        return _row_to_entity(row) if row else None

    def get_or_create_entity_by_source_id(
        self, user_id: str, slot_id: str, source_entity_id: str, name: str,
    ) -> tuple[RBEntity, bool]:
        """Find/create a relation node by left-brain entity.id.

        Returns
        -------
        (entity, created) -- created=True means a new entity was created this time.
        """
        existing = self.get_entity_by_source_id(user_id, slot_id, source_entity_id)
        if existing is not None:
            return existing, False
        with self._conn() as c:
            # A node with this name exists but its source_entity_id differs: the left brain may assign
            # more than one entity.id to the same real-world thing (a same-name entity re-extracted in a later
            # utterance / with a different entity_type), and rb_entities has UNIQUE(user_id,
            # slot_id, name) -- a plain INSERT would hit the unique constraint and raise, the caller
            # (the relation-node loop in core.py::_finish_ingest) would abort entirely, and this
            # utterance's remaining entity anchors and link_memory calls would all be lost. Same name means same thing,
            # so reuse the existing node; if it has not claimed a source_entity_id yet, claim it now
            # so later lookups can take the faster source_id path.
            row = c.execute(
                "SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? AND name=?",
                (user_id, slot_id, name),
            ).fetchone()
            if row is not None:
                ent = _row_to_entity(row)
                if not ent.source_entity_id:
                    now = _utc_iso()
                    c.execute(
                        "UPDATE rb_entities SET source_entity_id=?, updated_at=? WHERE id=?",
                        (source_entity_id, now, ent.id),
                    )
                    ent.source_entity_id = source_entity_id
                    ent.updated_at = now
                return ent, False
            ent = self._insert_entity(
                c, user_id, slot_id, name, "", None, source_entity_id=source_entity_id,
            )
        return ent, True

    def get_entities_for_slot(self, user_id: str, slot_id: str,
                              *, newest_first: bool = False) -> list[RBEntity]:
        """Sorted by name by default (stable, good for display). ``newest_first`` sorts by insertion order, newest first --
        the mind map reserves a few seats for newly grown entities, and when sorting by name where a new one lands depends only on its name."""
        order = "rowid DESC" if newest_first else "name"
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? ORDER BY {order}",
                (user_id, slot_id),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]

    def get_entity_by_name(self, user_id: str, slot_id: str, name: str) -> RBEntity | None:
        """Exact lookup by name (read-only, never creates) -- for retrieval, unlike get_or_create_entity,
        which creates an empty one when nothing is found."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rb_entities WHERE user_id=? AND slot_id=? AND name=?",
                (user_id, slot_id, name),
            ).fetchone()
        return _row_to_entity(row) if row else None

    def get_entity(self, entity_id: str) -> RBEntity | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM rb_entities WHERE id=?", (entity_id,)).fetchone()
        return _row_to_entity(row) if row else None

    def get_slot(self, slot_id: str) -> RBSlot | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM rb_slots WHERE id=?", (slot_id,)).fetchone()
        return _row_to_slot(row) if row else None

    def set_entity_description(self, entity_id: str, description: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE rb_entities SET description=?, updated_at=? WHERE id=?",
                (description, _utc_iso(), entity_id),
            )

    # ── Entity ↔ Memory ─────────────────────────────────────────────────────

    def link_memory(self, entity_id: str, user_id: str, memory_id: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO rb_entity_memories
                   (id, entity_id, user_id, memory_id, created_at)
                   VALUES (?,?,?,?,?)""",
                (_new_id(), entity_id, user_id, memory_id, _utc_iso()),
            )

    def get_memories_for_entity(self, entity_id: str) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT memory_id FROM rb_entity_memories WHERE entity_id=?", (entity_id,)
            ).fetchall()
        return [r["memory_id"] for r in rows]

    def get_entities_for_memory(self, user_id: str, memory_id: str) -> list[RBEntity]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT rbe.* FROM rb_entities rbe
                   JOIN rb_entity_memories rbem ON rbem.entity_id = rbe.id
                   WHERE rbem.user_id=? AND rbem.memory_id=?""",
                (user_id, memory_id),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]


def _row_to_slot(row: sqlite3.Row) -> RBSlot:
    return RBSlot(
        id=row["id"], user_id=row["user_id"], name=row["name"],
        description=row["description"], created_at=row["created_at"],
    )


def _row_to_entity(row: sqlite3.Row) -> RBEntity:
    return RBEntity(
        id=row["id"], user_id=row["user_id"], slot_id=row["slot_id"],
        name=row["name"], description=row["description"],
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        source_entity_id=row["source_entity_id"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )
