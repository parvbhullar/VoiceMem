"""Left-brain slot -> entity graph layer.

Hierarchy: slot (base SlotV2 / dynamic_slot emerged from the subgraph mechanism)
        └── entity (a concrete node under a slot, e.g. "Project X" under "work")
              └── memory (a concrete memory id attached to an entity)

slot_ref is an opaque string identifier that supports both slot sources:
  - base slot   : the string value of a SlotV2, e.g. "work"
  - dynamic_slot: the name used by DynamicSlotStore.create_dynamic_slot() (emerged via SubgraphManager)

It doesn't matter which kind a slot_ref comes from -- the caller decides what to pass.

Entity deduplication relies on semantic similarity (not exact string matching): a new fact
is first compared against the embeddings of existing entities under the same slot; if it is
similar enough the existing entity is reused, otherwise a new one is created.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

#: Warn about dimension mismatches only once, instead of spamming every round.
_WARNED_DIM: set = set()

from supermem.utils.common._graph_common import cosine as _cosine, new_id as _new_id, utc_iso as _utc_iso

DEFAULT_MATCH_THRESHOLD = 0.65


@dataclass
class GraphEntity:
    id: str
    user_id: str
    slot_ref: str
    name: str
    description: str = ""
    embedding: list[float] | None = None
    can_split: bool = True
    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)


class GraphEntityStore:
    """SQLite access to the left-brain three-level slot -> entity -> memory graph. Thread-safe: a new connection per operation."""

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
            CREATE TABLE IF NOT EXISTS graph_entities (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                slot_ref    TEXT NOT NULL,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                embedding   TEXT,
                can_split   INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                UNIQUE (user_id, slot_ref, name)
            );
            CREATE INDEX IF NOT EXISTS idx_ge_slot ON graph_entities(user_id, slot_ref);

            CREATE TABLE IF NOT EXISTS graph_entity_memories (
                id          TEXT PRIMARY KEY,
                entity_id   TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                memory_id   TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                UNIQUE (entity_id, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_gem_entity ON graph_entity_memories(entity_id);
            CREATE INDEX IF NOT EXISTS idx_gem_memory ON graph_entity_memories(user_id, memory_id);

            CREATE TABLE IF NOT EXISTS graph_query_activations (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                query_id    TEXT NOT NULL,
                entity_id   TEXT NOT NULL,
                session_id  TEXT,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gqa_user_query ON graph_query_activations(user_id, query_id);
            """)
            # The ALTER TABLE for the session_id column must finish first, before an
            # index can be built on that column -- previously the order was reversed
            # (the index referencing session_id was created inside executescript, with
            # ALTER TABLE after it), which blew up on old databases that existed before
            # Phase 4: CREATE TABLE IF NOT EXISTS is a no-op on an existing table and
            # doesn't add the new column, so the following CREATE INDEX fails with a
            # missing session_id column and the ALTER TABLE step never runs (real repro:
            # the graph_entities.sqlite used by openai_voice_demo since Phase 0 only
            # surfaced "no such column: session_id" during the Phase 8 full-pipeline
            # regression test). New databases (whose CREATE TABLE already includes
            # session_id) are unaffected; only old databases upgraded across versions
            # hit this ordering bug.
            try:
                c.execute("ALTER TABLE graph_query_activations ADD COLUMN session_id TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists — skip
            c.execute("CREATE INDEX IF NOT EXISTS idx_gqa_user_session ON graph_query_activations(user_id, session_id)")

    # ── Query Activation (used by the cluster-emergence ρ formula) ─────────────
    # Records "which graph_entities each retrieval activated", so SubgraphManager can
    # compute the paper's ρ(H) = (1/|Q|)·Σ_{q∈Q} |A_q∩H|/|A_q∪H| -- the entity space
    # here must be the same one SubgraphManager uses to judge candidate subsets
    # (graph_entities, not the NER entities of cognitive_graph), otherwise the
    # intersections/unions computed are meaningless.

    def record_query_activation(
        self, user_id: str, query_id: str, entity_ids: list[str], *, session_id: str | None = None
    ) -> None:
        """Record the set of graph_entities activated by one query (A_q). Nothing is recorded if empty.

        session_id: in the paper, Q is the set of queries in the "current session", not the
        lifetime history -- this stores the session in which the retrieval was triggered, so
        compute_rho() can filter by session. When the caller doesn't pass session_id (no
        session concept), NULL is stored; compute_rho() likewise falls back to the old
        behaviour (full history) when not given one, so not every caller is forced to upgrade.
        """
        if not entity_ids:
            return
        now = _utc_iso()
        with self._conn() as c:
            c.executemany(
                "INSERT INTO graph_query_activations (id, user_id, query_id, entity_id, session_id, created_at)"
                " VALUES (?,?,?,?,?,?)",
                [(_new_id(), user_id, query_id, eid, session_id, now) for eid in set(entity_ids)],
            )

    def compute_rho(
        self, user_id: str, candidate_entity_ids: set[str], *, session_id: str | None = None
    ) -> float:
        """ρ(H) = (1/|Q|)·Σ_{q∈Q} |A_q∩H|/|A_q∪H| -- H is the candidate entity subset.

        When session_id is passed, Q is restricted to the query activations recorded in
        that session (literally per the paper: Q is the set of queries in the current
        session, not the lifetime history); when not passed it falls back to the old
        behaviour and Q is all of the user's historical query activations (callers without
        a session concept, e.g. test/dev scripts not wired to SessionTracker, are
        unaffected). Returns 0.0 when there are no matching historical query activations
        (no query-behaviour signal, so it can't be evaluated -- the caller decides the
        fallback in that case)."""
        if not candidate_entity_ids:
            return 0.0
        with self._conn() as c:
            if session_id is not None:
                rows = c.execute(
                    "SELECT query_id, entity_id FROM graph_query_activations WHERE user_id=? AND session_id=?",
                    (user_id, session_id),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT query_id, entity_id FROM graph_query_activations WHERE user_id=?",
                    (user_id,),
                ).fetchall()
        if not rows:
            return 0.0
        by_query: dict[str, set[str]] = {}
        for r in rows:
            by_query.setdefault(r["query_id"], set()).add(r["entity_id"])
        total = 0.0
        for a_q in by_query.values():
            union = a_q | candidate_entity_ids
            if not union:
                continue
            total += len(a_q & candidate_entity_ids) / len(union)
        return total / len(by_query)

    # ── Entity (exact name matching, suited to seed/fixed-vocabulary scenarios) ──

    def get_or_create_entity(
        self,
        user_id: str,
        slot_ref: str,
        name: str,
        *,
        description: str = "",
        embedding: list[float] | None = None,
    ) -> GraphEntity:
        """Return the entity if the same (user_id, slot_ref, name) already exists, otherwise create it."""
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM graph_entities WHERE user_id=? AND slot_ref=? AND name=?",
                (user_id, slot_ref, name),
            ).fetchone()
            if row:
                return _row_to_entity(row)
            return self._insert_entity(c, user_id, slot_ref, name, description, embedding)

    # ── Entity (semantic similarity matching, suited to freely generated emotional/factual entities) ──

    def find_similar_entity(
        self,
        user_id: str,
        slot_ref: str,
        embedding: list[float],
        *,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> GraphEntity | None:
        """Find the most semantically similar entity under the same slot; return None if not similar enough."""
        best: GraphEntity | None = None
        best_sim = -1.0
        mismatched = 0
        for ent in self.get_entities_for_slot(user_id, slot_ref):
            if ent.embedding is None:
                continue
            if len(ent.embedding) != len(embedding):
                mismatched += 1          # embedder was changed; old vector dimensions don't match
                continue
            sim = _cosine(embedding, ent.embedding)
            if sim > best_sim:
                best_sim, best = sim, ent
        if mismatched and not _WARNED_DIM:
            _WARNED_DIM.add(1)
            print(f"[GraphEntity] ⚠ {mismatched} entity vectors have a dimension that doesn't match the current embedder; "
                  "skipped -- after switching embedders the old vectors are invalid and semantic dedup will not work. "
                  "Re-embed them or clear these vectors.", flush=True)
        return best if best is not None and best_sim >= threshold else None

    def get_or_create_entity_semantic(
        self,
        user_id: str,
        slot_ref: str,
        name: str,
        embedding: list[float],
        *,
        description: str = "",
        threshold: float = DEFAULT_MATCH_THRESHOLD,
    ) -> tuple[GraphEntity, bool]:
        """First look for an existing entity by semantic similarity and reuse it if found; otherwise create one.

        Returns
        -------
        (entity, created) — created=True means a new entity was created this time.
        """
        existing = self.find_similar_entity(user_id, slot_ref, embedding, threshold=threshold)
        if existing is not None:
            return existing, False
        with self._conn() as c:
            ent = self._insert_entity(c, user_id, slot_ref, name, description, embedding)
        return ent, True

    def _insert_entity(
        self,
        c: sqlite3.Connection,
        user_id: str,
        slot_ref: str,
        name: str,
        description: str,
        embedding: list[float] | None,
    ) -> GraphEntity:
        now = _utc_iso()
        ent = GraphEntity(
            id=_new_id(), user_id=user_id, slot_ref=slot_ref, name=name,
            description=description, embedding=embedding, can_split=True,
            created_at=now, updated_at=now,
        )
        c.execute(
            """INSERT INTO graph_entities
               (id, user_id, slot_ref, name, description, embedding, can_split, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (ent.id, ent.user_id, ent.slot_ref, ent.name, ent.description,
             json.dumps(embedding) if embedding is not None else None,
             1, ent.created_at, ent.updated_at),
        )
        return ent

    def get_entities_for_slot(self, user_id: str, slot_ref: str) -> list[GraphEntity]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM graph_entities WHERE user_id=? AND slot_ref=? ORDER BY name",
                (user_id, slot_ref),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]

    def update_slot_ref(self, entity_id: str, new_slot_ref: str) -> None:
        """Formally move an entity under a new slot (called after the subgraph judgement passes and the new slot is created).

        The target slot may already contain an entity with the same name -- this happens
        because once an entity is moved away, the semantic dedup at write time is scoped to
        the slot_ref, so when the same concept is mentioned again a new same-named entity is
        created under the original slot (it can't find the one that was moved); if that new
        one is later also moved to the same target slot by subgraph judgement, it collides
        with the earlier one on UNIQUE(user_id, slot_ref, name) -- in that case they are
        treated as the same concept and merged into one: this entity's memory links are
        transferred over and this entity is deleted, rather than letting the UPDATE crash.
        """
        with self._conn() as c:
            row = c.execute("SELECT * FROM graph_entities WHERE id=?", (entity_id,)).fetchone()
            if row is None:
                return
            name, user_id = row["name"], row["user_id"]
            existing = c.execute(
                "SELECT id FROM graph_entities WHERE user_id=? AND slot_ref=? AND name=? AND id!=?",
                (user_id, new_slot_ref, name, entity_id),
            ).fetchone()

            if existing is None:
                c.execute(
                    "UPDATE graph_entities SET slot_ref=?, updated_at=? WHERE id=?",
                    (new_slot_ref, _utc_iso(), entity_id),
                )
                return

            target_id = existing["id"]
            mem_rows = c.execute(
                "SELECT memory_id, created_at FROM graph_entity_memories WHERE entity_id=?",
                (entity_id,),
            ).fetchall()
            for mrow in mem_rows:
                c.execute(
                    """INSERT OR IGNORE INTO graph_entity_memories
                       (id, entity_id, user_id, memory_id, created_at)
                       VALUES (?,?,?,?,?)""",
                    (_new_id(), target_id, user_id, mrow["memory_id"], mrow["created_at"]),
                )
            c.execute("DELETE FROM graph_entity_memories WHERE entity_id=?", (entity_id,))
            c.execute("DELETE FROM graph_entities WHERE id=?", (entity_id,))

    def get_entity(self, entity_id: str) -> GraphEntity | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM graph_entities WHERE id=?", (entity_id,)
            ).fetchone()
        return _row_to_entity(row) if row else None

    def set_description(self, entity_id: str, description: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE graph_entities SET description=?, updated_at=? WHERE id=?",
                (description, _utc_iso(), entity_id),
            )

    def mark_cannot_split(self, entity_id: str) -> None:
        """Permanently set can_split to False. One-way operation -- no reverse set-True API is
        provided, to avoid misuse; new entities default to True, so reactivation should
        create/link a new entity rather than flipping an old entity's can_split back."""
        with self._conn() as c:
            c.execute(
                "UPDATE graph_entities SET can_split=0, updated_at=? WHERE id=?",
                (_utc_iso(), entity_id),
            )

    # ── Entity ↔ Memory ─────────────────────────────────────────────────────

    def link_memory(self, entity_id: str, user_id: str, memory_id: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO graph_entity_memories
                   (id, entity_id, user_id, memory_id, created_at)
                   VALUES (?,?,?,?,?)""",
                (_new_id(), entity_id, user_id, memory_id, _utc_iso()),
            )

    def get_entities_for_memory(self, user_id: str, memory_id: str) -> list[GraphEntity]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT ge.* FROM graph_entities ge
                   JOIN graph_entity_memories gem ON gem.entity_id = ge.id
                   WHERE gem.user_id=? AND gem.memory_id=?""",
                (user_id, memory_id),
            ).fetchall()
        return [_row_to_entity(r) for r in rows]


def _row_to_entity(row: sqlite3.Row) -> GraphEntity:
    return GraphEntity(
        id=row["id"], user_id=row["user_id"], slot_ref=row["slot_ref"],
        name=row["name"], description=row["description"],
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        can_split=bool(row["can_split"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
    )
