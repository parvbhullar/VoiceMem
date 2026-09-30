"""SQLite storage layer for the cognitive graph.

Tables:
  entities            — cognitive graph nodes (people/projects/knowledge/tasks, etc.)
  entity_edges        — directed edges between nodes (typed relation)
  entity_memory_links — links attaching memories back to nodes (with role)
  memories            — memory metadata wrapper (slot/sensitivity/TTL)
  slot_profiles       — summary snapshot per slot
  affective_edges     — right-brain affective edges (reserved)
"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from supermem.utils.common._graph_common import cosine as _cosine, new_id as _new_id, utc_iso as _utc_iso

from .slot_v2 import SlotV2
from .types import (
    AffectiveEdge,
    AnnotatedFact,
    Entity,
    EntityEdge,
    EntityMemoryLink,
    EntityMemoryRole,
    EntityType,
    MemoryRecord,
    SlotProfile,
)


def normalize_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(u"^[\u201c\u201d\u2018\u2019\"']+|[\u201c\u201d\u2018\u2019\"']+$", "", s)
    return s


# Real bug found via live testing: SlotV2(row["slot"]) crashes with
# ValueError on rows written under the OLDER, pre-unification SlotType
# taxonomy (types.py's 7-category "people/projects/knowledge/tasks/places/
# routines/assets" set, not SlotV2's "work/finance/relationships/health/
# goals/daily_life/knowledge") -- confirmed real rows in a live memory store
# with slot='people' (person entities from before the Phase 2 taxonomy
# unification, or some other still-live write path never fully migrated).
# Every caller of find_entities()/get_entity() has NO try/except around it,
# so one bad row raises out of a plain list comprehension and kills the
# whole query -- and because AnchorRouter._build_anchors() (the right
# brain's entity-anchor resolution, called on EVERY Search()) calls
# find_entities() unconditionally, this was silently zeroing out ALL right-
# brain results for the entire user on every single search, not just
# queries touching the bad entity. Mirrors voice_input.py's
# _VOICE_SLOT_TO_SLOTV2 mapping for the same old->new taxonomy migration;
# anything still unrecognized falls back to KNOWLEDGE rather than crashing.
_LEGACY_SLOT_TO_SLOTV2 = {
    "people": "relationships", "person": "relationships",
    "projects": "work", "project": "work", "tasks": "work", "task": "work",
    "places": "daily_life", "place": "daily_life",
    "routines": "daily_life", "routine": "daily_life",
    "assets": "daily_life", "asset": "daily_life",
}


def _coerce_slot_v2(raw: str) -> SlotV2:
    try:
        return SlotV2(raw)
    except ValueError:
        return SlotV2(_LEGACY_SLOT_TO_SLOTV2.get(raw, "knowledge"))


# Same threshold as slot_split/graph_entity_store.py -- both sides do "semantic dedup
# of the same entity", so the standard should not differ.
SEMANTIC_MATCH_THRESHOLD = 0.65

# entity_edges.weight is an accumulated count of "how many times this relation was observed/restated"
# (1 on first creation, then +1 each time the same edge is extracted again -- not capped by taking the
# max like confidence). strong/weak is a thresholded classification of this accumulated weight: a
# relation seen only once is still weak evidence (possibly a mis-extraction or a one-off mention); it
# is upgraded to strong (a real, stable relation) only after recurring >=2 times.
EDGE_STRONG_THRESHOLD = 2.0

# Exponential decay by updated_at at read time (no separate background decay job needed): the longer
# since the edge was last reinforced, the lower the effective weight returned on this read, reflecting
# "a relation not mentioned for a long time is less active" -- but the raw weight stored in the DB is
# not rewritten; decay only affects the value returned by this read.
EDGE_DECAY_HALFLIFE_DAYS = 30.0

# Memory heat: the same "exponential decay by updated_at/last_hit_at at read time, without changing the
# raw stored value" pattern (reusing the same _decayed_weight() implementation as entity_edges.weight),
# meaning "how long since it was last hit by retrieval". Each hit adds heat +1 (accumulated, not max);
# the half-life is shorter than entity_edges -- judging whether a memory is "still useful" should be more
# sensitive than relation strength: things the user hasn't asked about for weeks should cool down faster
# than a relationship with a person not mentioned for weeks.
MEMORY_HEAT_DECAY_HALFLIFE_DAYS = 14.0

# Archive threshold: only when the read-time decayed heat is below this value and the memory has existed
# long enough (see min_age_days of list_archivable_memories) is it considered "unused for a long time",
# and the caller decides whether to actually archive it (this only decides, it does not execute --
# executing requires mem0's expiration_date, see mem0_backend_store.py::archive_memory).
ARCHIVE_HEAT_THRESHOLD = 0.3


def _decayed_weight(weight: float, updated_at: str, *, halflife_days: float = EDGE_DECAY_HALFLIFE_DAYS) -> float:
    """Exponential decay: returns weight unchanged when elapsed=0, halved every half-life."""
    try:
        updated = datetime.fromisoformat(updated_at)
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        elapsed_days = (datetime.now(timezone.utc) - updated).total_seconds() / 86400.0
    except (TypeError, ValueError):
        return weight
    if elapsed_days <= 0 or halflife_days <= 0:
        return weight
    return weight * (0.5 ** (elapsed_days / halflife_days))


def make_entity_id(entity_type: EntityType, name_norm: str) -> str:
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "_", name_norm)[:32].strip("_")
    return f"{entity_type.value}_{slug}_{uuid.uuid4().hex[:6]}"


def normalize_relation(rel: str) -> str:
    s = rel.strip().lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "related_to"


def _normalize_role(raw: str | None) -> str:
    """Normalize the LLM's role description into one of the four EntityMemoryRole values."""
    if not raw:
        return EntityMemoryRole.CONTEXT.value
    r = raw.strip().lower()
    if r in ("subject", "speaker"):
        return EntityMemoryRole.SUBJECT.value
    if r in ("object", "target"):
        return EntityMemoryRole.OBJECT.value
    if r in ("owner", "owner of"):
        return EntityMemoryRole.OWNER.value
    return EntityMemoryRole.CONTEXT.value


class CognitiveGraphStore:
    """SQLite cognitive graph store. Thread-safe: opens a new connection per operation."""

    def __init__(self, db_path: Path | str, embedder: Any = None) -> None:
        """embedder: optional object with an `embed_texts(list[str]) -> list[list[float]]`
        method (same protocol as local_memory_store.TextEmbedder). Only when passed does
        upsert_entity do semantic dedup (merging the same entity phrased differently into one);
        when omitted it falls back to the old behaviour (exact string match) without error --
        staying backward compatible with any caller that has not upgraded its call style.
        """
        self._path = Path(db_path)
        self._embedder = embedder
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._migrate_schema()

    # ── Connection ────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        return c

    # ── Schema ────────────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                name        TEXT NOT NULL,
                name_norm   TEXT NOT NULL,
                slot        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                confidence  REAL NOT NULL DEFAULT 1.0,
                importance  REAL NOT NULL DEFAULT 0.5,
                aliases     TEXT NOT NULL DEFAULT '[]',
                properties  TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ent_user ON entities(user_id);
            CREATE INDEX IF NOT EXISTS idx_ent_norm ON entities(user_id, name_norm, entity_type);

            CREATE TABLE IF NOT EXISTS entity_edges (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                from_entity_id      TEXT NOT NULL,
                to_entity_id        TEXT NOT NULL,
                relation_type       TEXT NOT NULL,
                role_label          TEXT,
                confidence          REAL NOT NULL DEFAULT 1.0,
                weight              REAL NOT NULL DEFAULT 1.0,
                edge_type           TEXT NOT NULL DEFAULT 'weak',
                status              TEXT NOT NULL DEFAULT 'active',
                evidence_memory_ids TEXT NOT NULL DEFAULT '[]',
                created_at          TEXT NOT NULL,
                updated_at          TEXT NOT NULL,
                UNIQUE (user_id, from_entity_id, to_entity_id, relation_type)
            );
            CREATE INDEX IF NOT EXISTS idx_edge_from ON entity_edges(user_id, from_entity_id);
            CREATE INDEX IF NOT EXISTS idx_edge_to   ON entity_edges(user_id, to_entity_id);

            CREATE TABLE IF NOT EXISTS entity_memory_links (
                id             TEXT PRIMARY KEY,
                memory_id      TEXT NOT NULL,
                entity_id      TEXT NOT NULL,
                user_id        TEXT NOT NULL,
                role           TEXT NOT NULL DEFAULT 'context',
                relation_hint  TEXT,
                created_at     TEXT NOT NULL,
                UNIQUE (memory_id, entity_id)
            );
            CREATE INDEX IF NOT EXISTS idx_eml_mem ON entity_memory_links(memory_id);
            CREATE INDEX IF NOT EXISTS idx_eml_ent ON entity_memory_links(entity_id);

            CREATE TABLE IF NOT EXISTS memories (
                id           TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                slot         TEXT NOT NULL,
                memory_type  TEXT NOT NULL DEFAULT 'fact',
                content      TEXT NOT NULL,
                confidence   REAL NOT NULL DEFAULT 1.0,
                sensitivity  REAL NOT NULL DEFAULT 0.0,
                ttl          INTEGER,
                heat         REAL NOT NULL DEFAULT 1.0,
                last_hit_at  TEXT,
                created_at   TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, slot);

            CREATE TABLE IF NOT EXISTS slot_profiles (
                user_id      TEXT NOT NULL,
                slot         TEXT NOT NULL,
                summary      TEXT NOT NULL DEFAULT '',
                entity_ids   TEXT NOT NULL DEFAULT '[]',
                entity_count INTEGER NOT NULL DEFAULT 0,
                memory_count INTEGER NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (user_id, slot)
            );

            CREATE TABLE IF NOT EXISTS affective_edges (
                id                  TEXT PRIMARY KEY,
                user_id             TEXT NOT NULL,
                from_entity_id      TEXT NOT NULL,
                to_entity_id        TEXT,
                trigger_frame       TEXT,
                emotion             TEXT,
                appraisal           TEXT,
                response_policy     TEXT,
                confidence          REAL NOT NULL DEFAULT 1.0,
                evidence_memory_ids TEXT NOT NULL DEFAULT '[]',
                created_at          TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ae_user ON affective_edges(user_id, from_entity_id);

            CREATE TABLE IF NOT EXISTS query_activations (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                query_id    TEXT NOT NULL,
                entity_id   TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_qa_user_query ON query_activations(user_id, query_id);
            CREATE INDEX IF NOT EXISTS idx_qa_user_entity ON query_activations(user_id, entity_id);
            """)

    def _migrate_schema(self) -> None:
        """Add missing columns to an existing database via ALTER TABLE; safe and idempotent."""
        migrations = [
            "ALTER TABLE entities ADD COLUMN importance REAL NOT NULL DEFAULT 0.5",
            "ALTER TABLE entities ADD COLUMN aliases TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE entities ADD COLUMN embedding TEXT",
            "ALTER TABLE entities ADD COLUMN description TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE entity_edges ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "ALTER TABLE entity_edges ADD COLUMN weight REAL NOT NULL DEFAULT 1.0",
            "ALTER TABLE entity_edges ADD COLUMN edge_type TEXT NOT NULL DEFAULT 'weak'",
            "ALTER TABLE entity_memory_links ADD COLUMN role TEXT NOT NULL DEFAULT 'context'",
            "ALTER TABLE entity_memory_links ADD COLUMN relation_hint TEXT",
            "ALTER TABLE slot_profiles ADD COLUMN summary TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE slot_profiles ADD COLUMN entity_ids TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE affective_edges ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0",
            "ALTER TABLE affective_edges ADD COLUMN evidence_memory_ids TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE memories ADD COLUMN heat REAL NOT NULL DEFAULT 1.0",
            "ALTER TABLE memories ADD COLUMN last_hit_at TEXT",
        ]
        with self._conn() as c:
            for sql in migrations:
                try:
                    c.execute(sql)
                except sqlite3.OperationalError:
                    pass  # column already exists — skip

    # ── Entity CRUD ───────────────────────────────────────────────────────────

    def upsert_entity(
        self,
        user_id: str,
        name: str,
        entity_type: EntityType,
        slot: SlotV2 | None = None,
        description: str | None = None,
        confidence: float = 1.0,
        importance: float = 0.5,
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> Entity:
        """Update an existing same-type entity with the same or a semantically similar name, otherwise create one. Returns Entity.

        Dedup has two layers: first an exact string match (the most common case, avoids a full
        table scan); only on a miss, and if an embedder was passed at construction, fall back to
        semantic similarity matching (cosine >= SEMANTIC_MATCH_THRESHOLD) -- previously there was only
        the exact-match path, so when the LLM phrased the same entity inconsistently across turns
        ("Project A" vs "the A project", aliases, real semantic rewrites beyond casing/punctuation
        differences) each got its own new node, the graph filled with duplicate entities, and
        one-hop neighbour expansion missed edges because entities were split. Without an embedder the
        behaviour is exactly as before, not affecting callers that have not upgraded.

        description (d_v): filled with the caller's source sentence when the entity is first created;
        not overwritten when the entity already exists with a description (the first source sentence
        best captures "what this is"; rewriting on every mention would make it flip back and forth)
        -- old entities with an empty description (pre-migration data) get filled in when a new
        description arrives, which does not count as overwriting.
        """
        name_norm = normalize_name(name)
        effective_slot = slot or SlotV2.KNOWLEDGE
        now = _utc_iso()

        embedding: list[float] | None = None
        if self._embedder is not None:
            try:
                embedding = self._embedder.embed_texts([name])[0]
            except Exception:
                embedding = None

        with self._conn() as c:
            # person/user are deduped only by exact name, no semantic matching -- two real people's
            # names that sound similar (e.g. "Jon" and "John") are quite likely different people; exact match is safer.
            if entity_type in (EntityType.PERSON, EntityType.USER):
                row = c.execute(
                    "SELECT * FROM entities WHERE user_id=? AND name_norm=?"
                    " AND entity_type IN ('person','user')",
                    (user_id, name_norm),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT * FROM entities WHERE user_id=? AND name_norm=? AND entity_type=?",
                    (user_id, name_norm, entity_type.value),
                ).fetchone()
                if row is None and embedding is not None:
                    candidates = c.execute(
                        "SELECT * FROM entities WHERE user_id=? AND entity_type=? AND embedding IS NOT NULL",
                        (user_id, entity_type.value),
                    ).fetchall()
                    best_row, best_sim = None, -1.0
                    for cand in candidates:
                        try:
                            cand_emb = json.loads(cand["embedding"])
                        except (TypeError, ValueError):
                            continue
                        sim = _cosine(embedding, cand_emb)
                        if sim > best_sim:
                            best_sim, best_row = sim, cand
                    if best_row is not None and best_sim >= SEMANTIC_MATCH_THRESHOLD:
                        row = best_row

            if row:
                new_conf = max(float(row["confidence"]), confidence)
                new_imp  = max(float(row["importance"]), importance)
                props = json.loads(row["properties"] or "{}")
                if properties:
                    props.update(properties)
                existing_aliases = json.loads(row["aliases"] or "[]")
                for a in (aliases or []):
                    if a not in existing_aliases:
                        existing_aliases.append(a)
                # Semantic match hit but the literal name differs -- record this phrasing as an alias
                # so the same phrasing can hit via exact match later without recomputing similarity each time.
                if name.strip() and name.strip() not in existing_aliases and normalize_name(name) != row["name_norm"]:
                    existing_aliases.append(name.strip())
                existing_desc = row["description"] if "description" in row.keys() else ""
                new_desc = existing_desc or (description or "").strip()
                c.execute(
                    """UPDATE entities
                       SET confidence=?, importance=?, aliases=?, properties=?, description=?, updated_at=?
                       WHERE id=?""",
                    (new_conf, new_imp,
                     json.dumps(existing_aliases, ensure_ascii=False),
                     json.dumps(props, ensure_ascii=False), new_desc, now, row["id"]),
                )
                return Entity(
                    id=row["id"], user_id=user_id,
                    entity_type=entity_type, name=row["name"],
                    name_norm=row["name_norm"], slot=effective_slot,
                    description=new_desc,
                    confidence=new_conf, importance=new_imp,
                    aliases=existing_aliases, properties=props,
                    created_at=row["created_at"], updated_at=now,
                )
            else:
                eid = make_entity_id(entity_type, name_norm)
                props = properties or {}
                alias_list = aliases or []
                desc = (description or "").strip()
                c.execute(
                    """INSERT INTO entities
                       (id,user_id,entity_type,name,name_norm,slot,description,confidence,importance,aliases,properties,embedding,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (eid, user_id, entity_type.value, name.strip(), name_norm,
                     effective_slot.value, desc, confidence, importance,
                     json.dumps(alias_list, ensure_ascii=False),
                     json.dumps(props, ensure_ascii=False),
                     json.dumps(embedding) if embedding is not None else None,
                     now, now),
                )
                return Entity(
                    id=eid, user_id=user_id,
                    entity_type=entity_type, name=name.strip(),
                    name_norm=name_norm, slot=effective_slot,
                    description=desc,
                    confidence=confidence, importance=importance,
                    aliases=alias_list, properties=props,
                    created_at=now, updated_at=now,
                )

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM entities WHERE id=?", (entity_id,)).fetchone()
        return self._row_to_entity(row) if row else None

    def find_entities(
        self, user_id: str, *,
        slots: list[SlotV2] | None = None,
        entity_ids: list[str] | None = None,
        name_norm: str | None = None,
    ) -> list[Entity]:
        wheres, params = ["user_id=?"], [user_id]
        if slots:
            ph = ",".join("?" * len(slots))
            wheres.append(f"slot IN ({ph})")
            params.extend(s.value for s in slots)
        if entity_ids:
            ph = ",".join("?" * len(entity_ids))
            wheres.append(f"id IN ({ph})")
            params.extend(entity_ids)
        if name_norm:
            wheres.append("name_norm=?")
            params.append(name_norm)
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM entities WHERE " + " AND ".join(wheres), params
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def find_entities_by_name_fuzzy(
        self, user_id: str, name_norm: str, *, slots: list[SlotV2] | None = None
    ) -> list[Entity]:
        """Exact match first, then fall back to LIKE '%name%'."""
        exact = self.find_entities(user_id, name_norm=name_norm, slots=slots)
        if exact:
            return exact
        wheres = ["user_id=?", "name_norm LIKE ?"]
        params: list = [user_id, f"%{name_norm}%"]
        if slots:
            ph = ",".join("?" * len(slots))
            wheres.append(f"slot IN ({ph})")
            params.extend(s.value for s in slots)
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM entities WHERE " + " AND ".join(wheres), params
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        props = json.loads(row["properties"] or "{}")
        # importance/aliases/description may not exist in migrated rows (handled by DEFAULT)
        importance = float(row["importance"]) if "importance" in row.keys() else 0.5
        aliases = json.loads(row["aliases"] or "[]") if "aliases" in row.keys() else []
        description = row["description"] if "description" in row.keys() else ""
        return Entity(
            id=row["id"], user_id=row["user_id"],
            entity_type=EntityType(row["entity_type"]),
            name=row["name"], name_norm=row["name_norm"],
            slot=_coerce_slot_v2(row["slot"]),
            description=description or "",
            confidence=float(row["confidence"]),
            importance=importance,
            aliases=aliases,
            properties=props if isinstance(props, dict) else {},
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    # ── Entity Edge CRUD ──────────────────────────────────────────────────────

    def upsert_edge(
        self,
        user_id: str,
        from_entity_id: str,
        to_entity_id: str,
        relation_type: str,
        *,
        role_label: str | None = None,
        confidence: float = 1.0,
        status: str = "active",
        evidence_memory_id: str | None = None,
    ) -> EntityEdge:
        rel = normalize_relation(relation_type)
        now = _utc_iso()
        with self._conn() as c:
            row = c.execute(
                """SELECT * FROM entity_edges
                   WHERE user_id=? AND from_entity_id=? AND to_entity_id=? AND relation_type=?""",
                (user_id, from_entity_id, to_entity_id, rel),
            ).fetchone()
            if row:
                eids = json.loads(row["evidence_memory_ids"] or "[]")
                # weight is only +1 when this is "new evidence" (the same memory_id passed in again
                # -- e.g. the same fact re-annotated -- should not inflate the weight); without an
                # evidence_memory_id we cannot dedup, so each call counts as new evidence.
                is_new_evidence = (not evidence_memory_id) or evidence_memory_id not in eids
                if evidence_memory_id and evidence_memory_id not in eids:
                    eids.append(evidence_memory_id)
                new_conf = max(float(row["confidence"]), confidence)
                old_weight = float(row["weight"]) if "weight" in row.keys() else 1.0
                new_weight = old_weight + 1.0 if is_new_evidence else old_weight
                new_edge_type = "strong" if new_weight >= EDGE_STRONG_THRESHOLD else "weak"
                c.execute(
                    """UPDATE entity_edges
                       SET confidence=?, weight=?, edge_type=?, evidence_memory_ids=?,
                           role_label=COALESCE(?,role_label), updated_at=?
                       WHERE id=?""",
                    (new_conf, new_weight, new_edge_type, json.dumps(eids), role_label, now, row["id"]),
                )
                return EntityEdge(
                    id=row["id"], user_id=user_id,
                    from_entity_id=from_entity_id, to_entity_id=to_entity_id,
                    relation_type=rel, role_label=role_label or row["role_label"],
                    confidence=new_conf, weight=new_weight, edge_type=new_edge_type,
                    status=row["status"] if "status" in row.keys() else "active",
                    evidence_memory_ids=eids,
                    created_at=row["created_at"], updated_at=now,
                )
            else:
                eid = _new_id()
                eids = [evidence_memory_id] if evidence_memory_id else []
                c.execute(
                    """INSERT INTO entity_edges
                       (id,user_id,from_entity_id,to_entity_id,relation_type,
                        role_label,confidence,weight,edge_type,status,evidence_memory_ids,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (eid, user_id, from_entity_id, to_entity_id, rel,
                     role_label, confidence, 1.0, "weak", status, json.dumps(eids), now, now),
                )
                return EntityEdge(
                    id=eid, user_id=user_id,
                    from_entity_id=from_entity_id, to_entity_id=to_entity_id,
                    relation_type=rel, role_label=role_label,
                    confidence=confidence, weight=1.0, edge_type="weak", status=status,
                    evidence_memory_ids=eids,
                    created_at=now, updated_at=now,
                )

    def edges_for_entity(self, entity_id: str, user_id: str) -> list[EntityEdge]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT * FROM entity_edges
                   WHERE user_id=? AND (from_entity_id=? OR to_entity_id=?)
                     AND status='active'""",
                (user_id, entity_id, entity_id),
            ).fetchall()
        return [self._row_to_edge(r) for r in rows]

    def _row_to_edge(self, row: sqlite3.Row) -> EntityEdge:
        # weight/edge_type may not exist in migrated rows (handled by DEFAULT).
        # Decay only applies to the returned value; the raw stored weight/updated_at are unchanged --
        # the next time this edge is reinforced, accumulation continues from the raw value, uninterrupted by reads.
        raw_weight = float(row["weight"]) if "weight" in row.keys() else 1.0
        decayed = _decayed_weight(raw_weight, row["updated_at"])
        edge_type = row["edge_type"] if "edge_type" in row.keys() else (
            "strong" if raw_weight >= EDGE_STRONG_THRESHOLD else "weak"
        )
        return EntityEdge(
            id=row["id"], user_id=row["user_id"],
            from_entity_id=row["from_entity_id"],
            to_entity_id=row["to_entity_id"],
            relation_type=row["relation_type"],
            role_label=row["role_label"],
            confidence=float(row["confidence"]),
            weight=decayed, edge_type=edge_type or "weak",
            status=row["status"] if "status" in row.keys() else "active",
            evidence_memory_ids=json.loads(row["evidence_memory_ids"] or "[]"),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    # ── Entity-Memory Link ────────────────────────────────────────────────────

    def link_memory(
        self,
        memory_id: str,
        entity_id: str,
        user_id: str,
        *,
        role: EntityMemoryRole = EntityMemoryRole.CONTEXT,
        relation_hint: str | None = None,
    ) -> None:
        now = _utc_iso()
        with self._conn() as c:
            c.execute(
                """INSERT INTO entity_memory_links
                   (id, memory_id, entity_id, user_id, role, relation_hint, created_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(memory_id, entity_id) DO UPDATE SET
                     role=CASE WHEN excluded.role != 'context' THEN excluded.role ELSE role END,
                     relation_hint=COALESCE(excluded.relation_hint, relation_hint)""",
                (_new_id(), memory_id, entity_id, user_id,
                 role.value, relation_hint, now),
            )

    def memory_ids_for_entities(self, entity_ids: list[str]) -> list[str]:
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT DISTINCT memory_id FROM entity_memory_links WHERE entity_id IN ({ph})",
                entity_ids,
            ).fetchall()
        return [r["memory_id"] for r in rows]

    def memory_ids_for_slots(self, user_id: str, slots: list[SlotV2]) -> list[str]:
        """Return ids of memories directly tagged with the given slots (queries memories.slot, not entity type)."""
        if not slots:
            return []
        ph = ",".join("?" * len(slots))
        with self._conn() as c:
            rows = c.execute(
                f"SELECT id FROM memories WHERE user_id=? AND slot IN ({ph})",
                [user_id, *[s.value for s in slots]],
            ).fetchall()
        return [r["id"] for r in rows]

    def entity_ids_for_memory(self, memory_id: str) -> list[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT entity_id FROM entity_memory_links WHERE memory_id=?",
                (memory_id,),
            ).fetchall()
        return [r["entity_id"] for r in rows]

    def all_linked_memory_ids(self, user_id: str) -> list[str]:
        """Return all memory_ids for this user that appear in entity_memory_links."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT memory_id FROM entity_memory_links WHERE user_id=?",
                (user_id,),
            ).fetchall()
        return [r["memory_id"] for r in rows]

    def all_memory_record_ids(self, user_id: str) -> list[str]:
        """Return all memory_ids for this user that already have a record in the memories table."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id FROM memories WHERE user_id=?",
                (user_id,),
            ).fetchall()
        return [r["id"] for r in rows]

    def neighbor_entity_ids(self, user_id: str, entity_ids: list[str]) -> list[str]:
        """Return 1-hop neighbour entity ids of the given entity set (excluding the seeds themselves)."""
        if not entity_ids:
            return []
        ph = ",".join("?" * len(entity_ids))
        with self._conn() as c:
            rows = c.execute(
                f"""SELECT DISTINCT
                        CASE WHEN from_entity_id IN ({ph}) THEN to_entity_id
                             ELSE from_entity_id END AS neighbor_id
                    FROM entity_edges
                    WHERE user_id=? AND status='active'
                      AND (from_entity_id IN ({ph}) OR to_entity_id IN ({ph}))""",
                # Placeholder order: the IN({ph}) inside CASE comes before user_id=? -- it used to be
                # [user_id] + ids*3, so user_id was bound to CASE's first placeholder and the real
                # user_id=? got an entity id, never matching any rows: one-hop neighbour expansion
                # never took effect (measured: f_nbr was 0 on all 821 questions).
                entity_ids + [user_id] + entity_ids * 2,
            ).fetchall()
        seed_set = set(entity_ids)
        return [r["neighbor_id"] for r in rows if r["neighbor_id"] not in seed_set]

    # ── Memory Record (the cognitive graph's own memory metadata) ─────────────────────────────────

    def upsert_memory_record(
        self,
        user_id: str,
        memory_id: str,
        slot: SlotV2,
        content: str,
        *,
        memory_type: str = "fact",
        confidence: float = 1.0,
        sensitivity: float = 0.0,
        ttl: int | None = None,
    ) -> MemoryRecord:
        now = _utc_iso()
        with self._conn() as c:
            c.execute(
                """INSERT INTO memories
                   (id, user_id, slot, memory_type, content, confidence, sensitivity, ttl, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     slot=excluded.slot, content=excluded.content,
                     confidence=excluded.confidence, updated_at=excluded.updated_at""",
                (memory_id, user_id, slot.value, memory_type, content,
                 confidence, sensitivity, ttl, now, now),
            )
        return MemoryRecord(
            id=memory_id, user_id=user_id, slot=slot,
            memory_type=memory_type, content=content,
            confidence=confidence, sensitivity=sensitivity, ttl=ttl,
            created_at=now, updated_at=now,
        )

    def get_memory_record(self, memory_id: str) -> MemoryRecord | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        if not row:
            return None
        return MemoryRecord(
            id=row["id"], user_id=row["user_id"],
            slot=_coerce_slot_v2(row["slot"]), memory_type=row["memory_type"],
            content=row["content"], confidence=float(row["confidence"]),
            sensitivity=float(row["sensitivity"]), ttl=row["ttl"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    # ── Memory heat (retrieval hit -> heat accumulates; exponential decay by last_hit_at on read) ────────────

    def record_memory_hits(self, memory_ids: list[str]) -> None:
        """A batch of real retrieval hits (usually the top-k returned by one Rank() call): each one's heat
        accumulates +1 (not max) and last_hit_at is refreshed. Only affects ids that already have a
        record in the ``memories`` table -- ids without a record (e.g. ones that have not gone through
        ingest_annotated_fact yet) are silently skipped rather than raising for a missing row, so the
        caller does not need to check first whether the cognitive graph has recorded them. Uses one
        executemany in a single connection, not one connection per hit."""
        if not memory_ids:
            return
        now = _utc_iso()
        with self._conn() as c:
            c.executemany(
                "UPDATE memories SET heat = heat + 1.0, last_hit_at = ? WHERE id = ?",
                [(now, mid) for mid in memory_ids],
            )

    def get_memory_heat(self, memory_id: str) -> float | None:
        """Return the heat exponentially decayed by last_hit_at (created_at if never hit);
        returns None if there is no such memory (not 0 -- 0 means "exists but very cold", None means
        "no bookkeeping for this memory at all"; the two mean different things)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT heat, last_hit_at, created_at FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        if not row:
            return None
        anchor = row["last_hit_at"] or row["created_at"]
        return _decayed_weight(float(row["heat"]), anchor, halflife_days=MEMORY_HEAT_DECAY_HALFLIFE_DAYS)

    def list_archivable_memories(
        self, user_id: str, *, min_age_days: float = 30.0, heat_threshold: float = ARCHIVE_HEAT_THRESHOLD,
    ) -> list[str]:
        """Return ids of this user's memories whose decayed heat is below the threshold and that have existed
        long enough -- "long enough" avoids misjudging freshly ingested memories that have not been retrieved
        even once as cold (a new memory's heat defaults to 1.0 with created_at as the decay anchor; without a
        minimum age, a memory written a few minutes ago would be treated as "never hit, already cold" and archived).
        This only decides; it does not actually archive (to execute, the caller takes these ids to
        Mem0BackendStore.archive_memory, see its docstring)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, heat, last_hit_at, created_at FROM memories WHERE user_id=?",
                (user_id,),
            ).fetchall()
        now = datetime.now(timezone.utc)
        archivable: list[str] = []
        for r in rows:
            anchor = r["last_hit_at"] or r["created_at"]
            decayed = _decayed_weight(float(r["heat"]), anchor, halflife_days=MEMORY_HEAT_DECAY_HALFLIFE_DAYS)
            if decayed >= heat_threshold:
                continue
            try:
                created = datetime.fromisoformat(r["created_at"])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400.0
            except (TypeError, ValueError):
                continue
            if age_days >= min_age_days:
                archivable.append(r["id"])
        return archivable

    # ── Slot Profile ──────────────────────────────────────────────────────────

    def refresh_slot_profile(
        self, user_id: str, slot: SlotV2, *, summary: str = ""
    ) -> SlotProfile:
        now = _utc_iso()
        with self._conn() as c:
            ec = c.execute(
                "SELECT COUNT(*) FROM entities WHERE user_id=? AND slot=?",
                (user_id, slot.value),
            ).fetchone()[0]
            mc = c.execute(
                """SELECT COUNT(DISTINCT eml.memory_id)
                   FROM entity_memory_links eml
                   JOIN entities e ON eml.entity_id=e.id
                   WHERE e.user_id=? AND e.slot=?""",
                (user_id, slot.value),
            ).fetchone()[0]
            # collect entity_ids for this slot
            eid_rows = c.execute(
                "SELECT id FROM entities WHERE user_id=? AND slot=?",
                (user_id, slot.value),
            ).fetchall()
            entity_ids = [r["id"] for r in eid_rows]

            c.execute(
                """INSERT INTO slot_profiles
                   (user_id, slot, summary, entity_ids, entity_count, memory_count, last_updated)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(user_id,slot) DO UPDATE SET
                     summary=CASE WHEN excluded.summary!='' THEN excluded.summary ELSE summary END,
                     entity_ids=excluded.entity_ids,
                     entity_count=excluded.entity_count,
                     memory_count=excluded.memory_count,
                     last_updated=excluded.last_updated""",
                (user_id, slot.value, summary,
                 json.dumps(entity_ids, ensure_ascii=False), ec, mc, now),
            )
        return SlotProfile(
            user_id=user_id, slot=slot, summary=summary,
            entity_ids=entity_ids, entity_count=ec,
            memory_count=mc, last_updated=now,
        )

    def slot_profiles(self, user_id: str) -> list[SlotProfile]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM slot_profiles WHERE user_id=?", (user_id,)
            ).fetchall()
        result = []
        for r in rows:
            entity_ids = json.loads(r["entity_ids"] or "[]") if "entity_ids" in r.keys() else []
            summary = r["summary"] if "summary" in r.keys() else ""
            result.append(SlotProfile(
                user_id=r["user_id"], slot=_coerce_slot_v2(r["slot"]),
                summary=summary, entity_ids=entity_ids,
                entity_count=r["entity_count"], memory_count=r["memory_count"],
                last_updated=r["last_updated"],
            ))
        return result

    # ── Affective Edge (right brain stub) ─────────────────────────────────────

    def upsert_affective_edge(
        self,
        user_id: str,
        from_entity_id: str,
        *,
        to_entity_id: str | None = None,
        trigger_frame: str | None = None,
        emotion: str | None = None,
        appraisal: str | None = None,
        response_policy: str | None = None,
        confidence: float = 1.0,
        evidence_memory_ids: list[str] | None = None,
    ) -> AffectiveEdge:
        now = _utc_iso()
        eid = _new_id()
        eids = evidence_memory_ids or []
        with self._conn() as c:
            c.execute(
                """INSERT INTO affective_edges
                   (id,user_id,from_entity_id,to_entity_id,trigger_frame,
                    emotion,appraisal,response_policy,confidence,evidence_memory_ids,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (eid, user_id, from_entity_id, to_entity_id,
                 trigger_frame, emotion, appraisal, response_policy,
                 confidence, json.dumps(eids), now),
            )
        return AffectiveEdge(
            id=eid, user_id=user_id, from_entity_id=from_entity_id,
            to_entity_id=to_entity_id, trigger_frame=trigger_frame,
            emotion=emotion, appraisal=appraisal,
            response_policy=response_policy, confidence=confidence,
            evidence_memory_ids=eids, created_at=now,
        )

    def affective_edges_for_entity(self, entity_id: str, user_id: str) -> list[AffectiveEdge]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM affective_edges WHERE user_id=? AND from_entity_id=?",
                (user_id, entity_id),
            ).fetchall()
        result = []
        for r in rows:
            conf = float(r["confidence"]) if "confidence" in r.keys() else 1.0
            eids = json.loads(r["evidence_memory_ids"] or "[]") if "evidence_memory_ids" in r.keys() else []
            result.append(AffectiveEdge(
                id=r["id"], user_id=r["user_id"],
                from_entity_id=r["from_entity_id"],
                to_entity_id=r["to_entity_id"],
                trigger_frame=r["trigger_frame"],
                emotion=r["emotion"], appraisal=r["appraisal"],
                response_policy=r["response_policy"],
                confidence=conf, evidence_memory_ids=eids,
                created_at=r["created_at"],
            ))
        return result

    # ── Core ingest ───────────────────────────────────────────────────────────

    def ingest_annotated_fact(
        self,
        user_id: str,
        annotated: "AnnotatedFact",
        memory_ids: list[str],
    ) -> list[Entity]:
        """Persist all entities/relations/links of one AnnotatedFact; returns the list of Entity."""
        from .types import EntityAnnotation
        entities: list[Entity] = []
        name_to_entity: dict[str, Entity] = {}
        name_to_role: dict[str, str] = {}   # Records each entity's role in this fact

        for ann in annotated.entities:
            if not ann.name.strip():
                continue
            ent = self.upsert_entity(
                user_id, ann.name, ann.entity_type,
                slot=annotated.slot,
                description=annotated.fact_text,
                confidence=annotated.confidence,
            )
            entities.append(ent)
            name_to_entity[normalize_name(ann.name)] = ent
            name_to_role[normalize_name(ann.name)] = _normalize_role(ann.role)

        # Write the memory metadata wrapper
        for mid in memory_ids:
            self.upsert_memory_record(
                user_id, mid, annotated.slot, annotated.fact_text,
                confidence=annotated.confidence,
            )

        # Link entity <-> memory, with role and relation_hint
        for mid in memory_ids:
            for norm_name, ent in name_to_entity.items():
                role_str = name_to_role.get(norm_name, EntityMemoryRole.CONTEXT.value)
                # relation_hint: a short "entity_name --slot--> memory" description
                hint = f"{ent.name} ({ent.entity_type.value})"
                from .types import EntityMemoryRole as _R
                try:
                    role_enum = _R(role_str)
                except ValueError:
                    role_enum = _R.CONTEXT
                self.link_memory(mid, ent.id, user_id,
                                 role=role_enum, relation_hint=hint)

        # Write relation edges between entities
        for rel in annotated.relations:
            from_ent = name_to_entity.get(normalize_name(rel.from_name))
            to_ent = name_to_entity.get(normalize_name(rel.to_name))
            if from_ent and to_ent and from_ent.id != to_ent.id:
                self.upsert_edge(
                    user_id, from_ent.id, to_ent.id,
                    rel.relation_type,
                    role_label=rel.role_label,
                    confidence=rel.confidence,
                    evidence_memory_id=memory_ids[0] if memory_ids else None,
                )

        # Refresh the profiles of the affected slots (counts only, no LLM summary)
        for slot in {e.slot for e in entities}:
            self.refresh_slot_profile(user_id, slot)

        return entities

    # ── Graph context ─────────────────────────────────────────────────────────

    def entity_context(
        self, entity_id: str, user_id: str, *, depth: int = 1
    ) -> dict[str, Any]:
        """Return a node's full context: node data + related edges + related memory_ids + affective edges."""
        entity = self.get_entity(entity_id)
        if not entity:
            return {}
        edges = self.edges_for_entity(entity_id, user_id)
        memory_ids = self.memory_ids_for_entities([entity_id])
        affective = self.affective_edges_for_entity(entity_id, user_id)

        neighbor_ids: set[str] = set()
        for e in edges:
            neighbor_ids.add(
                e.from_entity_id if e.to_entity_id == entity_id else e.to_entity_id
            )

        neighbors: list[dict] = []
        if depth > 0 and neighbor_ids:
            for nid in neighbor_ids:
                ne = self.get_entity(nid)
                if ne:
                    neighbors.append({
                        "id": ne.id, "name": ne.name,
                        "entity_type": ne.entity_type.value,
                        "slot": ne.slot.value,
                    })

        return {
            "entity": {
                "id": entity.id, "name": entity.name,
                "entity_type": entity.entity_type.value,
                "slot": entity.slot.value,
                "description": entity.description,
                "confidence": entity.confidence,
                "importance": entity.importance,
                "aliases": entity.aliases,
                "properties": entity.properties,
                "created_at": entity.created_at, "updated_at": entity.updated_at,
            },
            "edges": [
                {
                    "from": e.from_entity_id, "to": e.to_entity_id,
                    "relation": e.relation_type, "role": e.role_label,
                    "confidence": e.confidence, "weight": e.weight,
                    "edge_type": e.edge_type, "status": e.status,
                }
                for e in edges
            ],
            "memory_ids": memory_ids,
            "neighbors": neighbors,
            "affective_edges": [
                {
                    "trigger": a.trigger_frame, "emotion": a.emotion,
                    "appraisal": a.appraisal, "policy": a.response_policy,
                    "confidence": a.confidence,
                }
                for a in affective
            ],
        }

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def delete_user(self, user_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM entity_memory_links WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM entity_edges WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM entities WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM slot_profiles WHERE user_id=?", (user_id,))
            c.execute("DELETE FROM affective_edges WHERE user_id=?", (user_id,))

    def unlink_memory(self, memory_id: str) -> None:
        """Drop this memory's entity links; ingest_annotated_fact re-creates them on edit."""
        with self._conn() as c:
            c.execute("DELETE FROM entity_memory_links WHERE memory_id=?", (memory_id,))

    def delete_memory(self, memory_id: str) -> None:
        """Remove one memory's graph record and entity links.

        Entities and edges stay: other memories may share them.
        """
        with self._conn() as c:
            c.execute("DELETE FROM entity_memory_links WHERE memory_id=?", (memory_id,))
            c.execute("DELETE FROM memories WHERE id=?", (memory_id,))
