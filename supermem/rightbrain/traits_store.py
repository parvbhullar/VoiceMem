"""Right brain v2: one node = one judgement about this person, with evidence hung underneath.

    rb_traits                          node
      claim      wants reassurance when stressed    fixed at write time, one short phrase
      slot       one of five (emotion/coping_style/expression_style/thinking_pattern/likes_dislikes)
      embedding  vector of the claim -- the right brain can finally search semantically
    rb_evidence                        evidence
      quote      don't give me a solution yet, let me finish   the user's own words
      emotion    irritated             emotion is an attribute of the evidence, not a node
      cause_id   <- the left-brain fact

**Why replace the slot -> entity -> heartnote design**: the ``entity`` layer had three jobs at once --
sometimes a judgement about the person ("hates being interrupted"), sometimes a topic ("pour-over coffee", "NUS"), sometimes an emotion word
("anxious"). Mixing three kinds of things in one layer led to what we actually measured:

  · every sad event piled into a single "sad" node (61 items), and everything linked to "Jiaqi" (52 items)
  · titles could not follow one format -- the three kinds of things never had a common way to be written
  · descriptions relied on a later consolidation batch, which ran rarely, so many nodes had no description

Only **judgements about the person** live here. Topic entities belong to the left brain's cognitive graph; emotion is demoted to an attribute of evidence;
the assistant's self-review (response_experience) does not go into this table.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Two claims this similar count as the same one, and their evidence is merged.
#: 0.95 is an empirically measured boundary: local E5's baseline similarity on short Chinese phrases is already 0.9+,
#: "likes pour-over coffee" <-> "prefers pour-over coffee" is 0.964 (should merge),
#: "hates people smacking their lips while eating" <-> "hates being interrupted" is 0.934 (should not merge).
MERGE_THRESHOLD = 0.95

#: The five slots. The old "people/places/attitudes" slot was removed -- it stored topics (pour-over coffee/NUS/Jiaqi),
#: which belong in the left brain anyway, and it was the source of the "Jiaqi x52" grab bag.
#: Warn about a dimension mismatch only once, not every turn.
_WARNED_DIM: set = set()

SLOTS = ("emotion", "coping_style", "expression_style", "thinking_pattern", "likes_dislikes")

#: five slots -> the three UI categories
SLOT_TO_CLUSTER = {
    "emotion":          "emotion",
    "coping_style":     "personality",
    "expression_style": "personality",
    "thinking_pattern": "personality",
    "likes_dislikes":   "preference",
}


@dataclass
class Evidence:
    quote: str
    emotion: str = ""
    cause: str = ""            # original text of the left-brain fact (rendered as the "why")
    cause_id: str = ""
    at: str = ""


@dataclass
class Trait:
    id: str
    slot: str
    claim: str
    confidence: float = 0.9
    evidence: list[Evidence] = field(default_factory=list)
    updated_at: str = ""

    @property
    def cluster(self) -> str:
        return SLOT_TO_CLUSTER.get(self.slot, "personality")


#: Common subjects in front of a claim. The node title is "hates being interrupted", not "the user hates being interrupted" --
#: the whole graph is about the same person, so prefixing every title with "the user" is pure noise.
#: Empty: English claims were never stripped of their subject, and stripping them now
#: would change titles of nodes that already exist.
_SUBJECTS: tuple[str, ...] = ()


def normalize_claim(claim: str) -> str:
    """Tidy a claim into what a node title should look like: no subject, no trailing period, one short phrase.

    The write paths differ in output quality -- the merged extraction path has an explicit format requirement, the assistant self-review path
    (user_trait from response_experience) does not, and in practice it produced
    whole sentences with a subject like "The user likes sharing their experiences and may not pay much attention to the assistant's greetings."
    Rather than restating the requirement on every path, normalise it once at the entry point.
    """
    c = (claim or "").strip().strip("\u300c\u300d\"'").rstrip("\u3002.\uff01!\uff1b;\uff0c,")
    for s in _SUBJECTS:
        if c.startswith(s) and len(c) > len(s) + 2:
            c = c[len(s):].lstrip("\uff0c,\u3001 ")
            break
    # For a two-clause "A, maybe B" keep only the first clause -- the second is almost always speculation added by the model
    if "\uff0c" in c and len(c) > 15:
        head = c.split("\uff0c")[0].strip()
        if len(head) >= 5:
            c = head
    return c.strip()


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TraitStore:
    """The rb_traits / rb_evidence tables, sharing the space's sqlite with the other structured stores."""

    def __init__(self, db_path, embed) -> None:
        self._db = str(db_path)
        self._embed = embed                 # fn(text) -> list[float]
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS rb_traits (
                id             TEXT PRIMARY KEY,
                user_id        TEXT NOT NULL,
                slot           TEXT NOT NULL,
                claim          TEXT NOT NULL,
                embedding      BLOB,
                confidence     REAL NOT NULL DEFAULT 0.9,
                created_at     TEXT NOT NULL,
                updated_at     TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS rb_evidence (
                id         TEXT PRIMARY KEY,
                trait_id   TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                quote      TEXT NOT NULL,
                emotion    TEXT NOT NULL DEFAULT '',
                cause      TEXT NOT NULL DEFAULT '',
                cause_id   TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_ev_trait ON rb_evidence(trait_id)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_tr_user ON rb_traits(user_id, slot)")

    def _conn(self):
        c = sqlite3.connect(self._db, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, user_id: str, slot: str, claim: str, ev: Evidence) -> str:
        """Add a judgement + its evidence. If a claim with the same meaning already exists, merge into it instead of creating a node."""
        claim = normalize_claim(claim)
        if not claim or slot not in SLOTS:
            return ""

        vec = self._vec(claim)
        tid = self._find_similar(user_id, slot, vec)
        now = _now()
        with self._conn() as c:
            if tid is None:
                tid = uuid.uuid4().hex
                c.execute("INSERT INTO rb_traits "
                          "(id,user_id,slot,claim,embedding,confidence,created_at,updated_at) "
                          "VALUES (?,?,?,?,?,?,?,?)",
                          (tid, user_id, slot, claim,
                           vec.astype(np.float32).tobytes() if vec is not None else None,
                           0.9, now, now))
            else:
                c.execute("UPDATE rb_traits SET updated_at=? WHERE id=?", (now, tid))
            # Pass every field through str(): evidence often comes from old data or LLM output,
            # missing fields are None, and these columns are all NOT NULL -- a raw insert would fail the whole turn's write.
            c.execute("INSERT INTO rb_evidence "
                      "(id,trait_id,user_id,quote,emotion,cause,cause_id,created_at) "
                      "VALUES (?,?,?,?,?,?,?,?)",
                      (uuid.uuid4().hex, tid, user_id, str(ev.quote or ""),
                       str(ev.emotion or ""), str(ev.cause or ""),
                       str(ev.cause_id or ""), str(ev.at or now)))
        return tid

    def _vec(self, text: str):
        try:
            v = np.asarray(self._embed(text), dtype=np.float32)
            n = float(np.linalg.norm(v))
            return v / n if n else v
        except Exception:
            return None

    def _find_similar(self, user_id: str, slot: str, vec) -> str | None:
        if vec is None:
            return None
        with self._conn() as c:
            rows = c.execute("SELECT id, embedding FROM rb_traits "
                             "WHERE user_id=? AND slot=? AND embedding IS NOT NULL",
                             (user_id, slot)).fetchall()
        best, best_sim = None, 0.0
        for r in rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            if v.shape != vec.shape:
                continue
            sim = float(v @ vec)
            if sim > best_sim:
                best, best_sim = r["id"], sim
        return best if best_sim >= MERGE_THRESHOLD else None

    # ── Read ──────────────────────────────────────────────────────────────────

    def all(self, user_id: str, *, per_slot: int = 8) -> list[Trait]:
        """For the mind map: per slot, the top entries by evidence count + the most recently added ones."""
        out: list[Trait] = []
        with self._conn() as c:
            for slot in SLOTS:
                rows = c.execute(
                    """SELECT t.*, COUNT(e.id) n FROM rb_traits t
                       LEFT JOIN rb_evidence e ON e.trait_id = t.id
                       WHERE t.user_id=? AND t.slot=? GROUP BY t.id
                       HAVING n > 0""", (user_id, slot)).fetchall()
                by_ev = sorted(rows, key=lambda r: -r["n"])
                by_new = sorted(rows, key=lambda r: r["updated_at"], reverse=True)
                fresh = max(1, per_slot // 2)
                picked, seen = [], set()
                # half the seats go to the most recently added (what was just said must show up immediately), half to the most evidenced
                for r in by_new[:fresh] + by_ev:
                    if r["id"] in seen:
                        continue
                    seen.add(r["id"])
                    picked.append(r)
                    if len(picked) >= per_slot:
                        break
                for r in picked:
                    out.append(self._to_trait(c, r))
        return out

    def search(self, user_id: str, query: str, *, top_k: int = 5) -> list[Trait]:
        """Search judgements semantically.

        The old right brain could only match on emotion anchors, so every turn returned the same few static profile items.
        With claim vectors, this is real retrieval.
        """
        return [t for t, _ in self.search_scored(user_id, query, top_k=top_k)]

    def search_scored(self, user_id: str, query: str, *, top_k: int = 5
                      ) -> list[tuple[Trait, float]]:
        """Same as :meth:`search`, but with cosine similarity.

        Retrieval uses it as the priority -- how relevant a judgement is to this utterance directly decides whether it deserves
        a top-N seat; a fixed priority lets irrelevant judgements crowd out the truly relevant ones.
        """
        q = self._vec(query)
        if q is None:
            return []
        # Retrieval picks its threshold from this -- the threshold is tied to the embedder, see brain.trait_min_sim.
        self.last_query_dim = int(q.shape[0])
        with self._conn() as c:
            rows = c.execute("SELECT * FROM rb_traits WHERE user_id=? AND embedding IS NOT NULL",
                             (user_id,)).fetchall()
            scored, stale = [], 0
            for r in rows:
                v = np.frombuffer(r["embedding"], dtype=np.float32)
                if v.shape != q.shape:
                    stale += 1           # the embedder was changed; old vector dimensions do not match
                    continue
                scored.append((float(v @ q), r))
            if stale and not _WARNED_DIM:
                _WARNED_DIM.add(1)
                print(f"[RBTraits] ⚠ {stale} judgement vectors do not match the current embedder's dimension, "
                      "skipped -- after switching embedders the old vectors are invalid and right-brain retrieval silently goes empty. "
                      "Re-embed them to fix this.", flush=True)
            scored.sort(key=lambda t: -t[0])
            return [(self._to_trait(c, r), s) for s, r in scored[:top_k]]

    def _to_trait(self, c, r) -> Trait:
        evs = c.execute("SELECT * FROM rb_evidence WHERE trait_id=? ORDER BY created_at DESC",
                        (r["id"],)).fetchall()
        return Trait(
            id=r["id"], slot=r["slot"], claim=r["claim"],
            confidence=r["confidence"], updated_at=r["updated_at"],
            evidence=[Evidence(quote=e["quote"], emotion=e["emotion"],
                               cause=e["cause"], cause_id=e["cause_id"],
                               at=e["created_at"]) for e in evs],
        )

    def counts(self, user_id: str) -> tuple[int, int]:
        with self._conn() as c:
            t = c.execute("SELECT COUNT(*) FROM rb_traits WHERE user_id=?", (user_id,)).fetchone()[0]
            e = c.execute("SELECT COUNT(*) FROM rb_evidence WHERE user_id=?", (user_id,)).fetchone()[0]
        return t, e
