"""Memory Space: one directory per user / per set of memories.

    supermem_memoryspace/
      demo/                     <- the space name is the directory name
        demo.json               overview + left brain / right brain / mem0 properties
        demo.sqlite             all structured storage (used to be split across 10 .sqlite files)
        multi_modal/            voiceprint vectors, audio embeddings, raw wav
        vectors/                left-brain text vector store (qdrant's own directory format)

When no space is set, it defaults to ``demo``.

**Why the sqlite files are merged into one**: a space used to have nine separate files --
cognitive_graph / graph_entities / rb_graph / right_brain / session_tracker / slot_splits /
scene_triggers / routine_memory / audio_archive -- each opening its own connection. They all
belong to the same memory; splitting them just turns "copy a space" into "don't forget any
file". Table names don't overlap (the one collision, ``query_activations``, was renamed on the
graph_entities side; see GRAPH_ACTIVATIONS_TABLE).

**Why vectors/ is not inside multi_modal/**: multi_modal holds audio derivatives (voiceprints,
audio embeddings, wav); vectors/ is the vector store for left-brain **text** memory, and it is
qdrant's own directory format, not a single file.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_SPACE = "demo"
ROOT_DIR_NAME = "supermem_memoryspace"

#: graph_entities' query_activations uses this table name instead -- it has one more
#: column (session_id) than cognitive_graph's, so sharing a name would let whichever
#: table was created first break the other's inserts.
GRAPH_ACTIVATIONS_TABLE = "graph_query_activations"


def _safe(name: str) -> str:
    """The space name must be usable as a directory name. Don't let things like "../x" escape the directory."""
    cleaned = "".join(c for c in (name or "") if c.isalnum() or c in "-_ .").strip()
    cleaned = cleaned.strip(".")
    return cleaned or DEFAULT_SPACE


def root() -> Path:
    """Parent directory of all spaces. ``SUPERMEM_MEMORYSPACE_ROOT`` relocates it wholesale."""
    env = os.environ.get("SUPERMEM_MEMORYSPACE_ROOT")
    return Path(env) if env else Path.cwd() / ROOT_DIR_NAME


class MemorySpace:
    """All paths of one space. Pass it to the Orchestrator as memory_root."""

    def __init__(self, name: str | None = None, *, root_dir: Path | str | None = None):
        self.name = _safe(name or os.environ.get("SUPERMEM_SPACE") or DEFAULT_SPACE)
        base = Path(root_dir) if root_dir else root()
        self.dir = base / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "multi_modal").mkdir(exist_ok=True)

    # ── Entry points ─────────────────────────────────────────────────────────
    @property
    def json_path(self) -> Path:
        return self.dir / f"{self.name}.json"

    @property
    def db_path(self) -> Path:
        return self.dir / f"{self.name}.sqlite"

    @property
    def multi_modal(self) -> Path:
        return self.dir / "multi_modal"

    @property
    def vectors(self) -> Path:
        return self.dir / "vectors"

    def __fspath__(self) -> str:        # lets it be used directly as memory_root
        return str(self.dir)

    def __str__(self) -> str:
        return str(self.dir)


# ── Component-side entry points ──────────────────────────────────────────────
# Components only get memory_root, not the MemorySpace object, so these functions
# derive paths from the directory itself: the directory name is the space name and
# file names follow it (demo/ contains demo.sqlite), so a copied folder is still
# recognisable.

def _pick(memory_root, suffix: str) -> Path:
    """Pick the file by directory name; if the directory already has one with this suffix, use it.

    File names follow the directory name (demo/ contains demo.sqlite), so a copied folder
    is still recognisable. But **after renaming or copying the folder the directory name
    changes**; looking strictly by the new name finds nothing and creates a fresh empty
    store -- memories seem to vanish (the left brain survives since vectors/ ignores names,
    which makes it even harder to debug). So look by name first, and otherwise use the
    existing file in the directory.
    """
    p = Path(memory_root)
    want = p / f"{p.name}{suffix}"
    if want.exists():
        return want
    existing = sorted(x for x in p.glob(f"*{suffix}") if x.is_file())
    return existing[0] if existing else want


def db(memory_root) -> Path:
    """This space's single sqlite. All structured storage lives in it; table names don't overlap."""
    return _pick(memory_root, ".sqlite")


def json_path(memory_root) -> Path:
    """Space description file: overview + left brain / right brain / mem0 properties.

    Holds only "what this space is", not runtime state -- runtime state (voiceprint
    registry, cleanup progress, left-brain json mirror) goes in the sqlite kv table, see
    ``kv_get`` / ``kv_set``. Those four pieces of state differ, and stuffing them into one
    json would make them overwrite each other.
    """
    return _pick(memory_root, ".json")


# ── Small state: stored in the sqlite kv table instead of a separate json each ──

def _kv_conn(memory_root):
    import sqlite3
    c = sqlite3.connect(db(memory_root))
    c.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    return c


def kv_get(memory_root, key: str, default=None):
    import json as _json
    try:
        with _kv_conn(memory_root) as c:
            row = c.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return _json.loads(row[0]) if row else default
    except Exception:
        return default


def kv_set(memory_root, key: str, value) -> None:
    import json as _json
    try:
        with _kv_conn(memory_root) as c:
            c.execute("INSERT INTO kv (k, v) VALUES (?,?) "
                      "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                      (key, _json.dumps(value, ensure_ascii=False)))
    except Exception as e:
        print(f"[space] failed to write {key}: {type(e).__name__}: {e}", flush=True)


def mm(memory_root, name: str = "") -> Path:
    """Things under multi_modal/: voiceprints, audio embeddings, wav."""
    d = Path(memory_root) / "multi_modal"
    d.mkdir(parents=True, exist_ok=True)
    return d / name if name else d


def vectors(memory_root) -> Path:
    """Left-brain text vector store (qdrant's directory format, not a single file)."""
    d = Path(memory_root) / "vectors"
    d.mkdir(parents=True, exist_ok=True)
    return d


def check_dims(memory_root, dims: int) -> None:
    """How many dimensions was this space built with? If it doesn't match the current embedder, say so plainly.

    Dimensionality is a **property of the space**: all vectors in a space must come from
    the same embedder. Without this check the error surfaces deep inside qdrant --
    ``shapes (227,384) and (1536,) not aligned`` -- with no hint that the embedding changed.
    """
    import json as _json
    path = json_path(memory_root)
    if not path.is_file():
        return
    try:
        old = (_json.loads(path.read_text(encoding="utf-8")).get("mem0") or {}).get("dims")
    except Exception:
        return
    if not old or int(old) == int(dims):
        return
    raise ValueError(
        f"This memory space was built with a {old}-dim embedding, but you are now using {dims} dims.\n"
        f"  space: {Path(memory_root)}\n"
        f"A space can only use one embedding. Two options:\n"
        f"  - switch to a new space: SuperMem(space=\"my_name\")\n"
        f"  - or switch back to the original embedding ({old} dims is usually local E5, "
        f"1536 dims is OpenAI text-embedding-3-small)")


def describe(memory_root, *, user_id: str = "", mode: str = "",
             dims: int | None = None, counts: dict | None = None) -> dict:
    """Write/update the space description file ``<space>.json`` and return what was written.

    Four sections, matching the structure in the paper:

        space       name, creation time, last update, total counts
        left_brain  left brain: factual memory + cognitive graph (slot / entity / relation)
        right_brain right brain: heartnote + rb_slots / rb_entities (personality and emotion)
        mem0        underlying memory engine: vector store location, collection, vector dims

    Metadata only, never touches the memories themselves; can be deleted and rebuilt at any time.
    """
    import json as _json
    from datetime import datetime, timezone

    p = Path(memory_root)
    path = json_path(p)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    old = {}
    if path.is_file():
        try:
            old = _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}

    doc = {
        "space": {
            "name": p.name,
            "created_at": (old.get("space") or {}).get("created_at") or now,
            "updated_at": now,
            "user_id": user_id or (old.get("space") or {}).get("user_id", ""),
            "mode": mode or (old.get("space") or {}).get("mode", ""),
            "counts": counts or (old.get("space") or {}).get("counts", {}),
            # Which language this space uses ("en"). Decided once at creation and never
            # changed -- memories and replies both follow it. Retrieval is vector-based, and
            # mixing languages in one store makes half the memories unretrievable, so language
            # is a **property of the store**, not of a single sentence.
            # Inherited from old like created_at: this file is rewritten every time the space opens.
            "language": (old.get("space") or {}).get("language", ""),
        },
        "left_brain": {
            "role": "Factual memory: what was said, when, and who was involved",
            "storage": f"{p.name}.sqlite",
            "tables": ["memories", "entities", "entity_edges", "memory_tags",
                       "slot_profiles", "graph_entities"],
        },
        "right_brain": {
            "role": "Personality and emotion: what this person is like and why they react the way they do",
            "storage": f"{p.name}.sqlite",
            "tables": ["right_brain_memories", "right_brain_anchor_links",
                       "rb_slots", "rb_entities", "rb_entity_memories"],
        },
        "mem0": {
            "role": "Underlying memory engine (vector retrieval)",
            # Dims are bound to this space: changing the embedding means changing the space, see check_dims()
            "dims": dims or (old.get("mem0") or {}).get("dims"),
            "vector_store": "qdrant (local)",
            "path": "vectors/",
            "collection": "supermem",
            "history": f"{p.name}.sqlite",
        },
        "multi_modal": {
            "role": "Voiceprints, audio embeddings, raw wav",
            "path": "multi_modal/",
        },
    }
    try:
        path.write_text(_json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    except Exception as e:
        print(f"[space] failed to write description file: {type(e).__name__}: {e}", flush=True)
    return doc


def resolve(space=None, memory_root=None) -> Path:
    """Normalise ``space=`` / ``memory_root=`` into one directory.

    If ``memory_root`` is given explicitly it is used as-is (old code and evals rely on it
    to open a separate store per conversation); otherwise the space name maps to
    ``supermem_memoryspace/<space>/``.
    """
    if memory_root:
        p = Path(memory_root)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return MemorySpace(space).dir
