# Memories Page Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> superpowers:subagent-driven-development) to implement this plan task-by-task.

**Goal:** A `/memories` operator page on the SuperMem web demo: pick or create a
brain, browse/search/edit/delete its facts, read its profile, load memories from
pasted text or a `.txt/.md/.json/.pdf/.docx` file in the background, clear or
delete a brain.

**Architecture:** Pure chunking + a one-thread job runner in `web/ingest.py`;
routes in `web/memories_api.py` that take every dependency as a callback (tested
on a bare `FastAPI()` with fakes); `web/run.py` supplies the callbacks and the
space lifecycle; five small core fixes make edit, delete, concurrent writes and
space deletion correct. One static page, `web/memories.html`.

**Tech Stack:** Python 3.12, FastAPI, unittest (no pytest in the venv), pypdf,
python-docx, plain HTML/JS.

**Spec:** `docs/superpowers/specs/2026-09-30-memories-page-design.md`. This plan
supersedes the spec where they differ; see "Amendments" below.

**Ground rules for every task**

- Run commands from the repo root (`SuperMem/`). Tests:
  `uv run --no-sync python tests/<file>.py`; whole suite:
  `uv run --no-sync python -m unittest discover -s tests -p 'test_*.py'`
  (72 tests pass before this work).
- Tests are `unittest.TestCase` classes ending in
  `if __name__ == "__main__": unittest.main(verbosity=2)`. No network, no model
  loading, no OpenAI key needed.
- New `web/` modules are imported **flat** (`from ingest import ...`), the way
  `web/run.py` imports its siblings. Tests put `ROOT / "web"` on `sys.path`.
  Never import `web/utils.py` or `web/run.py` from a test: `utils.py` builds an
  `AsyncOpenAI()` at import time.
- Match the surrounding style: comment density, `ponytail:` comments for
  deliberate simplifications that have a known ceiling, type hints on new code.
- Do **not** commit. The controller commits at the end if the user asks.
- Do not touch `supermem_memoryspace/` (real data). Tests use temp dirs.
- A server is running on :8787 from the pre-change code; leave it alone until the
  integration task.

---

## Amendments to the spec (from the code investigation)

1. **Per-chunk cost is 5–20 s, not 1–2 s** (extraction, conflict resolution,
   slot tagging, attribution). `MAX_CHUNKS` drops to **500**, and jobs get
   **cancel** (`POST /api/ingest/{id}/cancel`, checked between chunks). The page
   shows elapsed time and an ETA.
2. **Concurrent writes corrupt a brain.** Embedded Qdrant `_add_point`, the kv
   mirror and `TraitStore.add` are unlocked read-modify-write, and the voice loop
   already overlaps `_finish_ingest` threads. Core gets a per-instance
   `Orchestrator._write_lock` (RLock) around `_finish_ingest`; edit, delete and
   the job's prepare/finish hooks take it too. **Never take `Orchestrator._lock`**
   around ingest: `_finish_ingest` takes it itself (deadlock).
3. **Extraction failures are swallowed.** `_finish_ingest` returns no error, so a
   missing key looks like "0 facts". Its return dict gains `"error"`, and the job
   counts a non-empty `error` as a failed chunk.
4. **Delete must cascade into the cognitive graph** (its `memories` row still
   feeds the KV cartridge, replay and slot summaries). **Edit must refresh
   `memories.content` and replace entity links.**
5. **Deleting a space must evict the cached mem0 client** (`_MEM0_CLIENT_CACHE`),
   or a re-created space with the same name serves the old vectors and fails
   writes with "readonly database". Also pop `_SPACES`, `_CARTRIDGES`,
   `_OWNER_NAME_CACHE`.
6. **Exact-name matching.** The volume is case-insensitive; `Demo` must not
   resolve to `demo`. Existence = `d.is_dir() and safe in os.listdir(d.parent)`.
7. **One `_SPACES_LOCK` (RLock)** serializes opening and wiping spaces, so a
   request cannot rebuild a space mid-wipe. `get_space` keeps a lock-free fast
   path for already-open spaces.
8. **Clear of the live brain is allowed only when no voice session is connected
   and no memory write is pending** (else 409). The demo has no working brain
   switcher, so refusing it outright would make Clear useless on `demo`. The wipe
   holds that instance's `_write_lock`. **Delete of the live brain is refused.**
9. **Semantic search uses the repo's vector search directly**
   (`repo.search(q, user_id, top_k=200, include_assistant=True)`), never
   `SuperMem.search`, which bumps heat, books subgraph activations and dedupes
   near-duplicates.
10. **Jobs do not pass `session_id`.** Passing one fires a session-boundary batch
    and rescopes the live session's retrieval. Instead the job clears the reply
    history (`_exchanges`) before and after, and calls `mem.flush()` at the end,
    all under the write lock.
11. **The Profile tab shows evidence text from the live fact** (`cause_id`), so a
    deleted fact disappears from it and an edited one shows its new text.
12. **Routes that can block are sync `def`** (FastAPI runs them in the
    threadpool). The ingest route is `async def` only to read the raw body; it
    resolves the space and plans chunks with `asyncio.to_thread`.
13. **Uploads are the raw request body** (`await req.body()`), so no
    `python-multipart`.
14. **The picker shows brain names only.** `list_spaces().count` counts the graph
    table, which differs from the Facts total; the Facts tab shows its own total.
15. **Deleting a brain also deletes its turn recordings** in the shared
    `results/turn_audio/` (paths from the brain's `audio_archive` table).
16. **Transcript details.** An `owner` that is not among the speakers is a 400
    listing the speakers. Consecutive turns by the same speaker merge up to 800
    chars. A turn over 800 chars is cut at sentence ends. A `.json` upload that is
    not a turn list is a 400. Detection needs ≥60% `Name: text` lines **and** at
    most `max(2, turns // 2)` distinct speakers, so a `Key: value` notes list is
    prose.

---

## File ownership (for parallel execution)

| Owner | Files |
|---|---|
| Core | `supermem/orchestrator.py`, `supermem/leftbrain/cognitive_graph/store.py`, `supermem/leftbrain/cognitive_graph/store_v2.py`, `supermem/leftbrain/memory_repository.py`, `supermem/leftbrain/mem0_backend_store.py`, `tests/test_memory_delete.py`, `tests/test_write_lock.py`, `tests/test_space_evict.py` |
| Ingest | `web/ingest.py`, `tests/test_ingest_split.py`, `tests/test_ingest_jobs.py` |
| API | `web/memories_api.py`, `tests/test_memories_api.py` |
| Server | `web/run.py`, `web/utils.py` |
| Page | `web/memories.html`, `web/supermem.html`, `README.md` |

Interfaces between owners are fixed below; code against them even if the other
file is not written yet.

---

## Task 1 (Core): per-instance write lock + ingest error

**Files:** Modify `supermem/orchestrator.py`; Test `tests/test_write_lock.py`.

**Step 1: failing test.** `tests/test_write_lock.py`:

- Build `o = Orchestrator.__new__(Orchestrator)` (no `__init__`, no models) and
  set `o._write_lock = threading.RLock()`.
- Replace `o._finish_ingest_locked` with a function that increments a shared
  counter, records the max concurrent value, sleeps 0.05 s, decrements, and
  returns `{"facts_count": 0}`.
- Call `o._finish_ingest({})` from 4 threads; join. Assert max concurrency == 1.
- Second test: the lock is reentrant: call `o._finish_ingest({})` from inside a
  `with o._write_lock:` block on the same thread; it must not deadlock (run it in
  a thread with `join(timeout=2)` and assert the thread finished).
- Third test: `__init__` creates it: grep-free check via
  `inspect.getsource(Orchestrator.__init__)` containing `_write_lock` is weak;
  instead assert `"_write_lock" in Orchestrator.__init__.__code__.co_names`.

If `import supermem.orchestrator` is slow or needs a key, set
`os.environ.setdefault("OPENAI_API_KEY", "sk-test")` before importing.

**Step 2:** run, expect FAIL (`_finish_ingest_locked` missing).

**Step 3: implement.**

- In `Orchestrator.__init__`, next to `self._lock = threading.Lock()` (~line 298):

```python
        # _lock only guards building the lazy singletons in _cache, and _finish_ingest takes it
        # itself -- never hold it around an ingest (deadlock). _write_lock serialises the slow
        # write path per instance: the voice loop's background ingest threads, the /memories
        # job worker, and the page's edit/delete all go through it.
        self._write_lock = threading.RLock()
```

- Rename the existing `def _finish_ingest(self, ctx: dict) -> dict:` (~1219) to
  `_finish_ingest_locked` (keep its docstring), and add just above it:

```python
    def _finish_ingest(self, ctx: dict) -> dict:
        """Serialised entry to _finish_ingest_locked (see _write_lock in __init__)."""
        with self._write_lock:
            return self._finish_ingest_locked(ctx)
```

- In the return dict of `_finish_ingest_locked` (~1421), add after
  `"affect": result.affect,`:

```python
            # Extraction swallows its failures (missing key, network) and returns 0 facts;
            # surface the reason so callers like the /memories job can tell "nothing to store"
            # from "could not store".
            "error":               getattr(result, "error", None),
```

  Confirm `result` is the extraction result object that carries `error`
  (`supermem/utils/common/voice_input.py` ~536-545). If the early-return dict
  above it (~1171) also exists, add `"error": None` there for a uniform shape.

**Step 4:** run the test, expect PASS. Run the whole suite.

---

## Task 2 (Core): delete cascades into the graph; edit refreshes it

**Files:** Modify `supermem/leftbrain/cognitive_graph/store.py`,
`supermem/leftbrain/cognitive_graph/store_v2.py`,
`supermem/leftbrain/memory_repository.py`; Test `tests/test_memory_delete.py`.

**Step 1: failing tests** (real sqlite in a `tempfile.TemporaryDirectory()`; find
the store constructors' signatures in the files, e.g. `CognitiveGraphStoreV2(path)`):

1. `upsert_memory_record(uid, "m1", slot, "old text")` then again with
   `"new text"` → `SELECT content FROM memories WHERE id='m1'` is `"new text"`.
2. V2 store: create memory `m1` and `m2`, link each to an entity
   (`upsert_entity` + `link_memory`), tag each (`upsert_memory_tags`), then
   `store.delete_memory("m1")` → zero rows for `m1` in `memory_tags`,
   `entity_memory_links`, `memories`; `m2` rows intact; the entity row intact.
3. `unlink_memory("m2")` removes only `m2`'s `entity_memory_links` rows.
4. Repository wiring, without mem0: build
   `repo = LeftBrainMemoryRepository.__new__(LeftBrainMemoryRepository)` and set
   the attributes `delete_memory` / `update_memory` read (`_vector_store` = a fake
   whose `delete_memory`/`update_memory` return True; `_cognitive_store` = a fake
   recording calls; `_cognitive_annotator` = a fake whose `annotate([t])` returns
   `[object()]`; stub `load_json_store` → `{"results": []}` and
   `_write_json_store` → no-op). Assert:
   - `repo.delete_memory("m1")` calls `cognitive.delete_memory("m1")`;
   - a failing `cognitive.delete_memory` (raises) still returns True (logged);
   - vector delete returning False → cognitive delete **not** called;
   - `repo.update_memory("m1", "t", user_id="u")` calls `unlink_memory("m1")`
     **before** `ingest_annotated_fact`; with `user_id=None` it calls neither.

**Step 2:** run, expect FAIL.

**Step 3: implement.**

`store.py`, `upsert_memory_record` (~719-724): change the conflict clause to

```python
                   ON CONFLICT(id) DO UPDATE SET
                     slot=excluded.slot, content=excluded.content,
                     confidence=excluded.confidence, updated_at=excluded.updated_at""",
```

(All three callers pass the verbatim fact text, so refreshing it is safe.)

`store.py`, after `delete_user` (~1062):

```python
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
```

`store_v2.py`, after its `delete_user` (~336):

```python
    def delete_memory(self, memory_id: str) -> None:
        """Delete one memory's V2 tags, then its base graph rows.

        Tags go first: memory_tags.memory_id is a foreign key to memories(id).
        """
        with self._conn() as c:
            c.execute("DELETE FROM memory_tags WHERE memory_id=?", (memory_id,))
        super().delete_memory(memory_id)
```

`memory_repository.py` `delete_memory` (~203): inside `if deleted:` after
`self._write_json_store(results)`:

```python
            # The graph's memories row feeds the KV cartridge, replay and slot summaries
            # directly; leaving it would keep a deleted fact alive there. The vector store
            # stays the source of truth, so a graph failure is logged, not raised.
            if self._cognitive_store is not None:
                try:
                    self._cognitive_store.delete_memory(memory_id)
                except Exception as _cog_err:
                    import logging
                    logging.getLogger(__name__).warning("cognitive graph delete failed: %s", _cog_err)
```

`memory_repository.py` `update_memory` (~196-197):

```python
                    if annotated:
                        # Annotate first so a failed annotation keeps the old links.
                        self._cognitive_store.unlink_memory(memory_id)
                        self._cognitive_store.ingest_annotated_fact(user_id, annotated[0], [memory_id])
```

Update the `delete_memory` docstring to "Delete a single memory. Syncs the JSON
mirror and the cognitive graph."

**Step 4:** tests pass; whole suite passes.

---

## Task 3 (Core): evict a space's cached mem0 client

**Files:** Modify `supermem/leftbrain/mem0_backend_store.py`; Test
`tests/test_space_evict.py`.

**Step 1: failing test** (the regression that stays silent otherwise):

- Temp root; `os.environ` for `SUPERMEM_MEMORYSPACE_ROOT` (temp) and
  `OPENAI_API_KEY=sk-test` set **before** importing the module.
- A fake embedder:

```python
class _Emb:
    dimensions = 8
    model_name = "fake"
    def embed_texts(self, ts):
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in ts]
```

- `s = Mem0BackendStore(_Emb(), memory_root=root / "s1")`; `s.add_text("u", "Alice likes tea")`
  (check the real signature); assert `s.list_ids(user_id="u")` non-empty.
- `evict_client(root / "s1")`; `gc.collect()`; `shutil.rmtree(root / "s1")`.
- `s2 = Mem0BackendStore(_Emb(), memory_root=root / "s1")` → `list_ids(user_id="u") == []`,
  and `s2.add_text("u", "Bob likes coffee")` succeeds.
- Second test: `evict_client` on a root that was never opened is a no-op.

If building `Mem0BackendStore` needs more than this, follow what its constructor
needs; keep the test model-free and network-free.

**Step 2:** run, expect FAIL (`evict_client` missing).

**Step 3: implement** (module level, after `_close_on_exit`):

```python
def evict_client(memory_root) -> None:
    """Drop this space's cached mem0 client and close it (Qdrant lock + history sqlite).

    Call before deleting a space's files. The cache outlives the space, so without this a
    space re-created under the same name reuses the old client: it serves the deleted
    vectors from RAM and fails every write with "attempt to write a readonly database".
    """
    key = str((Path(memory_root) / "vectors").resolve())   # same key as __init__ builds
    with _MEM0_CLIENT_CACHE_LOCK:
        client = _MEM0_CLIENT_CACHE.pop(key, None)
    if client is None:
        return
    for close in (lambda: client.vector_store.client.close(), getattr(client, "close", None)):
        if close is None:
            continue
        try:
            close()
        except Exception:
            pass
```

Check `__init__` computes the key the same way (`_space.vectors(memory_root)`
then `.resolve()`); `space.vectors()` mkdirs, so do not call it here.

**Step 4:** tests pass.

---

## Task 4 (Ingest): extraction, detection, splitting

**Files:** Create `web/ingest.py`; Test `tests/test_ingest_split.py`.

**Interface (fixed):**

```python
@dataclass(frozen=True)
class Chunk:
    text: str
    speaker: str = "user"
    agent_reply: str | None = None

MAX_BYTES = 10 * 1024 * 1024
MAX_CHUNKS = 500
CHUNK_CHARS = 800

def plan_chunks(filename: str, data: bytes, fmt: str = "auto", owner: str = "") -> tuple[str, list[Chunk]]
def extract_text(filename: str, data: bytes) -> str
def detect_format(text: str) -> str                      # "prose" | "transcript"
def split_prose(text: str) -> list[Chunk]
def split_transcript(text: str, owner: str = "") -> list[Chunk]
```

All operator-facing failures raise `ValueError` with a readable message.

**Step 1: failing tests.** Fixtures are generated at runtime, never committed:

```python
def tiny_pdf(*lines: str) -> bytes:
    """One-page PDF with a real text layer. No lines -> blank page (the scanned-PDF case).
    Text must be ASCII without ( ) or backslash."""
    ops = "".join(f"({s}) Tj 0 -14 Td " for s in lines)
    stream = f"BT /F1 12 Tf 72 720 Td {ops}ET".encode() if lines else b""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1, xref)
    return bytes(out)


def tiny_docx(*paras: str) -> bytes:
    import io
    import docx
    d = docx.Document()
    d.add_heading("Notes", level=1)
    for p in paras:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()
```

Cases (one `test_*` each):

- extract: `.txt` UTF-8 (with BOM stripped); non-UTF-8 bytes → ValueError; `.exe`
  → ValueError naming the allowed types; `tiny_pdf("Alice is vegetarian.", "She lives in Pune.")`
  → text contains both sentences; `tiny_pdf()` → ValueError containing "scanned";
  garbage bytes as `.pdf` → ValueError; `tiny_docx(...)` → contains `"# Notes"`
  and both paragraphs.
- detect: `"Alice: hi\nBot: hello\nAlice: I am vegetarian"` → transcript; prose
  with one `Note: x` line among 5 → prose; `"Diet: veg\nCity: Pune\nJob: nurse"`
  → prose (3 distinct keys, 3 lines); JSON turn list → transcript;
  `"Alice: hi\nBot: hello"` → transcript.
- prose: two short paragraphs merge into one chunk; paragraphs totalling > 800
  chars become ≥ 2 chunks, none over 800 (+ heading); one 2,000-char paragraph is
  cut at sentence ends; a 1,000-char sentence with no punctuation is cut at a
  space; `# Diet\n\nShe is vegetarian.` → chunk text starts with `# Diet\n`;
  a new heading flushes the buffer (chunks never span headings);
  whitespace-only → `[]`.
- transcript: default owner is the first non-assistant speaker and becomes
  `"user"`; named owner (`owner="bob"`, case-insensitive) → Bob's chunks are
  `"user"`, Alice keeps `"Alice"`; an assistant turn after a user turn becomes
  `agent_reply` on it; two assistant turns in a row concatenate; an assistant turn
  first is dropped; consecutive same-speaker turns merge (and do not merge across
  a reply); a continuation line (no `Name:`) appends to the previous turn; JSON
  `[{"role":"user","content":"hi"},{"role":"assistant","content":"yo"}]` →
  one `Chunk("hi", "user", "yo")`; JSON `role: assistant` with
  `speaker: "Sam"` is still the assistant; a 1,500-char turn is cut into pieces
  with the reply on the last piece; unknown owner → ValueError listing speakers.
- plan: returns `("prose"|"transcript", chunks)`; `fmt="transcript"` overrides
  detection; bad `fmt` → ValueError; > MAX_BYTES → ValueError; > MAX_CHUNKS →
  ValueError (monkeypatch `ingest.MAX_CHUNKS = 2`); `.json` that is not a turn
  list → ValueError; empty text → ValueError.

**Step 2:** run, expect FAIL (no module).

**Step 3: implement** `web/ingest.py` (full module; Task 5 appends `Jobs`):

```python
"""Load memories from pasted text or a file: extract, detect, split, ingest in the background.

The pure functions turn bytes into Chunks (one ``mem.ingest`` call each); ``Jobs`` runs chunk
lists on one daemon thread, one job at a time. Each chunk costs several LLM calls (5-20 s),
which is why ingest never runs inside an HTTP request.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024
MAX_CHUNKS = 500          # ~5-20 s each: 500 is already over an hour of LLM calls
CHUNK_CHARS = 800
TEXT_EXTS = {"", ".txt", ".md", ".markdown", ".json"}
ASSISTANT_NAMES = {"assistant", "agent", "ai", "bot"}

_TURN_RE = re.compile(r"^([^:\n]{1,40}):\s+(\S.*)$")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class Chunk:
    """One ingest call: the text, who said it, and the assistant's reply to it (if any)."""
    text: str
    speaker: str = "user"
    agent_reply: str | None = None


def plan_chunks(filename: str, data: bytes, fmt: str = "auto",
                owner: str = "") -> tuple[str, list[Chunk]]:
    """Bytes -> (format, chunks). Raises ValueError with an operator-readable reason."""
    if len(data) > MAX_BYTES:
        raise ValueError(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    text = extract_text(filename, data)
    if Path(filename or "").suffix.lower() == ".json" and _json_turns(text) is None:
        raise ValueError("A .json file must be a list of {speaker or role, text or content} objects.")
    kind = detect_format(text) if fmt in ("", "auto") else fmt
    if kind == "transcript":
        chunks = split_transcript(text, owner)
    elif kind == "prose":
        chunks = split_prose(text)
    else:
        raise ValueError(f"Unknown format {fmt!r}. Use auto, prose or transcript.")
    if not chunks:
        raise ValueError("There is no text to add.")
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(f"{len(chunks)} chunks is over the limit of {MAX_CHUNKS}. "
                         "Split the file and add it in parts.")
    return kind, chunks


def extract_text(filename: str, data: bytes) -> str:
    """Plain text from an upload. Text types must be UTF-8; .pdf and .docx go through their readers."""
    ext = Path(filename or "").suffix.lower()
    if ext == ".pdf":
        return _pdf_text(data)
    if ext == ".docx":
        return _docx_text(data)
    if ext not in TEXT_EXTS:
        raise ValueError(f"Unsupported file type {ext}. Use .txt, .md, .json, .pdf or .docx.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("The file is not UTF-8 text.") from None


def _pdf_text(data: bytes) -> str:
    import pypdf
    try:
        pages = pypdf.PdfReader(io.BytesIO(data)).pages
        text = "\n\n".join((p.extract_text() or "").strip() for p in pages)
    except Exception as e:
        raise ValueError(f"Could not read the PDF: {e}") from None
    if not text.strip():
        raise ValueError("No extractable text. This looks like a scanned PDF.")
    return text


def _docx_text(data: bytes) -> str:
    """Paragraph text; Title/Heading N paragraphs become Markdown headings so split_prose keeps them.

    ponytail: tables, headers and footnotes are skipped; add them if operators load forms.
    """
    import docx
    try:
        doc = docx.Document(io.BytesIO(data))
    except Exception as e:
        raise ValueError(f"Could not read the DOCX: {e}") from None
    out = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        style = (p.style.name if p.style is not None else "") or ""
        if style == "Title" or style.startswith("Heading"):
            last = style.split()[-1]
            t = "#" * min(int(last) if last.isdigit() else 1, 6) + " " + t
        out.append(t)
    return "\n\n".join(out)


def detect_format(text: str) -> str:
    """'transcript' for a JSON turn list, or mostly 'Name: text' lines from a few speakers."""
    if _json_turns(text) is not None:
        return "transcript"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    speakers = [m.group(1).strip().lower() for m in map(_TURN_RE.match, lines) if m]
    if not lines or len(speakers) / len(lines) < 0.6:
        return "prose"
    # A 'Key: value' notes list has many keys used once; a conversation has a few voices that repeat.
    return "transcript" if len(set(speakers)) <= max(2, len(speakers) // 2) else "prose"


def split_prose(text: str) -> list[Chunk]:
    """Paragraphs merged up to CHUNK_CHARS, long ones cut at sentence ends, each prefixed
    with its nearest Markdown heading so a sentence keeps its subject."""
    chunks: list[Chunk] = []
    buf, heading = "", ""

    def flush() -> None:
        nonlocal buf
        if buf:
            chunks.append(Chunk(f"# {heading}\n{buf}" if heading else buf))
            buf = ""

    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        while lines and _HEADING_RE.match(lines[0]):
            flush()                          # never let a chunk span two sections
            heading = _HEADING_RE.match(lines[0]).group(1)
            lines = lines[1:]
        for piece in _pieces("\n".join(lines)):
            if buf and len(buf) + 2 + len(piece) <= CHUNK_CHARS:
                buf += "\n\n" + piece
            else:
                flush()
                buf = piece
    flush()
    return chunks


def split_transcript(text: str, owner: str = "") -> list[Chunk]:
    """One chunk per turn (same-speaker turns merged). The owner's turns are speaker "user";
    an assistant turn becomes agent_reply on the turn before it."""
    turns = _json_turns(text)
    if turns is None:
        turns = _line_turns(text)
    turns = [(w, s, a) for w, s, a in turns if s]
    people = list(dict.fromkeys(w for w, _, a in turns if not a))
    owner_key = owner.strip().lower()
    if owner_key and owner_key not in {p.lower() for p in people}:
        raise ValueError(f"Speaker {owner!r} is not in the transcript. "
                         f"Speakers: {', '.join(people) or 'none'}.")
    owner_key = owner_key or (people[0].lower() if people else "")
    chunks: list[Chunk] = []
    for who, said, is_assistant in turns:
        if is_assistant:
            if chunks:                       # a reply with no turn before it has nothing to attach to
                prev = chunks[-1]
                reply = f"{prev.agent_reply} {said}" if prev.agent_reply else said
                chunks[-1] = replace(prev, agent_reply=reply)
            continue
        speaker = "user" if who.lower() == owner_key else who
        for piece in _pieces(said):
            prev = chunks[-1] if chunks else None
            if (prev and prev.speaker == speaker and prev.agent_reply is None
                    and len(prev.text) + 1 + len(piece) <= CHUNK_CHARS):
                chunks[-1] = replace(prev, text=f"{prev.text} {piece}")
            else:
                chunks.append(Chunk(piece, speaker))
    return chunks


def _json_turns(text: str) -> list[tuple[str, str, bool]] | None:
    """[(speaker, text, is_assistant)] from a JSON list of turn objects, else None."""
    s = text.strip()
    if not s.startswith("["):
        return None
    try:
        data = json.loads(s)
    except ValueError:
        return None
    if not isinstance(data, list) or not data:
        return None
    turns = []
    for item in data:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role") or "").strip().lower()
        who = str(item.get("speaker") or item.get("name") or item.get("role") or "").strip()
        said = item.get("text", item.get("content"))
        if not who or not isinstance(said, str):
            return None
        turns.append((who, said.strip(), role == "assistant" or who.lower() in ASSISTANT_NAMES))
    return turns


def _line_turns(text: str) -> list[tuple[str, str, bool]]:
    """Turns from 'Name: text' lines; a line without a name continues the previous turn."""
    turns: list[list] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _TURN_RE.match(line)
        if m:
            who = m.group(1).strip()
            turns.append([who, m.group(2).strip(), who.lower() in ASSISTANT_NAMES])
        elif turns:
            turns[-1][1] += " " + line
    return [(w, s, a) for w, s, a in turns]


def _pieces(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Split text into pieces of at most ``limit`` chars, at sentence ends where possible."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    out, cur = [], ""
    for sent in _SENTENCE_END.split(text):
        while len(sent) > limit:             # one sentence longer than the limit: cut at a space
            cut = sent.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                out.append(cur)
                cur = ""
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if cur and len(cur) + 1 + len(sent) > limit:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}" if cur else sent
    if cur:
        out.append(cur)
    return [p for p in out if p]
```

**Step 4:** tests pass. Fix the code, not the tests, unless a test contradicts
this plan.

---

## Task 5 (Ingest): the job runner

**Files:** Modify `web/ingest.py` (append); Test `tests/test_ingest_jobs.py`.

**Interface (fixed):**

```python
class Jobs:
    def __init__(self, prepare=None, finish=None): ...
    def submit(self, mem, space: str, chunks: list[Chunk], observed_at: str | None = None,
               *, filename: str = "", kind: str = "", on_done=None) -> str   # job_id
    def get(self, job_id: str) -> dict | None
    def list(self, space: str) -> list[dict]          # newest first
    def busy(self, space: str) -> bool                # any queued/running job for space
    def cancel(self, job_id: str) -> bool             # False if unknown
```

Job record: `{id, space, filename, format, state, done, total, facts, errors,
error_count, created_at, started_at, finished_at}`; `state` ∈ `queued | running
| done | failed | cancelled`; `errors` = up to 50 `{index, text, error}`.

**Step 1: failing tests** with a fake:

```python
class FakeMem:
    def __init__(self, fail=(), soft_fail=(), gate=None):
        self.calls, self.fail, self.soft_fail, self.gate = [], set(fail), set(soft_fail), gate
    def ingest(self, text, speaker="user", agent_reply=None, observed_at=None, **kw):
        if self.gate is not None:
            self.gate.wait(5)
        self.calls.append((text, speaker, agent_reply, observed_at, kw))
        if text in self.fail:
            raise RuntimeError("llm down")
        if text in self.soft_fail:
            return {"facts_count": 0, "memory_ids": [], "error": "no key"}
        return {"facts_count": 2, "memory_ids": ["m"]}
```

and `_wait(jobs, job_id, timeout=5)` polling `get()` until a terminal state.
Cases: counts (`done`, `total`, `facts`) for 3 chunks; ingest kwargs carry
speaker, agent_reply and observed_at, and **no** `session_id`; a raising chunk
and a soft-fail chunk are both recorded in `errors` and skipped, job `done`;
the first 5 chunks all failing stops the job at 5 with `failed` (7 submitted);
every chunk failing in a 3-chunk job → `failed`; jobs run one at a time in
submit order (a gated first job keeps the second `queued`); `busy(space)` true
while queued/running, false after, false for another space; `cancel` of a
running job stops it before the next chunk (`cancelled`, `done < total`);
cancel of a queued job → `cancelled` with `done == 0`, no ingest calls; cancel
of unknown id → False; `prepare(mem)` runs once before the first chunk and
`finish(mem)` once after (also after failure and cancel); `on_done()` runs once
at the end; a raising `prepare` fails the job but the worker keeps serving the
next job; `list(space)` newest first and filtered; `get()` returns a copy
(mutating it does not change the job); `errors` capped at 50.

**Step 3: implement** (append to `web/ingest.py`; add `import queue, threading,
time, uuid` to the imports):

```python
FAIL_FAST = 5            # the first N chunks all failing means config (key, network), not bad text
MAX_ERRORS_KEPT = 50
KEEP_FINISHED = 50


class Jobs:
    """Background ingest: one daemon worker, jobs run one at a time in submit order.

    ``prepare(mem)`` / ``finish(mem)`` run before and after each job (run.py uses them to
    clear the voice loop's reply history and flush the session batch under the write lock).
    ponytail: state is in memory and lost on restart; persist it if jobs must survive restarts.
    """

    def __init__(self, prepare=None, finish=None) -> None:
        self._prepare = prepare or (lambda mem: None)
        self._finish = finish or (lambda mem: None)
        self._q: queue.Queue = queue.Queue()
        self._jobs: dict[str, dict] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(self, mem, space: str, chunks: list[Chunk], observed_at: str | None = None,
               *, filename: str = "", kind: str = "", on_done=None) -> str:
        job_id = uuid.uuid4().hex[:12]
        rec = {"id": job_id, "space": space, "filename": filename, "format": kind,
               "state": "queued", "done": 0, "total": len(chunks), "facts": 0,
               "errors": [], "error_count": 0, "created_at": time.time(),
               "started_at": None, "finished_at": None}
        with self._lock:
            self._jobs[job_id] = rec
            self._trim()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._loop, name="ingest-jobs", daemon=True)
                self._worker.start()
        self._q.put((job_id, mem, list(chunks), observed_at, on_done))
        return job_id

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            return _copy(rec) if rec else None

    def list(self, space: str) -> list[dict]:
        with self._lock:
            recs = [_copy(r) for r in self._jobs.values() if r["space"] == space]
        return sorted(recs, key=lambda r: r["created_at"], reverse=True)

    def busy(self, space: str) -> bool:
        with self._lock:
            return any(r["space"] == space and r["state"] in ("queued", "running")
                       for r in self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return False
            if rec["state"] in ("queued", "running"):
                self._cancelled.add(job_id)
            return True

    def _loop(self) -> None:
        while True:
            job_id, mem, chunks, observed_at, on_done = self._q.get()
            try:
                self._run(job_id, mem, chunks, observed_at, on_done)
            except Exception as e:          # never let one job kill the worker
                self._note(job_id, -1, "", f"{type(e).__name__}: {e}")
                self._update(job_id, state="failed", finished_at=time.time())

    def _run(self, job_id, mem, chunks, observed_at, on_done) -> None:
        if self._take_cancel(job_id):
            self._update(job_id, state="cancelled", finished_at=time.time())
            return
        self._update(job_id, state="running", started_at=time.time())
        state = "done"
        try:
            self._prepare(mem)
            for i, c in enumerate(chunks):
                if self._take_cancel(job_id):
                    state = "cancelled"
                    break
                facts, err = 0, None
                try:
                    # No session_id: passing one fires a session-boundary batch and rescopes
                    # the live session's retrieval. finish() flushes once at the end instead.
                    r = mem.ingest(c.text, speaker=c.speaker, agent_reply=c.agent_reply,
                                   observed_at=observed_at) or {}
                    err = r.get("error")
                    facts = int(r.get("facts_count") or 0)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                with self._lock:
                    rec = self._jobs[job_id]
                    rec["done"] = i + 1
                    rec["facts"] += facts
                    if err:
                        rec["error_count"] += 1
                        if len(rec["errors"]) < MAX_ERRORS_KEPT:
                            rec["errors"].append({"index": i, "text": c.text[:300],
                                                  "error": str(err)[:500]})
                    all_failed = rec["error_count"] == rec["done"]
                if all_failed and i + 1 == FAIL_FAST:
                    state = "failed"
                    break
        except Exception as e:
            state = "failed"
            self._note(job_id, -1, "", f"{type(e).__name__}: {e}")
        finally:
            for hook in (lambda: self._finish(mem), on_done):
                if hook is None:
                    continue
                try:
                    hook()
                except Exception as e:
                    self._note(job_id, -1, "", f"{type(e).__name__}: {e}")
        with self._lock:
            rec = self._jobs[job_id]
            if state == "done" and rec["done"] and rec["error_count"] == rec["done"]:
                state = "failed"            # nothing stored at all
        self._update(job_id, state=state, finished_at=time.time())

    def _take_cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._cancelled:
                self._cancelled.discard(job_id)
                return True
            return False

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def _note(self, job_id: str, index: int, text: str, error: str) -> None:
        with self._lock:
            rec = self._jobs[job_id]
            if len(rec["errors"]) < MAX_ERRORS_KEPT:
                rec["errors"].append({"index": index, "text": text, "error": error[:500]})

    def _trim(self) -> None:
        """Keep the newest KEEP_FINISHED finished jobs (caller holds the lock)."""
        done = sorted((r for r in self._jobs.values()
                       if r["state"] not in ("queued", "running")),
                      key=lambda r: r["created_at"])
        for r in done[:max(0, len(done) - KEEP_FINISHED)]:
            del self._jobs[r["id"]]


def _copy(rec: dict) -> dict:
    return dict(rec, errors=[dict(e) for e in rec["errors"]])
```

Note: hook-level errors (`index: -1`) do not count toward `error_count`.

**Step 4:** tests pass (no test may hang: every gate is released in
`tearDown`/`finally`).

---

## Task 6 (API): routes

**Files:** Create `web/memories_api.py`; Test `tests/test_memories_api.py`.

**Interface (fixed):**

```python
def register_routes(app, *, resolve, delete_space, clear_space, facts, profile,
                    semantic, update_fact, delete_fact, jobs, plan,
                    on_change=lambda space: None) -> None
```

Callbacks (all supplied by run.py; fakes in tests):

| Callback | Contract |
|---|---|
| `resolve(space) -> mem` | Existing brain only. `FileNotFoundError` unknown, `ValueError` bad name. |
| `facts(mem) -> list[dict]` | `{id, text, slot, entities, date, role}` for every fact. |
| `profile(mem) -> list[dict]` | Right-brain tree items `{cluster, slot, raw, text, desc, notes[{text, emotion, cause}]}`. |
| `semantic(mem, q) -> list[tuple[str, float]]` | Ranked `(id, score)`. |
| `update_fact(mem, id, text) -> bool` / `delete_fact(mem, id) -> bool` | False = no such id. |
| `clear_space(name) -> dict` / `delete_space(name) -> None` | Raise `FileNotFoundError` (404), `PermissionError` / `RuntimeError` / `FileExistsError` (409), `ValueError` (400). |
| `jobs` | a `Jobs` (Task 5). |
| `plan(filename, data, fmt, owner) -> (kind, chunks)` | `ingest.plan_chunks`. |
| `on_change(space)` | Called after any successful write (run.py drops the cached cartridge). |

Routes: see the table in the spec, plus `GET /api/ingest?space=` (job list),
`POST /api/ingest/{job_id}/cancel`, and `GET /memories` (serves
`web/memories.html` with `Cache-Control: no-store`).

**Step 1: failing tests.** Bare app, never import utils/run:

```python
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
from memories_api import register_routes  # noqa: E402

class FakeMem: ...          # holds rows: dict id -> row
class FakeJobs: ...         # records submit() args; busy_spaces set; get/list/cancel
```

`setUp` builds `spaces = {"a": FakeMem(rows...), "b": FakeMem(...)}`, callbacks
backed by it (`resolve` raises `FileNotFoundError` for other names), registers
routes on `FastAPI()`, wraps in `TestClient`. Cases:

- `GET /api/mem?space=a`: all rows, `total`, `slots`/`entities` computed before
  filtering, newest date first, blank dates last; `page`/`size` slicing;
  `size` clamped to 200; `q` substring case-insensitive; `slot`, `entity`,
  `role` filters; `mode=semantic` returns only ranked ids in rank order with a
  `score`; `mode=bogus` → 400; unknown space → 404 **and resolve was not asked
  to create anything** (the fake just raises).
- `GET /api/mem/profile?space=a` → `{"items": [...]}`.
- `PATCH /api/mem/m1?space=a` `{"text":" new "}` → 200, stored `"new"`,
  `on_change("a")` called; blank text → 400 and no update call; unknown id → 404.
- `DELETE /api/mem/m1?space=a` → 200, row gone in `a`, untouched in `b`;
  unknown id → 404.
- `POST /api/mem/clear?space=a` `{"confirm":"a"}` → 200; wrong confirm → 400
  and clear not called; busy space → 409; `clear_space` raising
  `PermissionError` → 409 with its message.
- `DELETE /api/spaces/b` with `client.request("DELETE", ..., json={"confirm":"b"})`
  → 200; wrong confirm → 400; `PermissionError` → 409; `FileNotFoundError` → 404;
  busy → 409.
- `POST /api/ingest?space=a&filename=notes.txt` with `content=b"..."` → 200
  `{job_id, total, format}`; the job got `mem` of space a, `observed_at` =
  today's ISO date when omitted; `observed_at=2026-01-02` passes through;
  `observed_at=01/02/2026` → 400; `plan` raising `ValueError` → 400 with the
  message; unknown space → 404; `on_done()` passed to submit calls
  `on_change("a")`.
- `GET /api/ingest?space=a`, `GET /api/ingest/{id}` (404 unknown),
  `POST /api/ingest/{id}/cancel` (404 unknown).
- `GET /memories` → 200 `text/html` once `web/memories.html` exists (skip with
  `skipUnless` if the file is absent).

**Step 3: implement** `web/memories_api.py`:

```python
"""/memories operator page: browse, edit and delete facts, read the profile, load text or files,
clear and delete brains.

Every route names its brain with ``space``; nothing here switches the voice demo's active brain.
run.py passes every dependency in as a callback, so this module imports only fastapi and the
stdlib and is tested on a bare FastAPI() with fakes (tests/test_memories_api.py).
"""
from __future__ import annotations

import asyncio
import re
from datetime import date
from pathlib import Path

from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
PAGE_SIZE, MAX_PAGE_SIZE = 50, 200
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NOCACHE = {"Cache-Control": "no-store"}


class FactEdit(BaseModel):
    text: str


class Confirm(BaseModel):
    confirm: str = ""


def _status(e: Exception) -> HTTPException | None:
    """The status code for an exception the callbacks raise on purpose; None means a real bug (500)."""
    if isinstance(e, FileNotFoundError):
        return HTTPException(404, "No brain with that name.")
    if isinstance(e, (PermissionError, FileExistsError, RuntimeError)):
        return HTTPException(409, str(e))
    if isinstance(e, ValueError):
        return HTTPException(400, str(e))
    return None


def _call(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        mapped = _status(e)
        if mapped is None:
            raise
        raise mapped from e


def _confirm(space: str, body: Confirm) -> None:
    if body.confirm != space:
        raise HTTPException(400, "Type the brain's name exactly to confirm.")


def register_routes(app, *, resolve, delete_space, clear_space, facts, profile, semantic,
                    update_fact, delete_fact, jobs, plan, on_change=lambda space: None) -> None:
    """Attach the /memories page and its /api/mem, /api/ingest and DELETE /api/spaces routes."""

    def mem_of(space: str):
        return _call(resolve, space)

    @app.get("/memories")
    def memories_page():
        return FileResponse(HERE / "memories.html", headers=_NOCACHE)

    # Sync defs on purpose: opening a brain, LLM-backed edits and waiting on the write lock all
    # block, and FastAPI runs sync routes in its threadpool instead of on the voice event loop.

    @app.get("/api/mem")
    def api_facts(space: str, q: str = "", mode: str = "substring", slot: str = "",
                  entity: str = "", role: str = "", page: int = 1,
                  size: int = PAGE_SIZE) -> dict:
        if mode not in ("substring", "semantic"):
            raise HTTPException(400, "mode must be substring or semantic.")
        mem = mem_of(space)
        rows = facts(mem)
        # Filter options come from the whole brain so the dropdowns don't shrink as you filter.
        slots = sorted({r["slot"] for r in rows})
        entities = sorted({e for r in rows for e in r["entities"]}, key=str.lower)
        q = q.strip()
        if q and mode == "semantic":
            ranked = semantic(mem, q)
            order = {mid: i for i, (mid, _) in enumerate(ranked)}
            score = dict(ranked)
            rows = sorted((dict(r, score=round(score[r["id"]], 3)) for r in rows if r["id"] in order),
                          key=lambda r: order[r["id"]])
        else:
            if q:
                ql = q.lower()
                rows = [r for r in rows if ql in r["text"].lower()]
            rows = sorted(rows, key=lambda r: r["date"], reverse=True)   # blank dates sort last
        if slot:
            rows = [r for r in rows if r["slot"] == slot]
        if entity:
            rows = [r for r in rows if entity in r["entities"]]
        if role:
            rows = [r for r in rows if r["role"] == role]
        size = max(1, min(size, MAX_PAGE_SIZE))
        page = max(1, page)
        start = (page - 1) * size
        return {"items": rows[start:start + size], "total": len(rows), "page": page,
                "size": size, "slots": slots, "entities": entities}

    @app.get("/api/mem/profile")
    def api_profile(space: str) -> dict:
        return {"items": profile(mem_of(space))}

    @app.patch("/api/mem/{mid}")
    def api_edit(mid: str, space: str, body: FactEdit) -> dict:
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "Text cannot be empty.")
        if not update_fact(mem_of(space), mid, text):
            raise HTTPException(404, "No such memory.")
        on_change(space)
        return {"id": mid, "text": text}

    @app.delete("/api/mem/{mid}")
    def api_delete(mid: str, space: str) -> dict:
        if not delete_fact(mem_of(space), mid):
            raise HTTPException(404, "No such memory.")
        on_change(space)
        return {"id": mid, "deleted": True}

    @app.post("/api/mem/clear")
    def api_clear(space: str, body: Confirm) -> dict:
        _confirm(space, body)
        if jobs.busy(space):
            raise HTTPException(409, "A load is running for this brain. Wait for it or cancel it.")
        out = _call(clear_space, space)
        on_change(space)
        return out or {"cleared": space}

    @app.delete("/api/spaces/{name}")
    def api_delete_space(name: str, body: Confirm) -> dict:
        _confirm(name, body)
        if jobs.busy(name):
            raise HTTPException(409, "A load is running for this brain. Wait for it or cancel it.")
        _call(delete_space, name)
        on_change(name)
        return {"deleted": name}

    @app.post("/api/ingest")
    async def api_ingest(req: Request, space: str, filename: str = "",
                         fmt: str = Query("auto", alias="format"), owner: str = "",
                         observed_at: str = "") -> dict:
        observed_at = observed_at.strip() or date.today().isoformat()
        if not _DATE_RE.match(observed_at):
            raise HTTPException(400, "The date must be YYYY-MM-DD.")
        data = await req.body()
        # Opening a brain (seconds) and reading a PDF are blocking: keep them off the event loop.
        mem = await asyncio.to_thread(mem_of, space)
        try:
            kind, chunks = await asyncio.to_thread(plan, filename, data, fmt, owner)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        job_id = jobs.submit(mem, space, chunks, observed_at,
                             filename=filename or "pasted text", kind=kind,
                             on_done=lambda: on_change(space))
        return {"job_id": job_id, "total": len(chunks), "format": kind}

    @app.get("/api/ingest")
    def api_jobs(space: str) -> dict:
        return {"jobs": jobs.list(space)}

    @app.get("/api/ingest/{job_id}")
    def api_job(job_id: str) -> dict:
        rec = jobs.get(job_id)
        if rec is None:
            raise HTTPException(404, "No such job.")
        return rec

    @app.post("/api/ingest/{job_id}/cancel")
    def api_cancel(job_id: str) -> dict:
        if not jobs.cancel(job_id):
            raise HTTPException(404, "No such job.")
        return jobs.get(job_id)
```

Check: FastAPI accepts a pydantic body on `DELETE` (it does); `TestClient`
needs `client.request("DELETE", url, json=...)`.

**Step 4:** tests pass.

---

## Task 7 (Server): run.py helpers, space lifecycle, wiring

**Files:** Modify `web/run.py`, `web/utils.py`. No unit tests (run.py cannot be
imported in tests); verified in Task 9. After editing:
`uv run --no-sync python -m py_compile web/run.py web/utils.py`.

**7a. Imports.** Add `import shutil` near the top imports (and `os` if missing).
Next to the flat imports (~102-105) add
`from ingest import Jobs, plan_chunks` and
`from supermem.leftbrain.mem0_backend_store import evict_client`.

**7b. Live-session counter** (next to `_REMEMBER_TASKS`, ~1947):

```python
# Voice websockets currently open. Clearing the live brain waits for 0 (see clear_space).
_LIVE_SESSIONS = 0


def _counted(session):
    """Wrap a websocket session loop so _LIVE_SESSIONS tracks it (single event loop: no lock)."""
    async def run(sock):
        global _LIVE_SESSIONS
        _LIVE_SESSIONS += 1
        try:
            return await session(sock)
        finally:
            _LIVE_SESSIONS -= 1
    return run
```

and in the `build_app(...)` call wrap the session:
`_counted(realtime_session if MODE == "realtime" else llm_tts_session)`.

**7c. Space lifecycle** (around `get_space`, ~1141-1230):

```python
# ponytail: one lock serialises opening and wiping spaces; per-space locks if opens contend.
_SPACES_LOCK = threading.RLock()


def get_space(name: str):
    """Get (creating if needed) the SuperMem for this space."""
    _, safe = space_dir(name)
    inst = _SPACES.get(safe)
    if inst is not None:          # fast path, no lock: async callers never wait on another brain's wipe
        return inst
    with _SPACES_LOCK:
        if safe not in _SPACES:
            ...existing build + warmup + print, unchanged...
        return _SPACES[safe]


def space_exists(name: str) -> bool:
    """A space directory with exactly this name. The volume may be case-insensitive, so
    is_dir() alone would let "Demo" match "demo" -- and then delete it."""
    try:
        d, safe = space_dir(name)
    except ValueError:
        return False
    return d.is_dir() and safe in os.listdir(d.parent)


def resolve_space(name: str):
    """The instance of an existing space. Never creates one: an unknown name is FileNotFoundError."""
    _, safe = space_dir(name)
    with _SPACES_LOCK:
        if not space_exists(safe):
            raise FileNotFoundError(safe)
        return get_space(safe)


def _space_audio(d: Path) -> list:
    """Turn recordings this space's audio_archive points at; they live in the shared TURN_AUDIO_DIR."""
    import sqlite3
    from supermem.utils.common import space as _sp
    try:
        c = sqlite3.connect(f"file:{_sp.db(d)}?mode=ro", uri=True)
        try:
            rows = c.execute("SELECT DISTINCT audio_path FROM audio_archive").fetchall()
        finally:
            c.close()
    except sqlite3.Error:
        return []
    base = TURN_AUDIO_DIR.resolve()
    return [p for p in (Path(r[0]).resolve() for r in rows if r[0]) if p.parent == base]


def _wipe_space(safe: str, d: Path) -> None:
    """Drop every in-process handle on the space, close what stays open, remove its files.

    Caller holds _SPACES_LOCK. The order matters: the cached mem0 client must be closed before
    rmtree, or a space re-created under this name reuses it (old vectors, readonly writes).
    """
    import gc
    inst = _SPACES.get(safe)
    if inst is not None and Path(inst._o._memory_root).resolve() != d.resolve():
        raise RuntimeError(f"\"{safe}\" is not stored under {d}; refusing to delete it.")
    _SPACES.pop(safe, None)
    _CARTRIDGES.pop(safe, None)          # a compiled copy of every fact in this brain
    _OWNER_NAME_CACHE.pop(safe, None)
    evict_client(d)
    gc.collect()                         # the stores never close their per-call sqlite connections
    audio = _space_audio(d)
    shutil.rmtree(d)
    for p in audio:
        p.unlink(missing_ok=True)


def delete_space(name: str) -> None:
    """Delete a brain and its recordings. The live brain is refused."""
    d, safe = space_dir(name)
    with _SPACES_LOCK:
        if not space_exists(safe):
            raise FileNotFoundError(safe)
        if safe == ACTIVE_SPACE:
            raise PermissionError("This brain is live in the demo. Start the server with "
                                  "another --space to delete it.")
        _wipe_space(safe, d)


def clear_space(name: str) -> dict:
    """Empty a brain: wipe it and re-create it empty under the same name.

    The live brain is cleared only while no voice session is connected and no memory write is
    queued: a live session holds the old instance and would write the old owner back into the
    new store. The wipe holds that instance's write lock, so an in-flight background ingest
    finishes first.
    """
    d, safe = space_dir(name)
    with _SPACES_LOCK:
        if not space_exists(safe):
            raise FileNotFoundError(safe)
        live = safe == ACTIVE_SPACE
        if live and (_LIVE_SESSIONS or any(not t.done() for t in list(_REMEMBER_TASKS))):
            raise PermissionError("A voice session is using this brain. End it, then clear.")
        inst = _SPACES.get(safe)
        if inst is not None:
            with inst._o._write_lock:
                _wipe_space(safe, d)
        else:
            _wipe_space(safe, d)
        d.mkdir(parents=True, exist_ok=True)
        _write_space_language(safe, "en")
        if live:
            use_space(safe)              # rebind the global vm to a fresh, empty instance
        return {"cleared": safe}
```

`TURN_AUDIO_DIR` is defined later in the module; it is read at call time, so the
forward reference is fine.

**7d. Fact and profile helpers.** Replace `fact_index` (~2962) with an
instance-aware version, add `_entity_names` + `left_entries` just above
`memory_snapshot`, rewrite `memory_snapshot`'s body on top of `left_entries`
(output keys, order and caps unchanged), and give `right_brain_tree` `mem` and
`per_slot` parameters. Keep the existing comments that still apply.

```python
def fact_index(uid: str, mem=None) -> dict:
    """Left-brain memory id -> original fact text. ``mem`` defaults to the live ``vm``.
    (keep the existing explanatory paragraph)"""
    mem = vm if mem is None else mem
    try:
        entries = mem._o._get_repo()._vector_store.list_entries(user_id=uid)
        return {e["id"]: e["text"] for e in entries}
    except Exception as e:
        print(f"[web] reading left-brain facts failed: {e}", flush=True)
        return {}


def _entity_names(cog) -> dict:
    """memory id -> entity names, in the order entity_ids_for_memory() returns them (by entity id).
    One JOIN per space instead of one lookup per memory."""
    with cog._conn() as c:
        rows = c.execute(
            "SELECT l.memory_id, l.entity_id, e.name FROM entity_memory_links l "
            "LEFT JOIN entities e ON e.id = l.entity_id "
            "ORDER BY l.memory_id, l.entity_id").fetchall()
    out: dict = {}
    for r in rows:
        out.setdefault(r["memory_id"], []).append((r["name"] or "").strip() or r["entity_id"])
    return out


def left_entries(mem=None) -> list[dict]:
    """Every left-brain entry of one space joined to its slot and entity names, uncapped.

    [{id, text, slot, entities, date, role}] in store order. memory_snapshot() caps this for the
    brain map; the /memories page filters and pages it. ``mem`` defaults to the live ``vm``.
    ponytail: list_entries reads at most 10,000 entries (mem0 get_all top_k).
    """
    from supermem.leftbrain.cognitive_graph.types import SlotV2

    mem = vm if mem is None else mem
    uid = mem._o._user_id
    repo = mem._o._get_repo()
    cog = repo._cognitive_store
    slot_of: dict = {}
    for slot in SlotV2:
        for mid in cog.memory_ids_for_slots(uid, [slot]):
            slot_of.setdefault(mid, slot.value)
    try:
        ents = _entity_names(cog)
    except Exception as e:
        print(f"[web] reading entity links failed: {e}", flush=True)
        ents = {}
    out = []
    for e in repo._vector_store.list_entries(user_id=uid):
        # date is time_start[:10]; a turn stored without observed_at has a time of day there
        # ("09:20:37"). If it doesn't look like a date, blank it.
        d = str(e.get("date", ""))
        mid = str(e["id"])
        out.append({"id": mid, "text": e["text"], "slot": slot_of.get(mid, "daily_life"),
                    "entities": ents.get(mid, []), "date": d if d[:4].isdigit() else "",
                    "role": e.get("role") or "user"})
    return out
```

`memory_snapshot` body (keep its docstring and comments):

```python
    uid = vm._o._user_id
    left, right = [], []
    try:
        entries = [e for e in left_entries(vm) if e["role"] != "assistant"]
        per_slot, kept = {}, []
        hit_first = ([e for e in entries if e["id"] in _LAST_HIT_IDS] +
                     [e for e in entries if e["id"] not in _LAST_HIT_IDS])
        for e in hit_first:
            hit = e["id"] in _LAST_HIT_IDS
            per_slot[e["slot"]] = per_slot.get(e["slot"], 0) + 1
            if hit or per_slot[e["slot"]] <= LB_ENTRIES_PER_SLOT:
                kept.append(e)
        head = [e for e in kept if e["id"] in _LAST_HIT_IDS]
        rest = [e for e in kept if e["id"] not in _LAST_HIT_IDS]
        left = [{"text": e["text"], "date": e["date"], "slot": e["slot"],
                 "hit": e["id"] in _LAST_HIT_IDS, "entities": e["entities"][:6]}
                for e in (head + rest)[:max(limit, len(head))]]
    except Exception as e:
        print(f"[web] left-brain snapshot read failed: {e}", flush=True)
    try:
        right = right_brain_tree(uid, fact_index(uid))
    except Exception as e:
        print(f"[web] right-brain snapshot read failed: {e}", flush=True)
    return {"left": left, "right": right}
```

`right_brain_tree` signature and body changes:

```python
def right_brain_tree(uid, facts=None, mem=None, per_slot: int = RB_ENTITIES_PER_SLOT):
    """(keep existing docstring) ``facts`` (id -> live fact text) resolves each evidence's
    cause, so a deleted fact drops out and an edited one shows its new text. ``mem`` defaults to
    the live ``vm``; ``per_slot`` caps judgements per slot (the /memories page passes a large one)."""
    mem = vm if mem is None else mem
    facts = facts or {}
    # ponytail: the plain-language rewrite is keyed to the live brain's owner name (owner_name()
    # reads ACTIVE_SPACE); other brains show the stored claim and cost no LLM calls.
    human = mem is vm
    try:
        store = mem._o._right._traits()
    except Exception as e:
        ...unchanged...
    traits = list(store.all(uid, per_slot=per_slot))
    if human:
        rb_human_batch([t.claim for t in traits])
    ...
            "text": rb_human(t.claim) if human else t.claim,
            ...
            "notes": [{"text": e.quote, "emotion": e.emotion,
                       "cause": facts.get(e.cause_id, "") if e.cause_id else e.cause}
                      for e in t.evidence],
```

Check `RB_ENTITIES_PER_SLOT` is defined above `right_brain_tree` (it is, ~2977).
If it were not, default `per_slot=None` and resolve inside.

**7e. Page callbacks** (after the helpers, before `app = utils.build_app(...)`):

```python
SEMANTIC_TOP_K = 200   # ponytail: mem0 searches with threshold 0.0, so this is a rank cutoff


def semantic_ids(mem, q: str) -> list:
    """Ranked (memory_id, score) for a query. Uses the vector store directly: SuperMem.search would
    bump heat, book subgraph activations for the live session and hide near-duplicates."""
    q = (q or "").strip()
    if not q:
        return []
    hits = mem._o._get_repo().search(q, user_id=mem._o._user_id, top_k=SEMANTIC_TOP_K,
                                     include_assistant=True)
    return [(h.memory_id, float(h.base_score)) for h in hits]


def _update_fact(mem, mid: str, text: str) -> bool:
    # user_id makes the repo re-annotate the graph (one LLM call); without it the graph keeps
    # the old entities.
    with mem._o._write_lock:
        return mem._o._get_repo().update_memory(mid, text, user_id=mem._o._user_id)


def _delete_fact(mem, mid: str) -> bool:
    with mem._o._write_lock:
        return mem._o._get_repo().delete_memory(mid)


def _brain_changed(space: str) -> None:
    """A write landed in this brain: drop its compiled cartridge so the next turn recompiles."""
    try:
        _CARTRIDGES.pop(space_dir(space)[1], None)
    except ValueError:
        pass


def _job_prepare(mem) -> None:
    # The voice loop's last reply would otherwise become prior_reply for the first chunk.
    with mem._o._write_lock:
        mem._o._exchanges.clear()


def _job_finish(mem) -> None:
    # Jobs pass no session_id, so the session-boundary batch never fires on its own; run it
    # once, and don't leave the transcript's last reply as the voice loop's prior_reply.
    with mem._o._write_lock:
        mem.flush()
        mem._o._exchanges.clear()


INGEST_JOBS = Jobs(prepare=_job_prepare, finish=_job_finish)
```

In the `build_app(...)` call add:

```python
                      memories=dict(
                          resolve=resolve_space, delete_space=delete_space,
                          clear_space=clear_space, facts=left_entries,
                          profile=lambda mem: right_brain_tree(
                              mem._o._user_id, fact_index(mem._o._user_id, mem), mem,
                              per_slot=10_000),
                          semantic=semantic_ids, update_fact=_update_fact,
                          delete_fact=_delete_fact, jobs=INGEST_JOBS, plan=plan_chunks,
                          on_change=_brain_changed))
```

**7f. `web/utils.py` `build_app`:** add `memories=None` to the signature and,
next to the compare wiring:

```python
    if memories:                                     # the /memories operator page and its routes
        from memories_api import register_routes as register_memories
        register_memories(app, **memories)
```

Update the docstring's parameter notes with one line for `memories`.

**Verification for Task 7:** `py_compile` both files; then
`uv run --no-sync python -c "import sys; sys.path.insert(0,'web'); import ingest, memories_api"`.

---

## Task 8 (Page): `web/memories.html`, demo link, README

**Files:** Create `web/memories.html`; Modify `web/supermem.html`, `README.md`.

**Look:** the demo's look, same tokens (copy `:root`, base rules, `.panel`,
`.panel-h`, `.btn` family, `.textin`, `.sel`, `.tag` chips, `.pill`, `.seg`,
`.mt-table`, `#toast`, `.turn-ctx` from `web/supermem.html` lines ~11-32, 82-110,
139-180, 273-276, 389-412). Differences: `body` scrolls (no `overflow:hidden`),
`:root{color-scheme:dark}`, textarea inherits fonts. Dark only. System fonts.
No external scripts or fonts. Works at 375px wide (single column below 900px).

**Structure:**

- **Top bar:** "SuperMem · Memories" title; brain picker `<select class="sel" id="brainSel">`
  (names only; the live brain suffixed " · live"); "New brain" button → inline
  name input + Create (POST `/api/spaces` `{name}`; on success select the
  returned `id`; show `detail` on 400/409); a `.pill.space` "Live in demo" when
  the selected brain is the active one; link "← Demo" to `/`. Remember the
  selection in `localStorage['vm-mem-space']` inside try/catch; default to the
  active brain. **Never call `/api/spaces/{name}/use`.**
- **Left column, panel "Add memories":** textarea `#pasteText`; drop zone
  `#drop` (click opens a hidden `<input type=file accept=".txt,.md,.json,.pdf,.docx">`;
  drag-and-drop too; shows the chosen file name with a remove ×; choosing a file
  disables the textarea and vice versa); fields: format select `#fmt`
  (Auto / Prose / Transcript), owner input `#owner` ("Owner speaker, for transcripts"),
  date `#obsDate` (`type=date`, default today); button `#addBtn` "Add to brain"
  (disabled with nothing to send). Submit = `fetch('/api/ingest?'+params, {method:'POST', body: file || new Blob([text])})`
  with `filename` = file name (empty for pasted text). On 4xx show `detail`
  inline under the button. On success clear the inputs.
- **Jobs list** under the form: `GET /api/ingest?space=` on load and every 1 s
  while any job is queued/running (stop polling otherwise). Each row: file name,
  format, state chip, "34 / 120 chunks · 51 facts · 2 errors", elapsed and ETA
  (`(elapsed / done) * (total - done)`, shown once `done > 0`), a Cancel button
  while queued/running (POST `.../cancel`). Errors in
  `<details class="turn-ctx"><summary>N errors</summary><pre>#index: error\n  text</pre></details>`.
  When a job leaves queued/running, reload Facts and Profile.
- **Right column, panel "Memories"** with `.seg` tabs **Facts** / **Profile**.
  - Facts toolbar: search `#q` (Enter, or 300 ms debounce), a "Semantic" toggle
    `.btn.sm` (`.on` when active), slot select `#slotSel`, entity select
    `#entSel` (options from the response's `slots` / `entities`, keep the
    current choice), role select `#roleSel` with "About the user" (`user`,
    default), "Assistant replies" (`assistant`), "All" (empty). A total line
    `#total` ("123 facts", "12 matches").
  - Table `.mt-table` columns: Text, Slot (`.tag.sch`), Entities (`.tag.ent`
    chips), Date, and a Score column only in semantic mode. Left-align all
    cells in this table (override the demo's right alignment).
  - Row actions: **Edit**: the text cell becomes a `<textarea>`; Enter (without
    Shift) saves via PATCH, Esc cancels; show "Saving…" while the request runs
    (it makes an LLM call, 1-2 s); on error show `detail` and keep the editor.
    **Delete**: inline "Delete? Yes / No" in the row; Yes → DELETE → reload
    page. Only one row in edit or confirm state at a time.
  - Pagination: Prev, "Page x of y", Next; 50 per page.
  - Empty states: "No memories in this brain yet. Add some on the left." / "No
    matches."
  - Profile tab: `GET /api/mem/profile?space=`; group by `slot` (heading per
    slot); each item: `text` (title attribute = `raw`), a `.tag` with `cluster`,
    then its notes: `.tag.emo` emotion chip + quote, and `cause` beneath in
    `--txt-3` when present. Show all notes. Empty: "No profile yet. It grows from
    conversations."
- **Danger zone** at the bottom of the right column: "Clear brain" and "Delete
  brain" `.btn.sm` with a rose outline. Each opens an inline confirm row: "Type
  the brain's name to confirm" input + a confirm button enabled only when the
  input equals the name exactly. Clear → POST `/api/mem/clear?space=` `{confirm}`;
  Delete → `fetch('/api/spaces/'+name, {method:'DELETE', headers:{'Content-Type':'application/json'}, body:JSON.stringify({confirm})})`.
  Show 409 `detail` inline (live session, running load, live brain). After
  delete, reload the picker and select the active brain.
- **Errors:** every fetch goes through one helper that throws `detail` (or the
  status text) so each caller shows a message; nothing fails silently. A
  `#toast` for successes ("Saved", "Deleted", "Brain cleared").
- **Accessibility:** real `<button>`s, labels on every input (visually hidden
  where the design has none), `aria-live="polite"` on the jobs list and the
  total line, visible focus rings (the demo's `:focus-visible` rule).
- **Escaping:** build rows with `textContent` / DOM APIs, never `innerHTML`
  with memory text.

**supermem.html:** inside `<div class="export">` (~line 531), before `#btnNew`,
add `<a class="btn sm" href="/memories" data-i18n="memoriesLink">Memories →</a>`;
add `memoriesLink:'Memories →',` to the English dict next to `download:` (~692);
add CSS `a.btn{text-decoration:none}`. Nothing else in that file changes.

**README.md:** after the "Interactive Demo" section's log paragraph, a short
"Managing memories" subsection: open `http://localhost:8787/memories`; what it
does (browse/search/edit/delete facts, profile, add text or `.txt/.md/.json/.pdf/.docx`,
transcripts `Name: text` or JSON turns, owner speaker, clear/delete brain); the
limits (10 MB, 500 chunks, ~5-20 s per chunk); the warning that the server has
no auth and these routes delete data, so keep it on a trusted network.

**Verification:** open the file directly in a browser is not enough (it needs
the API); Task 9 drives it against the server.

---

## Task 9 (Integration): suite, server, smoke, browser

1. Whole suite: `uv run --no-sync python -m unittest discover -s tests -p 'test_*.py'` → all pass
   (72 old + new).
2. Restart the server: stop the one on :8787
   (`lsof -ti tcp:8787 -sTCP:LISTEN | xargs kill`), then relaunch in the
   background exactly as before (OPENAI_API_KEY read from `../superkik/.env`,
   stripped of trailing comments; `uv run python web/run.py`; log to a file).
   Wait for "Uvicorn running". Confirm warmup shows no `skipped`.
3. Smoke with `curl` against a **new test brain** (never modify `demo`):
   create `qa-brain` via POST `/api/spaces`; `GET /memories` 200;
   `GET /api/mem?space=qa-brain` empty; `GET /api/mem?space=QA-BRAIN` → 404;
   ingest pasted prose (2 paragraphs) and a transcript; poll to done; facts
   listed with slots; semantic search returns rows with scores; PATCH one fact,
   confirm new text; DELETE one fact, confirm gone and
   `SELECT COUNT(*) FROM memories WHERE id=?` in the brain's sqlite is 0;
   profile route 200; clear `qa-brain` → empty; delete `qa-brain` → gone from
   `/api/spaces` and its directory removed; delete of `demo` → 409.
   Upload a generated PDF and DOCX (use the test helpers) to a second test brain.
4. Browser: drive `/memories` with the gstack browse tool (`$B`, headless):
   screenshot at 1280 and 375 widths, console errors empty; add pasted text,
   watch the job row progress, facts appear; edit and delete a row; switch to
   Profile; create and delete a brain from the page. Read every screenshot.
5. Leave the server running on the new code. Remove the test brains.
