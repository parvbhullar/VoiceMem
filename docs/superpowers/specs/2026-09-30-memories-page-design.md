# Memories page — browse, edit, and load memories per brain

Date: 2026-09-30
Status: approved, ready for implementation plan

## Problem

The web demo writes memories only as a side effect of talking to it, and shows
them only as dots on the brain map. An operator cannot read what a brain holds,
correct a wrong fact, remove one, or load a brain from existing material. The
only view of the store is the Download button, which saves a capped JSON
snapshot.

## Goal

A second page, `/memories`, where an operator manages any brain on the server:

- pick a brain, create one, clear one, delete one;
- browse and search its facts, edit a fact's text, delete a fact;
- read its profile (right brain);
- load memories from pasted text or a file (`.txt`, `.md`, `.pdf`, `.docx`, or a
  chat transcript as text or JSON), with progress.

A brain is a Memory Space: one person, one directory under
`supermem_memoryspace/<name>/`. The page works on the brain the operator picks,
independent of the brain the voice demo is talking to.

Non-goals for the first cut:

- Several people inside one brain. The core stores one user per space.
- Editing or deleting profile entries. See constraint 4.
- OCR, audio files, re-upload dedupe, cancelling a running job.
- Auth. See "Safety".
- Changes to `supermem.html` beyond one header link.

## Constraints discovered in the codebase

1. **The active space is a server-wide global.** `use_space()` in `web/run.py`
   rebinds the module global `vm`. If the page switched spaces that way, it
   would switch the live voice session too. The page must pass the space
   explicitly on every request and resolve it without calling `use_space()`.

2. **`get_space(name)` creates a space that does not exist.** A typo in a
   request would silently make an empty brain. Routes must check the name
   against `list_spaces()` first and return 404.

3. **`memory_snapshot()`, `fact_index()` and `right_brain_tree()` read the
   global `vm`,** and `memory_snapshot()` caps its output for the brain map (48
   rows, a per-slot cap, assistant turns dropped). The page needs the same
   joins (entry → slot → entity names) uncapped and for a chosen space. Extract
   the join into a helper that takes the instance; `memory_snapshot()` keeps
   its caps on top of it. Give the other two an instance parameter that
   defaults to the global.

4. **The profile cannot be edited.** `right_brain_tree()` reads the traits
   store, which exposes `add`, `all`, `search` and `counts` only.
   `RightBrainStore.update_content()` writes the old `right_brain_memories`
   table, which is no longer the source of the tree. The Profile tab is
   read-only.

5. **Deleting a fact does not cascade.** `MemoryRepository.delete_memory()`
   removes the vector entry and the JSON mirror. The cognitive graph has
   `delete_user()` but no per-memory delete, so slot tags and entity links for
   the deleted id remain, and a profile trait that cites the fact as evidence
   keeps it. The plan must confirm retrieval tolerates an orphan id; if it does
   not, add a per-memory delete to the graph store.

6. **`update_memory(id, text)` re-embeds and syncs the graph.** Edit needs no
   new core code.

7. **Text-only ingest must say who is speaking.** `SuperMem.ingest()` defaults
   `speaker="user"` when there is no audio; any other label is treated as an
   unverified voice and produces facts like "Unidentified speaker is
   vegetarian". The live loop calls
   `vm.ingest(text, agent_reply=reply, ...)` (`web/run.py:1909`), and
   `Ingest()` also accepts `session_id` and `observed_at`.

8. **A document is many LLM calls.** Each chunk is one fact-extraction call of
   roughly 1–2 s. A 200-paragraph file cannot run inside one HTTP request.

9. **No multipart parser is installed.** FastAPI's `UploadFile` needs
   `python-multipart`. Uploads are sent as the raw request body instead, which
   needs nothing.

## Architecture

Two new modules and one new page. `run.py` gains helpers, not routes.

### `web/ingest.py` — pure functions plus the job runner

```python
@dataclass
class Chunk:
    text: str
    speaker: str = "user"
    agent_reply: str | None = None

def extract_text(filename: str, data: bytes) -> str
def detect_format(text: str) -> str            # "prose" | "transcript"
def split_prose(text: str) -> list[Chunk]
def split_transcript(text: str, owner: str = "") -> list[Chunk]
def plan_chunks(filename, data, fmt="auto", owner="") -> list[Chunk]
```

**Extract.** `.txt`, `.md`, `.json` and pasted text decode as UTF-8. `.pdf`
uses `pypdf`; `.docx` uses `python-docx`. A PDF that yields no text raises
`ValueError("No extractable text. This looks like a scanned PDF.")`.

**Detect.** Transcript when the text parses as a JSON list of objects with
`speaker|role` and `text|content`, or when at least 60% of non-empty lines
match `^[^:\n]{1,40}:\s+\S`. Otherwise prose. `fmt` overrides detection.

**Split prose.** Split on blank lines. Merge consecutive short paragraphs up to
800 characters. Cut a longer paragraph at sentence ends. Prefix each chunk with
the nearest preceding Markdown heading, so a sentence keeps its subject.

**Split transcript.** `owner` names the speaker whose brain this is; default is
the first speaker seen. The owner's turns get `speaker="user"`. A speaker named
`assistant`, `agent`, `ai` or `bot` (any case), or a JSON `role` of
`assistant`, is the assistant: its turn is not a chunk but becomes
`agent_reply` on the owner turn before it. Any other speaker's turn is a chunk
with that speaker's label.

**Job runner.** One queue, one daemon worker thread, so jobs run one at a time
and do not contend with the voice loop's writes.

```python
class Jobs:
    def submit(self, mem, space: str, chunks: list[Chunk],
               observed_at: str | None) -> str      # job_id
    def get(self, job_id: str) -> dict | None
    def busy(self, space: str) -> bool              # queued or running
```

The worker calls, per chunk,
`mem.ingest(c.text, speaker=c.speaker, agent_reply=c.agent_reply,
session_id=job_id, observed_at=observed_at)` synchronously and adds the
returned `facts_count`. `mem` is the instance resolved at submit, so a job
stays in the brain it was sent to. A chunk that raises is appended to
`errors` as `{index, text, error}` and skipped. If the first five chunks all
fail, the job stops with `state="failed"`: that is a configuration fault, not a
bad paragraph. State is an in-memory dict and is lost on restart.

Job record: `{id, space, state, done, total, facts, errors}` with `state` one
of `queued | running | done | failed`.

Limits, checked before a job is created: 10 MB body, 2,000 chunks.

### `web/memories_api.py` — routes

Registered from `build_app()` the way `compare.register_routes` is. `run.py`
passes callbacks: `resolve(space)` (existence-checked, returns the instance),
`list_spaces`, `active_space`, `create_space`, `delete_space`, `clear_space`,
`facts(mem)`, `profile(mem)`.

| Route | Behaviour |
|---|---|
| `GET /memories` | Serves `web/memories.html`. |
| `GET /api/mem?space=&q=&mode=&slot=&entity=&role=&page=&size=` | Facts. `mode=substring` (default) filters in Python; `mode=semantic` runs the instance's search and keeps the hit ids in rank order. Returns `{items, total, page, size, slots, entities}`. Each item: `{id, text, slot, entities, date, role}`. `size` defaults to 50, max 200. |
| `GET /api/mem/profile?space=` | The right-brain tree for that space. |
| `PATCH /api/mem/{id}?space=` | Body `{text}`. Calls `update_memory`. Empty text is 400. |
| `DELETE /api/mem/{id}?space=` | Calls `delete_memory`. |
| `POST /api/mem/clear?space=` | Body `{confirm}` must equal the space name. |
| `DELETE /api/spaces/{name}` | Body `{confirm}` must equal the name. Refused for the active space. |
| `POST /api/ingest?space=&filename=&format=&owner=&observed_at=` | Raw body: file bytes, or UTF-8 text when `filename` is empty. Returns `{job_id, total}`. |
| `GET /api/ingest/{job_id}` | The job record. |

Brain creation and listing reuse the existing `GET`/`POST /api/spaces`.

### `web/run.py` — helpers

- `left_entries(mem) -> list[dict]`: the uncapped join, extracted from
  `memory_snapshot()`, which then applies its caps to the result.
- `fact_index(uid, mem=None)` and `right_brain_tree(uid, facts, mem=None)`.
- `space_exists(name)`, used by `resolve`.
- `delete_space(name)`: drop the instance from `_SPACES`, remove the
  directory.
- `clear_space(name)`: `delete_space`, then `create_space` with the same name;
  if it was the active space, `use_space(name)` so the demo rebinds to the new
  instance.

### `web/memories.html` — the page

Plain HTML and inline JS, reusing the CSS variables of `supermem.html`.

- **Header:** brain picker (name and memory count), "New brain", a link back to
  the demo, and the active-brain marker.
- **Left column, "Add memories":** a textarea, a file drop zone, and three
  fields: format (auto / prose / transcript), owner speaker (transcripts only),
  date (default today). After submit, a progress row — "34 / 120 chunks · 51
  facts · 2 errors" — polls `/api/ingest/{job_id}` every second. Errors expand
  to the failing chunk.
- **Right column, "Memories":** tabs **Facts** and **Profile**. Facts is a
  table of 50 rows per page with search (substring, with a semantic toggle),
  and slot, entity and role filters. Each row has Edit (the text cell becomes
  an input; Enter saves, Esc cancels) and Delete (inline "Delete? Yes / No").
  Profile is the read-only tree: slot → judgement → evidence.
- **Bottom:** "Clear brain" and "Delete brain", each requiring the brain's
  name typed in full.

The page holds the selected brain, the filters and the page number, and
nothing else. Every edit, delete or finished job re-fetches the current page.
No optimistic updates.

`supermem.html` gains one header link: "Memories →".

## Data flow

**Load a document.** Browser sends the file bytes → the route checks the space
exists and the size limit → `plan_chunks()` extracts, detects and splits →
chunk limit checked → `Jobs.submit()` returns a `job_id` → the worker ingests
chunk by chunk → the page polls until `done` or `failed` → the page re-fetches
facts.

**Edit a fact.** `PATCH` → `update_memory()` re-embeds and syncs the graph →
the page re-fetches.

## Error handling

| Case | Response |
|---|---|
| Unknown space | 404. The picker reloads. |
| Unsupported extension, body over 10 MB, over 2,000 chunks, empty text | 400 with the limit or reason stated. No job is created. |
| Scanned PDF | 400, "No extractable text. This looks like a scanned PDF." |
| A chunk fails | Recorded in `errors`, skipped. |
| First five chunks fail | Job stops, `state="failed"`. |
| Edit or delete of a missing id | 404. |
| `confirm` does not match | 400. Nothing is touched. |
| Delete the active brain | 409, "Switch the demo to another brain first." |
| Clear or delete while a job for that brain is queued or running | 409. |

## Safety

Space names pass through the existing `space_dir()` sanitizer, so a name
cannot leave `supermem_memoryspace/`. Uploads are parsed in memory and never
written to disk.

The server binds `0.0.0.0` with no auth, and this page adds routes that
destroy data. Until auth exists, run it on a trusted network only.

Clearing the active brain while someone is mid-conversation swaps the store
under that session. The typed-name guard is the only protection; an operator
should clear between sessions.

## Testing

The repo's pattern: `unittest`, run directly, fakes injected, no models and no
network (see `tests/test_compare.py`).

`tests/test_ingest_split.py`
- prose: blank-line split, merge to 800 characters, sentence cut, heading
  prefix, a single paragraph, whitespace-only input;
- detection: `Name: text` lines, JSON transcript, prose that contains a few
  colons, `fmt` override;
- transcript: default owner, named owner, assistant turn attached as
  `agent_reply`, third speaker keeps its label, consecutive owner turns, an
  assistant turn with no owner turn before it;
- extraction: one PDF and one DOCX fixture, a PDF with no text, an unsupported
  extension;
- limits.

`tests/test_memories_api.py`, with a fake instance and `TestClient`
- list: pagination, substring search, slot, entity and role filters, empty
  store;
- unknown space is 404 and creates nothing;
- edit, edit with empty text, edit of a missing id; delete, delete of a
  missing id;
- a write to space B does not reach space A;
- clear and delete: wrong `confirm`, active space, busy space;
- jobs: progress counts, a failing chunk is skipped, five failures stop the
  job, a job stays in its space after the active space changes.

## Dependencies

`uv add pypdf python-docx`. Nothing else.

## File-by-file change list

| File | Change |
|---|---|
| `web/ingest.py` | New. Extraction, detection, splitting, job runner. |
| `web/memories_api.py` | New. Routes. |
| `web/memories.html` | New. The page. |
| `web/run.py` | `left_entries`, `space_exists`, `delete_space`, `clear_space`; instance parameter on `fact_index` and `right_brain_tree`; pass callbacks to `build_app`. |
| `web/utils.py` | `build_app` accepts and registers the memories routes. |
| `web/supermem.html` | One header link. |
| `pyproject.toml` | `pypdf`, `python-docx`. |
| `tests/test_ingest_split.py`, `tests/test_memories_api.py` | New. |
| `README.md` | A short "Managing memories" section under the demo. |

## Open items for the plan

1. Confirm retrieval tolerates an orphan memory id left in the cognitive graph
   after a delete (constraint 5).
2. Confirm the search result exposes memory ids for `mode=semantic`; the
   orchestrator reads `h.memory_id` from `result.hits`, which suggests it
   does. If not, ship substring search only.
3. Confirm removing a space directory is safe while its SQLite files were
   opened by a now-dropped instance.

The three open items were resolved by a code investigation before implementation.
The answers, and the design changes they forced, are the "Amendments" section of
`docs/superpowers/plans/2026-09-30-memories-page.md`.

## Round two: adversarial review

Six review lenses (core, lifecycle, safety, ingest/API, frontend, regressions)
reported 34 findings. Two independent skeptics per finding confirmed 33 of them,
mostly at medium or low severity. What changed as a result:

- **A cleared or deleted brain is retired, not just dropped.** `_wipe_space`
  takes the brain's write lock and calls `Orchestrator.retire()`. Any later
  write through the old instance fails with "This brain was cleared or
  deleted." That covers queued voice `async_facts` threads, a remember task
  that still holds the old instance, an ingest whose upload was planned before
  the wipe, and an edit queued behind the clear. Before this, those writes
  recreated the directory or leaked rows into the new, empty brain.
- **Clearing the live brain refuses new voice sessions while it runs** (the
  `_CLEARING` set; the websocket is closed with 1013).
- **Uploads:** the body is read with a running 10 MB cap (413) after the brain
  is resolved; DOCX zip entries are size-checked before parsing (zip-bomb
  guard); cross-origin writes get 403 from a small ASGI middleware (Origin vs
  Host, not auth).
- **Detection and splitting:** `Key: value` notes, email headers and action
  lists stay prose unless an assistant speaks or the voices alternate;
  chat-export timestamps are allowed before `Name:`; JSON system, developer,
  tool and function turns are skipped; a `Me:`/`User:` speaker is the default
  owner; text before the first turn is an owner turn.
- **Edit keeps the old entity links when the annotator LLM fails.**
- **The page:** no double submit; a finishing load no longer destroys an open
  edit; empty states and totals follow the filters; loading states after Clear
  and on brain switch; IME-safe Enter; focus kept after row actions; row
  actions reachable at 375 px; drops outside the zone ignored; the demo's
  "Memories →" link opens a new tab so it does not end the voice session.

Left as is: a compare-mode turn that arrives during a live-brain clear can
block the event loop briefly while it opens its brain, and opening a brain that
is not yet open waits behind a slow clear. Both need compare-mode changes
outside this work.
