"""Load memories from pasted text or a file: extract, detect, split, ingest in the background.

The pure functions turn bytes into Chunks (one ``mem.ingest`` call each); ``Jobs`` runs chunk
lists on one daemon thread, one job at a time. Each chunk costs several LLM calls (5-20 s),
which is why ingest never runs inside an HTTP request.
"""
from __future__ import annotations

import io
import json
import queue
import re
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024
MAX_CHUNKS = 500          # ~5-20 s each: 500 is already over an hour of LLM calls
CHUNK_CHARS = 800
MAX_TEXT = MAX_CHUNKS * CHUNK_CHARS      # more text than this cannot fit in MAX_CHUNKS chunks
MAX_DOCX_XML = 20 * 1024 * 1024          # word/document.xml, uncompressed (zip-bomb guard)
MAX_DOCX_UNZIPPED = 200 * 1024 * 1024
TEXT_EXTS = {"", ".txt", ".md", ".markdown", ".json"}
ASSISTANT_NAMES = {"assistant", "agent", "ai", "bot"}
SKIP_ROLES = {"system", "developer", "tool", "function"}   # instructions and tool output, not people

# An optional chat-export timestamp ("[10:31]", "10/03/2024, 10:31 -") before "Name: text".
_TURN_RE = re.compile(r"^(?:\[?\d[\d/.,: ]*(?:[AaPp][Mm])?\]?\s*-?\s*)?([^:\n]{1,40}):\s+(\S.*)$")
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
        # A few hundred KB can inflate to GBs of XML: check sizes before python-docx parses it.
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            sizes = {i.filename: i.file_size for i in z.infolist()}
    except zipfile.BadZipFile as e:
        raise ValueError(f"Could not read the DOCX: {e}") from None
    if (sizes.get("word/document.xml", 0) > MAX_DOCX_XML
            or sum(sizes.values()) > MAX_DOCX_UNZIPPED):
        raise ValueError("The DOCX is too large once unpacked. Split it and add it in parts.")
    try:
        doc = docx.Document(io.BytesIO(data))
    except Exception as e:
        raise ValueError(f"Could not read the DOCX: {e}") from None
    # p.style rescans styles.xml on every call; map style id -> name once.
    names = {s.style_id: s.name or "" for s in doc.styles}
    out, size = [], 0
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        style = names.get(p._p.style, "")
        if style == "Title" or style.startswith("Heading"):
            last = style.split()[-1]
            t = "#" * min(int(last) if last.isdigit() else 1, 6) + " " + t
        out.append(t)
        size += len(t) + 2
        if size > MAX_TEXT:
            raise ValueError(f"The DOCX has more text than {MAX_CHUNKS} chunks. "
                             "Split it and add it in parts.")
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
    if len(set(speakers)) > max(2, len(speakers) // 2):
        return "prose"
    if set(speakers) & ASSISTANT_NAMES:
        return "transcript"
    # Without an assistant, voices must go back and forth (A, B, A): "Allergy: x / Blood type: y",
    # email headers and "Action: ... / Decision: ..." notes are prose.
    # ponytail: a naive turn-change count; the operator can still pick Transcript explicitly.
    changes = sum(a != b for a, b in zip(speakers, speakers[1:]))
    return "transcript" if changes >= 2 else "prose"


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

    def add(lines: list[str]) -> None:
        nonlocal buf
        for piece in _pieces("\n".join(lines)):
            if buf and len(buf) + 2 + len(piece) <= CHUNK_CHARS:
                buf += "\n\n" + piece
            else:
                flush()
                buf = piece

    for block in re.split(r"\n\s*\n", text):
        lines: list[str] = []
        for ln in (ln.strip() for ln in block.splitlines()):
            m = _HEADING_RE.match(ln)
            if m:                            # a heading anywhere in a block starts a new section
                add(lines)
                lines = []
                flush()                      # never let a chunk span two sections
                heading = m.group(1)
            elif ln:
                lines.append(ln)
        add(lines)
    flush()
    return chunks


def split_transcript(text: str, owner: str = "") -> list[Chunk]:
    """One chunk per turn (same-speaker turns merged). The owner's turns are speaker "user";
    an assistant turn becomes agent_reply on the turn before it."""
    turns = _json_turns(text)
    if turns is None:
        turns = _line_turns(text)
    turns = [(w, s, a) for w, s, a in turns if s]
    people = list(dict.fromkeys(w for w, _, a in turns if not a and w))
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
        speaker = "user" if not who or who.lower() == owner_key else who   # "": preamble
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
    except (ValueError, RecursionError):     # deep nesting overflows the parser
        return None
    if not isinstance(data, list) or not data:
        return None
    turns = []
    for item in data:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role") or "").strip().lower()
        if role in SKIP_ROLES:
            continue
        who = str(item.get("speaker") or item.get("name") or item.get("role") or "").strip()
        said = item.get("text", item.get("content"))
        if isinstance(said, list):           # OpenAI multi-part content: keep the text parts
            said = " ".join(p["text"] for p in said
                            if isinstance(p, dict) and isinstance(p.get("text"), str))
        if said is None:                     # e.g. an assistant turn that only calls a tool
            said = ""
        if not who or not isinstance(said, str):
            return None
        turns.append((who, said.strip(), role == "assistant" or who.lower() in ASSISTANT_NAMES))
    return turns


def _line_turns(text: str) -> list[tuple[str, str, bool]]:
    """Turns from 'Name: text' lines; a line without a name continues the previous turn.
    Lines before the first turn (a title, some context) become a turn with speaker ""."""
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
        else:
            turns.append(["", line, False])
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
        """Queue ``chunks`` for ``mem`` and return the job id; the worker starts on first use."""
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
        """A copy of the job record, or None if unknown."""
        with self._lock:
            rec = self._jobs.get(job_id)
            return _copy(rec) if rec else None

    def list(self, space: str) -> list[dict]:
        """Copies of this space's job records, newest first."""
        with self._lock:
            # Insertion order is submit order; created_at can tie within one clock tick.
            return [_copy(r) for r in reversed(self._jobs.values()) if r["space"] == space]

    def busy(self, space: str) -> bool:
        """True while any job for ``space`` is queued or running."""
        with self._lock:
            return any(r["space"] == space and r["state"] in ("queued", "running")
                       for r in self._jobs.values())

    def cancel(self, job_id: str) -> bool:
        """Stop a job: a queued one is cancelled at once (so busy() frees its brain), a running
        one before its next chunk. False if unknown."""
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return False
            if rec["state"] in ("queued", "running"):
                self._cancelled.add(job_id)      # _run still skips a queued one when it is dequeued
            if rec["state"] == "queued":
                rec.update(state="cancelled", finished_at=time.time())
            return True

    def _loop(self) -> None:
        while True:
            job_id, mem, chunks, observed_at, on_done = self._q.get()
            try:
                self._run(job_id, mem, chunks, observed_at, on_done)
            except Exception as e:          # never let one job kill the worker
                self._note(job_id, -1, "", f"{type(e).__name__}: {e}")
                self._update(job_id, state="failed", finished_at=time.time())
                with self._lock:
                    self._cancelled.discard(job_id)

    def _run(self, job_id, mem, chunks, observed_at, on_done) -> None:
        with self._lock:                    # atomic with cancel(): queued -> running or skipped
            if job_id in self._cancelled:   # cancel() already marked it cancelled
                self._cancelled.discard(job_id)
                return
            self._jobs[job_id].update(state="running", started_at=time.time())
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
            rec.update(state=state, finished_at=time.time())
            self._cancelled.discard(job_id)  # a cancel that came after the last chunk

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
        done = [r for r in self._jobs.values() if r["state"] not in ("queued", "running")]
        for r in done[:max(0, len(done) - KEEP_FINISHED)]:   # dict order is submit order
            del self._jobs[r["id"]]


def _copy(rec: dict) -> dict:
    return dict(rec, errors=[dict(e) for e in rec["errors"]])
