"""/memories operator page: browse, edit and delete facts, read the profile, load text or files,
clear and delete brains.

Every route names its brain with ``space``; nothing here switches the voice demo's active brain.
run.py passes every dependency in as a callback, so this module imports only fastapi, the
stdlib and ingest's size limit, and is tested on a bare FastAPI() with fakes
(tests/test_memories_api.py).
"""
from __future__ import annotations

import asyncio
import re
from datetime import date
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ingest import MAX_BYTES

HERE = Path(__file__).resolve().parent
PAGE_SIZE, MAX_PAGE_SIZE = 50, 200
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NOCACHE = {"Cache-Control": "no-store"}


class FactEdit(BaseModel):
    text: str


class Confirm(BaseModel):
    confirm: str = ""


def _status(e: Exception) -> HTTPException | None:
    """The HTTP error for an exception a callback raises on purpose; None means a real bug (500)."""
    if isinstance(e, FileNotFoundError):
        return HTTPException(404, "No brain with that name.")
    if isinstance(e, (PermissionError, FileExistsError, RuntimeError)):
        return HTTPException(409, str(e))
    if isinstance(e, ValueError):
        return HTTPException(400, str(e))
    return None


def _call(fn: Callable, *args: Any) -> Any:
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


def _cross_origin(req: Request) -> bool:
    """A browser request from another site. curl and same-origin fetches pass."""
    origin = req.headers.get("origin")
    return bool(origin) and urlsplit(origin).netloc != req.headers.get("host")


async def _read_capped(req: Request) -> bytes:
    """The request body, refused with 413 once it passes MAX_BYTES (before it is all in RAM)."""
    too_big = HTTPException(413, f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    length = req.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_BYTES:
        raise too_big
    buf = bytearray()
    async for part in req.stream():
        buf += part
        if len(buf) > MAX_BYTES:
            raise too_big
    return bytes(buf)


def register_routes(app, *, resolve, delete_space, clear_space, facts, profile, semantic,
                    update_fact, delete_fact, jobs, plan, on_change=lambda space: None) -> None:
    """Attach the /memories page and its /api/mem, /api/ingest and DELETE /api/spaces routes.

    Also refuses cross-site writes app-wide: /api/ingest takes a raw body, so a page on
    another site could POST text into a brain with a no-preflight "simple" request.
    """

    @app.middleware("http")
    async def refuse_cross_origin(req: Request, call_next):
        # ponytail: Origin vs Host only; add real auth before exposing the demo beyond a LAN.
        if req.method not in ("GET", "HEAD", "OPTIONS") and _cross_origin(req):
            return JSONResponse({"detail": "Cross-origin request refused."}, status_code=403)
        return await call_next(req)

    def mem_of(space: str) -> Any:
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
            rows = sorted((dict(r, score=round(score[r["id"]], 3))
                           for r in rows if r["id"] in order),
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
        if not _call(update_fact, mem_of(space), mid, text):
            raise HTTPException(404, "No such memory.")
        on_change(space)
        return {"id": mid, "text": text}

    @app.delete("/api/mem/{mid}")
    def api_delete(mid: str, space: str) -> dict:
        if not _call(delete_fact, mem_of(space), mid):
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
        # Opening a brain (seconds) and reading a PDF are blocking: keep them off the event loop.
        mem = await asyncio.to_thread(mem_of, space)      # 404 before reading any upload
        data = await _read_capped(req)
        try:
            kind, chunks = await asyncio.to_thread(plan, filename, data, fmt, owner)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        # plan() can take seconds; a clear or delete in that window replaced the brain. A job on
        # the old instance would write into a wiped (or deleted) directory.
        if await asyncio.to_thread(mem_of, space) is not mem:
            raise HTTPException(409, "The brain was cleared while the file was read. Add it again.")
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
