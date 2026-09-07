"""A/B compare: one turn, two arms, memory injected vs not.

The demo's claim is "inject retrieved memory into an existing LLM and the
replies get better". With a single reply on screen that claim is unproven — the
viewer cannot see what the same model would have said without the memory block.
This module answers it by running two arms over the **same** transcript and the
**same** retrieval result, differing only in whether ``memory_context`` is
injected.

Two things here are deliberate and easy to get wrong:

* Arms call the plain reply provider (``voicemem.reply.openai_reply``), **not**
  ``vm.reply_stream()``. The latter wraps the provider in ``capture(...)`` and
  registers the reply into memory (``voicemem/core.py``), so one call per arm
  would store two agent replies for a single user turn and corrupt the space.
  The turn is ingested exactly once, by the caller's existing ``remember_turn``.
* Both arms get the same ``system`` persona. Different personas would make the
  comparison dishonest — the difference on screen has to be the memory.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from fastapi import HTTPException, Request


@dataclass
class Arm:
    """One compare panel.

    ``model`` / ``base_url`` / ``api_key`` empty means "fall through to the
    server's own configuration" (``.env`` key, ``llm_config`` role resolution),
    so the common case needs no per-panel setup.
    """

    label: str
    model: str = ""
    memory: bool = True
    base_url: str = ""
    api_key: str = field(default="", repr=False)

    def __repr__(self) -> str:
        # api_key is repr=False above, but spell the whole repr out anyway: a
        # traceback or a debug print of an Arm must never be able to leak a
        # user-supplied key, and `repr=False` is one refactor away from gone.
        return (f"Arm(label={self.label!r}, model={self.model!r}, "
                f"memory={self.memory!r}, base_url={self.base_url!r}, "
                f"api_key={'***' if self.api_key else ''!r})")


@dataclass
class CompareState:
    """Process-level compare configuration, set over ``POST /api/compare``.

    Defaults are the interesting comparison: panel A with memory, panel B
    without, both on the server's default model.
    """

    enabled: bool = False
    arms: tuple[Arm, Arm] = (Arm("a", memory=True), Arm("b", memory=False))


def sanitize(state: CompareState) -> dict:
    """State as it goes over the wire: key presence, never the key."""
    return {
        "enabled": state.enabled,
        "arms": [{"label": a.label, "model": a.model, "memory": a.memory,
                  "base_url": a.base_url, "has_key": bool(a.api_key)}
                 for a in state.arms],
    }


def parse_arms(payload: Any, current: tuple[Arm, Arm]) -> tuple[Arm, Arm]:
    """Merge an ``/api/compare`` body into the current arms.

    An absent field keeps its current value, so the frontend can post just the
    switch the user flipped. An explicit empty ``api_key`` clears the stored
    key — that is how the user goes back to the server's own key; omitting the
    field keeps what is there.
    """
    by_label = {a.label: a for a in current}
    for item in (payload or []):
        if not isinstance(item, dict):
            raise ValueError("each arm must be an object")
        label = item.get("label")
        if label not in by_label:
            raise ValueError(f"unknown panel {label!r}, expected 'a' or 'b'")
        arm = by_label[label]
        for name in ("model", "base_url", "api_key"):
            if name in item:
                value = item[name]
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string")
                arm = replace(arm, **{name: value.strip()})
        if "memory" in item:
            arm = replace(arm, memory=bool(item["memory"]))
        by_label[label] = arm
    return (by_label["a"], by_label["b"])


def register_routes(app, get_state: Callable[[], CompareState],
                    set_state: Callable[[CompareState], None]) -> None:
    """Mount ``GET/POST /api/compare`` on *app*.

    Lives here rather than in ``build_app`` for two reasons: the routes are the
    compare feature's own surface, and this way they can be unit-tested on a
    bare FastAPI app without importing ``web.utils`` (which pulls in torch and
    the TTS stack).

    ``POST`` accepts a per-panel ``api_key``. It is kept in this process only —
    never written to disk, never returned by ``GET`` (which reports
    ``has_key``). That is fine for a localhost demo and is the reason the demo
    must not be bound to a public interface.
    """
    @app.get("/api/compare")
    def api_compare() -> dict:
        return sanitize(get_state())

    @app.post("/api/compare")
    async def api_compare_set(req: Request) -> dict:
        body = await req.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        state = get_state()
        try:
            arms = parse_arms(body.get("arms"), state.arms)
        except ValueError as e:
            # Reject the whole payload: a half-applied compare config would
            # silently compare something other than what the page shows.
            raise HTTPException(400, str(e))
        enabled = state.enabled if "enabled" not in body else bool(body["enabled"])
        updated = CompareState(enabled=enabled, arms=arms)
        set_state(updated)
        return sanitize(updated)


def default_provider(system: str = "") -> Callable[[Arm], Callable]:
    """Build the real reply provider for an arm.

    Imported lazily: ``import web.compare`` must not drag in openai, and the
    unit tests inject fakes instead of ever reaching this.
    """
    def build(arm: Arm) -> Callable:
        from voicemem.reply import openai_reply
        return openai_reply(model=arm.model or None,
                            api_key=arm.api_key or None,
                            base_url=arm.base_url or None,
                            system=system or None)
    return build


def context_for(arm: Arm, memory_context: str) -> str:
    """What this arm's system prompt gets.

    A ``memory=False`` arm gets "" — plain LLM, no memory wiring. Note it does
    **not** get run.py's ``_NO_MEMORY_NOTE`` ("say you don't know, don't make
    things up"): that note is for a memory-enabled turn that retrieved nothing,
    a third condition. Telling the baseline arm it has no memories would be
    testing a differently-prompted model, not a model without memory.
    """
    return memory_context if arm.memory else ""


async def _run_arm(arm: Arm, text: str, memory_context: str, send, provider) -> dict:
    """Stream one arm. Never raises — a dead arm must not take down the turn."""
    await send({"type": "cmp_start", "panel": arm.label,
                "model": arm.model, "memory": arm.memory})
    ctx = context_for(arm, memory_context)
    started = time.monotonic()
    reply, error = "", ""
    try:
        async for delta in provider(arm)(text, ctx):
            reply += delta
            await send({"type": "cmp_delta", "panel": arm.label, "text": delta})
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 -- surfaced to the panel, not raised
        error = f"{type(e).__name__}: {e}"
        print(f"[compare] panel {arm.label} failed: {error}", flush=True)
    ms = int((time.monotonic() - started) * 1000)
    done = {"type": "cmp_done", "panel": arm.label, "text": reply, "ms": ms}
    if error:
        done["error"] = error
    await send(done)
    return {"label": arm.label, "text": reply, "ms": ms, "error": error}


async def fan_out(text: str, memory_context: str, arms: tuple[Arm, Arm], send,
                  system: str = "", provider=None) -> dict:
    """Run both arms concurrently, streaming ``cmp_*`` messages as tokens land.

    Returns ``{"a": reply, "b": reply, "latency_ms": {...}, "errors": {...}}``.
    ``errors`` holds only the panels that failed, so ``not result["errors"]``
    means a clean turn and ``len(...) == 2`` means nothing was generated.
    """
    provider = provider or default_provider(system)
    await send({"type": "cmp_ctx", "context": memory_context,
                "chars": len(memory_context)})
    outcomes = await asyncio.gather(
        *(_run_arm(arm, text, memory_context, send, provider) for arm in arms))

    result: dict = {"latency_ms": {}, "errors": {}}
    for o in outcomes:
        result[o["label"]] = o["text"]
        result["latency_ms"][o["label"]] = o["ms"]
        if o["error"]:
            result["errors"][o["label"]] = o["error"]
    if len(result["errors"]) == len(arms):
        # Both dead: the page's existing error toast should fire, and the
        # caller must skip ingestion — there is no reply to remember.
        await send({"type": "error", "message": "compare: both panels failed"})
    return result
