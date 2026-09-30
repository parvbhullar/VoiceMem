"""Plumbing layer of the web demo (not the main flow) -- core dialogue logic lives in run.py, page rendering in index.html.

This holds: local E5 (memory embedding and slot classification share one model), audio resampling/VAD,
LLM/TTS/Realtime streams, and FastAPI + WebSocket wiring. run.py only assembles these into the
0-300ms speculative-prefetch dialogue flow.
"""
import asyncio
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel

# The local E5 embedder (memory embedding and slot classification share one model) moved into core, see
# supermem/leftbrain/local_e5_embedder.py; re-exported here so existing call sites of `utils.LocalE5Embedder`
# / `utils.shared_e5()` keep working (run.py uses it to inject SuperMem(embedding=...)).
from supermem.leftbrain.local_e5_embedder import LocalE5Embedder, shared_e5  # noqa: F401
from supermem.llm_config import resolve_model

HERE = Path(__file__).resolve().parent
#: The demo's reply model. Defaults one tier stronger than the background consolidation model -- the user hears this path directly.
#: SUPERMEM_REPLY_MODEL (or the legacy OPENAI_CHAT_MODEL) / models={"reply": ...} can override it.
CHAT_MODEL = resolve_model(role="reply", default="gpt-4o")
RT_MODEL = resolve_model(role="realtime")
#: Realtime voice. It was never set before, and the default (alloy) sounds the flattest.
#: Options on gpt-realtime: alloy / ash / ballad / coral / echo / sage / shimmer /
#: verse / marin / cedar -- marin and cedar are newer, with noticeably more intonation and breathiness.
RT_VOICE = os.environ.get("OPENAI_REALTIME_VOICE", "marin")
client = AsyncOpenAI()


# ── Audio helpers ─────────────────────────────────────────────────────────────
# resample uses the core copy (supermem/utils/audio/stream_io.py); do not duplicate it.
# There used to be a make_vad() here -- nobody called it after the demo switched to reusing vm.stream();
# VAD is now an injectable core capability (SuperMem(vad=...) / the config's vad section), so it was removed.
from supermem.utils.audio.stream_io import resample  # noqa: E402,F401

# TTS (online/offline backends + sentence splitting) moved into core, see supermem/tts.py; re-exported here
# so existing call sites of `utils.tts_stream(...)` keep working.
from supermem.tts import TTS_BACKEND, TTS_MODEL, cut_point, tts_stream  # noqa: E402,F401


# ── Reply models are configured in one place: the reply section of the unified config (run.py passes CONFIG["reply"]) ──
# If not passed, falls back to module-level env defaults (CHAT_MODEL / TTS_MODEL / TTS_BACKEND / RT_MODEL),
# so existing behaviour is unchanged. reply structure: {"llm": {"config": {"model": ...}},
# "tts": {"provider": "openai|local", "config": {"model": ...}},
# "realtime": {"config": {"model": ...}}}; every section is optional.
def _reply_seg(reply, name):
    seg = (reply or {}).get(name) or {}
    return seg.get("provider"), (seg.get("config") or {})


# ── Realtime stream ───────────────────────────────────────────────────────────
# There used to be an llm_stream() here -- it did the same thing as the core reply layer (openai_reply in
# supermem/reply.py): stream chat.completions with memory spliced into the system prompt. run.py now uses
# vm.reply_stream() directly, with the persona in CONFIG.reply.llm.config.system, so this copy was removed.


def realtime_connect(reply=None):
    """Option A: feed the full mic audio to it in parallel to get native speech out. Event names may shift
    slightly between SDK versions (compare openai_voice_demo/backend/providers/realtime.py)."""
    _, cfg = _reply_seg(reply, "realtime")
    return client.realtime.connect(model=resolve_model(cfg.get("model"), "realtime"))


# ── SearchResult -> memory_hits payload understood by the brain-map html ──────
def _emotion_of(rb_hits) -> str:
    """Emotion label carried by this turn's right-brain hits.

    The frontend used to regex a bracketed "x" out of content -- that is how the heartnote inner monologue
    is written, but the inner monologue is now gated (not generated every turn), so when nothing matched
    the label never showed. The emotion is already in metadata.emotion; hand it to the frontend directly
    instead of making it guess.
    """
    # **Only trust this turn's signal** (the affect_hint of current_signal).
    #
    # It used to fall back to "the emotion on retrieved old memories", which was a bug: the tag bar means
    # "how he feels now", but the fallback showed "how he felt about the recalled event back then" -- once
    # the store had many sad memories, every turn showed sad regardless of what the user said.
    # With no signal this turn, return empty and leave it to run.py's fill_tags (text keywords -> SenseVoice).
    for h in (rb_hits or []):
        if getattr(h, "source", "") != "current_signal":
            continue
        emo = ((getattr(h, "metadata", None) or {}).get("emotion") or "").strip()
        if emo:
            return emo
    return ""


#: Right-brain memories sent to the **model** get a date and slot prefix ("[2026-08-24] ⚠ avoid repeating: ...") --
#: the model needs to know when it was and which category it belongs to. But the page should not show that:
#: the left-brain column is one clean fact, while the right brain would carry a string of prefixes and look
#: like a different system. The category is already in the cluster field, so the body keeps only the body.
_RB_PREFIX = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?(?:[⚠✓✱*]\s*)?(?:[^\uff1a:\s]{2,8}[\uff1a:]\s*)?")


#: When rendered for the **model**, two parts are appended after the body: the response experience's
#: "(next time: ...)" is an actionable suggestion, and the heartnote's "(inner note: ...)" is extra interpretation.
#: The page only wants the body -- emotion is shown separately in brackets, the category is in the cluster field.
_RB_SUFFIX = re.compile(
    r"[\uff08(]\s*(?:next time|inner note)\s*[\uff1a:].*$", re.S | re.I)


def clean_rb(content: str) -> str:
    t = _RB_PREFIX.sub("", str(content or ""))
    t = _RB_SUFFIX.sub("", t)
    return t.strip()


def hits_payload(result, has_audio=None, cluster_of=None):
    """has_audio(memory_id) -> bool: whether this memory has archived original audio.
    The frontend uses it to decide whether to auto-play that original clip back this turn."""
    rb = getattr(result, "rb_hits", None) or []
    cls = getattr(result, "classification", None)
    return {
        # Slots and emotion are sent together with this turn's retrieval results. The frontend used to send a
        # separate /api/classify and wait for it -- a race: when memory_hits arrived first, curSlots was
        # still empty and the tag bar stayed blank. This reuses the classification Search already computed, zero extra cost.
        "slots": list(getattr(cls, "slots", []) or []),
        "entities": list(getattr(cls, "entities", []) or []),
        "emotion": _emotion_of(rb),
        "left_brain": [{"text": h.text, "score": h.score, "attributed_to": h.attributed_to,
                        "memory_id": h.memory_id,
                        "has_audio": bool(has_audio and has_audio(h.memory_id))}
                       for h in result.hits],
        # cluster is injected by run.py (same rules; the frontend no longer guesses from source)
        # content is for the page (prefix stripped), raw keeps the original -- the brain map needs it to match heartnotes
        # internal: response_experience is the assistant's note about **its own** behaviour ("didn't acknowledge the
        # emotion first this time, ask first next time"); useful to the model, but not a profile of the user -- shown in
        # the page's "Right brain / Profile" column it just confuses the user. The frontend skips it; brain-map matching still uses it.
        "right_brain_hits": [{"content": clean_rb(h.content), "raw": h.content,
                              "internal": h.source == "response_experience",
                              # Profile hits are **slot-level** portraits ("likes_dislikes: ..."),
                              # while brain-map nodes are **entity-level**, so matching by body never works --
                              # right-brain nodes never lit up and no rays linked left and right brain. Include the slot
                              # name so the frontend can place it under that slot's nodes.
                              "slot": ((getattr(h, "metadata", None) or {}).get("slot_name") or ""),
                              # The original claim. The page shows a first-person rewrite (run.py's
                              # rb_human); keep the original here for the rewrite and brain-map matching.
                              "claim": ((getattr(h, "metadata", None) or {}).get("claim") or ""),
                              "source": h.source, "priority": h.priority,
                              "cluster": cluster_of(h.content, h.source) if cluster_of else ""}
                             for h in (getattr(result, "rb_hits", None) or [])],
        "current_scene": getattr(result, "current_scene", None) or None,
        "related_summaries": getattr(result, "related_summaries", None) or {},
    }


# ── FastAPI + WS wiring (wiring only, all rendering is in index.html) ──────────
def build_app(mode, session, classify, snapshot=None, audio_of=None, spaces=None,
              set_lang=None, compare=None, memories=None):
    """session(sock): the session loop passed in by run.py (llm_tts / realtime). classify(query): used to grow the brain map.
    snapshot(): memories already in the store, so the frontend fills the brain map when the page opens.
    spaces=(list_fn, create_fn, use_fn, active_fn): create/list/switch Memory Spaces.
    set_lang(lang): syncs a UI language switch to the assistant (reply language + extraction language).
    compare=(get_state, set_state): the A/B comparison toggle and both arms' config. The routes live in
    web/compare.py -- it does not import this module (which pulls in torch/TTS), so it can be unit-tested.
    memories=dict of register_routes() callbacks: the /memories operator page (routes in web/memories_api.py)."""
    app = FastAPI()

    if compare:
        from compare import register_routes
        register_routes(app, *compare)

    if memories:                                     # the /memories operator page and its routes
        from memories_api import register_routes as register_memories
        register_memories(app, **memories)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        await sock.send_json({"type": "session_ready", "mode": mode})
        try:
            await session(sock)
        except WebSocketDisconnect:
            pass          # closing/refreshing the page is a normal end, don't spew a traceback

    class Q(BaseModel):
        query: str

    @app.post("/api/classify")                       # the brain-map html uses this to grow the left brain by slot
    def api_classify(body: Q) -> dict:
        c = classify(body.query)
        return {"slots": list(c.slots), "entities": list(c.entities)}

    class T(BaseModel):
        text: str

    @app.post("/api/title")                          # give the session a summarising name
    async def api_title(body: T) -> dict:
        """Summarise this conversation in one line to use as the sidebar title.

        Called only once, after the first turn, capped at 16 tokens -- a title must not
        slow the conversation down or cost noticeable money. On failure return an empty string; the frontend falls back to the user's first sentence.
        """
        try:
            r = await client.chat.completions.create(
                model=CHAT_MODEL, max_tokens=16, temperature=0,
                messages=[
                    {"role": "system", "content":
                     "Summarise what this conversation is about in at most 6 words, for use as a title. "
                     "Output only the title itself: no quotes, no punctuation, no openers like \"About\". "
                     # Openings are often "hello hello", "testing", "hi"; summarising literally gives "Voice test",
                     # while the rest of the conversation may be about something else entirely.
                     "Ignore opening small talk, mic checks and can-you-hear-me exchanges, "
                     "and capture what was actually discussed. Only when the whole thing is just greetings, call it \"Casual chat\"."},
                    {"role": "user", "content": body.text[:600]},
                ],
            )
            return {"title": (r.choices[0].message.content or "").strip()}
        except Exception as e:
            print(f"[web] Failed to generate title: {e}", flush=True)
            return {"title": ""}

    @app.get("/api/memories")                        # fill in existing memories when the page opens
    def api_memories() -> dict:
        return snapshot() if snapshot else {"left": [], "right": []}

    @app.post("/api/lang")                           # when the UI switches language, the assistant follows
    async def api_lang(req: Request) -> dict:
        lang = (await req.json()).get("lang", "en")
        set_lang(lang) if set_lang else None
        return {"lang": lang}

    if spaces:
        _list_spaces, _create_space, _use_space, _active_space = spaces

        @app.get("/api/spaces")                      # which spaces exist on disk
        def api_spaces() -> dict:
            return {"spaces": _list_spaces(), "active": _active_space()}

        @app.post("/api/spaces")                     # create an empty one
        async def api_space_new(req: Request) -> dict:
            body = await req.json()
            name, lang = body.get("name", ""), body.get("language", "")
            try:
                # Building + warming a brain takes seconds and may wait on _SPACES_LOCK behind
                # another brain's open or wipe: never on the event loop that carries voice.
                return await asyncio.to_thread(_create_space, name, lang)
            except FileExistsError as e:
                raise HTTPException(409, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))

        @app.post("/api/spaces/{name}/use")           # switch to it
        def api_space_use(name: str) -> dict:
            try:
                return {"active": _use_space(name)}
            except Exception as e:
                raise HTTPException(400, f"Could not switch: {e}")

    @app.get("/api/audio/{memory_id}")               # play back the original audio from that moment
    def api_audio(memory_id: str):
        path = audio_of(memory_id) if audio_of else None
        if not path or not Path(path).exists():
            raise HTTPException(404, "This memory has no archived audio")
        return FileResponse(path, media_type="audio/wav")

    (HERE / "images").mkdir(exist_ok=True)
    app.mount("/images", StaticFiles(directory=HERE / "images"), name="images")

    # no-store: demo_local also uses 8787, and same-origin caching would make the browser serve the previous demo's stale page
    _NOCACHE = {"Cache-Control": "no-store"}

    @app.get("/pcm-player-worklet.js")
    def pcm_player_worklet():
        return FileResponse(HERE / "pcm-player-worklet.js", headers=_NOCACHE,
                            media_type="application/javascript")

    @app.get("/")
    def index():
        return FileResponse(HERE / "supermem.html", headers=_NOCACHE)

    @app.get("/classic")                             # previous page version, kept for comparison
    def classic():
        return FileResponse(HERE / "index.html", headers=_NOCACHE)

    return app
