"""One-line memory for an EXISTING chat system.

The whole point of this module: someone who already has a working dialogue
loop should be able to add long-term memory by changing ONE line --

    from supermem import inject

    reply = llm.chat(inject(messages))       # <- the one line

``inject(messages)`` takes an OpenAI-style messages list (the de-facto
industry format -- OpenAI/Claude/Qwen/DeepSeek/ollama/vLLM/LangChain all
speak it), finds the latest user message, retrieves relevant left-brain
facts + right-brain persona from this user's memory, inserts them as a
system message right before that user message, schedules the new utterance
for background ingestion, and returns the augmented list. The caller's LLM
call is untouched -- this module never talks to the reply model.

For systems that don't use messages-format prompts, the same machinery is
exposed as two primitives:

    m = Memory(user_id="alice")
    context = m.recall("what did I say about my job?")   # -> str to splice
    m.remember("I got promoted today!")                   # store, background

Honest costs, not hidden: ``inject``/``recall`` run supermem's real
retrieval pipeline (Classify -> Search: one LLM call + one embedding call,
~1-2s wall time against OpenAI defaults), so the host conversation gains
that much latency per turn; ``remember`` is fire-and-forget in a daemon
thread and adds none. supermem's own internal calls need an
OpenAI-compatible endpoint (``OPENAI_API_KEY``, optionally
``OPENAI_BASE_URL`` for ollama/vLLM) -- the same requirement mem0-style
memory libraries have.
"""
from __future__ import annotations

import threading
from typing import Any


def build_memory_context(result: Any, max_rb: int = 5) -> str:
    """Render a SearchResult into the memory-context text block.

    Single source of truth for how retrieved memory becomes prompt text --
    the openai_voice_demo bridge delegates here too, so "what the demo does"
    and "what we tell integrators to do" can never drift apart. Returns ""
    when there is nothing worth injecting.

    ``max_rb``: right-brain hits are priority-sorted by Search(). This was 3 for
    a while -- letting all 5 through measurably delayed the reply model's first
    token in the demo. It is back to 5 because the prompt now says "top5 in right
    brain" and the header must not lie about what follows it. If first-token
    latency matters more than coverage, pass max_rb=3 and fix the header too.
    """
    lines: list[str] = []
    hits = getattr(result, "hits", None) or []
    if hits:
        lines.append("factual memory CONTEXT you know about the user (top5 in left brain):")
        for hit in hits:
            # Include the event date (same format as the right-brain heartnote [YYYY-MM-DD] prefix):
            # fact text is mostly relative ("last week", "over a month now") with no absolute date,
            # so "did this happen before that?" can't be answered otherwise.
            when = getattr(hit, "observed_at", "") or ""
            lines.append(f"- [{when}] {hit.text}" if when else f"- {hit.text}")
    rb_hits = getattr(result, "rb_hits", None) or []
    if rb_hits:
        # Right-brain content must be labelled separately from facts. It is profile / emotion
        # attribution / reply experience -- internal notes for the model ("avoid repeating: ...
        # (next time: ...)", "coping_style: ..."), not something to say to the user.
        # Unlabelled, it is just a few lines after the facts and the model reads them out, so the
        # reply turns into "your coping style is relieving stress at the gym" -- like reading a file.
        lines.append("")
        lines.append("user's emotion & characteristics (top5 in right brain):")
        lines.extend(f"- {h.content}" for h in rb_hits[:max_rb])
        lines.append("Let these shape your tone, what you bring up, and what you leave alone. "
                     "Never state them back to the user.")
    return "\n".join(lines)


class Memory:
    """Per-user memory handle wrapping the full SuperMem pipeline behind
    three calls: ``inject`` / ``recall`` / ``remember``.

    Audio-native perception (voiceprint/scene/emotion-from-audio) is OFF by
    default -- an existing text chat system shouldn't inherit those heavy
    optional dependencies. Pass ``audio_native=True`` (with the
    corresponding install extras) to enable them.
    """

    def __init__(self, user_id: str = "default", memory_root: str | None = None,
                 audio_native: bool = False, **supermem_kwargs: Any) -> None:
        from supermem.core import SuperMem
        self.user_id = user_id
        self._vm = SuperMem(
            memory_root=memory_root,
            user_id=user_id,
            enable_scene=audio_native,
            enable_music=audio_native,
            enable_abnormal_sound=audio_native,
            enable_voiceprint=audio_native,
            enable_emotion=audio_native,
            **supermem_kwargs,
        )

    # ── primitives ───────────────────────────────────────────────────────────

    def recall(self, text: str, top_k: int = 5) -> str:
        """Retrieve memory relevant to *text*, rendered as a prompt-ready
        block (may be ""). This is the officially-supported
        Classify() -> Search() pattern, packaged."""
        c = self._vm.Classify(text)
        result = self._vm.Search(text, slots=c.slots, entities=c.entities, top_k=top_k)
        return build_memory_context(result)

    def remember(self, text: str, speaker: str = "user", **ingest_kwargs: Any) -> None:
        """Store one user utterance. Runs in a daemon thread so the host
        conversation is never blocked; errors are printed, not raised
        (a memory-write failure must not break the host's turn)."""
        def _run() -> None:
            try:
                self._vm.Ingest(text, speaker=speaker, **ingest_kwargs)
            except Exception as e:  # noqa: BLE001 -- background, nothing to propagate to
                print(f"[supermem] background remember() failed: {e}", flush=True)
        threading.Thread(target=_run, daemon=True).start()

    def flush(self) -> None:
        """Optional: call when a conversation formally ends -- runs the
        session-boundary batch work (see SuperMem.Flush)."""
        self._vm.Flush()

    # ── the one line ─────────────────────────────────────────────────────────

    def inject(self, messages: list[dict], top_k: int = 5,
               remember: bool = True) -> list[dict]:
        """The one-line integration. Returns a NEW messages list (the
        caller's list is never mutated) with a system message of retrieved
        memory inserted immediately before the latest user message; that
        user message is also queued for background ingestion (disable with
        ``remember=False`` if the host stores turns itself elsewhere).

        No user message / no relevant memory -> the list comes back
        unchanged (aside from being a copy), so it is always safe to call
        unconditionally.
        """
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user" and isinstance(messages[i].get("content"), str):
                last_user_idx = i
                break
        if last_user_idx is None:
            return list(messages)

        user_text = messages[last_user_idx]["content"]
        context = self.recall(user_text, top_k=top_k)
        if remember:
            self.remember(user_text)
        if not context:
            return list(messages)

        out = list(messages)
        out.insert(last_user_idx, {"role": "system", "content": context})
        return out


# ── module-level lazy default, so the integration is literally one line ─────
# (`from supermem import inject` + `inject(messages)` -- no visible setup;
# the default Memory is built on first use, one per user_id.)

_DEFAULT_INSTANCES: dict[str, Memory] = {}
_DEFAULT_LOCK = threading.Lock()


def _default(user_id: str | None) -> Memory:
    import os
    uid = user_id or os.environ.get("SUPERMEM_USER_ID", "default")
    with _DEFAULT_LOCK:
        if uid not in _DEFAULT_INSTANCES:
            _DEFAULT_INSTANCES[uid] = Memory(user_id=uid)
        return _DEFAULT_INSTANCES[uid]


def inject(messages: list[dict], user_id: str | None = None, **kwargs: Any) -> list[dict]:
    return _default(user_id).inject(messages, **kwargs)


def recall(text: str, user_id: str | None = None, **kwargs: Any) -> str:
    return _default(user_id).recall(text, **kwargs)


def remember(text: str, user_id: str | None = None, **kwargs: Any) -> None:
    _default(user_id).remember(text, **kwargs)
