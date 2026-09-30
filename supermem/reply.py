"""Reply layer: the step after the core hands over a ``Turn``. Two paths, one interface.

``supermem/stream.py`` is the input side (audio → memory); this is the output side (memory → reply). Two paths:

    # Path A: use the built-in one (OpenAI-compatible API, streaming)
    vm = SuperMem.from_config({"reply": {"provider": "openai",
                                         "config": {"model": "gpt-4o-mini"}}})

    # Path B: use your own model/function
    vm = SuperMem(reply=my_fn)

Both paths get exactly the same call interface::

    answer = await vm.reply(turn)                      # collect everything, return the full string
    async for delta in vm.reply_stream(turn):  ...     # streaming, token by token

``my_fn`` can take any of the forms below; ``normalize()`` unifies them into an async generator::

    def       my_fn(text, memory_context) -> str          # sync: moved to a thread automatically, does not block the event loop
    async def my_fn(text, memory_context) -> str          # coroutine
    async def my_fn(text, memory_context): yield delta    # async generator (streaming)

**TTS is not here.** The reply layer only produces text; for audio use ``supermem/tts.py``:
``speak_stream(vm.reply_stream(turn))`` synthesises while generating, see examples/03_simple_agent_with_supermem_memory.py.
"""
from __future__ import annotations

import asyncio
import inspect
import os
from typing import AsyncIterator, Callable
from supermem.llm_config import resolve_api_key, resolve_base_url, resolve_model

# memory_context is only "what we remember about the user"; it carries no persona/style requirements, so the
# built-in provider appends it after this line rather than using it as the whole system prompt.
DEFAULT_SYSTEM = "You are a voice assistant. Answer briefly and naturally."


def compose_system(memory_context: str, system: str | None = None) -> str:
    """Persona + memory → system prompt. Either side may be empty."""
    parts = [system or DEFAULT_SYSTEM]
    if memory_context:
        parts.append(memory_context)
    return "\n\n".join(parts)


def openai_reply(model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, system: str | None = None) -> Callable:
    """Built-in reply provider: OpenAI-compatible API, streaming output. Returns an async generator function.

    The model uses the ``reply`` role: ``model`` argument → ``SUPERMEM_REPLY_MODEL`` → falls back to ``chat``.
    The reply is the path the user hears directly, so it has its own role that can be configured separately
    from the model that organises memory in the background; if unset it follows chat, so you never get the
    half-applied "set a model but replies still use the default".
    ``import supermem`` does not require a key because of this (the client is built on first call).
    """
    client = None

    async def fn(text: str, memory_context: str = "") -> AsyncIterator[str]:
        nonlocal client
        if client is None:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                api_key=resolve_api_key(api_key),
                base_url=resolve_base_url(base_url),
            )
        fn.last_usage = None
        stream = await client.chat.completions.create(
            model=resolve_model(model, "reply"),
            stream=True,
            stream_options={"include_usage": True},
            messages=[{"role": "system", "content": compose_system(memory_context, system)},
                      {"role": "user", "content": text}],
        )
        async for chunk in stream:
            if chunk.usage:
                fn.last_usage = usage_dict(chunk.usage)
            if not chunk.choices:        # the usage chunk carries no choices
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # The engine's own accounting, same shape the cartridge provider leaves, so the
    # compare panel can show prompt / cached tokens for plain arms too.
    fn.last_usage = None
    return fn


def usage_dict(usage) -> dict:
    """Prompt and cached tokens from an OpenAI ``usage`` object. ``cached_tokens``
    stays None when the engine did not report it (vLLM needs
    ``--enable-prompt-tokens-details``): unknown, not zero."""
    details = getattr(usage, "prompt_tokens_details", None)
    return {"prompt_tokens": usage.prompt_tokens,
            "cached_tokens": getattr(details, "cached_tokens", None)}


def normalize(fn: Callable) -> Callable:
    """Normalise a reply function of any shape into a single form: an async generator function.

    Sync functions go through ``asyncio.to_thread``: reply generation takes seconds, and running it directly
    on the event loop would stall the microphone-reading path. If the return value is itself an async iterable
    (e.g. a lambda wrapping someone else's generator), it is still expanded as a stream.
    """
    if inspect.isasyncgenfunction(fn):
        return fn

    if inspect.iscoroutinefunction(fn):
        async def gen(text: str, memory_context: str = "") -> AsyncIterator[str]:
            out = await fn(text, memory_context)
            if hasattr(out, "__aiter__"):
                async for delta in out:
                    yield delta
            else:
                yield out
        return gen

    async def gen(text: str, memory_context: str = "") -> AsyncIterator[str]:
        out = await asyncio.to_thread(fn, text, memory_context)
        if hasattr(out, "__aiter__"):
            async for delta in out:
                yield delta
        else:
            yield out
    return gen


async def capture(deltas: AsyncIterator[str], on_done: Callable[[str], None]) -> AsyncIterator[str]:
    """Pass each delta through unchanged, and hand the full sentence to ``on_done`` when finished.

    The agent's half of the conversation should also go into memory, without making the caller write an extra
    line and without waiting to collect everything before yielding. If interrupted, ``finally`` hands over the
    part already yielded: we remember exactly as much as the user heard.
    """
    parts: list[str] = []
    try:
        async for delta in deltas:
            parts.append(delta)
            yield delta
    finally:
        on_done("".join(parts))


def unpack(turn_or_text, memory_context: str = "") -> tuple[str, str]:
    """Convenience for ``vm.reply(turn)``: unpack a Turn / StreamState directly into (text, memory_context)."""
    text = getattr(turn_or_text, "text", None)
    if text is not None and hasattr(turn_or_text, "memory_context"):
        return text, (memory_context or turn_or_text.memory_context)
    return turn_or_text, memory_context
