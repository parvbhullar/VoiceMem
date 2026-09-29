"""Context Runtime: which cartridges a turn gets, in what order, and when to
pre-fill them.

Prompt layout (identical for every arm that carries memory, so arms differ
only in how the engine treats the KV, never in what the model reads)::

    system : persona + epoch
             [org cartridge]      shared by every caller of the tenant
             [caller cartridge]   stable for this caller
             [account cartridge]  caller x org state
    ... conversation history ...
    user   : this turn's memory hits (volatile) + the utterance

Everything up to the end of the system message is a stable prefix: exact
prefix caching (vLLM APC, LMCache, Dynamo KVBM) reuses it from the second
turn on, and ``prefetch_messages`` lets it be computed before the caller
speaks at all.
"""
from __future__ import annotations

from typing import AsyncIterator, Callable

from supermem.cartridge.contract import Cartridge, ordered

DEFAULT_PERSONA = (
    "You are the voice assistant on a phone line. Reply in one or two short spoken "
    "sentences, in the caller's language. When the context contains the answer, use "
    "the exact names, numbers, IDs, dates and times as written there. If the context "
    "does not contain it, say you will check; never invent details."
)

# LMCache CacheBlend splits the prompt on this string and caches each piece on
# its own, so a cartridge can be reused at any position (non-prefix reuse).
# Must match the engine's blend_special_str.
BLEND_SEPARATOR = " # # "

MODES = ("nomem", "cartridge", "blend")


class ContextRuntime:
    def __init__(self, org: Cartridge | None = None, persona: str = DEFAULT_PERSONA,
                 epoch: str = "") -> None:
        self.org = org
        self.persona = persona
        # A run-scoped line at the very top of the prompt. Changing it starts every
        # cache cold (vLLM and LMCache both hash from the first token), so each
        # benchmark arm starts from the same state without restarting the engine.
        self.epoch = epoch
        self._callers: dict[str, list[Cartridge]] = {}

    def register(self, caller_id: str, cartridges: list[Cartridge]) -> None:
        extra = [c for c in cartridges if c.kind != "org"]
        self._callers[caller_id] = ordered(([self.org] if self.org else []) + extra)

    def cartridges(self, caller_id: str) -> list[Cartridge]:
        if caller_id not in self._callers:
            raise KeyError(f"no cartridges registered for caller {caller_id!r}")
        return self._callers[caller_id]

    def callers(self) -> list[str]:
        return list(self._callers)

    def context_tokens(self, caller_id: str) -> int:
        return sum(c.tokens for c in self.cartridges(caller_id))

    def _system(self, caller_id: str, mode: str) -> str:
        head = f"[session {self.epoch}]\n{self.persona}" if self.epoch else self.persona
        if mode == "nomem":
            return head
        blocks = [c.render() for c in self.cartridges(caller_id)]
        joiner = BLEND_SEPARATOR if mode == "blend" else "\n\n"
        return head + "\n\n" + joiner.join(blocks)

    def messages(self, caller_id: str, utterance: str, *, mode: str = "cartridge",
                 history: list[dict] | None = None, turn_memory: str = "") -> list[dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        user = utterance
        if turn_memory and mode != "nomem":
            user = f"(Relevant memory for this turn)\n{turn_memory}\n\n(Caller says)\n{utterance}"
        return ([{"role": "system", "content": self._system(caller_id, mode)}]
                + list(history or [])
                + [{"role": "user", "content": user}])

    def prefetch_messages(self, caller_id: str, mode: str = "cartridge") -> list[dict]:
        """A request whose prompt shares the whole cartridge prefix with the
        caller's real turns. Send it with max_tokens=1 on ring / first ASR partial."""
        return [{"role": "system", "content": self._system(caller_id, mode)},
                {"role": "user", "content": "."}]


def cartridge_reply(engine, runtime: ContextRuntime, caller_id: str,
                    history: list[dict] | None = None) -> Callable:
    """A SuperMem reply provider (same shape as ``supermem.reply.openai_reply``):
    ``fn(text, memory_context)`` streams text deltas. SuperMem's per-turn
    recall becomes the volatile ``turn_memory``; the caller's long-lived memory
    rides in the cartridges."""
    hist = history if history is not None else []

    async def fn(text: str, memory_context: str = "") -> AsyncIterator[str]:
        msgs = runtime.messages(caller_id, text, history=hist, turn_memory=memory_context)
        reply = ""
        async for kind, val in engine.stream(msgs):
            if kind == "delta":
                reply += val
                yield val
        hist.extend([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])

    return fn
