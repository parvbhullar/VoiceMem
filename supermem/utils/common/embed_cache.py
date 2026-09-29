"""In-process cache for embedding results.

One ingest was measured sending 15 embedding calls: the same fact 3 times (identical string),
the same 3 entities twice (once by the left brain, again by the orchestrator), and 7 slot
descriptions recomputed every turn (slot definitions are static, independent of input).
After dedup, about 4 remain.

Keyed by (model, text). A text's vector is deterministic, so caching changes no semantics.
"""
from __future__ import annotations

import threading

_MAX = 4096                      # enough for one session; when exceeded, clear it all -- no LRU bookkeeping
_lock = threading.Lock()
_cache: dict[tuple[str, str], list[float]] = {}


def get(model: str, text: str) -> list[float] | None:
    with _lock:
        return _cache.get((model, text))


def put(model: str, text: str, vec: list[float]) -> None:
    if not vec:
        return
    with _lock:
        if len(_cache) >= _MAX:
            _cache.clear()
        _cache[(model, text)] = vec


def resolve(model: str, texts: list[str], compute) -> list[list[float]]:
    """Hits are returned directly; misses are **merged into one batched call** to ``compute(list) -> list``.

    Batching matters too: OpenAI's embeddings API already accepts arrays, and 5 inputs cost about
    the same as 1, yet callers send one at a time -- of the 15 calls only the slot-description one was batched.
    """
    out: list[list[float] | None] = [get(model, t) for t in texts]
    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        fresh = compute([texts[i] for i in missing])
        for i, vec in zip(missing, fresh):
            out[i] = vec
            put(model, texts[i], vec)
    return [v or [] for v in out]
