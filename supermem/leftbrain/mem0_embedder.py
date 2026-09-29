"""Adapt mem0's embedding providers into SuperMem embedders.

mem0 ships a dozen or so providers (ollama / huggingface / gemini / bedrock / azure /
vertexai / together / lmstudio / fastembed / langchain / openai), and mem0 is already a
SuperMem dependency -- no need to write each one again; wrapping the interface differences
makes them all usable.

The two interfaces differ in three places:
  · mem0 is ``embed(text, memory_action)``, one at a time; SuperMem wants ``embed_texts(list)``
    in batches (for cases like slot anchors, seven at once; sending one by one is seven round trips).
  · mem0 uses ``memory_action`` to distinguish ingest/retrieval; SuperMem uses two method names.
    Symmetric models (most) give the same vector either way; asymmetric ones (E5-style, needing
    query:/passage: prefixes) only line up via this parameter, so it must be passed through faithfully.
  · SuperMem must be able to query the dimension (after switching embedders it decides whether
    old vectors are invalid), which mem0 doesn't always provide; if unavailable, compute a probe.
"""
from __future__ import annotations

from typing import Any


class Mem0Embedder:
    """mem0 provider -> SuperMem embedder."""

    def __init__(self, provider: str, config: dict[str, Any] | None = None) -> None:
        from mem0.utils.factory import EmbedderFactory
        self._provider = provider
        self._inner = EmbedderFactory.create(provider, dict(config or {}), None)
        self._dims: int | None = None

    # -- SuperMem-side interface ------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Ingest side: a batch of texts -> a batch of vectors."""
        return [self._one(t, "add") for t in texts]

    def embed_query_text(self, text: str) -> list[float]:
        """Retrieval side. Asymmetric models take the query path via memory_action."""
        return self._one(text, "search")

    @property
    def dimensions(self) -> int:
        if self._dims is None:
            cfg = getattr(self._inner, "config", None)
            self._dims = int(getattr(cfg, "embedding_dims", 0) or 0) or len(
                self._one("dimension probe", "search"))
        return self._dims

    @property
    def model_name(self) -> str:
        cfg = getattr(self._inner, "config", None)
        return str(getattr(cfg, "model", "") or self._provider)

    # -- internals --------------------------------------------------------------
    def _one(self, text: str, action: str) -> list[float]:
        try:
            return list(self._inner.embed(text, action))
        except TypeError:      # a few providers' embed() don't accept memory_action
            return list(self._inner.embed(text))


def mem0_providers() -> set[str]:
    """Provider names mem0 recognizes. If mem0 can't be imported, return an empty set --
    that path is just unavailable; it shouldn't make import supermem.config fail."""
    try:
        from mem0.utils.factory import EmbedderFactory
        return set(getattr(EmbedderFactory, "provider_to_class", {}) or {})
    except Exception:
        return set()
