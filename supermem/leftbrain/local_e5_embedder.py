"""Local E5 embedder (offline memory embedding, no network).

Symmetric with the remote ``OpenAILocalEmbedder`` (``local_memory_store.py``, via the OpenAI
Embeddings API): here vectors come from a local ``intfloat/multilingual-e5-small``, so the
whole Rank/store path stays within the 0-300ms speculative budget without touching the network.

    from supermem import SuperMem
    from supermem.leftbrain.local_e5_embedder import LocalE5Embedder
    vm = SuperMem(embedding=lambda: LocalE5Embedder())   # memory vectors via local E5, zero network

``shared_e5()`` caches a single SentenceTransformer, shared by embedding and local slot
classification (``LocalQueryClassifier(model=shared_e5())``), saving a copy in memory.

E5's ``"query: "`` / ``"passage: "`` prefixes are required (not decoration). The model downloads automatically on first use.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

# Use the offline package (models/embedding/) if present, otherwise the HF id downloads on first run
def _e5_name() -> str:
    from supermem.utils.common.paths import hf_model
    return hf_model("embedding", "intfloat/multilingual-e5-small", "e5")


_E5_NAME = _e5_name()


@lru_cache(maxsize=1)
def shared_e5():
    """Cache a single local E5 (memory embedding + slot classification share one instance, saving memory)."""
    import os as _os
    # Every startup prints a "Loading weights: 100%|███| 199/199" line (tqdm, from
    # transformers). The model is local and loads instantly; the line is alarming and
    # carries no information. Set SUPERMEM_VERBOSE=1 to see loading details.
    if _os.environ.get("SUPERMEM_VERBOSE", "0") == "0":
        try:
            from transformers.utils import logging as _hf_logging
            _hf_logging.disable_progress_bar()
        except Exception:
            pass
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_E5_NAME)


class LocalE5Embedder:
    """Inject into SuperMem(embedding=...): Rank/store vectors via local E5, no network within the 0-300ms budget."""

    @property
    def model_name(self):
        return f"{_E5_NAME} (local)"

    @property
    def dimensions(self):
        # Newer sentence-transformers renamed it to get_embedding_dimension; the old name
        # still exists but emits a FutureWarning. Try both names so it's quiet across versions.
        m = shared_e5()
        fn = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        return fn()

    def embed_texts(self, texts):
        if not texts:
            return []
        return np.asarray(shared_e5().encode([f"passage: {t}" for t in texts],
                                             normalize_embeddings=True)).tolist()

    def embed_query_text(self, text):
        return np.asarray(shared_e5().encode([f"query: {text}"], normalize_embeddings=True)[0]).tolist()
