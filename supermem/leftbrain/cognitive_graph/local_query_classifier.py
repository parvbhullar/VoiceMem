"""Local query classifier: query -> slots(+entities), never touches an LLM/network.

Interface-compatible with the built-in ``QuerySlotClassifier`` (single LLM call); it can be
injected via ``SuperMem(slots=...)`` to replace it, moving the ``Classify`` step (slot +
entity extraction) from the OpenAI API to a local model -- fully symmetric with
``embedding`` injection:

    from supermem import SuperMem
    from supermem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
    vm = SuperMem(slots=lambda: LocalQueryClassifier())    # slots use local E5, 0 LLM
    vm.search("where do I work")                            # Classify no longer hits the LLM

Design trade-offs (stated honestly, nothing hidden):
- **slots**: E5 cosine vs the 7 base-7 slot descriptions, take top-k (measured ~93% agreement with the LLM).
- **entities**: empty by default -> falls back to supermem's existing slot-only narrowing (not a new
  bad state). Open-domain entity recognition **does not require an LLM**: pass
  ``ner=<callable: query -> list[str]>`` to plug in local NER (gliner / spaCy etc.) and keep
  entities local too.
- **classify_child is not implemented**: "drilling down" into emergent child slots is left to the LLM
  version; child slots are created and maintained on the write side, not on the search hot path.
  ``engine.Classify`` detects the missing ``classify_child`` and skips the drill-down, using base-7 only.

E5's ``"query: "`` / ``"passage: "`` prefixes are required (not decoration). The model is downloaded
automatically on first use; pass ``model=`` to reuse an already-loaded SentenceTransformer (e.g. shared
with the local embedder, saving a copy in memory).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from supermem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification

# Shares the same E5 as the memory vectors (models/embedding/), saving a copy of the weights
def _model_name() -> str:
    from supermem.utils.common.paths import hf_model
    return hf_model("embedding", "intfloat/multilingual-e5-small", "e5")


_MODEL_NAME = _model_name()

# Same base-7 slot descriptions as the built-in LLM classifier (kept identical on purpose: this is a local approximation of the same classification decision).
_SLOT_DESCRIPTIONS = {
    "work": "career, job, company, projects, colleagues, workplace",
    "finance": "money, salary, income, expenses, investments, savings",
    "relationships": "friends, family, romantic, social connections",
    "health": "physical health, exercise, diet, sleep, medical",
    "goals": "future plans, dreams, aspirations, self-improvement",
    "daily_life": "daily routines, hobbies, leisure, lifestyle",
    "knowledge": "learning, concepts, skills, facts, technology",
}


class LocalQueryClassifier:
    """query -> QueryClassification(slots, entities), local E5, no LLM/network."""

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        model=None,
        ner: Callable[[str], Sequence[str]] | None = None,
        top_k: int = 2,
    ) -> None:
        self._model_name = model_name
        self._model = model                 # Can reuse an already-loaded SentenceTransformer
        self._ner = ner                     # Optional local entity recognition (default none -> slot-only)
        self._top_k = top_k
        self._slot_names = list(_SLOT_DESCRIPTIONS)
        self._slot_embs = None              # Slot description vectors, lazily computed once

    def _m(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _slots_matrix(self):
        if self._slot_embs is None:
            texts = [f"passage: {k}: {v}" for k, v in _SLOT_DESCRIPTIONS.items()]
            self._slot_embs = np.asarray(self._m().encode(texts, normalize_embeddings=True))
        return self._slot_embs

    def classify(self, query: str, extra_slots=None) -> QueryClassification:
        """Pick top-k slots from base-7 (E5 cosine) + optional local entities. extra_slots (dynamic
        child-slot candidates) are ignored by the local version -- child-slot drill-down is left to the LLM version/write side."""
        q = np.asarray(self._m().encode([f"query: {query}"], normalize_embeddings=True)[0])
        order = np.argsort(-(self._slots_matrix() @ q))[: self._top_k]
        slots = [self._slot_names[i] for i in order]
        entities = list(self._ner(query)) if self._ner else []
        return QueryClassification(slots=slots, entities=entities)
