"""Experience Repository: the high-level interface for right-brain retrieval + writes.

Retrieval order (spec section 9):
  1. response_experience    — highest priority, lessons from responses that worked or failed
  2. heartnote                   — situational emotional patterns
  3. user_interaction_profile    — global user style

Writes: upsert_experience + link_anchors
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .anchor_router import AnchorRouter
from .store import RightBrainStore
from .types import (
    CurrentSignals, MemoryAnchor, MemoryClass,
    MemoryQueryPlan, RightBrainContext, RightBrainMemory, TTL,
)

_DEFAULT_DB_NAME = "right_brain.sqlite"


class ExperienceRepository:
    """The complete entry point to the right-brain Experience Layer.

    Usage (minimal)::

        repo = ExperienceRepository.create(db_path="memory/right_brain.sqlite")
        plan = repo.build_query_plan("Lang's plan from last time", user_id="u1")
        ctx  = repo.retrieve(plan)
        print(ctx.to_prompt_block())
    """

    def __init__(
        self,
        store: RightBrainStore,
        anchor_router: AnchorRouter,
    ) -> None:
        self._store = store
        self._router = anchor_router

    @classmethod
    def create(
        cls,
        db_path: Path | str,
        *,
        cognitive_store=None,     # CognitiveGraphStore | None
    ) -> "ExperienceRepository":
        store  = RightBrainStore(db_path)
        router = AnchorRouter(cognitive_store)
        return cls(store, router)

    # ── Query Plan ────────────────────────────────────────────────────────────

    def build_query_plan(
        self,
        query: str,
        user_id: str,
        *,
        signals: CurrentSignals | None = None,
        entities: list[str] | None = None,
        emotion: str | None = None,
        context: str | None = None,
    ) -> MemoryQueryPlan:
        """``context``: the agent's previous line (which the user is responding to); used only for anchors, not included in clean_text."""
        return self._router.build_query_plan(
            query, user_id, signals=signals, entities=entities, emotion=emotion,
            context=context,
        )

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        plan: MemoryQueryPlan,
        *,
        experience_limit: int = 3,
        pattern_limit: int = 3,
    ) -> RightBrainContext:
        """Retrieve the three memory classes in parallel and return them combined by priority."""
        anchors = plan.anchors
        uid     = plan.user_id
        sigs    = plan.current_signals

        # 1. The response_experience class is **no longer written or retrieved** (see
        #    learn_from_reaction in brain.py): it was write-only. The genuinely useful next_time went into
        #    metadata with no reader at all; the assistant_did that reached the prompt looked like "the
        #    assistant guided the user in a relaxed tone", taking a slot every turn while giving no
        #    information. What the assistant should do can already be inferred from user-side traits
        #    ("when he is down he wants understanding and validation" already says what to give).
        #    We return an empty list here instead of removing the whole path: entries already stored in
        #    old databases should not suddenly become unhandled orphans; types and frontend compat remain.
        experiences: list = []

        # 2. heartnote
        patterns = self._store.search_by_anchors(
            uid, anchors, memory_class="heartnote", limit=pattern_limit,
        )

        # user_interaction_profile is managed separately by the prestimulus layer UserProfileStore, not retrieved here
        return RightBrainContext(
            response_experiences=experiences,
            situation_patterns=patterns,
            current_signals=sigs,
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_experience(
        self,
        user_id: str,
        memory_class: MemoryClass,
        content: str,
        anchors: list[MemoryAnchor],
        *,
        condition: str | None = None,
        priority: float = 0.5,
        confidence: float = 1.0,
        ttl: TTL = "long_term",
        metadata: dict[str, Any] | None = None,
        evidence_turn_ids: list[str] | None = None,
        evidence_memory_ids: list[str] | None = None,
        memory_id: str | None = None,
        created_at: str | None = None,
    ) -> RightBrainMemory:
        """Write one right-brain memory and attach anchors.

        ``created_at``: when the event actually happened (required for backfill/benchmark scenarios,
        otherwise the rendered date is the write wall-clock and temporal questions are skewed; see store.upsert_memory).
        """
        mem = self._store.upsert_memory(
            user_id, memory_class, content,
            condition=condition,
            priority=priority, confidence=confidence, ttl=ttl,
            metadata=metadata,
            evidence_turn_ids=evidence_turn_ids,
            evidence_memory_ids=evidence_memory_ids,
            memory_id=memory_id,
            created_at=created_at,
        )
        for anchor in anchors:
            self._store.link_anchor(mem.id, user_id, anchor)
        return mem

    # ── Convenience helpers ───────────────────────────────────────────────────

    def write_response_experience(
        self,
        user_id: str,
        content: str,
        anchors: list[MemoryAnchor],
        *,
        condition: str | None = None,
        failed: bool = False,
        metadata: dict[str, Any] | None = None,
        evidence_turn_ids: list[str] | None = None,
        **kwargs,
    ) -> RightBrainMemory:
        """Shortcut to write a response_experience; failed experiences automatically get a higher priority."""
        priority = kwargs.pop("priority", 0.9 if failed else 0.6)
        meta = dict(metadata or {})
        if failed:
            meta.setdefault("previous_failure", True)
        return self.write_experience(
            user_id, "response_experience", content, anchors,
            condition=condition, priority=priority, metadata=meta,
            evidence_turn_ids=evidence_turn_ids, **kwargs,
        )

    def write_situation_pattern(
        self,
        user_id: str,
        content: str,
        anchors: list[MemoryAnchor],
        *,
        condition: str | None = None,
        confidence: float = 0.8,
        **kwargs,
    ) -> RightBrainMemory:
        """Shortcut to write an emotional-situation pattern."""
        return self.write_experience(
            user_id, "heartnote", content, anchors,
            condition=condition, confidence=confidence, **kwargs,
        )

    # ── Inspect ───────────────────────────────────────────────────────────────

    def list_all(self, user_id: str) -> list[RightBrainMemory]:
        return self._store.get_all(user_id)

    @property
    def db_path(self) -> Path:
        return self._store._path  # noqa: SLF001
