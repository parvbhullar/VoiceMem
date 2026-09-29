"""Right-brain Experience Layer data types.

The three memory classes:
  user_interaction_profile   — the user's long-term style preferences
  heartnote                   — situational emotional patterns
  response_experience         — experience of responses that worked or failed (highest priority)

Retrieval entry: MemoryAnchor → attached to left-brain entities
Query plan: MemoryQueryPlan → Preprocessor signals + anchors
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ── Anchor ────────────────────────────────────────────────────────────────────

AnchorType = Literal[
    "user", "person", "project", "task", "knowledge",
    "place", "routine", "asset", "event",
    "entity_edge", "slot", "current_session", "global_style",
]

AnchorRole = Literal[
    "topic", "subject", "object", "context",
    "trigger", "relationship", "evidence", "global_profile",
]

MemoryClass = Literal[
    "heartnote",
    "response_experience",
]

TTL = Literal["session", "short_term", "long_term"]


@dataclass
class MemoryAnchor:
    anchor_type: AnchorType
    anchor_id: str | None        # None = global (e.g. user_self, global_style)
    role: AnchorRole
    weight: float = 1.0
    confidence: float = 1.0


# ── Query Plan ────────────────────────────────────────────────────────────────

@dataclass
class CurrentSignals:
    """Signals the Preprocessor detected for the current turn; not stored, used only for retrieval priority."""
    affect_hint: str | None = None                          # "frustrated" / "satisfied" ...
    affect_intensity: Literal["low", "medium", "high"] | None = None
    dissatisfaction_signal: bool = False
    correction_signal: bool = False
    short_answer_needed: bool = False


@dataclass
class MemoryQueryPlan:
    user_id: str
    clean_text: str
    anchors: list[MemoryAnchor] = field(default_factory=list)
    current_signals: CurrentSignals = field(default_factory=CurrentSignals)


# ── Right Brain Memory ────────────────────────────────────────────────────────

@dataclass
class RightBrainMemory:
    id: str
    user_id: str
    memory_class: MemoryClass
    content: str
    condition: str | None           # description of the applicable situation (optional)
    priority: float                 # 0~1, response_experience is usually highest
    confidence: float
    ttl: TTL
    metadata: dict[str, Any]        # failure reason, next_time_policy, etc.
    evidence_turn_ids: list[str]
    evidence_memory_ids: list[str]
    created_at: str
    updated_at: str
    #: Anchor-hit score for this retrieval (SUM(link.weight*link.confidence), see
    #: store.search_by_anchors). Not persisted: the same memory scores differently under different queries,
    #: so it is only carried in retrieval results, letting final ranking blend "retrieval relevance" into the static priority.
    anchor_score: float = 0.0


@dataclass
class RightBrainAnchorLink:
    id: str
    user_id: str
    right_memory_id: str
    anchor_type: AnchorType
    anchor_id: str | None
    role: AnchorRole
    weight: float
    confidence: float
    created_at: str


# ── Retrieval Result ──────────────────────────────────────────────────────────

@dataclass
class RightBrainContext:
    """Right-brain retrieval result, used by the Prompt Builder.

    Contains only heartnote and response_experience.
    user_interaction_profile is managed separately by the prestimulus layer (UserProfileStore).
    """
    response_experiences: list[RightBrainMemory] = field(default_factory=list)
    situation_patterns: list[RightBrainMemory] = field(default_factory=list)
    current_signals: CurrentSignals = field(default_factory=CurrentSignals)

    def is_empty(self) -> bool:
        return not (self.response_experiences or self.situation_patterns)

    def to_prompt_block(self) -> str:
        """Build the text block injected into the LLM prompt.

        Each memory is prefixed with a [YYYY-MM-DD] date. Without it, the downstream model has no way to
        answer "when did this happen" questions (temporal reasoning); the date was always stored in
        created_at, it was just dropped when rendering before.
        """
        lines: list[str] = []

        def _date(m: RightBrainMemory) -> str:
            d = (m.created_at or "")[:10]
            return f"[{d}] " if d else ""

        if self.response_experiences:
            lines.append("[Response experience]")
            for m in self.response_experiences:
                cond = f" (when: {m.condition})" if m.condition else ""
                lines.append(f"- {_date(m)}{m.content}{cond}")

        if self.situation_patterns:
            lines.append("[Situation pattern]")
            for m in self.situation_patterns:
                cond = f" (when: {m.condition})" if m.condition else ""
                lines.append(f"- {_date(m)}{m.content}{cond}")

        sigs = self.current_signals
        hints: list[str] = []
        if sigs.dissatisfaction_signal:
            hints.append("user shows dissatisfaction this turn")
        if sigs.correction_signal:
            hints.append("user is correcting assistant")
        if sigs.affect_hint:
            intensity = f" ({sigs.affect_intensity})" if sigs.affect_intensity else ""
            hints.append(f"affect={sigs.affect_hint}{intensity}")
        if hints:
            lines.append(f"[Current signals: {'; '.join(hints)}]")

        return "\n".join(lines)
