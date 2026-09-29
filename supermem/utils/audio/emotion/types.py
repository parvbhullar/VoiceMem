"""Right-brain data structures: VAD, anomaly attribution and emotion-graph deltas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class VAD:
    """Acoustic emotion state: Valence and Arousal.

    Convention: valence ∈ [-1, 1], negative = more negative, positive = more positive; arousal ∈ [0, 1], higher = stronger arousal.
    Values may come from a deep model or a heuristic estimate; upper-layer logic is decoupled from the concrete estimator.
    """

    valence: float
    arousal: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", float(max(-1.0, min(1.0, self.valence))))
        object.__setattr__(self, "arousal", float(max(0.0, min(1.0, self.arousal))))


TriggerKind = Literal["anomaly"]
EmotionNodeType = Literal["User", "EmotionEpisode", "Topic", "Event", "Entity", "Person", "Project", "Organization", "Place"]
EmotionEdgeType = Literal["EXPERIENCED", "EMOTIONAL_REACTION_TO"]
EmotionIntensity = Literal["low", "medium", "high"]


@dataclass
class TurnEmotionRecord:
    """Emotion-side record for a single user utterance turn."""

    turn_id: str
    session_id: str
    vad: VAD
    timestamp_s: float | None = None
    user_utterance_index: int = 0
    time_gap_from_prev_turn_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmotionSignal:
    """Structured emotion state, from multimodal attribution of an anomalous turn."""

    label: str
    valence: float
    arousal: float
    intensity: EmotionIntensity = "medium"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "valence", float(max(-1.0, min(1.0, self.valence))))
        object.__setattr__(self, "arousal", float(max(0.0, min(1.0, self.arousal))))
        object.__setattr__(self, "confidence", float(max(0.0, min(1.0, self.confidence))))


@dataclass(frozen=True)
class EmotionGraphNodeInput:
    """Entity/topic node the anomaly attribution model suggests writing into the emotion graph."""

    local_id: str
    name: str
    node_type: EmotionNodeType = "Entity"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmotionGraphEdgeInput:
    """Edge the anomaly attribution model suggests writing into the emotion graph."""

    source: str
    target: str
    edge_type: EmotionEdgeType = "EMOTIONAL_REACTION_TO"
    description: str = ""
    emotion_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmotionGraphDelta:
    """Graph delta produced by one anomaly attribution."""

    nodes: list[EmotionGraphNodeInput] = field(default_factory=list)
    edges: list[EmotionGraphEdgeInput] = field(default_factory=list)


@dataclass(frozen=True)
class TurnAttributionLLMResult:
    """Output of the multimodal attribution model for an anomalous turn."""

    analysis_text: str
    emotion: EmotionSignal
    acoustic_evidence: list[str] = field(default_factory=list)
    semantic_evidence: list[str] = field(default_factory=list)
    related_nodes: list[EmotionGraphNodeInput] = field(default_factory=list)
    graph_delta: EmotionGraphDelta = field(default_factory=EmotionGraphDelta)
    retrieval_snippet: list[str] = field(default_factory=list)


@dataclass
class EmotionAttribution:
    """Emotion attribution result; only produced when VAD is negatively significant."""

    turn_id: str
    session_id: str
    trigger: TriggerKind
    analysis_text: str
    vad_at_trigger: VAD
    left_context_summary: str | None = None
    emotion: EmotionSignal | None = None
    acoustic_evidence: list[str] = field(default_factory=list)
    semantic_evidence: list[str] = field(default_factory=list)
    related_nodes: list[EmotionGraphNodeInput] = field(default_factory=list)
    graph_delta: EmotionGraphDelta = field(default_factory=EmotionGraphDelta)
    user_utterance_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
