"""Anchor Router: turns user input and Preprocessor signals into a MemoryQueryPlan.

Core logic:
  1. Reuse the left brain CognitiveGraphStore's entity matching; no extra LLM call
  2. entity.entity_type -> anchor_type (person -> person, project -> project ...)
  3. entity_edges -> entity_edge anchor ("Boss assigns a task" is more precise than "Boss" alone)
  4. When nothing matches, add user_self + global_style fallback

Does not depend on right-brain tables; only reads the left-brain SQLite.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .types import CurrentSignals, MemoryAnchor, MemoryQueryPlan

if TYPE_CHECKING:
    from supermem.leftbrain.cognitive_graph.store import CognitiveGraphStore

# Left brain EntityType.value -> MemoryAnchor anchor_type
# This used to map from the left brain's SlotType (a separate 7-class
# entity-category taxonomy independent of SlotV2); SlotType was removed when
# the slot taxonomy was unified (SlotV2 now represents "life domains"), and
# what this code actually needed all along was "what type is this entity" --
# EntityType expresses that directly, no need to translate through slots.
# organization folds into person (the old ENTITY_TYPE_TO_SLOT also folded
# ORGANIZATION into the people slot); event folds into knowledge (the old
# mapping likewise put EVENT under knowledge).
# user/preference entities are not in the table and take the .get(..., "knowledge")
# default below, matching the fallback behaviour of the old SlotType path.
_ENTITY_TYPE_TO_ANCHOR: dict[str, str] = {
    "person":       "person",
    "organization": "person",
    "project":      "project",
    "task":         "task",
    "knowledge":    "knowledge",
    "event":        "knowledge",
    "place":        "place",
    "routine":      "routine",
    "asset":        "asset",
}

# anchor_type -> default role
_ANCHOR_ROLE: dict[str, str] = {
    "person":       "subject",
    "user":         "global_profile",
    "project":      "context",
    "task":         "topic",
    "knowledge":    "context",
    "place":        "context",
    "routine":      "context",
    "asset":        "context",
    "entity_edge":  "trigger",
    "global_style": "global_profile",
}

# anchor_type -> retrieval weight
_ANCHOR_WEIGHT: dict[str, float] = {
    "task":         1.0,
    "entity_edge":  0.9,
    "project":      0.9,
    "person":       0.7,
    "knowledge":    0.6,
    "place":        0.5,
    "routine":      0.5,
    "asset":        0.4,
    "user":         0.3,
    "global_style": 0.3,
}

_CANONICAL_EMOTIONS = {"anxious", "sad", "wronged", "lonely", "conflicted", "calm", "happy", "tired"}

# English keyword table -- emotion labels (e.g. English emotion_tag values
# produced by ASR/TTS) are normalised onto the canonical keys; otherwise
# English input would all be misclassified as "calm".
_EMOTION_KEYWORDS_EN: list[tuple[str, str]] = [
    # anxious
    ("anxious", "anxious"), ("anxiety", "anxious"), ("nervous", "anxious"), ("worried", "anxious"),
    ("worry", "anxious"), ("stressed", "anxious"), ("stress", "anxious"), ("tense", "anxious"),
    ("fearful", "anxious"), ("afraid", "anxious"), ("scared", "anxious"), ("panicked", "anxious"),
    ("panic", "anxious"), ("uneasy", "anxious"), ("apprehensive", "anxious"),
    # sad
    ("sad", "sad"), ("sadness", "sad"), ("upset", "sad"), ("depressed", "sad"),
    ("disappointed", "sad"), ("heartbroken", "sad"), ("miserable", "sad"),
    ("dejected", "sad"), ("despair", "sad"), ("sorrowful", "sad"), ("grief", "sad"),
    # wronged / angry
    ("wronged", "wronged"), ("angry", "wronged"), ("anger", "wronged"), ("mad", "wronged"),
    ("furious", "wronged"), ("irritated", "wronged"), ("annoyed", "wronged"), ("frustrated", "wronged"),
    ("resentful", "wronged"), ("indignant", "wronged"), ("unfair", "wronged"), ("bitter", "wronged"),
    # lonely
    ("lonely", "lonely"), ("loneliness", "lonely"), ("isolated", "lonely"), ("empty", "lonely"),
    ("alone", "lonely"),
    # conflicted
    ("conflicted", "conflicted"), ("torn", "conflicted"), ("confused", "conflicted"), ("uncertain", "conflicted"),
    ("hesitant", "conflicted"), ("ambivalent", "conflicted"), ("indecisive", "conflicted"), ("perplexed", "conflicted"),
    ("lost", "conflicted"),
    # calm
    ("calm", "calm"), ("relaxed", "calm"), ("peaceful", "calm"), ("composed", "calm"),
    ("serene", "calm"), ("settled", "calm"), ("neutral", "calm"),
    # happy
    ("happy", "happy"), ("happiness", "happy"), ("joy", "happy"), ("joyful", "happy"),
    ("excited", "happy"), ("excitement", "happy"), ("glad", "happy"), ("pleased", "happy"),
    ("delighted", "happy"), ("proud", "happy"), ("grateful", "happy"), ("thankful", "happy"),
    ("relieved", "happy"), ("hopeful", "happy"), ("cheerful", "happy"), ("satisfied", "happy"),
    ("content", "happy"), ("amused", "happy"),
    # tired
    ("tired", "tired"), ("exhausted", "tired"), ("fatigue", "tired"), ("fatigued", "tired"),
    ("weary", "tired"), ("drained", "tired"), ("sleepy", "tired"), ("worn out", "tired"),
]

_EN_KEYWORD_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), canonical)
    for kw, canonical in _EMOTION_KEYWORDS_EN
]


def normalize_emotion_strict(emotion: str) -> str | None:
    """Map a free-form emotion string to a canonical label; return None when
    nothing matches.

    Anchor-related callers should use this version: unrecognised emotion words
    (guilty / jealous / nostalgic...) used to all fall back to "calm" and then
    get written into the retrieval anchors at the highest weight (1.2) --
    effectively injecting a high-weight wrong signal into the query. If it
    can't be recognised, adding no emotion anchor is better than adding a
    wrong one."""
    e = emotion.strip()
    if not e:
        return None
    if e in _CANONICAL_EMOTIONS:
        return e
    for pattern, canonical in _EN_KEYWORD_RE:
        if pattern.search(e):
            return canonical
    return None


def normalize_emotion(emotion: str) -> str:
    """Like normalize_emotion_strict, but falls back to "calm" for callers that
    need a guaranteed canonical label (e.g. graph node naming)."""
    return normalize_emotion_strict(emotion) or "calm"


_STOP = {
    "who", "is", "are", "was", "were", "what", "when", "where", "why",
    "how", "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "my", "your", "his", "her", "their", "our", "tell",
    "me", "about", "did", "do", "does", "has", "have", "had", "can",
    "could", "would", "should", "with", "from", "that", "this", "it",
    "be", "been", "being", "not", "no", "any", "some", "which",
}


class AnchorRouter:
    """Builds a MemoryQueryPlan from the current input.

    cognitive_store may be None (then only fallback anchors are returned).
    """

    def __init__(
        self,
        cognitive_store: "CognitiveGraphStore | None" = None,
    ) -> None:
        self._store = cognitive_store

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
        """``context``: the agent's previous line. The user's line often only makes
        sense inside it ("never mind then"), so entities it mentions also become
        anchors, but demoted to the context role at half weight -- it is
        background, not the subject of the user's turn. clean_text is still
        only the user's own words."""
        anchors = self._build_anchors(
            query, user_id, hint_entities=entities, hint_emotion=emotion,
            context_text=context,
        )
        return MemoryQueryPlan(
            user_id=user_id,
            clean_text=query.strip(),
            anchors=anchors,
            current_signals=signals or CurrentSignals(),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    # Entities matched in the agent's line get their anchor weight scaled by this (background < subject of the user's turn)
    _CONTEXT_WEIGHT_SCALE = 0.5

    def _build_anchors(self, query: str, user_id: str, hint_entities: list[str] | None = None,
                       hint_emotion: str | None = None,
                       context_text: str | None = None) -> list[MemoryAnchor]:
        anchors: list[MemoryAnchor] = []
        seen_ids: set[str] = set()

        if self._store is not None:
            from supermem.leftbrain.cognitive_graph.store import normalize_name

            # Scan the user's line first: entities mentioned in both are claimed at full weight and not demoted by the background pass
            sources: list[tuple[str, float]] = [(query, 1.0)]
            if context_text and context_text.strip():
                sources.append((context_text, self._CONTEXT_WEIGHT_SCALE))

            matched_entities: list[tuple[Any, float]] = []
            all_ents = self._store.find_entities(user_id)   # for the reverse name lookup; one query shared by both passes
            for text, scale in sources:
                # Candidates: English words + bigrams (original logic)
                raw_words = re.findall(r"\b\w{2,}\b", text.lower())
                candidates = [w for w in raw_words if w not in _STOP]
                for i in range(len(raw_words) - 1):
                    a, b = raw_words[i], raw_words[i + 1]
                    if a not in _STOP and b not in _STOP:
                        candidates.append(f"{a} {b}")

                for cand in candidates:
                    ents = self._store.find_entities_by_name_fuzzy(user_id, cand)
                    for e in ents:
                        if e.id not in seen_ids:
                            seen_ids.add(e.id)
                            matched_entities.append((e, scale))

                # Reverse lookup: walk all entity names and check whether each appears in the input
                # (catches names the word-boundary regex misses, e.g. text without spaces)
                text_lower = text.lower()
                for e in all_ents:
                    if e.id in seen_ids:
                        continue
                    name_l = e.name.lower()
                    name_n = (e.name_norm or "").lower()
                    if len(name_l) >= 2 and (name_l in text_lower or name_n in text_lower):
                        seen_ids.add(e.id)
                        matched_entities.append((e, scale))

            # Entities supplied by the voice module: use the name directly as the anchor (consistent with Ingest writes)
            if hint_entities:
                for name in hint_entities:
                    key = name.lower().strip()
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    anchors.append(MemoryAnchor(
                        anchor_type="entity",
                        anchor_id=key,
                        role="subject",
                        weight=1.0,
                        confidence=1.0,
                    ))

            for ent, scale in matched_entities:
                anchor_type = _ENTITY_TYPE_TO_ANCHOR.get(ent.entity_type.value, "knowledge")
                anchors.append(MemoryAnchor(
                    anchor_type=anchor_type,
                    anchor_id=ent.id,
                    # Only appeared in the agent's line: record as context, half weight
                    role=("context" if scale < 1.0
                          else _ANCHOR_ROLE.get(anchor_type, "context")),
                    weight=_ANCHOR_WEIGHT.get(anchor_type, 0.5) * scale,
                    confidence=ent.confidence,
                ))
                # Right-brain writes use name.lower().strip() for entity anchors (the
                # entity anchor_id convention in core.py::Ingest), not the left brain's
                # entity.id -- the two ID systems don't interoperate. So we also add an
                # "entity" anchor normalised the same way, letting fuzzily matched
                # entities find the anchors attached when they were written.
                name_key = ent.name.lower().strip()
                if name_key not in seen_ids:
                    seen_ids.add(name_key)
                    anchors.append(MemoryAnchor(
                        anchor_type="entity",
                        anchor_id=name_key,
                        role="subject",
                        weight=1.0 * scale,
                        confidence=ent.confidence,
                    ))

            # entity_edge anchors: when >=2 entities match, also add the edges between them
            if len(matched_entities) >= 2:
                entity_ids = [e.id for e, _ in matched_entities]
                for e, _scale in matched_entities:
                    edges = self._store.edges_for_entity(e.id, user_id)
                    for edge in edges:
                        if (edge.from_entity_id in entity_ids
                                and edge.to_entity_id in entity_ids
                                and edge.id not in seen_ids):
                            seen_ids.add(edge.id)
                            anchors.append(MemoryAnchor(
                                anchor_type="entity_edge",
                                anchor_id=edge.id,
                                role="trigger",
                                weight=_ANCHOR_WEIGHT["entity_edge"],
                                confidence=edge.confidence,
                            ))

        # emotion anchor: retrieve past emotional events by the current emotion (highest weight).
        # Strict version: unrecognised emotion words add no anchor (instead of falling back to "calm").
        if hint_emotion:
            canonical = normalize_emotion_strict(hint_emotion)
            if canonical is not None:
                anchors.append(MemoryAnchor(
                    anchor_type="emotion",
                    anchor_id=canonical,
                    role="trigger",
                    weight=1.2,
                    confidence=1.0,
                ))

        # Fallback: user_self + global_style are always added (low weight)
        anchors.append(MemoryAnchor(
            anchor_type="user",
            anchor_id="user_self",
            role="global_profile",
            weight=_ANCHOR_WEIGHT["user"],
            confidence=1.0,
        ))
        anchors.append(MemoryAnchor(
            anchor_type="global_style",
            anchor_id="global_style",
            role="global_profile",
            weight=_ANCHOR_WEIGHT["global_style"],
            confidence=1.0,
        ))

        return anchors
