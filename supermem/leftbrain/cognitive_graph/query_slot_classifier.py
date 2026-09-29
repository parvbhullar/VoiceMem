"""LLM-based query slot classifier.

classify(): a single LLM call that only picks slots from base-7 + extracts entities; dynamic slots are not flattened in.
classify_child(): the "drill-down" step of hierarchical classification -- given the already-selected parent slot and
its list of child slots, decide whether any child slot is more precise than the parent itself; called recursively by
the caller (core.py::Classify()), implementing "pick the broad category first, then drill level by level into more
precise children, and stop at the current level when no child fits better".

Returns: QueryClassification(slots=["work"], entities=["Alibaba"])
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .slot_v2 import ALL_SLOT_V2_VALUES
from supermem.llm_config import resolve_api_key, resolve_model

logger = logging.getLogger(__name__)

_BASE_SLOTS = """- work: career, job, company, projects, colleagues, workplace
- finance: money, salary, income, expenses, investments, savings
- relationships: friends, family, romantic, social connections
- health: physical health, exercise, diet, sleep, medical
- goals: future plans, dreams, aspirations, self-improvement
- daily_life: daily routines, hobbies, leisure, lifestyle
- knowledge: learning, concepts, skills, facts, technology"""

_SYSTEM_PROMPT_TEMPLATE = """You are a memory router. Given a user's utterance, identify:
1. slots: which life domains the query is about (pick 1-2 from the list below)
2. entities: specific named things mentioned — people, organizations, places, AND
   specific objects/assets/projects/collections the user owns or refers to
   (e.g. "my fish tank", "the coin collection", "my car", "the report").
   Do NOT include generic activities or topics with no specific referent
   (e.g. "exercise" alone is not an entity, but "my fish tank" or "coin collection" is).

Available slots:
{slot_list}

ALWAYS return at least one slot — an empty list is never a valid answer.
If the utterance asks about a specific event or moment ("When did X go to the
museum?", "Where did Y camp?"), do not stop at "it's asking about a date/place".
Classify by what kind of life activity that event is:
- attending a support group / meeting friends → relationships
- running a race / a medical visit → health
- giving a talk at school / a work trip → work
- a pottery class / a museum trip / painting → daily_life
- reading a book / learning a skill → knowledge
Only when nothing fits at all, pick the single closest slot anyway.

Return JSON only: {{"slots": [...], "entities": [...]}}

Examples:
- "I want to quit my job" → {{"slots": ["work"], "entities": []}}
- "When did she go to the support group?" → {{"slots": ["relationships"], "entities": ["support group"]}}
- "When did he run the charity race?" → {{"slots": ["health"], "entities": ["charity race"]}}
- "Running has been really tiring lately" → {{"slots": ["health", "daily_life"], "entities": []}}
- "How's the project with Alibaba going" → {{"slots": ["work"], "entities": ["Alibaba"]}}
- "I want to save money to buy a house" → {{"slots": ["finance", "goals"], "entities": []}}
- "How is Mom's health" → {{"slots": ["health", "relationships"], "entities": ["Mom"]}}
- "How do I learn Python async programming" → {{"slots": ["knowledge"], "entities": []}}
- "What's in my fish tank" → {{"slots": ["daily_life"], "entities": ["fish tank"]}}
- "How many coins are in my coin collection now" → {{"slots": ["daily_life"], "entities": ["coins", "coin collection"]}}
- "Can you check whether that report has been revised yet" → {{"slots": ["work"], "entities": ["report"]}}"""


def _build_system_prompt(extra_slots: list[tuple[str, str]] | None) -> str:
    slot_list = _BASE_SLOTS
    if extra_slots:
        extras = "\n".join(f"- {name}: {desc}" for name, desc in extra_slots)
        slot_list = slot_list + "\n" + extras
    return _SYSTEM_PROMPT_TEMPLATE.format(slot_list=slot_list)


@dataclass
class SlotClassifierConfig:
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = 15.0

    def resolved_model(self) -> str:
        return resolve_model(self.model)


@dataclass
class QueryClassification:
    slots: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)

    def primary_slot(self) -> str | None:
        return self.slots[0] if self.slots else None


class QuerySlotClassifier:
    """Single-call LLM classifier: query → domain slots + named entities."""

    def __init__(self, config: SlotClassifierConfig | None = None) -> None:
        self._cfg = config or SlotClassifierConfig()
        self._model = self._cfg.resolved_model()
        self._client = self._build_client()

    def _build_client(self) -> Any:
        from openai import OpenAI
        kw: dict[str, Any] = {
            "api_key": resolve_api_key(self._cfg.api_key),
            "timeout": self._cfg.timeout,
        }
        if self._cfg.base_url:
            kw["base_url"] = self._cfg.base_url
        return OpenAI(**kw)

    def classify(
        self,
        query: str,
        extra_slots: list[tuple[str, str]] | None = None,
    ) -> QueryClassification:
        """Classify query into life-domain slots and named entities.

        extra_slots: [(name, description), ...] — dynamically emerged new slots, appended to the prompt.
        Returns empty QueryClassification on any failure — callers must handle
        the no-slot fallback (full corpus search).
        """
        known = set(ALL_SLOT_V2_VALUES)
        if extra_slots:
            known.update(name for name, _ in extra_slots)
        system_prompt = _build_system_prompt(extra_slots)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                max_tokens=512,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            slots = [s for s in (data.get("slots") or []) if s in known]
            entities = [
                str(e).strip()
                for e in (data.get("entities") or [])
                if str(e).strip()
            ]
            return QueryClassification(slots=slots[:2], entities=entities)
        except Exception as exc:
            print(f"[slot_classifier] ERROR: {exc}")
            logger.warning("QuerySlotClassifier.classify failed: %s", exc)
            return QueryClassification()

    def classify_child(
        self,
        query: str,
        parent_name: str,
        children: list[tuple[str, str]],
    ) -> str | None:
        """After parent_name has been selected, choose again among its child slots -- the
        "drill-down" step of hierarchical classification. When children is empty, or none is
        more precise than the parent slot itself, return None (the caller then falls back to
        parent_name instead of forcing a drill into an unsuitable child level).
        """
        if not children:
            return None
        child_list = "\n".join(f"- {name}: {desc}" for name, desc in children)
        prompt = (
            f"The user's question has already been classified under the broad category \"{parent_name}\". "
            f"This category is further divided into these more specific subtopics:\n{child_list}\n\n"
            f"User question: {query}\n\n"
            f"Among these subtopics, is there one that matches this question more precisely than the broad category \"{parent_name}\" itself? "
            f"If so, pick the single best fit; if none is more accurate than just using the broad category \"{parent_name}\" "
            "(e.g. the question is fairly general and does not specifically point to any one subtopic), say there is no more precise match.\n"
            'Output JSON only: {"child": "<subtopic name>"} or {"child": null}'
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=256,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            child = data.get("child")
            valid_names = {name for name, _ in children}
            return child if child in valid_names else None
        except Exception as exc:
            print(f"[slot_classifier] classify_child ERROR: {exc}")
            logger.warning("QuerySlotClassifier.classify_child failed: %s", exc)
            return None


__all__ = [
    "QueryClassification",
    "QuerySlotClassifier",
    "SlotClassifierConfig",
]
