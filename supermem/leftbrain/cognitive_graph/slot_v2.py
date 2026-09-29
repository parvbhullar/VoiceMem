"""Slot V2 taxonomy — life-domain slots for memory retrieval.

Seven slots that describe *which domain of life* a memory belongs to.
Used for LLM-based query classification and embedding-based memory tagging.
"""

from __future__ import annotations

from enum import Enum


class SlotV2(str, Enum):
    """Seven life-domain slots for memory retrieval (V2).

    Each value is the canonical string stored in the ``memory_tags`` table.
    Using ``str, Enum`` lets the values be compared and serialised directly
    as plain strings without calling ``.value``.
    """

    WORK          = "work"
    FINANCE       = "finance"
    RELATIONSHIPS = "relationships"
    HEALTH        = "health"
    GOALS         = "goals"
    DAILY_LIFE    = "daily_life"
    KNOWLEDGE     = "knowledge"


# ---------------------------------------------------------------------------
# Semantic descriptions used to build slot embeddings (write-side tagging).
# Rich descriptions so the embedder maps memories correctly.
# ---------------------------------------------------------------------------

SLOT_V2_DESCRIPTIONS: dict[str, str] = {
    SlotV2.WORK: (
        "Work, career, job, company, colleagues, projects, meetings, work performance, "
        "professional development, workplace, business tasks"
    ),
    SlotV2.FINANCE: (
        "Money, salary, income, expenses, investments, savings, debt, financial goals, "
        "spending, budget"
    ),
    SlotV2.RELATIONSHIPS: (
        "Friends, family, romantic relationships, social connections, interpersonal dynamics, "
        "people I care about, social events"
    ),
    SlotV2.HEALTH: (
        "Physical health, exercise, diet, sleep, mental health, fitness, wellness, "
        "illness, doctor"
    ),
    SlotV2.GOALS: (
        "Goals, ambitions, future plans, dreams, aspirations, long-term vision, "
        "self-improvement, personal development"
    ),
    SlotV2.DAILY_LIFE: (
        "Daily life, hobbies, leisure, entertainment, routines, habits, personal preferences, "
        "lifestyle, daily activities"
    ),
    SlotV2.KNOWLEDGE: (
        "Knowledge, learning, concepts, skills, facts, technology, ideas, "
        "education, research"
    ),
}

# Convenience list of all canonical slot value strings.
ALL_SLOT_V2_VALUES: list[str] = [s.value for s in SlotV2]

# Related slots — when retrieving for a primary slot, also fetch summaries of these.
SLOT_RELATIONS: dict[str, list[str]] = {
    "work":          ["finance", "relationships", "goals"],
    "finance":       ["work", "goals"],
    "health":        ["daily_life", "goals"],
    "relationships": ["work", "daily_life"],
    "goals":         ["work", "finance", "health"],
    "daily_life":    ["health", "relationships"],
    "knowledge":     ["work", "goals"],
}

__all__ = [
    "SlotV2",
    "SLOT_V2_DESCRIPTIONS",
    "ALL_SLOT_V2_VALUES",
    "SLOT_RELATIONS",
]
