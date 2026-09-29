"""Core data types for the cognitive graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .slot_v2 import SlotV2


class EntityType(str, Enum):
    USER = "user"
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    TASK = "task"
    KNOWLEDGE = "knowledge"
    PREFERENCE = "preference"
    PLACE = "place"
    ROUTINE = "routine"
    ASSET = "asset"
    EVENT = "event"


# Slots uniformly use SlotV2 (work/finance/relationships/health/goals/daily_life/
# knowledge, see slot_v2.py) -- this is the only taxonomy actually used by the
# Classify()/retrieval path. There used to be a separate 7-category SlotType here
# (people/projects/knowledge/tasks/places/routines/assets), read only by
# AnchorRouter; it was purely derived from EntityType (ENTITY_TYPE_TO_SLOT was a
# deterministic mapping), entirely disconnected from the SlotV2 used for retrieval,
# and a redundant second taxonomy. The "what type is this entity" info AnchorRouter
# needs already lives in EntityType, so it now reads entity_type directly and this
# translation layer is no longer needed.


class EntityMemoryRole(str, Enum):
    """The role an entity plays in a given memory."""
    SUBJECT = "subject"    # Subject: the initiator of the action ("Lang said...")
    OBJECT = "object"      # Object: the target of the action ("a discussion about Lang")
    CONTEXT = "context"    # Context: mentioned as background ("in project X")
    OWNER = "owner"        # Owner: the memory belongs directly to this entity ("Caroline's preference")


@dataclass
class Entity:
    id: str
    user_id: str
    entity_type: EntityType
    name: str               # Display name (original casing)
    name_norm: str          # Normalized name (used for dedup/merging)
    slot: SlotV2
    description: str = ""   # d_v: entity description, filled with the source sentence on first creation, can be appended to later
    confidence: float = 1.0
    importance: float = 0.5                        # 0-1, affects display ordering and node size
    aliases: list[str] = field(default_factory=list)   # Alias list (used when merging entities)
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class EntityEdge:
    id: str
    user_id: str
    from_entity_id: str
    to_entity_id: str
    relation_type: str
    role_label: str | None = None
    confidence: float = 1.0
    weight: float = 1.0                             # Accumulates with co-occurrence count; time-decayed by updated_at on read
    edge_type: str = "weak"                         # strong | weak, derived from a weight threshold
    status: str = "active"                         # active | deprecated | merged
    evidence_memory_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class EntityMemoryLink:
    id: str
    memory_id: str
    entity_id: str
    user_id: str
    role: EntityMemoryRole = EntityMemoryRole.CONTEXT  # Role of the entity in this memory
    relation_hint: str | None = None               # Relation description ("collaborates_on")
    created_at: str = ""


@dataclass
class MemoryRecord:
    """Memory metadata owned by the cognitive graph (one-to-one with memory_id in the vector store)."""
    id: str              # Same as memory_id in supermem_leftbrain.sqlite
    user_id: str
    slot: SlotV2
    memory_type: str     # fact | event | preference | routine | …
    content: str         # Memory text
    confidence: float = 1.0
    sensitivity: float = 0.0   # Sensitivity 0-1 (higher = more private; affects display and sharing)
    ttl: int | None = None     # Expiry in seconds, None = never expires
    created_at: str = ""
    updated_at: str = ""


@dataclass
class SlotProfile:
    user_id: str
    slot: SlotV2
    summary: str = ""                              # LLM-generated slot summary text
    entity_ids: list[str] = field(default_factory=list)  # Entity ids under this slot
    entity_count: int = 0
    memory_count: int = 0
    last_updated: str = ""


@dataclass
class AffectiveEdge:
    """Right-brain affective edge (reserved interface)."""
    id: str
    user_id: str
    from_entity_id: str
    to_entity_id: str | None = None
    trigger_frame: str | None = None
    emotion: str | None = None
    appraisal: str | None = None
    response_policy: str | None = None
    confidence: float = 1.0                        # Confidence of the emotion judgement
    evidence_memory_ids: list[str] = field(default_factory=list)  # Source memories that triggered it
    created_at: str = ""


@dataclass
class EntityAnnotation:
    """A single entity identified by the LLM from one fact."""
    name: str
    entity_type: EntityType
    role: str | None = None    # Role description from the LLM; normalized to EntityMemoryRole


@dataclass
class RelationAnnotation:
    """A relation between entities identified by the LLM from one fact."""
    from_name: str
    to_name: str
    relation_type: str
    role_label: str | None = None
    confidence: float = 1.0


@dataclass
class AnnotatedFact:
    """The LLM's full annotation result for one fact."""
    fact_text: str
    slot: SlotV2
    entities: list[EntityAnnotation] = field(default_factory=list)
    relations: list[RelationAnnotation] = field(default_factory=list)
    confidence: float = 1.0
