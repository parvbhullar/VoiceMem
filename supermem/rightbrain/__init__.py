"""Right-brain Experience Layer.

Two layers:

  · **Memory layer** heartnote + response_experience, retrieved by anchor (store/experience_repository)
  · **Judgement layer** rb_traits + rb_evidence, one node = one judgement about this person;
    the claim carries a vector, is retrieved by query semantics, and is returned as an rb_hit with source="profile"
    (see traits_store.py and brain._rb_trait_hits)

The judgement layer replaces the old slot→entity→heartnote graph. In that structure the entity layer did
three jobs at once (judgement / topic / emotion word), and in practice it snowballed into grab-bags like
"sad ×61" and "Jiaqi ×52", with descriptions only filled in later by consolidation batches. The old tables
(rb_slots/rb_entities) are kept read-only for one version and are no longer written.
"""
from .anchor_router import AnchorRouter
from .attribution_manager import AttributionManager
from .brain import RightBrain, RightBrainHit
from .experience_repository import ExperienceRepository
from .graph_store import RBEntity, RBSlot, RightBrainGraphStore
from .store import RightBrainStore
from .types import (
    CurrentSignals,
    MemoryAnchor,
    MemoryQueryPlan,
    RightBrainContext,
    RightBrainMemory,
)

__all__ = [
    "AnchorRouter",
    "AttributionManager",
    "RightBrain",
    "RightBrainHit",
    "ExperienceRepository",
    "RightBrainGraphStore",
    "RBSlot",
    "RBEntity",
    "RightBrainStore",
    "CurrentSignals",
    "MemoryAnchor",
    "MemoryQueryPlan",
    "RightBrainContext",
    "RightBrainMemory",
]
