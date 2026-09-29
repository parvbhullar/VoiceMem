"""Corpus callosum fusion: dual-channel retrieval, reply memory, and turn orchestration.

Note: ``orchestrator`` is not imported at package init, to avoid a package-level import cycle.
Use explicit imports such as ``from supermem.utils.fusion.orchestrator import process_user_turn``.
"""

from supermem.utils.fusion.config import FusionConfig
from supermem.utils.fusion.left_channel import (
    LeftBrainGraphSearchClient,
    LeftBrainSearchClient,
    search_left_channel,
    search_left_for_reply,
)
from supermem.utils.fusion.prompt_builder import (
    build_left_only_reply_context_prompt,
    build_reply_context_prompt,
)
from supermem.utils.fusion.reply_memory import (
    PersonaProvider,
    StubPersonaProvider,
    build_normal_left_reply_memory,
    build_omni_attribution_context,
    build_reply_memory,
    retrieve_for_reply,
    retrieve_left_for_normal_turn,
    run_anomaly_turn,
    run_rightbrain_memory_extract,
)
from supermem.utils.fusion.right_channel import (
    FixtureRightChannelRelevanceFilter,
    OpenAIRightChannelRelevanceFilter,
    RightChannelRelevanceFilter,
    search_right_channel,
)
from supermem.utils.fusion.types import (
    AnomalyTurnResult,
    FusionRetrievalResult,
    LeftMemoryHit,
    ReplyContextBundle,
    ReplyContextPrompt,
    ReplyRetrievalBundle,
    RightMemoryHit,
)

__all__ = [
    "AnomalyTurnResult",
    "FixtureRightChannelRelevanceFilter",
    "FusionConfig",
    "FusionRetrievalResult",
    "LeftBrainGraphSearchClient",
    "LeftBrainSearchClient",
    "LeftMemoryHit",
    "OpenAIRightChannelRelevanceFilter",
    "PersonaProvider",
    "ReplyContextBundle",
    "ReplyContextPrompt",
    "ReplyRetrievalBundle",
    "RightChannelRelevanceFilter",
    "RightMemoryHit",
    "StubPersonaProvider",
    "build_left_only_reply_context_prompt",
    "build_normal_left_reply_memory",
    "build_omni_attribution_context",
    "build_reply_context_prompt",
    "build_reply_memory",
    "retrieve_for_reply",
    "retrieve_left_for_normal_turn",
    "run_anomaly_turn",
    "run_rightbrain_memory_extract",
    "search_left_channel",
    "search_left_for_reply",
    "search_right_channel",
]
