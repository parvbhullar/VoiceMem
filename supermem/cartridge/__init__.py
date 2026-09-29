"""Reusable KV context cartridges.

SuperMem decides *what* the agent should know about a caller; this package
compiles that into cartridges whose KV cache the inference engine (vLLM,
LMCache, NVIDIA Dynamo) keeps and reuses, so carrying rich memory stops
costing a full prefill on every turn.

    compiler  SuperMem memory -> canonical, versioned cartridges
    runtime   cartridge order, prompt layout, pre-ring prefetch
    engine    measuring client for any OpenAI-compatible engine
    report    p50/p95 summaries and the showcase table
"""
from supermem.cartridge.compiler import ContextCompiler, Fact, TokenCounter, facts_from_space
from supermem.cartridge.contract import Cartridge, ordered
from supermem.cartridge.engine import Engine, TurnResult
from supermem.cartridge.runtime import ContextRuntime, cartridge_reply

__all__ = [
    "Cartridge", "ContextCompiler", "ContextRuntime", "Engine", "Fact", "TokenCounter",
    "TurnResult", "cartridge_reply", "facts_from_space", "ordered",
]
