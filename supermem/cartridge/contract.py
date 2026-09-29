"""The cartridge contract: what a compiled block of context *is*.

A cartridge is a piece of long-lived context (an organisation's policy, one
caller's memory, the caller x organisation account state) rendered into a
canonical text block whose KV cache the inference engine can keep and reuse.

KV is model-specific inference state, not portable data. A cartridge compiled
for one model / tokenizer is garbage for another, so everything that changes
the KV goes into the id. Two cartridges with the same id MUST produce the same
tokens; anything that could change a token bumps the id.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# Canonical attach order: most-shared first, most-volatile last. Exact prefix
# caching can only reuse a leading run of identical tokens, so the block shared
# by every caller of the tenant goes first, then the one stable for this caller,
# then the caller x org state. Per-turn memory hits and the utterance follow the
# cartridges and are never cached.
KIND_ORDER = ("org", "user", "rel")

KIND_TITLES = {
    "org": "ORGANISATION CONTEXT",
    "user": "CALLER MEMORY",
    "rel": "CALLER x ORGANISATION ACCOUNT",
}


def sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


@dataclass(frozen=True)
class Cartridge:
    kind: str                     # org | user | rel
    tenant: str                   # cartridges never cross tenants
    scope: dict                   # {"org": ...} / {"user": ...} / {"org": ..., "user": ...}
    body: str                     # canonical text, exactly what the model reads
    version: int                  # bumps when the underlying memory changes
    model: str                    # exact model the KV is valid for
    tokenizer: str                # tokenizer identity (defaults to the model id)
    tokens: int                   # token count of the rendered block
    tokens_estimated: bool = False
    codec: str = "fp16"           # KV storage format the engine keeps it in
    compiled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if self.kind not in KIND_ORDER:
            raise ValueError(f"unknown cartridge kind {self.kind!r}; expected one of {KIND_ORDER}")

    @property
    def chunk_sha(self) -> str:
        return sha(self.body)

    @property
    def id(self) -> str:
        # compiled_at and the token count are deliberately left out: recompiling
        # identical content for the same model must yield the same id, or the
        # engine would treat a warm cartridge as a new one.
        key = json.dumps({
            "kind": self.kind, "tenant": self.tenant, "scope": self.scope,
            "chunk_sha": self.chunk_sha, "version": self.version,
            "model": self.model, "tokenizer": self.tokenizer, "codec": self.codec,
        }, sort_keys=True)
        return sha(key)

    def render(self) -> str:
        """The block as it appears in the prompt. The header carries the id so a
        trace shows exactly which cartridge version the model saw."""
        return f"### {KIND_TITLES[self.kind]} [cartridge {self.id} v{self.version}]\n{self.body}"

    def manifest(self) -> dict:
        d = asdict(self)
        d.pop("body")
        d.update(id=self.id, chunk_sha=self.chunk_sha)
        return d


def ordered(cartridges: list[Cartridge]) -> list[Cartridge]:
    """Sort into canonical attach order and refuse a mixed-tenant or mixed-model set."""
    if not cartridges:
        return []
    tenants = {c.tenant for c in cartridges}
    if len(tenants) > 1:
        raise ValueError(f"cartridges from more than one tenant in one request: {sorted(tenants)}")
    models = {(c.model, c.tokenizer) for c in cartridges}
    if len(models) > 1:
        raise ValueError(f"cartridges compiled for different models in one request: {sorted(models)}")
    return sorted(cartridges, key=lambda c: KIND_ORDER.index(c.kind))
