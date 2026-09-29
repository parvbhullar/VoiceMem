"""Shared utilities for the left-brain vector store: embedder interface/implementation,
retrieval hit type, memory root directory resolution, and lexical/time bonus helpers
for time-related questions. Actual storage and retrieval are now implemented by
``mem0_backend_store.py`` (the real mem0/Qdrant backend); this file no longer holds the
storage layer itself, only these utilities still reused by that backend and upper layers.

Default store root: ``memory/leftbrain/`` under the repo root. Can be overridden with the
``SUPERMEM_MEMORY_ROOT`` environment variable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable
from supermem.llm_config import resolve_api_key, resolve_model

# This file lives in supermem/leftbrain/, parents[2] == the repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MEMORY_SQLITE = "supermem_leftbrain.sqlite"
_DEFAULT_LEFTBRAIN_MEMORY_ROOT = _REPO_ROOT / "memory" / "leftbrain"


def default_memory_root() -> Path:
    """Root directory for local memory files.

    Priority: ``SUPERMEM_MEMORY_ROOT`` -> repo ``memory/leftbrain/``.
    """
    env = (os.environ.get("SUPERMEM_MEMORY_ROOT") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    _DEFAULT_LEFTBRAIN_MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_LEFTBRAIN_MEMORY_ROOT.resolve()


def default_local_memory_db_path() -> Path:
    """Default SQLite path: ``default_memory_root() / supermem_leftbrain.sqlite``."""
    return default_memory_root() / _DEFAULT_MEMORY_SQLITE


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- Lexical + question-type retrieval boost --------------------------------------
# Pure cosine handles "how long / when" questions poorly: the question contains no words
# like years/months, yet the answer sits exactly in the memories carrying a duration or
# date, whose vectors aren't close, so they get buried dozens of places down.
# Here two near-zero-cost signals are layered on top of cosine (pure regex, no extra LLM
# call, no re-ingest):
#   1) lexical overlap -- how many of the question's content words (art / hike / friends)
#      appear in the memory text;
#   2) question type -- "how long" boosts memories with a duration expression, "when"
#      boosts memories with a date.

_DURATION_RE = re.compile(
    r"\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)[\s-]*"
    r"(?:year|month|week|day|hour|minute|decade)s?\b"
    r"|\bsince\s+(?:19|20)\d{2}\b"
    r"|\bhalf\s+a\s+(?:year|month)\b",
    re.I,
)
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}-\d{2}-\d{2}\b"
    r"|\b(?:last|next)\s+(?:year|month|week)\b",
    re.I,
)
_DURATION_Q_RE = re.compile(
    r"\bhow\s+long\b|\bhow\s+many\s+(?:year|month|week|day|hour)s?\b", re.I
)
_DATE_Q_RE = re.compile(
    r"\bwhen\b|\bwhat\s+(?:date|day|time)\b"
    # Relative time words. "what do I have next week?" used to not be recognized as a
    # time question, so _widen_for_time_question never fired -- once an appointment was
    # filtered out by slot (a dinner plan belongs to relationships, but the question is
    # about schedule) it could never be recovered; in testing only two of three
    # next-week appointments survived.
    r"|\b(?:today|tonight|tomorrow|yesterday|(?:this|next|last)\s+week)\b"
    r"|\b(?:coming\s+days|next\s+few\s+days|schedule|agenda|plans)\b",
    re.I,
)

# High-frequency words with no discriminative power in questions; including them in lexical matching only adds noise
_STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at for from by with about as into over
    is are was were be been being do does did have has had can could will would
    shall should may might must what when where who whom which why how many much
    long time you your their his her its it they them he she we us our i me my
    that this these those there here not no yes so than then get got make made
    """.split()
)

_LEX_WEIGHT = 0.15   # bonus when all content words match (cosine itself is ~0.2-0.6)
_TIME_WEIGHT = 0.10  # extra bonus when the question type matches
_DATE_MATCH_WEIGHT = 0.12  # bonus when a date expanded from the question matches a date in the memory text

#: Date literals like "August 26, 2026". This is how extraction normalizes facts,
#: and how ``time_expand`` writes expanded dates.
_QUERY_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+(?:19|20)\d{2}\b"
)


def query_dates(query: str) -> frozenset[str]:
    """Date literals appearing in the question.

    Most questions contain no date themselves -- ``time_expand.expand_relative_dates()``
    expands "next week" into those seven days and appends them. They are extracted here
    for exact comparison in ``date_overlap_bonus``.
    """
    return frozenset(_QUERY_DATE_RE.findall(query or ""))


def date_overlap_bonus(q_dates: frozenset[str], mem_text: str) -> float:
    """Whether a date in the memory text falls within the days the question refers to.

    Expanded dates only go into the vector, and cosine is weak at telling "is this date in
    range": in testing, "what do I have next week?" ranked only two of three next-week
    appointments, pushed out by entries dated outside next week but semantically closer
    (interview 8/22, cafe 8/20). This adds an exact literal comparison -- the bonus is for
    the hard fact "in range", not for similarity.
    """
    if not q_dates:
        return 0.0
    return _DATE_MATCH_WEIGHT if any(d in mem_text for d in q_dates) else 0.0


def time_question_kind(query: str) -> str | None:
    """Whether the question asks about time: ``"duration"`` (how long) / ``"date"`` (when) / ``None``."""
    if _DURATION_Q_RE.search(query):
        return "duration"
    if _DATE_Q_RE.search(query):
        return "date"
    return None


def _content_words(text: str) -> set[str]:
    """Extract content words usable for lexical matching: lowercased, stopwords removed, length >= 3."""
    return {
        w for w in re.findall(r"[a-z0-9']+", text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def _lexical_time_bonus(q_words: set[str], want_dur: bool, want_date: bool,
                        mem_text: str) -> tuple[float, bool]:
    """"Lexical overlap + time type" bonus for one memory; returns ``(bonus, whether the time type matched)``."""
    if not q_words:
        return 0.0, False
    overlap = len(q_words & _content_words(mem_text)) / len(q_words)
    bonus = _LEX_WEIGHT * overlap
    # The time bonus only goes to memories related to the question, otherwise every dated entry in the store would be lifted too
    time_hit = bool(overlap > 0 and (
        (want_dur and _DURATION_RE.search(mem_text))
        or (want_date and _DATE_RE.search(mem_text))
    ))
    if time_hit:
        bonus += _TIME_WEIGHT
    return bonus, time_hit


@runtime_checkable
class TextEmbedder(Protocol):
    """Batch-encode texts into vectors (matching the storage dimensions)."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class MemorySearchHit:
    memory_id: str
    text: str
    score: float
    attributed_to: str
    metadata: dict[str, Any]
    # Pure cosine before bonuses. Upper layers use it to tell "semantically relevant anyway"
    # from "only surfaced by lexical/time bonuses", so bonuses are used only to rescue buried
    # memories, not to push out the most semantically relevant ones.
    base_score: float = 0.0
    # This hit matches the time type asked about ("how long" and it has a duration expression / "when" and it has a date)
    time_boost: bool = False
    #: **Which day** the event in this memory happened (YYYY-MM-DD, from Ingest's observed_at).
    #: Without it, questions like "did this happen before that?" have no basis -- fact text
    #: usually only has relative phrases like "last week" or "over a month ago", from which
    #: order can't be inferred without an absolute date.
    observed_at: str = ""


@dataclass
class OpenAILocalEmbedderConfig:
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    dimensions: int | None = None  # skip probe if known

    def resolved_model(self) -> str:
        return resolve_model(self.model, "embedding")


class OpenAILocalEmbedder:
    """OpenAI Embeddings API (dev default ``text-embedding-3-small``)."""

    def __init__(self, config: OpenAILocalEmbedderConfig | None = None) -> None:
        self._cfg = config or OpenAILocalEmbedderConfig()
        self._model = self._cfg.resolved_model()
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Local vectors require: pip install openai>=1.0") from e

        api_key = resolve_api_key(self._cfg.api_key)
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY (or OpenAILocalEmbedderConfig(api_key=...))")

        # timeout must be set explicitly: the openai client's default timeout is 10 minutes,
        # and one hung request would stall the whole pipeline
        kw: dict[str, Any] = {"api_key": api_key, "timeout": 60.0, "max_retries": 2}
        if self._cfg.base_url:
            kw["base_url"] = self._cfg.base_url
        self._client = OpenAI(**kw)

        # Use pre-known dimensions to skip the probe API call
        self._dims = self._cfg.dimensions if self._cfg.dimensions else self._probe_dimensions()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dims

    def _probe_dimensions(self) -> int:
        v = self.embed_texts(["probe"])[0]
        return len(v)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # The same text is requested repeatedly within one ingest (fact 3 times, entities
        # 2 rounds, slot descriptions recomputed each round), so cache it. Vectors are
        # deterministic, so semantics are unchanged.
        from supermem.utils.common import embed_cache
        return embed_cache.resolve(self._model, texts, self._embed_uncached)

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        kw = {"model": self._model, "input": texts, "encoding_format": "float"}
        # When going through OpenRouter, pin the provider to OpenAI: in testing it occasionally
        # routed text-embedding-3-small to Google (gemini-embedding, 3072 dims), changing the
        # vector dimension of the whole store, after which every query failed with "shapes not
        # aligned". allow_fallbacks=False prefers an error over switching providers.
        if "openrouter" in str(getattr(self._client, "base_url", "") or "").lower():
            kw["extra_body"] = {"provider": {"order": ["OpenAI"], "allow_fallbacks": False}}
        r = self._client.embeddings.create(**kw)
        _exp = int(os.environ.get("SUPERMEM_EMBED_DIM", "1536"))
        if r.data and len(r.data[0].embedding) != _exp:
            raise RuntimeError(
                f"embedding dimension {len(r.data[0].embedding)} != expected {_exp} (model={self._model}, "
                f"response model={getattr(r, 'model', '?')}) -- provider was switched, refusing to write to avoid polluting the store")
        # Some OpenAI-compatible backends (e.g. Gemini) return index=None in batches;
        # only reorder by index when all are present, otherwise keep the API's order
        data = r.data
        if all(d.index is not None for d in data):
            data = sorted(data, key=lambda d: d.index)
        return [list(map(float, row.embedding)) for row in data]


def mock_embedder(dim: int = 8, seed: int = 0) -> TextEmbedder:
    """Deterministic fake vectors, for unit tests only (no API needed)."""

    class _Mock:
        model_name = "mock-deterministic"
        dimensions = dim

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            out: list[list[float]] = []
            state = seed
            for s in texts:
                vec: list[float] = []
                x = state + sum(ord(c) for c in s[:200])
                for j in range(dim):
                    x = (1103515245 * x + 12345) % (2**31)
                    vec.append(float(x % 1000) / 1000.0)
                state = x
                out.append(vec)
            return out

    return _Mock()
