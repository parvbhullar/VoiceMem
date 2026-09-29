"""LeftBrain component.

The **entire left-brain block** extracted from the SuperMem god class: cognitive-graph
slot filtering / entity narrowing / candidate widening for time questions / vector
ranking / query classification (with dynamic slot drill-down) / LLM slot tagging /
slot->entity graph-layer writes / subgraph activation bookkeeping and checkpoint /
schema description refresh / cold-memory archiving, plus the left-brain lazy-loaded
singletons around them (repo / extractor / dynamic_slot_store /
graph_entity_store / subgraph_manager).

Follows mem0's composition pattern:
  * **The component owns its parts** -- the 5 left-brain lazy singletons (repo /
    extractor / dynamic_slot_store / graph_entity_store / subgraph_manager), together
    with the cache and lock they share with the host, live inside this component;
    the engine no longer holds its own _get_* for them.
  * **Dependencies are injected explicitly** -- anything that needs a **cross-domain or
    runtime** capability ("text embedding / LLM(JSON) / LLM(text) / injectable
    classifier / session tracker") gets it injected in __init__ as a getter/function
    reference (lazy-loading semantics unchanged) and calls it via self._dep().

Logic is unchanged word for word: method bodies were moved as-is; only "how
dependencies are obtained" changed.

brain.py does not import engine (to avoid a cycle) -- module-level helpers
(_search_mode / _pool_mode / _RESCUE_K etc.) live in this module and engine imports
them back from here.
"""

from __future__ import annotations

from supermem.utils.common import space as _space

import os
import threading
from pathlib import Path
from typing import Any, Callable

from supermem.leftbrain.cognitive_graph.slot_v2 import SLOT_RELATIONS
from supermem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
from supermem.leftbrain.local_memory_store import MemorySearchHit


# ── Module-level constants / helpers ───────────────────────────────────────────

# Max number of memories "rescued" by lexical/time bonuses (given on top of top_k, not taking semantic slots)
_RESCUE_K = 3


# ── Recency weight ────────────────────────────────────────────────────────────
# Before this, retrieval ranking looked **only at semantic similarity**: something from
# three months ago and something said yesterday were weighted exactly the same.
# (The store's heat decay was implemented and accumulates on every hit, but its only
#  downstream is an archiving function nobody in the repo calls, so zero effect on ranking.)
#
# Two deliberate limits, both to avoid breaking what already works:
#   · Only re-rank **candidates already retrieved** -- recall is unchanged, so we never
#     swap an irrelevant old memory for an irrelevant new one; we only lean toward
#     recent ones when similarities are close.
#   · Decay has a floor -- long-term attributes like "allergic to nuts" or "doesn't eat
#     spicy food" come up only every few months, but are critical every time, and should
#     not be pushed out by last week's trivia just for being old. So the weight bottoms
#     out at RECENCY_FLOOR and never approaches 0.
#
# Uses observed_at (the day the thing **happened**), not ingestion time: a note added
# today about something last month should count as last month, not today.
#: Half-life (days). Set to 0 or negative to disable recency weighting.
RECENCY_HALFLIFE_DAYS = float(os.environ.get("SUPERMEM_RECENCY_HALFLIFE_DAYS", "30"))
#: Decay floor; decides how much say time can have at most.
#: The weight range is [FLOOR, 1], so **the largest similarity gap that can be flipped is 1/FLOOR**:
#:   0.60 -> 1.67x (even a 67% higher similarity gets pushed out; too aggressive)
#:   0.75 -> 1.33x (default: breaks near-ties, can't overturn clearly more relevant ones)
#:   0.90 -> 1.11x (almost only matters on exact ties)
#: For a memory system, recalling the wrong thing is worse than recalling an old thing, so pick the conservative tier.
RECENCY_FLOOR = float(os.environ.get("SUPERMEM_RECENCY_FLOOR", "0.75"))


#: Apply recency weighting only for "how have things been lately" style questions.
#:
#: At first **every query** was multiplied by the recency weight, and it broke in testing:
#: the correct answer to "where do I study"
#: ("Jiaqi is an undergraduate in computer science at NUS", recorded half a year ago)
#: had the top similarity, 0.810; after multiplying by 0.752 it dropped to 0.609 and was
#: pushed to sixth place by a 0.773 memory "user prefers to communicate in Chinese".
#: The reason: **answers about long-term attributes are naturally old** -- major, allergies,
#: hometown, personality have no newer version, so penalising old means penalising correct.
#: Only questions like "what have you been busy with lately" should really favour new ones.
#: So we split by question: weight when asking about recency, not when asking about attributes.
#:
#: Contradicting facts ("interning at TikTok" / "interning at NVIDIA") are not resolved here --
#: both go into the prompt with dates and the reply model picks by time; that path is more
#: accurate and has no side effects.
_RECENCY_CUES = (
    "recent", "lately", "today", "yesterday", "just now", "right now",
    "currently", "these days", "this week", "last week", "latest",
)


def wants_recency(query: str) -> bool:
    """Whether this question asks about "recent status" or an "attribute"."""
    q = (query or "").lower()
    return any(c in q for c in _RECENCY_CUES)


def _recency_weight(observed_at: str) -> float:
    """The more recently it happened, the higher the weight; no date means no discount (treated as a time-independent attribute)."""
    if RECENCY_HALFLIFE_DAYS <= 0 or not observed_at:
        return 1.0
    from datetime import date, datetime
    try:
        d = datetime.fromisoformat(str(observed_at)[:10]).date()
    except (TypeError, ValueError):
        return 1.0                       # can't parse the date -> don't guess, apply no discount
    age = (date.today() - d).days
    if age <= 0:                         # today or (bad data) the future
        return 1.0
    return RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * (0.5 ** (age / RECENCY_HALFLIFE_DAYS))


# Candidate pool construction mode (SUPERMEM_POOL_MODE):
#   union  -- slot pool ∪ macro-related slot pool ∪ entity pool ∪ one-hop neighbour pool.
#   strict -- schema routing -> entity narrowing -> graph expansion: on an entity hit take
#             slot pool ∩ (entities ∪ one-hop neighbours) (if the intersection is too small
#             fall back to the entity pool, and only if that is empty fall back to the slot
#             pool); macro slot spreading only kicks in when the slot pool is smaller than
#             _STRICT_MACRO_MIN_POOL.
_POOL_MODE_ENV = "SUPERMEM_POOL_MODE"
_STRICT_MIN_INTERSECTION = 3      # use the intersection only if it has at least this many
_STRICT_MACRO_MIN_POOL = 30       # do macro slot spreading only when the slot pool is smaller than this


def _pool_mode() -> str:
    return os.environ.get(_POOL_MODE_ENV, "union").strip().lower()


def _search_mode(slot_ids: set, final_ids: set) -> str:
    if not slot_ids and not final_ids:
        return "fallback"
    if final_ids and final_ids < slot_ids:
        return "entity+slot-intersection"
    if final_ids == slot_ids:
        return "slot-only"
    return "entity+slot-union"


# ── LeftBrain component ────────────────────────────────────────────────────────

#: Lexical signals that the question asks "what did you (the assistant) say before". Zero LLM --
#: this check sits on the speculative-prefetch path and can't afford a network request.
_ASKS_ASSISTANT = (
    "you said", "you told", "you mentioned", "you recommend", "you suggest",
    "did you say", "did you tell", "you talked about", "you promised",
)


def asks_about_assistant(query: str) -> bool:
    """Whether this question asks about something the assistant itself said."""
    q = (query or "").lower()
    return any(w in q or w in (query or "") for w in _ASKS_ASSISTANT)


#: If the character-trigram Jaccard of two memories exceeds this, they "say the same thing".
#: 0.30 was measured on a real 47-memory store -- every pair >= 0.20 was **always** a different
#: wording of the same thing, with no false positives; genuinely different pairs fell below 0.07.
#: Measured samples (originally Chinese text, translated here):
#:   0.658  "Feeling very tired today, maybe related to tomorrow afternoon's meeting..." / "Feeling very tired lately, maybe related to tomorrow afternoon's meeting..."
#:   0.317  "Doesn't like crowded places, prefers quiet chats with one or two friends" / "Doesn't like crowded places, gets tired after long parties..."
#:   0.046  "Recently started working out regularly..." / "Pulled three all-nighters, the demo finally runs..."        <- genuinely different
#: 0.30 rather than lower leaves headroom for other stores, where different memories may look more alike on the surface.
_DEDUPE_JACCARD = 0.30


def _trigrams(text: str) -> set[str]:
    t = "".join(ch for ch in str(text or "") if ch.strip())
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))} or {t}


def _dedupe_near(hits: list) -> list:
    """Keep hits from highest to lowest score, skipping any too similar to one already kept.

    Extraction produces several similarly worded memories for the same thing, with nearly
    identical scores. Without dedup, two or three of the top-5 say the same thing, and the
    user's impression is "it only remembers this little".
    """
    kept, grams = [], []
    for h in hits:
        g = _trigrams(getattr(h, "text", ""))
        dup = False
        for g0 in grams:
            inter = len(g & g0)
            if inter and inter / len(g | g0) >= _DEDUPE_JACCARD:
                dup = True
                break
        if not dup:
            kept.append(h)
            grams.append(g)
    return kept


class LeftBrain:
    """Left-brain component that owns its parts and takes dependencies by explicit injection.

    Constructor parameters come in two kinds:

    **Runtime parameters** owned by the component (left-brain paths/identity)::

        memory_root, user_id, base_url, cognitive_db

    Explicitly **injected cross-domain/runtime dependencies** (all passed as getter/function
    references; lazy-loading semantics unchanged)::

        embed              -> self._embed_text          text embedding (slot/graph-layer writes)
        llm_json           -> self._llm_json            LLM(JSON) (tagging/subgraph/classification fallback)
        llm_text           -> self._llm_text            LLM(text) (schema description refresh)
        classifier         -> self._classifier          injectable query classifier (used by Classify)
        tracker            -> self._get_session_tracker  session tracker shared by left/right brain (bookkeeping/checkpoint)

    The 5 left-brain lazy singletons (repo / extractor / dynamic_slot_store /
    graph_entity_store / subgraph_manager), together with the cache/lock shared with the
    host, are owned by this component; see self._get_repo() etc. below.
    """

    def __init__(
        self,
        *,
        memory_root: Path,
        user_id: str,
        base_url: str | None,
        cognitive_db: Path,
        embedder: Any,
        vector_store: Any,
        embed: Callable[[str], list[float]],
        llm_json: Callable[[str], str],
        llm_text: Callable[..., str],
        classifier: Any,
        tracker: Callable[[], Any],
        cache: dict[str, Any] | None = None,
        lock: Any = None,
    ) -> None:
        # ── Runtime parameters ──
        self._memory_root = memory_root
        self._user_id = user_id
        self._base_url = base_url
        self._cognitive_db = cognitive_db
        # The two injectable building blocks repo construction needs (embedder / vector_store), aligned with the host.
        self._embedder = embedder
        self._vector_store = vector_store

        # ── Injected cross-domain/runtime dependencies (getter/function references) ──
        self._embed_text = embed
        self._llm_json = llm_json
        self._llm_text = llm_text
        self._classifier = classifier
        self._get_session_tracker = tracker

        # ── Cache of left-brain parts owned by the component ──
        # The host may share the same cache/lock (left-brain lazy singletons and the host's
        # _get_* land in the same dict, so existing call sites/tests that read/write the
        # host's _cache directly see the same view as this component).
        self._cache: dict[str, Any] = cache if cache is not None else {}
        self._lock = lock if lock is not None else threading.Lock()

    # ── Left-brain lazy singletons ──────────────────────────────────────────────

    def _get_repo(self):
        with self._lock:
            if "repo" not in self._cache:
                from supermem.leftbrain.cognitive_graph import CognitiveAnnotator, CognitiveAnnotatorConfig
                from supermem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
                from supermem.leftbrain.memory_repository_v2 import LeftBrainMemoryRepositoryConfig, LeftBrainMemoryRepositoryV2
                annotator = CognitiveAnnotator(CognitiveAnnotatorConfig(base_url=self._base_url))
                embedder  = self._embedder or OpenAILocalEmbedder(OpenAILocalEmbedderConfig(base_url=self._base_url))
                cfg = LeftBrainMemoryRepositoryConfig(
                    json_path=_space.json_path(self._memory_root),
                    db_path=_space.db(self._memory_root),
                    cognitive_db_path=self._cognitive_db,
                    enable_cognitive_graph=True,
                    base_url=self._base_url,
                )
                self._cache["repo"] = LeftBrainMemoryRepositoryV2(
                    embedder, config=cfg, cognitive_annotator=annotator,
                    vector_store=self._vector_store,
                )
        return self._cache["repo"]

    def _get_extractor(self):
        with self._lock:
            if "extractor" not in self._cache:
                from supermem.leftbrain.extract_facts_openai import (
                    OpenAIAdditiveExtractorConfig,
                    OpenAIMem0V3AdditiveExtractor,
                )
                self._cache["extractor"] = OpenAIMem0V3AdditiveExtractor(
                    OpenAIAdditiveExtractorConfig(base_url=self._base_url)
                )
        return self._cache["extractor"]

    def _get_dynamic_slot_store(self):
        with self._lock:
            if "dynamic_slot_store" not in self._cache:
                from supermem.leftbrain.slot_split import DynamicSlotStore
                self._cache["dynamic_slot_store"] = DynamicSlotStore(
                    _space.db(self._memory_root)
                )
        return self._cache["dynamic_slot_store"]

    def _get_dynamic_slots(self) -> list[tuple[str, str]]:
        """Return this user's emerged dynamic slots [(name, description), ...]."""
        try:
            return [(s.name, s.description)
                    for s in self._get_dynamic_slot_store().get_dynamic_slots(self._user_id)]
        except Exception:
            return []

    def _get_graph_entity_store(self):
        with self._lock:
            if "graph_entity_store" not in self._cache:
                from supermem.leftbrain.slot_split import GraphEntityStore
                self._cache["graph_entity_store"] = GraphEntityStore(
                    _space.db(self._memory_root)
                )
        return self._cache["graph_entity_store"]

    def _get_subgraph_manager(self):
        graph_store = self._get_graph_entity_store()   # fetch outside the lock first to avoid a nested acquire
        dyn_store = self._get_dynamic_slot_store()
        with self._lock:
            if "subgraph_manager" not in self._cache:
                from supermem.leftbrain.slot_split import SubgraphManager

                def _tag_new_slot(user_id: str, memory_id: str, slot_name: str) -> None:
                    cog_store = self._get_repo()._cognitive_store
                    if cog_store is not None and hasattr(cog_store, "upsert_memory_tags"):
                        cog_store.upsert_memory_tags(memory_id, user_id, [(slot_name, 0.9)])

                self._cache["subgraph_manager"] = SubgraphManager(
                    graph_store, dyn_store, llm_fn=self._llm_json, tag_fn=_tag_new_slot,
                )
        return self._cache["subgraph_manager"]

    # ── Step 1: slot filtering ─────────────────────────────────────────────────

    def SearchCogGraph(
        self,
        slots: list[str],
        entities: list[str] | None = None,
        scene_filter: str | None = None,
        speaker_filter: str | None = None,
    ) -> tuple[set[str], QueryClassification]:
        """Slot filtering; returns all memory IDs under the given slots.

        Parameters
        ----------
        slots:
            Slot list provided by the voice module, e.g. ``["work"]``.
        entities:
            Entity list provided by the voice module, e.g. ``["Alibaba"]``. May be empty.
        scene_filter:
            Optional scene filter (audiomem), e.g. ``"office"``.
        speaker_filter:
            Optional speaker filter (audiomem); pass a person_id (e.g. ``"person_3a2f1b"``).
            Only memories spoken by that speaker are returned.

        Returns
        -------
        (slot_mem_ids, classification)
            ``slot_mem_ids`` -- set of candidate memory_ids.
            ``classification`` -- data container wrapping slots and entities.
        """
        classification = QueryClassification(
            slots=slots,
            entities=entities or [],
        )
        store = self._get_repo()._cognitive_store

        # Union of the memory pools of all slots: Classify() returns both precise child slots and broad parent slots.
        slot_mem_ids: set[str] = set()
        if classification.slots and store is not None and hasattr(store, "memory_ids_for_slots_v2"):
            slot_mem_ids = set(store.memory_ids_for_slots_v2(self._user_id, classification.slots))

        # Scene filter (audiomem): intersect memories tagged scene:<tag> with the slot result.
        # "unknown" is not a scene, it means "not detected" -- block it defensively here too,
        # in case another caller passes it in as a valid value.
        if scene_filter and scene_filter != "unknown" and slot_mem_ids:
            try:
                if store and hasattr(store, "memory_ids_for_slots_v2"):
                    scene_ids = set(
                        store.memory_ids_for_slots_v2(
                            self._user_id, [f"scene:{scene_filter}"]
                        )
                    )
                    narrowed = slot_mem_ids & scene_ids
                    if narrowed:
                        slot_mem_ids = narrowed
            except Exception:
                pass

        # Speaker filter (audiomem): intersect the speaker:<person_id> tag with the slot result.
        # When slot_mem_ids is empty (no slot given), use that speaker's memories as the base candidate pool.
        if speaker_filter:
            try:
                if store and hasattr(store, "memory_ids_for_slots_v2"):
                    spk_ids = set(
                        store.memory_ids_for_slots_v2(
                            self._user_id, [f"speaker:{speaker_filter}"]
                        )
                    )
                    if slot_mem_ids:
                        narrowed = slot_mem_ids & spk_ids
                        if narrowed:
                            slot_mem_ids = narrowed
                    elif spk_ids:
                        slot_mem_ids = spk_ids
            except Exception:
                pass

        return slot_mem_ids, classification

    # ── Step 2: entity matching (pure cognitive graph, no vectors) ─────────────

    def SearchData(
        self,
        slot_mem_ids: set[str],
        classification: QueryClassification,
    ) -> set[str]:
        """Narrow slot_mem_ids by entity names and return the final candidate ID set.

        Pure cognitive-graph operation: does not call the vector store and needs no raw query.

        Parameters
        ----------
        slot_mem_ids:
            Slot candidate ID set returned by SearchCogGraph.
        classification:
            Classification result returned by SearchCogGraph (its entities field is used).

        Returns
        -------
        set[str]
            Final candidate ID set.
            - with entities -> slot ∪ entity
            - without entities -> slot_mem_ids returned as-is
        """
        final_ids, _activated_names = self._search_data_impl(slot_mem_ids, classification)
        return final_ids

    def _search_data_impl(
        self, slot_mem_ids: set[str], classification: QueryClassification,
    ) -> tuple[set[str], list[str]]:
        """The real implementation of SearchData(); additionally returns "the list of entity
        names the left brain actually activated" (fuzzy-match hits + one-hop neighbour
        spreading), which Search() passes on to the right brain internally.

        This differs from classification.entities (literal entity mentions in the query
        text) -- the right brain relies on the entity set the left-brain retrieval pipeline
        actually confirmed/spread to. The public SearchData() method returns only memory
        ids, keeping the original step-by-step pipeline contract unchanged.
        """
        store = self._get_repo()._cognitive_store
        if not classification.entities or store is None:
            return set(slot_mem_ids), []

        entity_mids: set[str] = set()
        matched_entity_ids: set[str] = set()
        activated_names: list[str] = []
        if hasattr(store, "find_entities_by_name_fuzzy"):
            for ent_name in classification.entities:
                ents = store.find_entities_by_name_fuzzy(self._user_id, ent_name)
                if ents:
                    ids = [e.id for e in ents]
                    matched_entity_ids.update(ids)
                    activated_names.extend(e.name for e in ents)
                    mids = store.memory_ids_for_entities(ids)
                    entity_mids.update(mids)

        # One-hop neighbour spreading: also merge memories of the directly matched entities'
        # one-hop neighbours (entity_edges) into the candidate pool, unweighted; ranking is
        # left to Rank()'s vector similarity. Neighbours also count toward activated_names.
        if matched_entity_ids and hasattr(store, "neighbor_entity_ids"):
            neighbor_ids = store.neighbor_entity_ids(self._user_id, list(matched_entity_ids))
            if neighbor_ids:
                entity_mids.update(store.memory_ids_for_entities(neighbor_ids))
                for nid in neighbor_ids:
                    ne = store.get_entity(nid)
                    if ne:
                        activated_names.append(ne.name)

        if not entity_mids:
            return set(slot_mem_ids), activated_names

        if slot_mem_ids:
            if _pool_mode() == "strict":
                # entity narrowing: intersect the entity pool with the slot pool; if the intersection is too small, trust entities over slots.
                inter = entity_mids & slot_mem_ids
                if len(inter) >= _STRICT_MIN_INTERSECTION:
                    return inter, activated_names
                return entity_mids, activated_names
            return entity_mids | slot_mem_ids, activated_names
        return entity_mids, activated_names

    # ── Step 2.5: widen candidates for time questions ─────────────────────────

    def _widen_for_time_question(self, query: str, final_ids: set[str]) -> set[str]:
        """For "how long / when" questions, merge memories containing duration or date expressions into the candidate pool.

        Entities and slots are indexed by semantic content and can't catch time expressions.
        Here we do an extra regex scan of the store based on question type and merge memories
        with duration/date expressions into the candidates. If final_ids is empty the
        full-store fallback is used, so no widening.
        """
        if not final_ids:
            return final_ids
        from supermem.leftbrain.local_memory_store import time_question_kind

        kind = time_question_kind(query)
        if kind is None:
            return final_ids
        store = self._get_repo()._vector_store
        if not hasattr(store, "memory_ids_with_time_expr"):
            return final_ids
        extra = store.memory_ids_with_time_expr(self._user_id, kind=kind)
        return (final_ids | extra) if extra else final_ids

    # ── Step 3: vector ranking ────────────────────────────────────────────────

    def Rank(
        self,
        query: str,
        candidate_ids: set[str],
        top_k: int = 5,
        speaker_filter: str | None = None,
    ) -> list[MemorySearchHit]:
        """Rank by vector similarity within candidate_ids and return the top-N memories."""
        fetch_k = max(top_k * 3, 20)   # fetch more candidates for the full-store fallback
        repo = self._get_repo()
        # Assistant utterances are not recalled by default; only let them in when the question asks "what did you say before".
        want_assistant = asks_about_assistant(query)

        if candidate_ids:
            # Slot allocation is left to the storage layer: top_k by pure cosine, plus up to
            # _RESCUE_K extra rescued by lexical/time bonuses (must run on the full candidate
            # set to avoid losing scores to a second truncation).
            hits = repo._vector_store.search(
                query,
                user_id=self._user_id,
                top_k=top_k * 3,          # fetch extra, truncate after dedup (see _dedupe_near)
                rescue_k=_RESCUE_K,
                memory_id_filter=candidate_ids,
                include_assistant=want_assistant,
            )
            # If short, top up from the full store -- but not when filtering by speaker (that
            # would mix in other people's memories); in that case fewer than top_k is preferable.
            if len(hits) < top_k and not speaker_filter:
                seen = {h.memory_id for h in hits}
                for h in repo.search(query, user_id=self._user_id, top_k=fetch_k,
                                     include_assistant=want_assistant):
                    if h.memory_id not in seen:
                        hits.append(h)
                        seen.add(h.memory_id)
                        if len(hits) >= top_k:
                            break
        else:
            hits = repo.search(query, user_id=self._user_id, top_k=fetch_k,
                               include_assistant=want_assistant)[:top_k]

        # Remove near-duplicates, then truncate. Extraction produces several similarly worded
        # memories for the same thing ("Feeling very tired today, maybe related to tomorrow
        # afternoon's meeting" / "Feeling very tired lately, maybe related to tomorrow
        # afternoon's meeting") with nearly identical scores, so two or three of the top-5 say
        # the same thing -- the user's impression is "it only remembers this little".
        # Truncate after dedup. Only "recent status" questions get recency re-ranking (see the
        # note above wants_recency); "attribute" questions stay pure similarity -- the correct
        # answer for long-term attributes is naturally old, so penalising old means penalising
        # correct. Re-ranking happens only inside the candidate pool; recall is unchanged.
        deduped = _dedupe_near(hits)
        if wants_recency(query):
            deduped = sorted(deduped,
                             key=lambda h: h.score * _recency_weight(h.observed_at),
                             reverse=True)
        final_hits = deduped[:top_k]
        # Memory lifecycle: retrieval hits add heat; on read it decays exponentially by last_hit_at, and low-heat memories get archived.
        cog_store = repo._cognitive_store
        if cog_store is not None and hasattr(cog_store, "record_memory_hits"):
            try:
                cog_store.record_memory_hits([h.memory_id for h in final_hits])
            except Exception as e:
                print(f"[MemoryHeat] Failed to record: {e}")
        return final_hits

    # ── v5: LLM tagging (replaces embedding similarity) ───────────────────────

    # English aliases for the base-7 slots -- used to build short "slot / alias" anchor
    # texts for embedding. Only short-label-vs-short-label cosine similarity is high enough
    # to fold wording variants back into the same slot.
    _BASE_SLOT_ALIASES: dict[str, str] = {
        "work": "job", "finance": "money", "relationships": "family and friends",
        "health": "wellbeing", "goals": "plans", "daily_life": "daily life",
        "knowledge": "learning",
    }

    def _get_slot_base_embeddings(self) -> dict[str, list[float]]:
        """Embeddings of the base-7 slots, cached once. Keys use the literal enum values
        ("relationships"), not str(enum member) -- SlotV2.RELATIONSHIPS's __str__ is
        "SlotV2.RELATIONSHIPS", not "relationships", which would write back a nonexistent
        slot name after a fold hit."""
        with self._lock:
            if "slot_base_embeddings" not in self._cache:
                self._cache["slot_base_embeddings"] = {
                    value: self._embed_text(f"{value} / {alias}")
                    for value, alias in self._BASE_SLOT_ALIASES.items()
                }
        return self._cache["slot_base_embeddings"]

    def _get_slot_dyn_embeddings(self, dynamic: list[tuple[str, str]]) -> dict[str, list[float]]:
        """Embeddings of emerged dynamic slots, cached incrementally (computed only when a new slot appears)."""
        with self._lock:
            cache = self._cache.setdefault("slot_dyn_embeddings", {})
        for name, desc in dynamic:
            if name not in cache:
                cache[name] = self._embed_text(f"{name}: {desc}" if desc else name)
        return cache

    def _normalize_slot_name(
        self, candidate: str, known_all: set[str], dynamic: list[tuple[str, str]],
        threshold: float = 0.65,
    ) -> str:
        """When exact match fails, fold the candidate slot back into the closest known slot by
        semantic similarity (so translation/wording drift doesn't split one category in two);
        only treat it as a brand-new slot if nothing close is found.
        """
        if candidate in known_all:
            return candidate

        from supermem.leftbrain.slot_split.split_manager import cosine_sim
        cand_emb = self._embed_text(candidate)

        best_name, best_sim = None, -1.0
        for name, emb in self._get_slot_base_embeddings().items():
            sim = cosine_sim(cand_emb, emb)
            if sim > best_sim:
                best_sim, best_name = sim, name
        for name, emb in self._get_slot_dyn_embeddings(dynamic).items():
            sim = cosine_sim(cand_emb, emb)
            if sim > best_sim:
                best_sim, best_name = sim, name

        return best_name if best_name is not None and best_sim >= threshold else candidate

    def _llm_tag_memories(self, text: str, memory_ids: list[str]) -> list[str]:
        """Use the LLM to tag these memories with slots, picking only 1-2 from the known slots
        (fixed + already-built dynamic slots); the LLM may not invent new categories (new
        slots can only come from SubgraphManager's co-occurrence subgraph decisions).
        Returns the list of slot names actually applied.
        """
        import json as _json
        from supermem.leftbrain.cognitive_graph.slot_v2 import ALL_SLOT_V2_VALUES, SLOT_V2_DESCRIPTIONS

        dynamic = self._get_dynamic_slots()  # [(name, description), ...]
        dyn_names = {n for n, _ in dynamic}
        known_all = set(ALL_SLOT_V2_VALUES) | dyn_names

        # Build the slot list description
        slot_lines = [f"- {s}: {SLOT_V2_DESCRIPTIONS[s][:60]}" for s in ALL_SLOT_V2_VALUES]
        if dynamic:
            slot_lines += [f"- {n}: {d}" for n, d in dynamic]
        slot_desc = "\n".join(slot_lines)

        prompt = (
            f"The user said:\n\"{text}\"\n\n"
            f"Pick the 1-2 closest life domains from the list below (you must pick ones already "
            f"in the list; just pick the closest, do not invent new categories):\n{slot_desc}\n\n"
            'Output only JSON: {"slots": ["category1", "category2"]}'
        )
        raw = self._llm_json(prompt)
        if not raw:
            return []

        try:
            slots = _json.loads(raw).get("slots", [])
        except Exception:
            return []

        slots = [s.strip() for s in slots if s.strip()][:2]
        if not slots:
            return []

        # Candidates that fail exact match are first folded back into known slots by semantic
        # similarity; any still not in the known list after folding (the LLM invented a new
        # name) are dropped -- creating new slots is left entirely to the subgraph mechanism.
        slots = [self._normalize_slot_name(s, known_all, dynamic) for s in slots]
        slots = [s for s in slots if s in known_all]
        slots = list(dict.fromkeys(slots))  # dedupe, keep order
        if not slots:
            return []

        cog_store = self._get_repo()._cognitive_store

        # Overwrite tags (replacing the old embedding-based tags)
        if cog_store and hasattr(cog_store, "upsert_memory_tags"):
            for mid in memory_ids:
                cog_store.upsert_memory_tags(
                    mid, self._user_id, [(s, 0.95) for s in slots]
                )
        return slots

    # ── Query classification (with dynamic slots) ─────────────────────────────

    def Classify(self, query: str) -> QueryClassification:
        """LLM-classify the query -> slots + entities, hierarchically:
        1. First choose only among base-7 (don't flatten all dynamic slots, so the list doesn't keep growing).
        2. For each chosen slot, drill one level into its child slots (split off by the subgraph
           mechanism); keep drilling while a more precise child exists, otherwise stop.
        3. Child slots reached are appended to the result and parent slots are kept -- parents
           secure recall, children add precision, and retrieval takes the union over slots.
        Entities are extracted only once, in step 1.
        """
        from supermem.leftbrain.cognitive_graph.query_slot_classifier import (
            QuerySlotClassifier, SlotClassifierConfig, QueryClassification,
        )
        # Injectable classifier (defaults to the built-in LLM version). Symmetric with embedder:
        # pass a local implementation to switch to a local model without touching the LLM/network
        # -- this step (slot + entity extraction) can now be OpenAI or local.
        clf = self._classifier or QuerySlotClassifier(SlotClassifierConfig(base_url=self._base_url))
        top = clf.classify(query)

        dyn_store = self._get_dynamic_slot_store()
        final_slots = []

        def _add(name: str) -> None:
            if name not in final_slots:
                final_slots.append(name)

        # Child-slot drill-down needs classifier support for classify_child (skipped entirely if the local version lacks it).
        _emergence_on = hasattr(clf, "classify_child")
        for slot in top.slots:
            _add(slot)
            if not _emergence_on:
                continue
            current = slot
            seen = {current}
            while True:
                children = dyn_store.get_children(self._user_id, current)
                children = [c for c in children if c.name not in seen]
                if not children:
                    break
                choice = clf.classify_child(
                    query, current, [(c.name, c.description) for c in children]
                )
                if choice is None:
                    break
                current = choice
                seen.add(current)
                _add(current)

        return QueryClassification(slots=final_slots, entities=top.entities)

    _SUBGRAPH_POOL_NS = "subgraph_pool"

    def _record_subgraph_activation(self, hits: list) -> None:
        """Bookkeeping for retrieval results: record the graph_entities of the hit memories into
        the session's subgraph candidate pool + query activation history (used by the
        cluster-emergence density formula). Cheap, no LLM call.
        Run automatically by Search() itself after every real retrieval.
        """
        memory_ids = {h.memory_id for h in hits}
        if not memory_ids:
            return
        tracker = self._get_session_tracker()
        for mid in memory_ids:
            tracker.touch(self._user_id, self._SUBGRAPH_POOL_NS, mid)
        try:
            graph_store = self._get_graph_entity_store()
            activated: set[str] = set()
            for mid in memory_ids:
                activated.update(e.id for e in graph_store.get_entities_for_memory(self._user_id, mid))
            if activated:
                import uuid as _uuid
                session_id = self._get_session_tracker().get_current_session(self._user_id)
                graph_store.record_query_activation(
                    self._user_id, _uuid.uuid4().hex, list(activated), session_id=session_id,
                )
        except Exception as e:
            print(f"[QueryActivation] Failed to record: {e}")

    def RunSubgraphCheckpoint(self) -> dict:
        """Take out (and clear) the whole accumulated memory_id list and do one real
        build-graph -> decide pass -- this is the "expensive" step of subgraph decisions; in the
        real product it should be called once at the end of each session.
        """
        tracker = self._get_session_tracker()
        memory_ids = set(tracker.pop_touched(self._user_id, self._SUBGRAPH_POOL_NS))
        if not memory_ids:
            return {"status": "no_memories"}

        cog_store = self._get_repo()._cognitive_store

        def _mem_lookup(mid: str) -> str | None:
            if cog_store is None:
                return None
            rec = cog_store.get_memory_record(mid)
            return rec.content if rec else None

        session_id = self._get_session_tracker().get_current_session(self._user_id)
        return self._get_subgraph_manager().run_for_retrieved_pool(
            self._user_id, memory_ids, memory_content_lookup=_mem_lookup, session_id=session_id,
        )

    def ArchiveColdMemories(
        self, *, min_age_days: float = 30.0, heat_threshold: float | None = None,
    ) -> dict:
        """The archiving step of the memory lifecycle: scan memories whose decayed heat is below
        the threshold and that have existed long enough, and archive them via mem0's
        expiration_date (mem0's search()/get_all() automatically hide expired memories). The
        decision lives in list_archivable_memories; this only executes it. An explicitly invoked
        batch operation, not run automatically on every Ingest()/Search().
        """
        cog_store = self._get_repo()._cognitive_store
        if cog_store is None or not hasattr(cog_store, "list_archivable_memories"):
            return {"status": "no_cognitive_store", "archived": []}

        from supermem.leftbrain.cognitive_graph.store import ARCHIVE_HEAT_THRESHOLD
        threshold = ARCHIVE_HEAT_THRESHOLD if heat_threshold is None else heat_threshold

        candidate_ids = cog_store.list_archivable_memories(
            self._user_id, min_age_days=min_age_days, heat_threshold=threshold,
        )
        if not candidate_ids:
            return {"status": "nothing_to_archive", "archived": []}

        vector_store = self._get_repo()._vector_store
        archived: list[str] = []
        for mid in candidate_ids:
            try:
                if hasattr(vector_store, "archive_memory") and vector_store.archive_memory(mid):
                    archived.append(mid)
            except Exception as e:
                print(f"[Archive] {mid} failed to archive: {e}")
        return {"status": "archived" if archived else "archive_failed", "archived": archived}

    # ── Left-brain retrieval entry point (the left-brain half of Search) ───────

    def search(
        self,
        query: str,
        slots: list[str] | None,
        entities: list[str] | None,
        scene_filter: str | None,
        speaker_filter: str | None,
    ) -> dict:
        """Left-brain retrieval stage: SearchCogGraph -> SearchData -> time widening -> related slot summaries.

        Does not include vector ranking (Rank) -- the orchestration that runs Rank concurrently
        with the right brain stays in engine.Search; after getting final_ids/activated_names
        from here, engine runs self._left.rank ‖ self._right.search concurrently. Returns a dict
        with the left-brain fields and per-step timestamps engine needs to assemble SearchResult.
        Logic is unchanged word for word from the left-brain steps of the original Search().
        """
        import time

        # (1) slot filtering
        t0 = time.time()
        slot_mem_ids, classification = self.SearchCogGraph(
            slots or [], entities, scene_filter=scene_filter, speaker_filter=speaker_filter,
        )
        t1 = time.time()

        # (2) entity narrowing -- finishes before the right brain. The right brain depends on the
        # left brain's "activated" entity set, so it must wait for _search_data_impl() to produce
        # the entities actually found/spread to in the left-brain graph.
        final_ids, activated_names = self._search_data_impl(slot_mem_ids, classification)
        final_ids = self._widen_for_time_question(query, final_ids)
        t2 = time.time()

        # Related slot summaries: prefer macro links learned automatically from data
        # co-occurrence; when the learned links are insufficient (cold start) fall back to the
        # static table (dynamic slots aren't in the static table, so link back to their parent slot).
        primary = classification.primary_slot()
        related_summaries: dict[str, str] = {}
        if primary:
            store = self._get_repo()._cognitive_store
            # All routed slots + up to 3 strongly linked slots of the primary slot, each with a one-line schema description.
            wanted: list[str] = list(classification.slots or [primary])
            related_slots: list[str] = []
            if store is not None and hasattr(store, "get_macro_related_slots"):
                related_slots = store.get_macro_related_slots(self._user_id, primary)
            if not related_slots:
                if primary in SLOT_RELATIONS:
                    related_slots = SLOT_RELATIONS[primary]
                else:
                    related_slots = self._get_dynamic_slot_store().get_parent_slots(self._user_id, primary)
            for r_ in related_slots[:3]:
                if r_ not in wanted:
                    wanted.append(r_)
            if wanted and store is not None and hasattr(store, "get_slot_summaries"):
                got = store.get_slot_summaries(self._user_id, wanted)
                related_summaries = {s_: got[s_] for s_ in wanted if got.get(s_)}

        return {
            "slot_mem_ids": slot_mem_ids,
            "final_ids": final_ids,
            "activated_names": activated_names,
            "classification": classification,
            "related_summaries": related_summaries,
            "t0": t0, "t1": t1, "t2": t2,
        }

    def rank(
        self, query: str, candidate_ids: set[str], top_k: int = 5,
        speaker_filter: str | None = None,
    ) -> list[MemorySearchHit]:
        """Vector ranking (the half of the Search orchestration that runs concurrently with the right brain); forwards to Rank."""
        return self.Rank(query, candidate_ids, top_k, speaker_filter=speaker_filter)

    def record_activation(self, hits: list) -> None:
        """Bookkeeping for retrieval results; forwards to _record_subgraph_activation."""
        return self._record_subgraph_activation(hits)

    # ── Left-brain writes ───────────────────────────────────────────────────────

    def ingest_facts(self, vi, *, registry, session_id, extra_metadata):
        """Left-brain fact extraction + storage: ingest_voice_input (synthesise messages -> extract -> write).
        registry is the audio side's voiceprint-to-name mapping, injected by engine during orchestration (cross-domain, not owned)."""
        from supermem.utils.common.voice_input import ingest_voice_input
        return ingest_voice_input(
            vi, self._user_id,
            registry=registry,
            repo=self._get_repo(),
            extractor=self._get_extractor(),
            session_id=session_id,
            extra_metadata=extra_metadata,
        )

    def write(self, result, text) -> None:
        """Left-brain write stage: LLM slot tagging + slot->entity graph-layer write."""
        if not result.memory_ids:
            return
        # LLM tagging (overrides embedding tags)
        try:
            llm_slots = self._llm_tag_memories(text, result.memory_ids)
            primary_slot = llm_slots[0] if llm_slots else None
        except Exception as e:
            print(f"[v5] LLM tagging failed: {e}", flush=True)
            primary_slot = None
            llm_slots = []

        # Semantic-cluster macro links: if this memory got 2+ slot tags, those slots are genuinely
        # related; learned automatically from data co-occurrence, not a hand-written relation table.
        if len(llm_slots) >= 2:
            try:
                self._get_repo()._cognitive_store.record_slot_cooccurrence(
                    self._user_id, llm_slots
                )
            except Exception as e:
                print(f"[SlotMacro] Failed to record co-occurrence: {e}", flush=True)

        # Left-brain slot->entity graph layer: attach this memory to entity nodes under the slot
        # (entity names reuse those CognitiveAnnotator already extracted; no extra LLM call)
        try:
            cog_store = self._get_repo()._cognitive_store
            graph_store = self._get_graph_entity_store()
            if cog_store is not None and primary_slot:
                for mid in result.memory_ids:
                    for eid in cog_store.entity_ids_for_memory(mid):
                        ent = cog_store.get_entity(eid)
                        if ent is None:
                            continue
                        ent_emb = self._embed_text(ent.name)
                        g_ent, _created = graph_store.get_or_create_entity_semantic(
                            self._user_id, primary_slot, ent.name, ent_emb,
                        )
                        graph_store.link_memory(g_ent.id, self._user_id, mid)
        except Exception as e:
            print(f"[GraphEntity] Left-brain graph-layer write failed: {e}")

    # ── Schema description refresh ──────────────────────────────────────────────

    _SCHEMA_DESC_MIN_NEW = 1      # rewrite the description only after >= N new memories in the slot
    _SCHEMA_DESC_MAX_FACTS = 80   # cap on summary input (most recent first)

    def refresh_schema(self) -> None:
        """Forwards to _refresh_schema_descriptions (public name)."""
        return self._refresh_schema_descriptions()

    def _refresh_schema_descriptions(self) -> None:
        """Rewrite a one-sentence overall description for slots whose memory count changed, and write
        it to the cognitive store's slot_summaries. Includes the date of the domain's most recent
        memory so temporal questions aren't misled by an undated summary."""
        repo = self._get_repo()
        cog = repo._cognitive_store
        if cog is None or not hasattr(cog, "memory_ids_for_slots_v2"):
            return
        entries = {}
        try:
            for e in repo._vector_store.list_entries(user_id=self._user_id):
                entries[e["id"]] = e
        except Exception:
            entries = {}
        slots = list(SLOT_RELATIONS.keys())
        try:
            slots += [d.name for d in self._get_dynamic_slot_store().get_dynamic_slots(self._user_id)]
        except Exception:
            pass
        for slot in dict.fromkeys(slots):
            try:
                mids = cog.memory_ids_for_slots_v2(self._user_id, [slot])
                n = len(mids)
                if n < 3:
                    continue
                last = cog.get_slot_summary_mem_count(self._user_id, slot) if hasattr(cog, "get_slot_summary_mem_count") else 0
                if n - last < self._SCHEMA_DESC_MIN_NEW:
                    continue
                facts = []
                for mid in mids:
                    e = entries.get(mid)
                    if e is not None and e["text"]:
                        facts.append((e["date"], e["text"]))
                    else:
                        rec = cog.get_memory_record(mid) if hasattr(cog, "get_memory_record") else None
                        if rec and rec.content:
                            facts.append(("", rec.content))
                if len(facts) < 3:
                    continue
                facts.sort(key=lambda t: t[0], reverse=True)
                facts = facts[: self._SCHEMA_DESC_MAX_FACTS]
                latest = next((d for d, _ in facts if d), "")
                sample = "\n".join(f"- {('[' + d + '] ') if d else ''}{t}" for d, t in facts)
                prompt = (
                    f"Below are memory facts about a user, all under the life domain '{slot}'.\n"
                    "Write ONE concise sentence (max 40 words) summarizing the overall picture in this domain: "
                    "the main people, ongoing situations, and how things changed over time. Plain and factual, "
                    "no fluff. Mention the most recent date if relevant. Output only the sentence.\n\n"
                ) + sample
                text = (self._llm_text(prompt) or "").strip()
                if not text:
                    continue
                if latest and latest not in text:
                    text = f"{text} (latest: {latest})"
                cog.upsert_slot_summary(self._user_id, slot, text, n)
            except Exception as e:
                print(f"[SchemaDesc] {slot} failed: {e}")

    # ── User name extraction ────────────────────────────────────────────────────

    def _get_user_name(self) -> str | None:
        """Extract the user's name from left-brain memories; cached once found."""
        with self._lock:
            if "user_name" in self._cache:
                return self._cache["user_name"]

        import re, sqlite3 as _sql
        name: str | None = None
        try:
            db_path = _space.db(self._memory_root)
            if db_path.exists():
                conn = _sql.connect(db_path)
                rows = conn.execute(
                    "SELECT text FROM memories WHERE user_id=? LIMIT 300",
                    (self._user_id,),
                ).fetchall()
                conn.close()
                patterns = [
                    r"[Mm]y name is ([A-Za-z]{2,15})",
                    r"[Ii]'?m ([A-Z][a-z]{1,14})",
                ]
                for (text,) in rows:
                    for pat in patterns:
                        m = re.search(pat, text)
                        if m:
                            name = m.group(1).strip()
                            break
                    if name:
                        break
        except Exception:
            pass

        with self._lock:
            self._cache["user_name"] = name
        return name


__all__ = ["LeftBrain", "_search_mode", "_pool_mode"]
