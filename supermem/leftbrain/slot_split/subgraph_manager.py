"""Left-brain subgraph judgement: retrieval-triggered (run_for_retrieved_pool); decides
whether an entity co-occurrence subgraph in a retrieval's candidate pool should be formally
promoted to a new slot. There is no longer a session-boundary trigger path that scans by
slot_ref at ingest time -- the candidate pool comes only from actual retrieval results.

Flow:
  1. The candidate pool is a batch of memory_ids passed in by the caller (e.g. the top-k
     results of one retrieval), not scanned separately per slot_ref -- the selection step
     shouldn't pre-guess "which are more relevant" on behalf of the co-occurrence
     structure; that is the job of connected components + the association coefficient
  2. Build the entity co-occurrence graph: node=entity, an edge whenever two entities
     appear in the same memory
  3. Find connected subgraphs and keep those with >= 3 entities
  4. Exclude subgraphs where every member has can_split=False (unless a new
     can_split=True entity mixes in and reactivates it)
  5. Greedily peel each connected component (repeatedly removing the lowest-degree node
     in the current subset), computing the association coefficient ρ at each layer (the
     paper's formula: the average Jaccard similarity between the candidate subset and the
     entity sets activated by each of the user's historical queries, see
     GraphEntityStore.compute_rho), and keep the layer with the highest ρ in that
     component's peeling sequence -- a large component doesn't necessarily deserve to be
     its own cluster; ρ may actually rise after peeling off one or two weakly linked nodes.
  6. Compare the best subsets peeled from each component; the one with the globally
     highest ρ becomes this round's candidate (one call yields only one candidate; the
     other components wait until they are hit by a later retrieval)
  7. Candidates whose ρ is below MIN_SYNERGY_THRESHOLD are skipped outright (no LLM call,
     saving cost), but not permanently blacklisted -- insufficient evidence is different
     from being judged unimportant; can_split stays True so they can be reconsidered once
     enough co-occurrence evidence accumulates
  8. LLM judgement: is it the same event/topic, and important enough -> if yes, create a
     new slot and move this batch of entities over (no new memory is produced); if no,
     permanently set can_split=False on this batch of entities
"""

from __future__ import annotations

import json
from typing import Callable

from .graph_entity_store import GraphEntity, GraphEntityStore

MIN_SUBGRAPH_SIZE = 3

# Lower bound on ρ -- ρ is a mean of Jaccard similarities in the range [0,1], a completely
# different magnitude from the previous "density x count" (which could reach dozens), so the
# old threshold can't be reused. A loose value for now: candidates below the line don't call
# the LLM anyway (saving cost), and the real quality gate is the following LLM judgement, so
# the threshold itself isn't a hard cutoff yet. Tighten it once real query data has
# accumulated and the actual distribution of ρ is known.
MIN_SYNERGY_THRESHOLD = 0.05

# The maximum fraction of the candidate pool's memories a single subgraph may cover -- this
# is independent of the scoring formula and kept separately: peeling must still prevent
# "one big blob" candidates from entering ρ scoring (when a component spans half the store,
# it is first cut by coverage fraction, forcing the greedy peel down to a tight enough
# layer, where ρ is meaningful to evaluate).
MAX_TOUCHED_FRACTION = 0.25


def _connected_components(adjacency: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components: list[set[str]] = []
    for start in adjacency:
        if start in seen:
            continue
        comp = {start}
        queue = [start]
        seen.add(start)
        while queue:
            node = queue.pop()
            for nb in adjacency.get(node, ()):
                if nb not in seen:
                    seen.add(nb)
                    comp.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def _synergy(
    entity_ids: set[str],
    mem_to_entities: dict[str, set[str]],
    rho_fn: Callable[[set[str]], float],
) -> tuple[float, set[str]]:
    """Given a batch of entity ids, return (ρ association coefficient, set of memories touched).
    ρ is computed by the caller-supplied rho_fn (GraphEntityStore.compute_rho, the paper's
    formula: average Jaccard similarity between the candidate subset and each of the user's
    historical query activation sets)."""
    touched = {mid for mid, eids in mem_to_entities.items() if eids & entity_ids}
    return rho_fn(entity_ids), touched


def _densest_subset(
    component: set[str],
    adjacency: dict[str, set[str]],
    mem_to_entities: dict[str, set[str]],
    min_size: int,
    rho_fn: Callable[[set[str]], float],
    max_touched: int | None = None,
) -> tuple[set[str], float, set[str]]:
    """Greedy peeling (Charikar peeling): repeatedly remove the lowest-degree node in the
    current subset, computing ρ at each layer, and return the layer with the highest ρ along
    the whole peeling path (including no peeling, i.e. the whole connected component itself,
    as the starting point of the path).

    When max_touched is set, only layers covering no more memories than it are eligible as
    the best solution -- see the note on MAX_TOUCHED_FRACTION. If every layer on the peeling
    path exceeds it (meaning these entities are spread too widely no matter how they are
    peeled, so they are essentially not one activity thread), return ρ=0.0 so it falls below
    MIN_SYNERGY_THRESHOLD and is skipped. Note that skipping is not blacklisting:
    can_split stays True, so it can be reconsidered if the co-occurrence structure changes.
    """
    def _ok(touched: set[str]) -> bool:
        return max_touched is None or len(touched) <= max_touched

    current = set(component)
    best_set: set[str] = set(current)
    best_synergy = 0.0
    best_touched: set[str] = set()
    found_ok = False

    synergy, touched = _synergy(current, mem_to_entities, rho_fn)
    if _ok(touched):
        best_set, best_synergy, best_touched, found_ok = set(current), synergy, touched, True

    while len(current) > min_size:
        degrees = {eid: len(adjacency[eid] & current) for eid in current}
        weakest = min(degrees, key=degrees.get)
        current = current - {weakest}
        synergy, touched = _synergy(current, mem_to_entities, rho_fn)
        if _ok(touched) and (not found_ok or synergy > best_synergy):
            best_set, best_synergy, best_touched, found_ok = set(current), synergy, touched, True

    if not found_ok:
        return set(component), 0.0, set()
    return best_set, best_synergy, best_touched


class SubgraphManager:
    """tag_fn: when a new slot passes judgement, the batch of memories that gave rise to it
    is also tagged with the new slot in cog_store.memory_tags -- without this, the slot_ref
    migration in GraphEntityStore is only internal bookkeeping (for the next subgraph
    analysis); the retrieval path (SearchCogGraph/memory_ids_for_slots_v2) never reads
    GraphEntityStore, so from retrieval's point of view the new slot is always empty, and
    even if Classify() classifies into the new slot it finds nothing.
    tag_fn has the signature (user_id, memory_id, slot_name) -> None and tags additively
    (upsert_memory_tags itself upserts on the unique key (memory_id, slot) and doesn't clear
    the memory's existing tags), so both the old broad slot ("relationships") and the new
    narrow slot can find this memory -- broad queries keep recall, narrow queries gain precision."""

    def __init__(
        self,
        graph_store: GraphEntityStore,
        dynamic_slot_store,
        llm_fn: Callable[[str], str],
        tag_fn: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self._graph = graph_store
        self._dyn = dynamic_slot_store
        self._llm = llm_fn
        self._tag = tag_fn

    def run_for_retrieved_pool(
        self,
        user_id: str,
        memory_ids: set[str],
        memory_content_lookup: Callable[[str], str | None],
        session_id: str | None = None,
    ) -> dict:
        """The candidate pool is simply a batch of memory_ids (e.g. the top-k results of a real
        retrieval), not scanned separately per slot_ref -- retrieval itself doesn't go
        through GraphEntityStore's slot_ref (SearchCogGraph uses cog_store's entity index and
        vector search), so even if the same entity was given different slot tags by
        _llm_tag_memories in different ingest batches and registered under different
        slot_refs, as long as they appear in the same batch of memory_ids they can be put
        back together to build the co-occurrence graph. The logic that actually judges
        co-occurrence/importance (_promote_or_reject) treats them all the same, regardless of
        how the candidate pool was assembled.
        """
        if not memory_ids:
            return {"status": "no_memories"}

        ent_by_id: dict[str, GraphEntity] = {}
        mem_to_entities: dict[str, set[str]] = {}
        for mid in memory_ids:
            ents = self._graph.get_entities_for_memory(user_id, mid)
            if not ents:
                continue
            for e in ents:
                ent_by_id[e.id] = e
            mem_to_entities[mid] = {e.id for e in ents}

        if not ent_by_id:
            return {"status": "no_memories"}

        return self._promote_or_reject(user_id, ent_by_id, mem_to_entities, memory_content_lookup, session_id)

    def _promote_or_reject(
        self,
        user_id: str,
        ent_by_id: dict[str, GraphEntity],
        mem_to_entities: dict[str, set[str]],
        memory_content_lookup: Callable[[str], str | None],
        session_id: str | None = None,
    ) -> dict:
        """Shared core: starting from (ent_by_id, mem_to_entities), build the co-occurrence
        graph -> connected components -> greedy peeling for the best subset -> LLM judgement
        -> create a new slot (promote) or can_split=False (reject).

        The parent_slots of a new slot are derived directly from the current slot_ref of each
        promoted entity (update_slot_ref hasn't been called yet, so what is read is the
        slot_ref where they originally lived) -- entities in the candidate pool may span
        several slot_refs (the same entity tagged differently in different ingest batches),
        so the derived result is the true set of multiple parent slots, with no need for the
        caller to specify or guess; if the candidate pool happens to be all under one
        slot_ref, it naturally collapses to a single-element set."""
        adjacency: dict[str, set[str]] = {eid: set() for eid in ent_by_id}
        for eids in mem_to_entities.values():
            eid_list = list(eids)
            for i in range(len(eid_list)):
                for j in range(i + 1, len(eid_list)):
                    adjacency[eid_list[i]].add(eid_list[j])
                    adjacency[eid_list[j]].add(eid_list[i])

        components = [c for c in _connected_components(adjacency) if len(c) >= MIN_SUBGRAPH_SIZE]
        components = [c for c in components if any(ent_by_id[eid].can_split for eid in c)]
        if not components:
            return {"status": "no_eligible_subgraph"}

        # Each connected component peels out its own highest-ρ subset; take the globally
        # highest across components. If session_id is passed, Q is restricted to the current
        # session (literally per the paper: Q is the set of queries in the current session,
        # not the lifetime history) -- when not passed (callers without a session concept)
        # compute_rho falls back to the old full-history behaviour itself.
        rho_fn = lambda h: self._graph.compute_rho(user_id, h, session_id=session_id)  # noqa: E731
        max_touched = max(MIN_SUBGRAPH_SIZE,
                          int(len(mem_to_entities) * MAX_TOUCHED_FRACTION))
        candidates = [
            _densest_subset(c, adjacency, mem_to_entities, MIN_SUBGRAPH_SIZE, rho_fn, max_touched)
            for c in components
        ]
        best, synergy, touched_memories = max(candidates, key=lambda t: t[1])
        synergy_score = round(synergy, 4)

        # Candidates whose ρ is below the line skip LLM judgement entirely -- "not enough
        # evidence yet" is different from "the LLM judged it unimportant"; can_split stays
        # True so it can be reconsidered once enough co-occurrence evidence accumulates,
        # unlike an LLM rejection, which is a permanent blacklist.
        if synergy_score < MIN_SYNERGY_THRESHOLD:
            names = [ent_by_id[eid].name for eid in best]
            return {"status": "below_threshold", "entities": names, "synergy": synergy_score}

        names = [ent_by_id[eid].name for eid in best]
        sample_texts = [memory_content_lookup(mid) for mid in list(touched_memories)[:5]]
        sample_texts = [t for t in sample_texts if t]

        # Existing dynamic slots are handed to the LLM judgement too -- when the topic
        # heavily overlaps an existing slot it is merged into that one, and a near-duplicate
        # new name is not allowed ("Family & Growth" / "Family & Love" / "Care & Acceptance"
        # each created separately would shred the same body of memories, and querying any
        # of them would only show one corner)
        existing = [(s.name, s.description) for s in self._dyn.get_dynamic_slots(user_id)]
        verdict = self._judge_subgraph(names, sample_texts, existing)

        if verdict:
            kind, slot_name, slot_desc = verdict
            if kind == "new":
                parent_slots = sorted({ent_by_id[eid].slot_ref for eid in best if ent_by_id[eid].slot_ref})
                self._dyn.create_dynamic_slot(user_id, slot_name, slot_desc, parent_slots=parent_slots)
            for eid in best:
                self._graph.update_slot_ref(eid, slot_name)
            if self._tag is not None:
                for mid in touched_memories:
                    try:
                        self._tag(user_id, mid, slot_name)
                    except Exception:
                        pass
            return {
                "status": "slot_created" if kind == "new" else "merged_into_slot",
                "name": slot_name, "entities": names, "synergy": synergy_score,
            }

        for eid in best:
            if ent_by_id[eid].can_split:
                self._graph.mark_cannot_split(eid)
        return {"status": "rejected", "entities": names, "synergy": synergy_score}

    def _judge_subgraph(
        self,
        entity_names: list[str],
        sample_texts: list[str],
        existing_slots: list[tuple[str, str]] | None = None,
    ) -> tuple[str, str, str] | None:
        """Three independent LLM checks (relevance/importance/completeness); it passes only if
        all three pass; if any fails it is rejected immediately without asking the rest
        (saves cost, the rejection reason is already settled).
        Returns ("existing", existing slot name, "") / ("new", new name, description) / None (rejected).

        The three criteria are asked separately rather than folded into one question because
        they focus on different things and tend to dilute each other: relevance asks "is this
        the same thing", importance asks "does this carry enough weight in this user's
        memory to get its own bucket", and completeness asks "is the content accumulated so
        far enough to support a useful category rather than an empty label". When folded
        into one question, the model tends to answer only the most salient reason and reach
        a conclusion, implicitly glossing over the other two, so the basis for
        rejecting/passing is unclear.
        """
        relevance = self._check_relevance(entity_names, sample_texts, existing_slots)
        if relevance is None:
            return None
        kind, slot_name, slot_desc = relevance
        # Merging into an existing topic: that topic's importance/completeness were already
        # independently verified when it was first created; this time it's just "this new
        # evidence also belongs to the same already-verified topic", so there's no need to
        # re-run them -- otherwise an established activity thread could be wrongly rejected by
        # the latter two checks just because this batch happened to bring few samples, and the
        # new evidence could never be merged in.
        if kind == "existing":
            return relevance
        if not self._check_importance(slot_name, sample_texts):
            return None
        if not self._check_completeness(slot_name, slot_desc, sample_texts):
            return None
        return relevance

    def _check_relevance(
        self,
        entity_names: list[str],
        sample_texts: list[str],
        existing_slots: list[tuple[str, str]] | None,
    ) -> tuple[str, str, str] | None:
        """Criterion 1 (relevance): do these concepts jointly point to the same concrete activity thread?
        If so, identify which one (merge into an existing one, or otherwise give it a name + description).
        """
        snippets = "\n".join(f"- {t}" for t in sample_texts) or "(none)"
        existing_part = ""
        if existing_slots:
            listing = "\n".join(f"- {n}: {d}" for n, d in existing_slots)
            existing_part = (
                f"\nThe user already has these fine-grained topics:\n{listing}\n\n"
                "If the topic these concepts point to is essentially the same as one of the existing ones above "
                "(even if worded differently), it must be merged into that existing topic; do not create a new name with the same meaning.\n"
            )
        # The criterion is an "activity thread", not a "topic". The difference is large and in
        # testing directly decided success or failure: judging by topic splits off
        # emotion/value labels like "Pride & Self-Acceptance" -- these cut across every topic,
        # grow ever larger (covering half the store), and don't match how questions are asked.
        # Users ask "which day did she make that plate" or "which day in July did she go
        # camping"; the retrieval unit is a **concrete activity**, not an emotion. Judging by
        # activity thread splits off things like "Pottery", "Camping", "Adoption process" --
        # the same thing done repeatedly, each time a distinguishable occasion, which matches
        # exactly this kind of question.
        prompt = (
            f"The following concepts often appear together in the user's memories: {', '.join(entity_names)}\n\n"
            f"Related memory snippets:\n{snippets}\n"
            f"{existing_part}\n"
            "Judge only one thing: do these concepts jointly point to **the same concrete activity thread** -- "
            "i.e. a specific thing the user repeatedly invests in (a hobby, a kind of outing, an ongoing "
            "process) that has happened multiple times over time, each time a distinguishable occasion.\n\n"
            "Examples that count: pottery (signed up for a class -> made a bowl -> made a plate -> got hurt and stopped), "
            "camping (several different camping trips, each with its own date and place), adoption (researched agencies -> "
            "attended an info session -> submitted an application -> passed the interview), "
            "running, keeping a pet.\n"
            "Examples that do not count:\n"
            "- Emotions, values, states of mind: self-acceptance, growth, happiness, gratitude, courage, belonging\n"
            "- Generic interpersonal or family concepts: family love, support from friends, intimate relationships\n"
            "These cut across every topic; once grouped into a category they match everything, which makes retrieval less precise.\n"
            "The name must be the activity itself (a noun), not the feeling it brings.\n\n"
            "At this step, don't worry about whether this is important enough or whether the content is rich enough -- "
            "those are judged separately in the next two steps; here only answer \"is this the same activity thread\".\n"
            "- Belongs to an existing topic -> {\"split\": true, \"existing\": \"existing topic name\"}\n"
            "- Is a brand-new activity thread -> {\"split\": true, \"name\": \"short English name, 1-3 words\", \"description\": \"one-sentence description\"}\n"
            "- Not the same activity thread, just several unrelated concepts that happened to appear together -> {\"split\": false}\n"
            "Output JSON only."
        )
        raw = self._llm(prompt)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except Exception:
            return None
        if not data.get("split"):
            return None
        existing_names = {n for n, _ in (existing_slots or [])}
        chosen = str(data.get("existing", "") or "").strip()
        if chosen and chosen in existing_names:
            return ("existing", chosen, "")
        name = str(data.get("name", "")).strip()
        desc = str(data.get("description", "")).strip()
        if not name:
            return None
        # The model didn't use the existing field but its name collides with an existing one -> treat it as a merge too
        if name in existing_names:
            return ("existing", name, "")
        return ("new", name, desc)

    def _check_importance(self, slot_name: str, sample_texts: list[str]) -> bool:
        """Criterion 2 (importance): does this activity thread carry enough weight in the user's
        memory to deserve its own retrievable category, rather than being a trivial matter
        mentioned only once or twice?"""
        snippets = "\n".join(f"- {t}" for t in sample_texts) or "(none)"
        prompt = (
            f"The user's memories contain a possible activity thread: \"{slot_name}\"\n\n"
            f"Related memory snippets:\n{snippets}\n\n"
            "Judge only one thing: is this important enough in the user's life/memory, and mentioned "
            "repeatedly enough, to deserve its own retrievable category (rather than a trivial matter "
            "mentioned in passing, or something that came up once and never again)?\n"
            "Something only mentioned in passing once or twice and never developed further does not count as important.\n\n"
            "{\"important\": true} or {\"important\": false}. Output JSON only."
        )
        raw = self._llm(prompt)
        if not raw:
            return False
        try:
            data = json.loads(raw)
        except Exception:
            return False
        return bool(data.get("important"))

    def _check_completeness(self, slot_name: str, slot_desc: str, sample_texts: list[str]) -> bool:
        """Criterion 3 (completeness): do the memory snippets accumulated so far carry enough
        concrete information to support a genuinely useful category, rather than an empty
        label with no substance?"""
        snippets = "\n".join(f"- {t}" for t in sample_texts) or "(none)"
        prompt = (
            f"The user's memories contain an activity thread: \"{slot_name}\" ({slot_desc or 'no description'})\n\n"
            f"Related memory snippets:\n{snippets}\n\n"
            "Judge only one thing: is the information in these memory snippets already concrete enough "
            "(e.g. substantive content such as specific times, places, progress, details) to support a "
            "category that will be genuinely useful for future retrieval? If the snippets are all vague and generic "
            "(e.g. they just repeatedly say \"this was mentioned\" without any concrete details), there isn't enough information, "
            "and creating this category now would just be an empty shell with no content.\n\n"
            "{\"complete\": true} or {\"complete\": false}. Output JSON only."
        )
        raw = self._llm(prompt)
        if not raw:
            return False
        try:
            data = json.loads(raw)
        except Exception:
            return False
        return bool(data.get("complete"))
