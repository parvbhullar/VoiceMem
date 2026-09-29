"""Left-brain dual-write storage: Mem0 Platform-style JSON + SQLite vector store + cognitive graph.

Semantic memory: ``memories.json`` + ``supermem_leftbrain.sqlite`` (vectors).
Cognitive graph: ``cognitive_graph.sqlite`` (entities / edges / slot_profiles / right-brain stub).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from supermem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
from supermem.leftbrain.local_memory_store import (
    MemorySearchHit,
    OpenAILocalEmbedder,
    OpenAILocalEmbedderConfig,
    TextEmbedder,
    default_memory_root,
)
from supermem.leftbrain.mem0_backend_store import Mem0BackendStore
from supermem.leftbrain.cognitive_graph import (
    CognitiveAnnotator,
    CognitiveAnnotatorConfig,
    CognitiveGraphStore,
    NullAnnotator,
)
from supermem.llm_config import resolve_base_url

_DEFAULT_JSON_NAME = "memories.json"
_DEFAULT_COGNITIVE_DB_NAME = "cognitive_graph.sqlite"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LeftBrainMemoryRepositoryConfig:
    """When ``json_path`` / ``db_path`` is ``None``, use the default file names under ``default_memory_root()``."""

    json_path: Path | None = None
    db_path: Path | None = None
    existing_limit: int = 50
    # cognitive graph
    cognitive_db_path: Path | None = None
    enable_cognitive_graph: bool = False
    #: LLM endpoint for background consolidation. Falls back to OPENAI_BASE_URL if not given --
    #: previously there was no entry point here; the slot summary path only read env, so
    #: SuperMem(base_url=...) had no effect on it.
    base_url: str | None = None


class LeftBrainMemoryRepository:
    """Mem0 JSON mirror, vector store, cognitive graph (Cognitive Graph).

    ``search``: pure vector retrieval;
    ``search_with_cognitive_scope``: first narrows scope via the cognitive graph, then vector retrieval.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        *,
        config: LeftBrainMemoryRepositoryConfig | None = None,
        cognitive_annotator: CognitiveAnnotator | NullAnnotator | None = None,
        experience_repo: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        self._embedder = embedder
        cfg = config or LeftBrainMemoryRepositoryConfig()
        self._base_url = resolve_base_url(cfg.base_url)
        # cfg.db_path is the "<memory_root>/supermem_leftbrain.sqlite" passed by the caller (core.py)
        # -- before the mem0 migration this file itself was the vector store; after it, mem0 needs
        # a whole directory (qdrant collection + history db), so take its parent directory to keep
        # the original "each SuperMem instance's memory_root is isolated" semantics; only fall
        # back to the global default directory when db_path isn't passed explicitly.
        root = cfg.db_path.parent if cfg.db_path is not None else default_memory_root()
        root.mkdir(parents=True, exist_ok=True)
        self._json_path = (
            Path(cfg.json_path).expanduser().resolve()
            if cfg.json_path is not None
            else root / _DEFAULT_JSON_NAME
        )
        self._existing_limit = cfg.existing_limit
        # Raw fact storage really goes through mem0 (paper: entity nodes point to "underlying
        # Mem0 backend raw memory entry indices"). See the top-of-file comment in
        # mem0_backend_store.py, which spells out the behavioural differences from the old
        # hand-rolled SQLite vector store (especially that id generation changed -- mem0
        # generates ids itself, the file cfg.db_path points to is no longer used).
        # mem0 by default; passing vector_store (e.g. zep or any object with the same interface) replaces the memory engine.
        self._vector_store = vector_store or Mem0BackendStore(embedder, memory_root=root)

        # cognitive graph
        self._cognitive_store: CognitiveGraphStore | None = None
        if cfg.enable_cognitive_graph:
            cog_db = (
                Path(cfg.cognitive_db_path).expanduser().resolve()
                if cfg.cognitive_db_path is not None
                else root / _DEFAULT_COGNITIVE_DB_NAME
            )
            self._cognitive_store = CognitiveGraphStore(cog_db, embedder=self._embedder)
        self._cognitive_annotator: CognitiveAnnotator | NullAnnotator | None = cognitive_annotator

        # right brain (optional injection, doesn't affect any existing method)
        self._experience_repo = experience_repo

    @property
    def json_path(self) -> Path:
        return self._json_path

    @property
    def db_path(self) -> Path:
        return self._vector_store._path  # noqa: SLF001 — dev scripts need to show the path

    @property
    def vector_store(self) -> Mem0BackendStore:
        return self._vector_store

    # This left-brain memory mirror used to be a separate memories.json under memory_root. A
    # space keeps only one json (the space description file), so the mirror moved into the
    # sqlite kv table -- it's internal state anyway, not meant for humans. If an old
    # memories.json is still there, it is read in automatically once.
    _KV_KEY = "leftbrain_store"

    def load_json_store(self) -> dict[str, Any]:
        from supermem.utils.common import space as _space
        data = _space.kv_get(self._json_path.parent, self._KV_KEY)
        if data is None and self._json_path.is_file():
            try:
                with self._json_path.open(encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = None
        if not isinstance(data, dict):
            return {"count": 0, "results": []}
        results = data.get("results")
        if not isinstance(results, list):
            return {"count": 0, "results": []}
        return {"count": len(results), "results": results}

    def _write_json_store(self, results: list[dict[str, Any]]) -> None:
        from supermem.utils.common import space as _space
        _space.kv_set(self._json_path.parent, self._KV_KEY,
                      {"count": len(results), "results": results})

    def existing_for_extractor(self, *, user_id: str, limit: int | None = None) -> list[dict[str, str]]:
        """For additive extraction: ``[{"id", "text"}, ...]`` (JSON preferred, aligned with Mem0 fields)."""
        cap = limit if limit is not None else self._existing_limit
        store = self.load_json_store()
        rows: list[dict[str, str]] = []
        for obj in store["results"]:
            if not isinstance(obj, dict):
                continue
            if obj.get("user_id", user_id) != user_id:
                continue
            mid = str(obj.get("id", "")).strip()
            memory = str(obj.get("memory", "")).strip()
            if mid and memory:
                rows.append({"id": mid, "text": memory})
        return rows[-cap:]

    def update_memory(self, memory_id: str, new_text: str,
                      session_id: int | str | None = None,
                      observed_at: str | None = None,
                      user_id: str | None = None) -> bool:
        """Update memory text in place (including re-embedding). Syncs the JSON mirror + cognitive graph.

        When session_id / observed_at are not None, refresh metadata (observed_at is created_at;
        a new fact should carry the date of the session it came from).

        When user_id is passed, re-runs cognitive-graph entity/relation extraction (the same step
        append_extracted does for ADD facts) -- previously only the text was updated here, so facts
        from an UPDATE decision never entered the cognitive graph; entities/relations stayed as they
        were when this memory was first written (possibly with completely different wording), and
        changed entities in the new version were never synced. When user_id is None (old callers
        don't pass it), keep the original behaviour: only update text, don't touch the graph.
        """
        updated = self._vector_store.update_memory(
            memory_id, new_text, session_id=session_id, observed_at=observed_at)
        if updated:
            store = self.load_json_store()
            for obj in store["results"]:
                if isinstance(obj, dict) and str(obj.get("id", "")) == memory_id:
                    obj["memory"] = new_text
                    if observed_at is not None:      # refresh the date in the mirror too, so it stays consistent with the store
                        obj["created_at"] = observed_at
                    break
            self._write_json_store(store["results"])

            if user_id is not None and self._cognitive_store is not None and self._cognitive_annotator is not None:
                try:
                    annotated = self._cognitive_annotator.annotate([new_text])
                    if annotated:
                        self._cognitive_store.ingest_annotated_fact(user_id, annotated[0], [memory_id])
                except Exception as _cog_err:
                    import logging
                    logging.getLogger(__name__).warning("UPDATE cognitive graph rewrite failed: %s", _cog_err)
        return updated

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory. Syncs the JSON mirror."""
        deleted = self._vector_store.delete_memory(memory_id)
        if deleted:
            store = self.load_json_store()
            results = [obj for obj in store["results"]
                       if not (isinstance(obj, dict) and str(obj.get("id", "")) == memory_id)]
            self._write_json_store(results)
        return deleted

    def append_extracted(
        self,
        memories: Sequence[ExtractedAdditiveMemory],
        *,
        user_id: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Append memories: vector ingest (mem0, the real source of ids) + JSON mirror + cognitive graph write.

        ids are now generated by mem0, no longer computed locally by format_memory_id() -- mem0's
        add() has no "use the id I give you" option. Previously this computed a local id first
        and fed it to both the JSON mirror and vector_store, but never looked at the return value
        of add_records_with_ids(), so the ids used by the JSON mirror / cognitive-graph links and
        the ids actually stored in mem0 were two different sets that never matched (discovered by
        actually running Search(): memory_ids linked in the cognitive graph didn't exist in mem0
        at all, and the GraphEntityStore links used by right-brain bookkeeping Algorithm 1 were
        all broken too). Now: assemble text+metadata and hand it to mem0 first, then write the
        JSON mirror and cognitive graph with the ids it really returns -- one set of ids throughout.
        """
        base_md = dict(extra_metadata or {})

        vector_items: list[tuple[str, str, str, dict[str, Any]]] = []
        pending: list[tuple[str, str, dict[str, Any]]] = []  # (body, attributed_to, metadata), in matching order
        for m in memories:
            body = (m.text or "").strip()
            if not body:
                continue
            md: dict[str, Any] = dict(base_md)
            if m.local_id:
                md["extractor_local_id"] = m.local_id
            if m.linked_memory_ids:
                md["linked_memory_ids"] = list(m.linked_memory_ids)
            md["mem0_attributed_to"] = m.attributed_to or "user"
            attributed_to = m.attributed_to or "user"
            vector_items.append(("", body, attributed_to, md))
            pending.append((body, attributed_to, md))

        if not vector_items:
            return []

        saved_ids = self._vector_store.add_records_with_ids(user_id, vector_items)
        if len(saved_ids) != len(pending):
            # add_records_with_ids() calls mem0 one item at a time; if some fail it returns fewer
            # than the input -- truncate by position to those actually written, so the whole
            # thing doesn't get misaligned and linked to other facts.
            import logging
            logging.getLogger(__name__).warning(
                "append_extracted: mem0 wrote %d/%d items successfully, continuing with the successful ones",
                len(saved_ids), len(pending),
            )
            pending = pending[: len(saved_ids)]

        store = self.load_json_store()
        results: list[dict[str, Any]] = list(store["results"])
        now = _utc_iso()
        for mid, (body, attributed_to, md) in zip(saved_ids, pending):
            results.append(
                {
                    "id": mid,
                    "memory": body,
                    "user_id": user_id,
                    "metadata": md,
                    "categories": [],
                    "created_at": now,
                    "updated_at": None,
                }
            )
        self._write_json_store(results)

        # cognitive graph write: annotate facts → entities + slots + edges
        if self._cognitive_store is not None and self._cognitive_annotator is not None:
            fact_texts = [body for body, _, _ in pending]
            try:
                annotated_facts = self._cognitive_annotator.annotate(fact_texts)
                for annotated, mid in zip(annotated_facts, saved_ids):
                    self._cognitive_store.ingest_annotated_fact(user_id, annotated, [mid])
            except Exception as _cog_err:
                import logging
                logging.getLogger(__name__).warning("cognitive graph write failed: %s", _cog_err)

        return saved_ids

    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float | None = None,
        include_assistant: bool = False,
    ) -> list[MemorySearchHit]:
        return self._vector_store.search(
            query,
            user_id=user_id,
            top_k=top_k,
            threshold=threshold,
            include_assistant=include_assistant,
        )

    def search_with_graph(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        threshold: float | None = None,
        relation_depth: int = 1,
    ) -> list[GraphSearchHit]:
        """After semantic retrieval, attach local graph relation context."""
        hits = self.search(query, user_id=user_id, top_k=top_k, threshold=threshold)
        if self._graph_store is None:
            return [
                GraphSearchHit(
                    memory=h,
                    graph=GraphMemoryContext(memory_id=h.memory_id),
                )
                for h in hits
            ]
        return self._graph_store.enrich_hits(
            hits,
            user_id=user_id,
            relation_depth=relation_depth,
        )

    @property
    def cognitive_store(self) -> "CognitiveGraphStore | None":
        return self._cognitive_store

    @property
    def experience_repo(self):
        """Right-brain ExperienceRepository (may be None)."""
        return self._experience_repo

    def search_combined(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 5,
        scope_min: int = 3,
        scope_ratio_max: float = 0.60,
        use_slot_filtering: bool = True,
        signals=None,   # CurrentSignals | None
    ) -> tuple[list, Any, dict]:
        """Parallel left+right brain retrieval, returns (left_hits, right_context, trace).

        - left_hits: exactly the same as search_with_cognitive_scope
        - right_context: RightBrainContext (empty when experience_repo=None)
        - trace: left-brain trace + right_brain_empty flag

        Existing callers that only use the left brain need no code changes;
        call this method only when the right brain is needed.
        """
        from concurrent.futures import ThreadPoolExecutor

        # ── left brain (existing logic, unchanged) ────────────────────────────────────
        def _left():
            return self.search_with_cognitive_scope(
                query,
                user_id=user_id,
                top_k=top_k,
                scope_min=scope_min,
                scope_ratio_max=scope_ratio_max,
                use_slot_filtering=use_slot_filtering,
            )

        # ── right brain (returns empty fast when there is no experience_repo) ────────────────────────────
        def _right():
            if self._experience_repo is None:
                from supermem.rightbrain.types import RightBrainContext, CurrentSignals
                return RightBrainContext(current_signals=signals or CurrentSignals())
            plan = self._experience_repo.build_query_plan(
                query, user_id, signals=signals,
            )
            return self._experience_repo.retrieve(plan)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_left  = pool.submit(_left)
            fut_right = pool.submit(_right)
            left_hits, trace = fut_left.result()
            right_context    = fut_right.result()

        trace["right_brain_active"] = self._experience_repo is not None
        trace["right_brain_empty"]  = right_context.is_empty()
        return left_hits, right_context, trace

    def backfill_cognitive_graph_from_json(
        self,
        *,
        user_id: str,
        batch_size: int = 10,
        force: bool = False,
    ) -> dict[str, int]:
        """Batch-backfill cognitive-graph annotations for existing memories in memories.json.

        Checks one by one whether entity_memory_links already exist, skipping processed memory_ids.
        With ``force=True``, first clears this user's cognitive-graph data and rebuilds.

        Returns:
            {"processed": n, "skipped": n, "entities_created": n}
        """
        if self._cognitive_store is None:
            raise ValueError("cognitive_store not configured (enable_cognitive_graph=False)")
        if self._cognitive_annotator is None:
            raise ValueError("cognitive_annotator not configured")

        if force:
            self._cognitive_store.delete_user(user_id)

        store = self.load_json_store()
        processed = skipped = entities_created = 0

        # collect unprocessed memory_ids
        pending: list[tuple[str, str]] = []  # (memory_id, text)
        for obj in store["results"]:
            if not isinstance(obj, dict) or obj.get("user_id") != user_id:
                continue
            mid = str(obj.get("id", "")).strip()
            text = str(obj.get("memory", "")).strip()
            if not mid or not text:
                continue
            # skip condition: entity links exist AND the memory record exists too
            # if there are entity links but no memory record, still reprocess to backfill the slot
            has_links = bool(self._cognitive_store.entity_ids_for_memory(mid))
            has_record = bool(self._cognitive_store.get_memory_record(mid))
            if not force and has_links and has_record:
                skipped += 1
                continue
            pending.append((mid, text))

        # annotate in batches
        for i in range(0, len(pending), batch_size):
            batch = pending[i: i + batch_size]
            texts = [t for _, t in batch]
            try:
                annotated_facts = self._cognitive_annotator.annotate(texts)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("backfill batch %d annotation failed: %s", i, e)
                skipped += len(batch)
                continue

            for (mid, _), ann in zip(batch, annotated_facts):
                entities = self._cognitive_store.ingest_annotated_fact(user_id, ann, [mid])
                entities_created += len(entities)
                processed += 1

        return {"processed": processed, "skipped": skipped, "entities_created": entities_created}

    def backfill_memory_slots_from_entities(self, *, user_id: str) -> dict[str, int]:
        """Backfill the slot for memories that have entity_memory_links but no memories record.

        No LLM call. Takes the most frequent slot among the entities linked to the memory as its slot.
        Used to repair historical data where "the cognitive graph has backfilled entities but the memories table is empty".
        """
        if self._cognitive_store is None:
            raise ValueError("cognitive_store not configured")

        linked_mids = set(self._cognitive_store.all_linked_memory_ids(user_id))
        existing_mids = set(self._cognitive_store.all_memory_record_ids(user_id))
        missing_mids = linked_mids - existing_mids

        # read text from the JSON store
        store = self.load_json_store()
        mid_to_text: dict[str, str] = {}
        for obj in store["results"]:
            if isinstance(obj, dict) and obj.get("user_id") == user_id:
                mid = str(obj.get("id", "")).strip()
                text = str(obj.get("memory", "")).strip()
                if mid and text:
                    mid_to_text[mid] = text

        written = skipped = 0
        for mid in missing_mids:
            text = mid_to_text.get(mid, "")
            if not text:
                skipped += 1
                continue
            entity_ids = self._cognitive_store.entity_ids_for_memory(mid)
            if not entity_ids:
                skipped += 1
                continue
            entities = self._cognitive_store.find_entities(user_id, entity_ids=entity_ids)
            if not entities:
                skipped += 1
                continue
            # take the most frequent slot among linked entities
            from collections import Counter
            dominant_slot = Counter(e.slot for e in entities).most_common(1)[0][0]
            self._cognitive_store.upsert_memory_record(user_id, mid, dominant_slot, text)
            written += 1

        return {"written": written, "skipped": skipped, "total_missing": len(missing_mids)}

    def sync_vectors_from_json(self, *, user_id: str) -> int:
        """Fill in vectors for JSON entries not yet written to SQLite (for migration or repair)."""
        store = self.load_json_store()
        existing_ids = set(self._vector_store.list_ids(user_id=user_id))
        items: list[tuple[str, str, str, dict[str, Any]]] = []
        for obj in store["results"]:
            if not isinstance(obj, dict) or obj.get("user_id") != user_id:
                continue
            mid = str(obj.get("id", "")).strip()
            memory = str(obj.get("memory", "")).strip()
            if not mid or not memory or mid in existing_ids:
                continue
            md = obj.get("metadata")
            meta = md if isinstance(md, dict) else {}
            attributed = str(meta.get("mem0_attributed_to", "user"))
            items.append((mid, memory, attributed, meta))
        if not items:
            return 0
        self._vector_store.add_records_with_ids(user_id, items)
        return len(items)


def create_openai_memory_repository(
    *,
    config: LeftBrainMemoryRepositoryConfig | None = None,
    embedder_config: OpenAILocalEmbedderConfig | None = None,
    cognitive_annotator_config: "CognitiveAnnotatorConfig | None" = None,
) -> LeftBrainMemoryRepository:
    """Default repository using OpenAI Embeddings (requires ``OPENAI_API_KEY``).

    If config.enable_cognitive_graph=True, a cognitive-graph annotator is created automatically.
    """
    from supermem.leftbrain.cognitive_graph import CognitiveAnnotatorConfig as _CogCfg
    embedder = OpenAILocalEmbedder(embedder_config)
    annotator: CognitiveAnnotator | NullAnnotator | None = None
    cfg = config or LeftBrainMemoryRepositoryConfig()
    if cfg.enable_cognitive_graph:
        annotator = CognitiveAnnotator(cognitive_annotator_config or _CogCfg())
    return LeftBrainMemoryRepository(embedder, config=cfg, cognitive_annotator=annotator)
