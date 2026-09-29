"""Right-brain attribution batch jobs: short-term attribution (entity.description, every 3 turns) + long-term attribution (slot.description, at session boundaries).

Short-term: look at which memories an entity has newly been linked to over the last few turns,
synthesise them into an updated description for that entity, and while at it refine each memory
item itself (strip redundancy, keep the core).

Long-term: at session end, look at all entities under a slot (including their descriptions)
and synthesise a higher-level slot description, something like "the personality profile shown in this area".
"""

from __future__ import annotations

from typing import Callable

from .graph_store import RightBrainGraphStore
from .store import RightBrainStore


def _is_cjk(text: str) -> bool:
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cjk / max(sum(1 for c in text if c.isalpha()) + cjk, 1) >= 0.3


class AttributionManager:
    def __init__(
        self,
        graph_store: RightBrainGraphStore,
        rb_store: RightBrainStore,
        llm_fn: Callable[[str], str],
    ) -> None:
        self._graph = graph_store
        self._rb_store = rb_store
        self._llm = llm_fn

    # ── Short-term attribution: entity.description ────────────────────────────────────

    def run_short_term(self, user_id: str, entity_ids: list[str]) -> None:
        for eid in entity_ids:
            ent = self._graph.get_entity(eid)
            if ent is None:
                continue
            mem_ids = self._graph.get_memories_for_entity(eid)
            contents = []          # (mid, content, already_refined)
            for mid in mem_ids:
                mem = self._rb_store.get_memory(mid)
                if mem is not None and mem.content:
                    contents.append((mid, mem.content, bool((mem.metadata or {}).get("refined"))))
            if not contents:
                continue

            src = [c for _, c, _ in contents]
            new_desc = self._summarize_entity(ent.name, src)
            # Same as below: if the language does not match the evidence, don't write it and keep the
            # previous description. The profile is spliced into the system prompt every turn, and mixed
            # languages are worse than one entry fewer.
            if new_desc and _is_cjk(new_desc) != _is_cjk(" ".join(src)):
                new_desc = ""
            if new_desc:
                self._graph.set_entity_description(eid, new_desc)

            # Refine each memory item (strip redundancy), but only once per item (tagged via
            # metadata.refined). Previously every short-term pass rewrote the same memories again: each
            # rewrite is lossy, so over many turns concrete details like numbers/names/times got worn
            # away, and each item was a synchronous LLM call, burning money linearly with turns.
            for mid, content, already_refined in contents:
                if already_refined:
                    continue
                refined = self._refine_memory_item(ent.name, content)
                # Discard if the language changed: the prompt says "keep the original language" but the model
                # does not always comply. Once the user's original words are refined into another language,
                # what is stored is no longer what the user said (memory is evidence, and translated evidence
                # is useless), and the profiles summarised from it end up mixing languages too.
                if refined and _is_cjk(refined) != _is_cjk(content):
                    print(f"[Attribution] Language changed after refinement, keeping original: {content[:30]}")
                    refined = ""
                if refined and refined != content:
                    self._rb_store.update_content(mid, refined)
                self._rb_store.merge_metadata(mid, {"refined": True})

    def _summarize_entity(self, entity_name: str, contents: list[str]) -> str:
        snippets = "\n".join(f"- {c}" for c in contents[:20])
        prompt = (
            f"Below is a set of memory snippets related to the trait \"{entity_name}\":\n{snippets}\n\n"
            "In one sentence (at most 20 words), summarise how this trait concretely shows up in the user.\n"
            "Hard rules: describe concrete behaviour and facts, no lyrical adjectives; plain language; "
            "do not re-list the original sentences; do not write anything the snippets do not support; "
            "write in the same language as the snippets (if the snippets are in English, write in English)."
        )
        return self._llm_text(prompt)

    def _refine_memory_item(self, entity_name: str, content: str) -> str:
        prompt = (
            f"This memory relates to the user's trait \"{entity_name}\":\n\"{content}\"\n\n"
            "If this sentence has redundancy, conversational filler or unnecessary adjectives, refine it into a "
            "more concise, plain sentence, keeping the original meaning and key details without losing information; "
            "if it is already concise, return it unchanged.\n"
            "You must keep the original sentence's language (if it is in English, output English; do not translate).\n"
            "Output only the refined sentence, without any explanation."
        )
        return self._llm_text(prompt)

    # ── Long-term attribution: slot.description ───────────────────────────────────────

    def run_long_term(self, user_id: str, slot_ids: list[str]) -> None:
        for sid in slot_ids:
            slot = self._graph.get_slot(sid)
            if slot is None:
                continue
            # Only summarise entities that actually have memory evidence. Empty seed-placeholder entities (e.g.
            # emotion labels that never fired) should not take part, or traits never observed get invented.
            entities = [
                e for e in self._graph.get_entities_for_slot(user_id, sid)
                if self._graph.get_memories_for_entity(e.id)
            ]
            if not entities:
                continue
            new_desc = self._summarize_slot(slot.name, entities)
            src = " ".join(f"{e.name}{e.description or ''}" for e in entities)
            if new_desc and _is_cjk(new_desc) != _is_cjk(src):
                new_desc = ""
            if new_desc:
                self._graph.set_slot_description(sid, new_desc)

    def _summarize_slot(self, slot_name: str, entities) -> str:
        lines = []
        for e in entities:
            if e.description:
                lines.append(f"- {e.name}: {e.description}")
            else:
                lines.append(f"- {e.name}")
        listing = "\n".join(lines[:30])
        # This used to ask only for "a 2-3 sentence summary of the overall profile", with no length or
        # style limit, so GPT wrote 100+ character lyrical paragraphs full of parallelisms ("shows a complex
        # state, both longing for... and..."). That description is spliced into the system prompt every turn
        # (prestimulus persona block + profile hit); in measurements it was the largest token cost in the
        # whole prompt, and it was mostly empty words with low information density.
        prompt = (
            f"Below are the concrete traits the user currently shows on the \"{slot_name}\" dimension:\n{listing}\n\n"
            f"Summarise the user's profile on the \"{slot_name}\" dimension in one sentence, at most 35 words.\n"
            "Hard rules: only write concrete observations supported by the entries above; plain language, like a "
            "note between colleagues, not a psychological report; no parallelisms, metaphors or lyrical language; "
            "if there is not enough information, write less, and do not inflate or fill in gaps; write in the same "
            "language as the entries (if the entries are in English, write in English)."
        )
        return self._llm_text(prompt)

    # ── LLM helpers ──────────────────────────────────────────────────────────

    def _llm_text(self, prompt: str) -> str:
        """Unlike core.py::_llm_json, this wants plain-text output, not JSON."""
        raw = self._llm(prompt)
        return raw.strip() if raw else ""
