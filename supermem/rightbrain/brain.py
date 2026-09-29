"""Right brain component, RightBrain.

The **whole right-brain block** extracted from the SuperMem god class -- heartnote emotional memory writes,
inner-monologue generation, right-brain graph layer writes (emotion/relation/personality traits), right-brain
search (the structured top-N behind the situational guidance rb_directive), and the LLM cleanup around them
(duplicate deletion / contradiction supersede).

Follows mem0's composition pattern:
  * **The component owns its parts** -- the 3 right-brain lazy singletons (rb_repo / rb_graph_store /
    attribution_manager), together with the cache and lock they share with the host, live inside this
    component; engine no longer holds its own _get_* for them.
  * **Dependencies are injected explicitly** -- anywhere that needs **non-right-brain** capabilities such as
    "text embedding / LLM(JSON) / LLM(text) / session tracker / left-brain repo / inner-monologue generation /
    trait extraction" gets them injected in __init__ as getters/function references (lazy-loading semantics
    unchanged), and the component calls them via self._dep().

Logic unchanged word for word: method bodies moved as-is, only "how dependencies are obtained" changed.

brain.py does not import engine (to avoid cycles) -- the RightBrainHit dataclass and the module-level _rb_*
helpers live in this module, and engine imports them from here.
"""

from __future__ import annotations

from supermem.utils.common import space as _space

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from supermem.llm_config import resolve_api_key, resolve_model


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class RightBrainHit:
    """A single structured right-brain search result. rb_directive is rendered from a list of these."""
    content: str
    source: str                          # response_experience | situation_pattern | relation | emotion_trait | profile
    priority: float
    metadata: dict = field(default_factory=dict)


# ── language detection helpers ────────────────────────────────────────────────

def _is_en_text(text: str) -> bool:
    """True if text is predominantly English (low CJK ratio)."""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    alpha = sum(1 for c in text if c.isalpha())
    return alpha > 0 and cjk / max(alpha, 1) < 0.3


def _rb_lang(rb_ctx) -> bool:
    """Return True if right brain content is predominantly English."""
    samples = [m.content for m in rb_ctx.situation_patterns[:2]]
    samples += [m.content for m in rb_ctx.response_experiences[:1]]
    return _is_en_text(" ".join(samples))


def _rb_mem_date(m) -> str:
    """created_at ISO -> '[YYYY-MM-DD] ' prefix; empty string if there's no date (for temporal reasoning)."""
    d = (getattr(m, "created_at", "") or "")[:10]
    return f"[{d}] " if d else ""


def _rb_blended_priority(m) -> float:
    """Static priority + this search's anchor relevance (normalized, then weighted).

    anchor_score = SUM(link.weight*confidence), squashed into [0,1) via s/(1+s) then multiplied by 0.5:
    concrete evidence that strongly hits an anchor can compete with relation/emotion_trait; misses stay as they are."""
    s = getattr(m, "anchor_score", 0.0) or 0.0
    return m.priority + 0.5 * (s / (1.0 + s))


def _rb_ctx_to_hits(rb_ctx) -> list["RightBrainHit"]:
    """rb_ctx (heartnote / response_experience search results) -> structured hit list.
    Current signals (dissatisfaction/correction/emotion hint) are this turn's live state, so they get a fairly high fixed priority."""
    hits: list[RightBrainHit] = []
    for m in rb_ctx.response_experiences:
        meta = m.metadata or {}
        failed = meta.get("previous_failure", False)
        prefix = (
            ("⚠ " + "Avoid repeating: ")
            if failed else
            ("✓ " + "Effective approach: ")
        )
        # content only says "what was done at the time"; next_time_policy is the actionable half.
        # Why the user reacted that way isn't appended here -- that's a user trait, served by the graph layer's profile hit.
        body = m.content
        policy = str(meta.get("next_time_policy") or "").strip()
        if policy:
            body += f" (next time: {policy})"
        hits.append(RightBrainHit(
            content=f"{_rb_mem_date(m)}{prefix}{body}", source="response_experience",
            priority=_rb_blended_priority(m),
            metadata={"failed": failed, "anchor_score": getattr(m, "anchor_score", 0.0),
                      "next_time_policy": policy},
        ))
    for m in rb_ctx.situation_patterns:
        meta = m.metadata or {}
        prefix = "Emotional note: "
        inner = str(meta.get("inner_os") or "").strip()
        # content stores the original words; inner_os is appended as a supplement when rendering. An overly long
        # original (>400) would blow up the prompt, so fall back to the inner-monologue summary, or truncate if there is none.
        if len(m.content) > 400:
            body = inner if inner else (m.content[:400] + "…")
        else:
            body = m.content
            if inner and inner != m.content:
                body += f" (inner note: {inner})"
        content = f"{_rb_mem_date(m)}{prefix}{body}"
        priority = _rb_blended_priority(m)
        # An old situation superseded by a later record: keep it but tag it + lower its weight, so the model doesn't treat it as current.
        if meta.get("superseded_by"):
            until = str(meta.get("superseded_at") or "")[:10]
            tag = (
                f" [outdated{f', changed around {until}' if until else ''} — see newer note]"
            )
            content += tag
            priority *= 0.75
        hits.append(RightBrainHit(
            content=content, source="situation_pattern", priority=priority,
            # Carry emotion out too. It was always in this memory's metadata but never made it into the hit --
            # callers (the web demo's emotion tag) couldn't get it and had to regex it out of content.
            metadata={"anchor_score": getattr(m, "anchor_score", 0.0),
                      "emotion": str(meta.get("emotion") or "")},
        ))
    sigs = rb_ctx.current_signals
    now: list[str] = []
    if sigs.dissatisfaction_signal:
        now.append("user is dissatisfied; get to the point")
    if sigs.correction_signal:
        now.append("user is correcting; just accept it")
    if sigs.affect_hint:
        now.append(f"current emotion={sigs.affect_hint}")
    if now:
        sep = "; "
        head = "Current signals: "
        hits.append(RightBrainHit(
            content=head + sep.join(now), source="current_signal", priority=0.95,
            # This is **this turn's** emotion (affect_hint); it's more worth showing than the one on a retrieved old memory.
            metadata={"emotion": str(sigs.affect_hint or "")},
        ))
    return hits


# ── the user's reaction to the agent's previous line ─────────────────────────
# The search path must be 0 LLM (README: read/write separation), so this is purely lexical. Only explicit
# reaction words count, and there must be a previous agent line -- the user feeling bad != the agent said something wrong.
_DISSATISFIED_CUES = (
    "not what i", "that's not", "thats not", "you don't get", "you dont get",
    "never mind", "nevermind", "forget it", "useless", "not helpful",
    "didn't ask", "didnt ask",
)
_CORRECTION_CUES = (
    "i said", "i meant", "you got it wrong", "that's wrong", "thats wrong",
    "no, i ", "actually, i", "actually i ",
)
_APPRECIATION_CUES = (
    "thanks", "thank you", "exactly", "perfect", "that helps", "helpful",
    "good point", "appreciate",
)


def _hits_any(text: str, cues: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(c in low for c in cues)


def _reaction_signals(user_text: str, agent_reply: str = ""):
    """(user's line, agent's previous line) -> CurrentSignals: is the user dissatisfied with / correcting the agent?"""
    from supermem.rightbrain.types import CurrentSignals
    if not (agent_reply or "").strip() or not (user_text or "").strip():
        return CurrentSignals()
    return CurrentSignals(
        dissatisfaction_signal=_hits_any(user_text, _DISSATISFIED_CUES),
        correction_signal=_hits_any(user_text, _CORRECTION_CUES),
    )


#: Whether it's worth spending an LLM on analysis -- inner monologue and the 4-category trait extraction are one api call each,
#: while in voice settings much of the input is mic checks, acknowledgements, interruptions ("oops"/"can you hear me"/"hold on").
#: These are still stored as heartnotes (original words + emotion tag; the emotion is tagged by the model every turn at no extra cost),
#: we just no longer invent an empathetic narration for them -- the prompt asks to "be moved", the model can't answer "no emotion",
#: so a mic check gets read as "must be feeling very complicated inside", and the noise also leaks through the emotion slot
#: into the personality description.
_FILLER = {
    "test", "hello", "hi", "ok", "okay",
}
#: Anything shorter than this that also didn't let the left brain extract any fact is treated as filler.
_MIN_ANALYZE_CHARS = 8


def _worth_analyzing(text: str, has_fact: bool) -> bool:
    """Whether this utterance is worth spending more LLM on. 0 cost, only a word list and length.

    Short utterances can't all be cut -- "I'm devastated" / "so sad" are only a few words, yet exactly what most
    deserves analysis. So before length, ask anchor_router's emotion keyword table: if the text has an explicit
    emotion word, let it through ("fine" / "nope" / "got it" don't match, "devastated" / "sad" do; the two groups separate cleanly).
    """
    s = re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")
    if not s:
        return False
    if s.lower() in _FILLER:
        return False
    if has_fact or len(s) >= _MIN_ANALYZE_CHARS:
        return True
    from supermem.rightbrain.anchor_router import normalize_emotion_strict
    return normalize_emotion_strict(text) is not None      # short, but the emotion was stated


def _is_self_entity(ent, user_id: str, owner_names) -> bool:
    """Whether this entity is "the speaker themself".

    Three cases count: entity_type is explicitly user; the name is the memory store's user_id; the name is
    registered in the voiceprint registry (in the demo that's "Jiaqi"). The speaker's entity_type in the graph
    is often person (same as "boss"), so the type alone isn't enough.
    """
    if str(getattr(getattr(ent, "entity_type", None), "value", "")) == "user":
        return True
    name = (getattr(ent, "name", "") or "").strip()
    return bool(name) and (name == user_id or name in owner_names)


#: A trait must be at least this similar to the utterance to be returned.
#:
#: Measured boundary: "meetings run too long" <-> "doesn't like long meetings" is 0.62, "I don't really want to go
#: to the meeting" is 0.69 -- true hits are all above 0.6. Whereas "I have a two-hour meeting this afternoon and I'm
#: a bit dreading it" scores 0.36-0.45 against the top eight in the store, with only 0.08 between first and eighth:
#: that's a miss, and returning them just fills the top-N slots with noise (which is what the right-brain quota comment is about).
#: Better for the right brain to give no profile this turn than three irrelevant ones.
#: This number must be re-measured after changing the embedder -- similarity distributions differ completely across
#: models, so a hard-coded constant only works for one of them. Measured (same batch of 130 traits):
#:   OpenAI text-embedding-3-small   noise ceiling ~0.24  -> 0.45 is strict but usable
#:   local multilingual-e5-small     true hits 0.89~0.92   noise 0.82~0.86  -> 0.88
#: E5 raises and compresses short-text similarity overall, so gating it at 0.45 is no threshold at all (noise reaches 0.86).
#: Pick a tier by vector dimension: 384 = local E5, everything else as OpenAI. Dimension isn't a strict identifier of
#: the model, but it's enough between the two current built-in implementations; for precision set SUPERMEM_RB_TRAIT_MIN_SIM explicitly.
_TRAIT_MIN_SIM_BY_DIM = {384: 0.88}
_TRAIT_MIN_SIM_DEFAULT = 0.45
RB_TRAIT_MIN_SIM = float(os.environ.get("SUPERMEM_RB_TRAIT_MIN_SIM",
                                        _TRAIT_MIN_SIM_DEFAULT))


def trait_min_sim(dim: int | None) -> float:
    """How high the threshold should be for this embedder. If the env var is set explicitly, it wins."""
    if os.environ.get("SUPERMEM_RB_TRAIT_MIN_SIM"):
        return RB_TRAIT_MIN_SIM
    return _TRAIT_MIN_SIM_BY_DIM.get(dim or 0, _TRAIT_MIN_SIM_DEFAULT)


def _rb_trait_hits(store, user_id: str, query: str, top_k: int = 4) -> list["RightBrainHit"]:
    """Semantic search over the trait table (rb_traits): the few traits that best fit this utterance.

    This replaces the old ``_rb_graph_hits`` -- it returned each slot's whole description regardless of the query,
    so every turn got the same five static summaries ("the user prefers pour-over coffee and quiet gatherings,
    loves AI work..."), and asking "what do I do when people interrupt me" still returned coffee.
    Trait claims have vectors, so this is real query-driven retrieval.

    priority is the similarity rather than a fixed value: how relevant a trait is to this utterance directly decides
    whether it deserves a top-N slot.
    """
    out: list[RightBrainHit] = []
    for t, sim in store.search_scored(user_id, query, top_k=top_k):
        if sim < trait_min_sim(getattr(store, "last_query_dim", None)):
            break                      # already sorted by similarity descending, the rest only get lower
        # Pick the most recent evidence as support -- from a bare claim the model can't tell where it came from.
        ev = t.evidence[0].quote if t.evidence else ""
        content = f"{t.claim} ({t.slot})" + (f" | he said: {ev[:60]}" if ev else "")
        out.append(RightBrainHit(
            content=content, source="profile",
            priority=round(float(sim), 3),
            metadata={"slot_name": t.slot, "trait_id": t.id, "claim": t.claim},
        ))
    return out




#: Maximum number of top-N slots each source may take. Unlisted sources are unlimited.
#:
#: Why quotas: both of these have **query-independent** constant priorities that nothing else can beat --
#:   · response_experience records "how the assistant answered last time" ("✓ Effective approach: the assistant used
#:     a relaxed tone to get the user talking"), an internal note for the reply layer, not an understanding of the user;
#:   · profile used to be each slot's static description, all returned unconditionally with a fixed priority of 0.5,
#:     **the same few items every turn**. Measured: together these two could take every top-5 slot, so the right
#:     brain gave the same thing for every question: replies never read like "it remembers this about me", and every
#:     search on the brain map fired at the same set of nodes.
#:
#: profile is now semantic search over the trait table (``_rb_trait_hits``) with similarity as priority, so it's
#: query-relevant and its quota is relaxed to 3; response_experience is still a query-independent internal note and keeps 1 slot.
_SOURCE_QUOTA = {
    # 0 = not included in the prompt. This category is no longer written or searched (see learn_from_reaction),
    # the quota is kept only so the display layer can still recognise old stores; set to 1 to see old data.
    "response_experience": max(0, int(os.environ.get("SUPERMEM_RB_RESPONSE_MAX", "0"))),
    "profile": max(0, int(os.environ.get("SUPERMEM_RB_PROFILE_MAX", "3"))),
    # heartnotes need a quota too. Their priority follows **anchor freshness**, so the just-stored ones are strongest --
    # ask three questions in a row and the third one's right-brain column is full of your own first two questions
    # ("Emotional note: can you tell me your impression of me"), and not a single trait about the person fits in.
    # Just-said words naturally beat a settled profile, so they must be capped.
    "situation_pattern": max(0, int(os.environ.get("SUPERMEM_RB_HEARTNOTE_MAX", "2"))),
}


def _apply_source_quota(hits: list["RightBrainHit"]) -> list["RightBrainHit"]:
    """Cap by source; anything over quota is **dropped outright**.

    It used to "move to the back without dropping", but there are often only five or six candidates -- moved to the
    back they still make the top-5. Measured: of the five items returned for "I'm so tired", three were
    response_experience (internal notes on how the assistant answered last time) and two were static profile, not a
    single real emotional memory, and identical every turn. Better to give only three real items than fill the slots with internal notes.
    """
    kept, used = [], {}
    for h in hits:
        src = getattr(h, "source", "")
        cap = _SOURCE_QUOTA.get(src)
        if cap is None:
            kept.append(h)
            continue
        used[src] = used.get(src, 0) + 1
        if used[src] <= cap:
            kept.append(h)
    return kept


def _render_rb_directive(hits: list["RightBrainHit"]) -> str:
    """Structured top-N -> a text block to splice into the prompt."""
    return "\n".join(h.content for h in hits) if hits else ""


# ── RightBrain component ──────────────────────────────────────────────────────

class RightBrain:
    """Right-brain component that owns its parts and receives dependencies by explicit injection.

    Constructor arguments come in two kinds:

    **Runtime parameters** owned by the component (right-brain paths/identity)::

        memory_root, user_id, base_url, cognitive_db

    Explicitly **injected non-right-brain dependencies** (all passed as getters/function references, lazy-loading semantics unchanged)::

        embed              -> self._embed_text          text embedding (trait/graph layer writes)
        llm_json           -> self._llm_json            LLM(JSON) (trait extraction)
        llm_text           -> self._llm_text            LLM(text) (attribution manager)
        tracker            -> self._get_session_tracker cross left/right brain session tracker (touch)
        repo               -> self._get_repo            left-brain repo (looks up left-brain entity links on write)
        generate_inner_os  -> self._generate_inner_os   inner-monologue generation (resolved lazily, patchable in tests)
        extract_rb_traits  -> self._extract_rb_traits   trait extraction (resolved lazily, patchable in tests)

    The 3 right-brain lazy singletons (rb_repo / rb_graph_store / attribution_manager), together with the
    cache/lock shared with the host, are owned by this component; see self._rb_repo() etc. below.
    """

    def __init__(
        self,
        *,
        memory_root: Path,
        user_id: str,
        base_url: str | None,
        cognitive_db: Path,
        embed: Callable[[str], list[float]],
        llm_json: Callable[[str], str],
        llm_text: Callable[..., str],
        tracker: Callable[[], Any],
        repo: Callable[[], Any],
        generate_inner_os: Callable[..., str],
        extract_rb_traits: Callable[..., list[tuple[str, str]]],
        cache: dict[str, Any] | None = None,
        lock: Any = None,
    ) -> None:
        # ── runtime parameters ──
        self._memory_root = memory_root
        self._user_id = user_id
        self._base_url = base_url
        self._cognitive_db = cognitive_db

        # ── injected non-right-brain dependencies (getters/function references) ──
        self._embed = embed
        self._llm_json = llm_json
        self._llm_text = llm_text
        self._tracker = tracker
        self._repo = repo
        # generate_inner_os / extract_rb_traits are resolved lazily (getter form),
        # so existing tests' patch.object on the host instance takes effect (write goes through the entry point the host exposes).
        self._generate_inner_os = generate_inner_os
        self._extract_rb_traits = extract_rb_traits

        # ── cache for the right-brain parts owned by this component ──
        # The host may share the same cache/lock (right-brain lazy singletons and the host's _get_* land in the same
        # dict, so existing call sites/tests reading and writing the host's _cache see the same view as this component).
        self._cache: dict[str, Any] = cache if cache is not None else {}
        self._lock = lock if lock is not None else threading.Lock()

    # ── right-brain lazy singletons ─────────────────────────────────────────────

    def _rb_repo(self):
        with self._lock:
            if "rb_repo" not in self._cache:
                from supermem.leftbrain.cognitive_graph import CognitiveGraphStore
                from supermem.rightbrain import ExperienceRepository
                cog_store = CognitiveGraphStore(self._cognitive_db)
                self._cache["rb_repo"] = ExperienceRepository.create(
                    _space.db(self._memory_root),
                    cognitive_store=cog_store,
                )
        return self._cache["rb_repo"]

    def _rb_graph_store(self):
        with self._lock:
            if "rb_graph_store" not in self._cache:
                from supermem.rightbrain import RightBrainGraphStore
                store = RightBrainGraphStore(_space.db(self._memory_root))
                store.ensure_seed_slots(self._user_id)
                self._cache["rb_graph_store"] = store
        return self._cache["rb_graph_store"]

    def _attribution_manager(self):
        rb_graph = self._rb_graph_store()
        rb_repo = self._rb_repo()
        with self._lock:
            if "attribution_manager" not in self._cache:
                from supermem.rightbrain import AttributionManager
                self._cache["attribution_manager"] = AttributionManager(
                    rb_graph, rb_repo._store, llm_fn=self._llm_text,
                )
        return self._cache["attribution_manager"]

    # ── right-brain search ──────────────────────────────────────────────────────

    def search(
        self, query: str, activated_names: list[str], emotion: str | None, top_k: int,
        agent_reply: str = "",
    ) -> tuple[list["RightBrainHit"], str]:
        """Concurrent right-brain search section (formerly the _run_rb closure in Search()): build_query_plan ->
        retrieve -> relation/emotion-trait/profile graph layers -> sort by priority and truncate, returning
        (rb_hits, rb_directive). On error, degrade to no right brain (empty list + empty directive).

        ``agent_reply``: the agent's line from the previous turn. The user is responding to it; it's used in two places --
        (1) reaction signals (dissatisfaction/correction -> retrieve allows one more response-experience slot); (2) the entities
        it mentions become anchors with the context role at half weight. Purely lexical; the search path is still 0 LLM.
        """
        try:
            rb_repo = self._rb_repo()
            signals = _reaction_signals(query, agent_reply)
            # The right brain takes the left brain's "activated entities" as anchors (joint search: the right brain depends on left-brain activation)
            plan    = rb_repo.build_query_plan(
                query, self._user_id,
                signals=signals,
                entities=activated_names or None,
                emotion=emotion,
                context=agent_reply or None,
            )
            rb_ctx = rb_repo.retrieve(plan)
            collected: list[RightBrainHit] = _rb_ctx_to_hits(rb_ctx) if not rb_ctx.is_empty() else []

            # Profile: look up the best-fitting traits for this utterance in the trait table.
            #
            # This used to attach three sources from the old slot->entity graph (relation nodes / same-named emotion
            # entities / slot descriptions). The first two are no longer written, and the third is query-independent --
            # the same five static summaries every turn, no matter what was asked. Trait claims carry vectors,
            # so this is now real retrieval.
            collected.extend(_rb_trait_hits(self._traits(), self._user_id, query))

            # Sort by priority and truncate; rb_directive is rendered from the truncated list so the two stay consistent.
            collected.sort(key=lambda h: h.priority, reverse=True)
            # Right-brain structured top-N (default 5, adjustable via SUPERMEM_RB_TOPN).
            try:
                _rb_topn = max(1, int(os.environ.get("SUPERMEM_RB_TOPN", "5")))
            except ValueError:
                _rb_topn = 5
            rb_hits = _apply_source_quota(collected)[:_rb_topn]
            return rb_hits, _render_rb_directive(rb_hits)
        except Exception as e:
            import traceback as _tb
            print(f"[Search] right-brain search failed (this turn degrades to no right brain): {e}\n{_tb.format_exc()}", flush=True)
            return [], ""

    # ── right-brain write ───────────────────────────────────────────────────────

    def write(self, emotion, result, text, entities, observed_at,
              agent_reply: str = "") -> str | None:
        """Right-brain write section: one heartnote per utterance, with emotion + entity anchors +
        relation nodes + the right-brain slot->entity graph layer. Not bound to result.memory_ids -- a purely emotional
        utterance may yield no left-brain fact yet still be worth remembering; when mid is empty no evidence is attached
        and no left-brain entity links are looked up, but the emotion anchor + text entity-name anchors are still written.

        ``agent_reply``: the agent's line from the previous turn. The same "fine, whatever" after empathy and after being
        handed a solution are two different emotions -- it's fed to inner-monologue generation and stored in metadata as evidence.
        """
        # This used to be `if not emotion: return` -- emotion was treated as the right brain's master switch. But emotion is
        # only one of the right brain's five categories: "I hate it when people chew loudly" is a preference, "I list pros and
        # cons before deciding" is a thinking pattern; neither has a detectable emotion, so the right brain wrote nothing and
        # the graph never reacted to anything new the user said. Now: write even without emotion if traits can be extracted.
        # traits are computed once here and used below -- on the merged-extraction path they're already available (0 extra
        # calls); only without merging is an LLM call actually made, and only when there's no emotion.
        # Emotion: the merged-extraction call also has the model judge it by **what was said**; prefer that.
        #
        # It's more accurate than the upstream keyword table: the table only checks whether a word appears, so "I'm so angry,
        # my boss keeps pressuring me" gets labelled [anxious] because of the word "pressure", even though the person said
        # they're angry; the model reads the whole sentence and gives [angry]. When it can't tell, the model returns an empty
        # string and we fall back to the upstream value. A wrong label is worse than none -- it gets printed as [x] next to the
        # user's own words (that's how "I like strawberries" got labelled [sad] in old data).
        from supermem.leftbrain import merged_extraction
        judged = (merged_extraction.take_emotion(text) or "").strip()
        if judged:
            emotion = judged

        worth = _worth_analyzing(text, has_fact=bool(getattr(result, "memory_ids", None)))
        traits = self._extract_rb_traits(text, emotion) if worth else []
        if not emotion and not traits:
            return
        try:
            from supermem.rightbrain.types import MemoryAnchor
            rb_repo = self._rb_repo()
            mid = result.memory_ids[0] if result.memory_ids else None

            # content stores the original words, inner_os goes into metadata (appended after the original as a supplement
            # when rendering, see _rb_ctx_to_hits) -- so an empathetic rewrite doesn't erase details like numbers/names/times.
            # Filler (mic checks/acknowledgements/interruptions) isn't worth an inner monologue, see _worth_analyzing.
            # worth was already computed at the top of the function (needed there to decide whether to extract traits); don't recompute.
            inner_os = (self._generate_inner_os(text, emotion, entities or [], agent_reply)
                        if worth else "")
            content = text

            # Event time uses observed_at (same source as the left brain's time_start), not the wall clock at write time
            _obs = str(observed_at) if observed_at and re.match(r"^\d{4}-\d{2}-\d{2}", str(observed_at)) else None
            rb_mem = rb_repo._store.upsert_memory(
                user_id=self._user_id,
                memory_class="heartnote",
                content=content,
                metadata={"emotion": emotion, "entities": entities or [],
                          "left_memory_id": mid, "inner_os": inner_os or "",
                          # the emotion's trigger context: which agent line this utterance followed
                          "agent_reply": (agent_reply or "").strip()},
                evidence_memory_ids=[mid] if mid else [],
                created_at=_obs,
            )
            # emotion anchor: for searching by emotion. Strict version: unrecognised emotion words get no anchor.
            from supermem.rightbrain.anchor_router import normalize_emotion_strict
            canonical_emotion = normalize_emotion_strict(emotion)
            if canonical_emotion is not None:
                rb_repo._store.link_anchor(
                    rb_mem.id, self._user_id,
                    MemoryAnchor(anchor_type="emotion", anchor_id=canonical_emotion,
                                 role="trigger", weight=1.0, confidence=1.0),
                )
            # entity anchors: prefer the entity.id actually linked to this left-brain memory (stable),
            # with name-string anchors as fallback. Anchors are a search index, separate from brain-map nodes; kept.
            #
            # This used to also create a node per entity under the "people/places/attitudes" slot. Those nodes are no
            # longer created -- they stored **topics** (pour-over coffee / NUS / Jiaqi), which belong to the left brain's
            # cognitive graph; stuffing them into the right brain just snowballs into a hodgepodge (measured: the "Jiaqi"
            # node alone had 52 items). The right brain now only holds traits about the person, see traits_store.py.
            try:
                from supermem.rightbrain.anchor_router import _ENTITY_TYPE_TO_ANCHOR
                cog_store = self._repo()._cognitive_store
                if mid and cog_store is not None:
                    for eid in cog_store.entity_ids_for_memory(mid):
                        ent = cog_store.get_entity(eid)
                        if ent is None:
                            continue
                        rb_repo._store.link_anchor(
                            rb_mem.id, self._user_id,
                            MemoryAnchor(
                                anchor_type=_ENTITY_TYPE_TO_ANCHOR.get(ent.entity_type.value, "knowledge"),
                                anchor_id=ent.id, role="subject",
                                weight=1.0, confidence=ent.confidence,
                            ),
                        )
            except Exception as e:
                print(f"[RBAnchor] entity ID anchor write failed: {e}")

            for name in (entities or []):
                rb_repo._store.link_anchor(
                    rb_mem.id, self._user_id,
                    MemoryAnchor(anchor_type="entity", anchor_id=name.lower().strip(),
                                 role="subject", weight=0.8, confidence=1.0),
                )

            # Right-brain brain map: traits seen this turn are written to the trait table (rb_traits/rb_evidence).
            #
            # The old slot->entity graph layer is no longer written. Its entities did triple duty -- traits, topics, and bare
            # emotion words -- all mixed together, so the "sad" node swallowed 61 items and "Jiaqi" 52, and titles couldn't be
            # made consistent. In the trait table one node is one trait about the person, and claims carry vectors, so the
            # right brain can finally search semantically. The old tables are kept read-only and no longer written.
            try:
                from supermem.rightbrain.traits_store import Evidence
                # cause = the left-brain fact behind this trait. vector_store has no way to fetch text by id,
                # so search list_entries once (a few dozen items, fast).
                left_fact = ""
                if mid:
                    try:
                        for e in (self._repo()._vector_store
                                  .list_entries(user_id=self._user_id)):
                            if str(e.get("id")) == str(mid):
                                left_fact = str(e.get("text", ""))
                                break
                    except Exception:
                        left_fact = ""
                ev = Evidence(quote=text, emotion=emotion or "",
                              cause=left_fact, cause_id=mid or "",
                              at=str(observed_at or ""))
                for slot_name, label in traits:
                    self._traits().add(self._user_id, slot_name, label, ev)
            except Exception as e:
                print(f"[RBGraph] trait table write failed: {e}", flush=True)
            return rb_mem.id
        except Exception as e:
            print(f"[Ingest] right brain write skipped: {e}")
            return None

    def _traits(self):
        """Right brain v2 trait table (see traits_store.py). Built lazily, shares the same sqlite as the other stores."""
        if "traits" not in self._cache:
            from supermem.rightbrain.traits_store import TraitStore
            from supermem.utils.common import space as _space
            self._cache["traits"] = TraitStore(_space.db(self._memory_root), self._embed)
        return self._cache["traits"]

    def _registry_names(self) -> set:
        """Names registered in the voiceprint registry -- used to recognise "the speaker themself".

        RightBrain doesn't hold the registry directly (it's an audio-side thing), so read the space's
        multi_modal/voiceprint_registry.json directly and return an empty set if it can't be read:
        failing to recognise yourself costs at most one extra node and shouldn't make the write fail.
        """
        try:
            import json
            from supermem.utils.common import space as _space
            p = _space.mm(self._memory_root, "voiceprint_registry.json")
            if not p.is_file():
                return set()
            data = json.loads(p.read_text(encoding="utf-8"))
            out = set()
            for k, v in data.items():
                out.add(k)
                if isinstance(v, dict) and v.get("name"):
                    out.add(v["name"])
            return out
        except Exception:
            return set()

    def _write_trait(self, slot_name: str, label: str, memory_id: str) -> bool:
        """Attach a semantically deduplicated trait entity under a graph-layer slot and link this memory to it as evidence.
        The description is left to AttributionManager to summarise from the evidence; this only attaches + touches."""
        if not slot_name or not label:
            return False
        rb_graph = self._rb_graph_store()
        slot = rb_graph.get_slot_by_name(self._user_id, slot_name)
        if slot is None:
            return False
        ent, _created = rb_graph.get_or_create_entity_semantic(
            self._user_id, slot.id, label, self._embed(label),
        )
        rb_graph.link_memory(ent.id, self._user_id, memory_id)
        tracker = self._tracker()
        tracker.touch(self._user_id, "rb_entity_short", ent.id)
        tracker.touch(self._user_id, "rb_slot_long", slot.id)
        return True

    # ── response success/failure experience (the agent's own lines, scored by the user's reaction) ──

    #: Per-field character cap for attribution -- these fields get spliced into the system prompt every turn, so any slack is a fixed cost.
    _EXPERIENCE_MAX_CHARS = 60

    _ATTRIBUTION_PROMPTS = {"en": """Decide whether the user is reacting to the assistant's reply. Output JSON only.

The assistant's reply: {reply}
The user's response: {user}{emotion_line}

{{"significant": bool,
  "assistant_helped": "whether the reply helped (true) or made things worse (false)",
  "user_reaction": "the user's reaction, with the user as the subject",
  "why": "what specifically in the assistant's reply caused that reaction",
  "user_trait": {{"slot": "expression_style|coping_style|thinking_pattern|likes_dislikes", "label": "a lasting
    user trait revealed by the reaction, such as 'shuts down when given solutions' or
    'speaks plainly when dissatisfied'; use null for a one-off situational reaction"}}}}

significant is false by default. Set it to true only for explicit dissatisfaction,
correction, explicit thanks or approval, an obvious emotional change caused by the
assistant's reply, or a dismissive ending such as "whatever" or "never mind".
Continuing the story, answering a question, making a new request, and small talk are
all false. The user feeling bad does not mean the assistant did something wrong.
Fill every field even when significant is false; callers make additional decisions.
Keep each text field at most {n} characters and write it in English."""}

    def _attribute_reaction(self, user_text: str, agent_reply: str, emotion: str) -> dict:
        """(assistant's previous line + user's turn) -> attribution: whether to record + reaction/why/what to do next time.

        The division of labour is deliberate -- **whether to record isn't left entirely to the model**: small models are shaky
        on this call; measured, the same "that's not what I meant, you didn't get it at all" gave opposite conclusions on two
        runs, and the ones missed were exactly the ones most worth recording.
          · lexical hit (dissatisfaction/correction/thanks) -> forced record, good/bad also decided lexically, the model only writes text;
          · no lexical hit -> grey zone ("...never mind, you go ahead") is left to the model to judge significant.
        """
        sigs = _reaction_signals(user_text, agent_reply)
        forced_failed = bool(sigs.dissatisfaction_signal or sigs.correction_signal)
        forced = forced_failed or _hits_any(user_text, _APPRECIATION_CUES)

        emotion_line = (
            f"\nDetected emotion: {emotion}" if emotion else ""
        )
        raw = self._llm_json(self._ATTRIBUTION_PROMPTS["en"].format(
            reply=agent_reply[:300], user=user_text[:300],
            emotion_line=emotion_line, n=self._EXPERIENCE_MAX_CHARS,
        ))
        data = {}
        if raw:
            try:
                import json as _json
                parsed = _json.loads(raw)
                if isinstance(parsed, dict):
                    data = parsed
            except Exception as e:
                print(f"[RBExperience] attribution parse failed: {e}")

        if forced:
            data["significant"] = True
            data["assistant_helped"] = not forced_failed
            data.setdefault("assistant_did", agent_reply[:self._EXPERIENCE_MAX_CHARS])
            if not str(data.get("user_reaction") or "").strip():
                data["user_reaction"] = (
                    ("user corrected it" if sigs.correction_signal else "user pushed back")
                    if forced_failed else "user said it helped")
        return data if "significant" in data else {"significant": False}

    def learn_from_reaction(self, text: str, emotion: str, entities, agent_reply: str,
                            memory_id: str | None = None, observed_at=None,
                            heartnote_id: str | None = None) -> None:
        """(assistant's previous line + user's turn) -> emotion attribution; attach a trait to the trait layer only when warranted.

        There is only **one outlet**: the user-side "lasting trait revealed" -> the trait layer's 5 slots
        (coping_style/expression_style/thinking_pattern/likes_dislikes), usable across topics. If no lasting trait can be
        extracted (just a one-off situational reaction), nothing is attached; nothing is forced.

        There used to be a second outlet: the assistant-side "approach + what to do next time" written as response_experience.
        That line was **write-only** -- `next_time` (the genuinely useful half) was stored in metadata with no reader anywhere in
        the repo; what reached the prompt was `assistant_did`, which in practice looked like
        "The assistant uses a relaxed tone to guide the user", taking a _SOURCE_QUOTA slot every turn while carrying no information.
        And what the assistant should do can already be inferred from user-side traits -- "when he's down he wants understanding
        and validation" already tells the assistant what to give, no separate copy needed. So that whole line was removed.

        Not governed by write()'s ``if not emotion`` -- "that's not what I meant" is a behavioural signal, independent of whether
        acoustic emotion produced output (in text_mode emotion is often empty). Returns immediately without a previous assistant
        line, so the first turn makes no LLM call.
        """
        reply = (agent_reply or "").strip()
        if not reply or not (text or "").strip():
            return
        try:
            attribution = self._attribute_reaction(text, reply, emotion or "")
            if not attribution.get("significant"):
                return

            def _clip(v) -> str:
                s = str(v or "").strip()
                return s if len(s) <= self._EXPERIENCE_MAX_CHARS else s[:self._EXPERIENCE_MAX_CHARS] + "…"

            failed = not bool(attribution.get("assistant_helped", False))
            print(f"[RBReaction] {'failed' if failed else 'effective'}: "
                  f"{_clip(attribution.get('user_reaction'))}", flush=True)

            # "This person shuts down when handed a solution" is a lasting trait; it settles into a trait-layer slot,
            # gets summarised into a personality description by attribution, and is usable across topics.
            trait = attribution.get("user_trait") or {}
            if isinstance(trait, dict):
                slot_name, label = str(trait.get("slot") or ""), _clip(trait.get("label"))
                # Evidence is the **user's own words**, not this assistant experience -- attribution reads evidence to write
                # descriptions, so attaching exp would mean "describing the user's traits with the assistant's approach".
                if label and label.lower() not in ("null", "none"):
                    from supermem.lang import is_zh
                    label_is_zh = any("\u4e00" <= ch <= "\u9fff" for ch in label)
                    if label_is_zh != is_zh():
                        print(f"[RBTrait] language mismatch, dropped: {slot_name} ← {label}", flush=True)
                        return
                    from supermem.rightbrain.traits_store import Evidence
                    if self._traits().add(
                            self._user_id, slot_name, label,
                            Evidence(quote=text.strip()[:200], emotion=emotion or "",
                                     cause_id=memory_id or "", at=str(observed_at or ""))):
                        print(f"[RBTrait] {slot_name} ← {label}", flush=True)
        except Exception as e:
            print(f"[RBReaction] reaction attribution failed: {e}")

    # ── right-brain cleanup ─────────────────────────────────────────────────────

    def check_and_cleanup(self) -> None:
        """Trigger a right-brain cleanup every 50 new heartnotes."""
        try:
            import json as _json
            state = {"last_count": _space.kv_get(self._memory_root, "rb_cleanup_last_count", 0)}

            rb_repo = self._rb_repo()
            all_mems = rb_repo._store.get_all(self._user_id)
            current_count = sum(1 for m in all_mems if m.memory_class == "heartnote")

            if current_count - state.get("last_count", 0) >= 50:
                _space.kv_set(self._memory_root, "rb_cleanup_last_count", current_count)
                self.run_cleanup()
        except Exception as e:
            print(f"[Cleanup] check error: {e}")

    def run_cleanup(self) -> None:
        """Clean right-brain heartnotes with an LLM: duplicate/meaningless -> delete; contradiction -> mark supersede.

        Contradictions are not "delete old, keep new" (preference-evolution questions need both old and new + their order):
        the old entry is kept with superseded_by/superseded_at marks, and is tagged "outdated" and down-weighted when rendered."""
        try:
            import json as _json
            import sqlite3

            rb_repo = self._rb_repo()
            heartnotes = [
                m for m in rb_repo._store.get_all(self._user_id)
                if m.memory_class == "heartnote"
            ]
            if len(heartnotes) < 10:
                return

            # Build a compact list for the LLM (first 8 chars of the ID to save tokens)
            lines = []
            id_map: dict[str, str] = {}  # short_id -> full_id
            for i, m in enumerate(heartnotes):
                short = m.id[:8]
                id_map[short] = m.id
                emotion  = (m.metadata or {}).get("emotion", "")
                entities = (m.metadata or {}).get("entities", [])
                lines.append(
                    f"[{i}] ID:{short} | emotion:{emotion} | entities:{','.join(entities)} | {m.content}"
                )

            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
                timeout=60.0,
            )
            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[
                    {"role": "system", "content": (
                        "You are a memory cleanup assistant. Analyze the following list of emotional memories and make two kinds of decisions.\n"
                        "I. Delete (when in doubt, delete less; never delete valuable records by mistake):\n"
                        "1. Duplicates: highly similar content; keep one, delete the rest;\n"
                        "2. Meaningless: extremely low information (e.g. pure punctuation, a single word, a broken sentence).\n"
                        "II. Supersede (do not delete): contradictory descriptions of the same entity/same preference; "
                        "the one with the later index is the new state -- do not delete the old one, mark it as superseded by the new one, "
                        "to preserve the preference's evolution.\n"
                        "Return JSON: {\"delete_ids\": [\"8-char ID\", ...], "
                        "\"supersede\": [{\"old_id\": \"8-char ID\", \"new_id\": \"8-char ID\"}, ...]}\n"
                        "If nothing needs to be done, return {\"delete_ids\": [], \"supersede\": []}"
                    )},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            result      = _json.loads(resp.choices[0].message.content)
            short_ids   = result.get("delete_ids", [])
            full_ids    = [id_map[s] for s in short_ids if s in id_map]

            if full_ids:
                with sqlite3.connect(rb_repo._store._path) as conn:
                    for mid in full_ids:
                        conn.execute(
                            "DELETE FROM right_brain_anchor_links WHERE right_memory_id=?", (mid,)
                        )
                        conn.execute(
                            "DELETE FROM right_brain_memories WHERE id=?", (mid,)
                        )
                print(f"[Cleanup] cleanup done, deleted {len(full_ids)} right-brain memories")
            else:
                print("[Cleanup] nothing to delete")

            # Contradiction pairs: mark the old entry superseded (kept; tagged + down-weighted when rendered).
            pairs = result.get("supersede", []) or []
            keep_old = True
            marked = 0
            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat()
            for p in pairs:
                old_full = id_map.get(str(p.get("old_id", "")))
                new_full = id_map.get(str(p.get("new_id", "")))
                if not old_full or not new_full or old_full == new_full:
                    continue
                if old_full in full_ids or new_full in full_ids:
                    continue  # already deleted, don't mark
                if keep_old:
                    rb_repo._store.merge_metadata(
                        old_full, {"superseded_by": new_full, "superseded_at": now_iso},
                    )
                else:
                    with sqlite3.connect(rb_repo._store._path) as conn:
                        conn.execute(
                            "DELETE FROM right_brain_anchor_links WHERE right_memory_id=?",
                            (old_full,),
                        )
                        conn.execute(
                            "DELETE FROM right_brain_memories WHERE id=?", (old_full,)
                        )
                marked += 1
            if marked:
                action = "marked as superseded (evolution preserved)" if keep_old else "deleted (old behaviour)"
                print(f"[Cleanup] {marked} contradicted old entries {action}")

            # update last_count
            remaining = sum(
                1 for m in rb_repo._store.get_all(self._user_id)
                if m.memory_class == "heartnote"
            )
            _space.kv_set(self._memory_root, "rb_cleanup_last_count", remaining)

        except Exception as e:
            print(f"[Cleanup] run error: {e}")


__all__ = ["RightBrain", "RightBrainHit"]
