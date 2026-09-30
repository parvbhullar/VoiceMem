"""Orchestration implementation (mem0-style): Orchestrator = left brain + right brain + audio perception + a set of swappable capabilities (utils).

This is the implementation layer hidden behind the facade ``supermem.core.SuperMem``. The facade only exposes the
lowercase user-facing API; the actual pipeline (Search/Ingest orchestration, helper methods, forwarding to the three components,
the SearchResult dataclass, the Utils capability table) all lives in this module.

    left brain   factual memory: entities + cognitive graph (slot classification/retrieval), backed by a mem0 vector store
    right brain  emotional memory: per-turn valence-arousal, emotion attribution, personality profile
    utils        pluggable capabilities: embedding / schema (classification) / entity / emotion / voiceprint / asr / memory_engine
                 each has a built-in default; pass a function to swap in your own (local model, another vector store...)

mode decides which capabilities get loaded: left_brain_single / text_mode / multi_modal (with audio).

Orchestrator directly holds and constructs three self-contained components (self._left/_right/_audio) and orchestrates them
to run the full Search/Ingest pipelines.

Usage (orchestration layer, normally called through the facade)::

    from supermem.orchestrator import Orchestrator

    o = Orchestrator()

    # the voice module provides slots and entities, pass them in directly:
    result = o.Search(query, slots=["work"], entities=["Alibaba"])

    # step by step:
    slot_ids, clf = o.SearchCogGraph(slots=["work"], entities=["Alibaba"])
    candidate_ids  = o.SearchData(slot_ids, clf)
    hits           = o.Rank(query, candidate_ids, top_k=5)
"""

from __future__ import annotations

from supermem.utils.common import space as _space

import functools
import inspect
import os
import time
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from supermem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
from supermem.leftbrain.local_memory_store import MemorySearchHit
# The whole left-brain block (slot filtering/entity narrowing/time-based candidate widening/vector ranking/query classification/LLM tagging/
# slot->entity graph layer/subgraph bookkeeping and checkpoint/schema description refresh/cold memory archiving) moved into
# the LeftBrain component; the _search_mode helper moved with it to supermem.leftbrain.brain. It is
# imported back here for Search() to use when assembling SearchResult.
from supermem.leftbrain.brain import LeftBrain, _search_mode
from supermem.utils.audio.perceiver import AudioPerception, AudioPerceiver
# The whole right-brain block (heartnote writes/inner monologue/graph layer/search/cleanup) moved into the RightBrain component;
# the RightBrainHit dataclass and _rb_* helpers moved with it to supermem.rightbrain.brain.
from supermem.rightbrain.brain import (
    RightBrain,
    RightBrainHit,
    _is_en_text,
    _rb_blended_priority,
    _rb_ctx_to_hits,
    _rb_lang,
    _rb_mem_date,
    _rb_trait_hits,
    _render_rb_directive,
)
from supermem.utils.defaults import default_utils
from supermem.llm_config import resolve_api_key, resolve_base_url, resolve_model

# How many new memories (left-brain facts + right-brain heartnotes) before consolidating once. See the comment in Ingest():
# consolidation re-summarises historical memories; running it every turn is slow (7s) with little to summarise. At session end
# the backlog is cleared no matter how much has accumulated, so it never piles up unprocessed.
SHORT_TERM_MIN_MEMORIES = int(os.environ.get("SUPERMEM_ATTRIBUTION_MIN_MEMORIES", "20"))

# Which utils each mode needs (only these are loaded).
# tts is in none of them: the core path stops at text, speech output is an optional layer, whoever wants audio calls
# utils.get("tts") -- including it here would make warmup crash for users without piper/voxcpm installed.
_NEED = {
    "left_brain_single": ["embedding", "slots", "entity", "memory_engine"],
    "text_mode":         ["embedding", "slots", "entity", "emotion", "memory_engine"],
    "multi_modal":       ["embedding", "slots", "entity", "emotion", "voiceprint", "asr", "memory_engine"],
}


#: After saying "let me play you a song", how long sound-only turns still count as that song.
#: Shorter is safer: the music usually follows right after; after a long gap it's probably something else.
EXPECT_MUSIC_S = float(os.environ.get("SUPERMEM_EXPECT_MUSIC_S", "120"))


#: Historical aliases of a capability -> the canonical capability name.
#:
#: ``schema`` is really the "query slot classifier", always called ``slots`` in from_config; both names mean
#: the same thing. ``embedder`` / ``vector_store`` / ``classifier`` are the names used on the constructor-argument
#: path. These two paths used to be separate and merged with pick() in __init__, i.e. two entry points and two sets of code
#: for the same thing. Now names are normalised to the capability name at the door, leaving one path. Old names still work.
_ALIASES = {"schema": "slots", "embedder": "embedding",
            "vector_store": "memory_engine", "classifier": "slots"}


def _canon(overrides: dict) -> dict:
    """Normalise aliases to canonical capability names. If both old and new names are given, the canonical name wins."""
    out = {}
    for k, v in overrides.items():
        out.setdefault(_ALIASES.get(k, k), v)
    for k, v in overrides.items():
        if k not in _ALIASES:
            out[k] = v
    return out


class Utils:
    """Capability table: built-in defaults (see utils/defaults.py) + user overrides, lazily loaded on demand and cached."""
    def __init__(self, mode, base_url, memory_root, overrides):
        self._factory = {**default_utils(base_url, memory_root), **_canon(overrides)}
        self.need = _NEED[mode]
        self._cache = {}
    def get(self, name):
        """A capability value can be a **factory** (function / lambda / class, lazy-loaded; all built-in defaults are this),
        or directly an **already-built object** -- both are accepted, so users don't have to wrap a ready object in a
        lambda. The test is "is it a function/class", not callable(): a component object may itself have
        ``__call__``, and callable() would call it once as a factory."""
        if name not in self._cache:
            f = self._factory[name]
            self._cache[name] = f() if (inspect.isfunction(f) or inspect.ismethod(f)
                                        or inspect.isclass(f)
                                        or isinstance(f, functools.partial)) else f
        return self._cache[name]


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Full return value of Search()."""
    hits: list[MemorySearchHit]
    classification: QueryClassification
    related_summaries: dict[str, str]   # {slot: summary_text}
    slot_mem_ids: set[str]              # raw slot IDs returned by SearchCogGraph
    final_candidate_ids: set[str]       # final candidate IDs after SearchData entity narrowing
    search_mode: str = "fallback"
    rb_directive: str = ""              # right-brain situational guidance text (rendered from rb_hits)
    rb_hits: list[RightBrainHit] = field(default_factory=list)  # right-brain structured top-N
    scene_directive: str = ""          # reply style suggestion for the current acoustic scene
    current_scene: str = ""            # current scene tag, e.g. "transit"
    timing: dict = None                 # {slot_filter, entity_narrow, rank, rb, total} in ms

    # ── what each brain retrieved (the user-facing view) ────────────────────────
    # hits / rb_hits are structured results with scores and metadata; the two below are the "readable and printable"
    # layer, matching result.result_leftbrain / result.result_rightbrain in the docs.

    @property
    def result_leftbrain(self) -> list[str]:
        """Facts retrieved by the left brain (sorted by relevance)."""
        return [h.text for h in self.hits]

    @property
    def result_rightbrain(self) -> list[str]:
        """Emotional/personality context retrieved by the right brain."""
        return [h.content for h in self.rb_hits]


# The left-brain candidate pool construction/rescue constants (_RESCUE_K / _POOL_MODE_ENV / _pool_mode /
# _STRICT_* etc.) and the _search_mode helper moved with the left-brain block to supermem.leftbrain.brain
# (_search_mode is imported back at the top of this module for Search() to use when assembling SearchResult).
# RightBrainHit / _rb_* helpers and _is_en_text moved with the right-brain block to
# supermem.rightbrain.brain (imported back at the top of this module so Search() etc. can keep calling them directly);
# AudioPerception moved to supermem.utils.audio.perceiver.


# ── Orchestrator class ────────────────────────────────────────────────────────

class Orchestrator:
    """Left-brain + right-brain personal memory system (mem0-style orchestrator).

    Directly holds and constructs three self-contained components (``self._left`` / ``self._right`` /
    ``self._audio``) and orchestrates them to run the full Search/Ingest pipelines.

    Parameters
    ----------
    api_key:
        OpenAI API key; if given, it is written to the ``OPENAI_API_KEY`` environment variable.
    mode:
        ``left_brain_single`` / ``text_mode`` / ``multi_modal``, decides which utils are loaded and
        whether audio/emotion capabilities are enabled.
    memory_root:
        Memory storage directory.  Defaults to
        ``<current working directory>/supermem_memory`` (overridable via env ``SUPERMEM_MEMORY_ROOT``).
    user_id:
        Owner of all memories managed by this instance.
    base_url:
        OpenAI-compatible API base URL (e.g. a proxy).  Falls back to the
        ``OPENAI_BASE_URL`` environment variable.
    enable_scene / enable_music / enable_abnormal_sound / enable_voiceprint / enable_emotion:
        5 capability switches. Default ``None`` -> derived from ``mode`` (``multi_modal`` turns all on, other modes turn audio
        off; emotion is on for anything other than ``left_brain_single``). Passing True/False explicitly overrides the mode default.
    embedder / vector_store / classifier:
        **Old parameter names** for the ``embedding`` / ``memory_engine`` / ``slots`` capabilities, equivalent,
        kept for compatibility. New code should use the capability names.
    util_overrides:
        Override built-in defaults by capability name: ``embedding`` / ``slots`` / ``entity`` / ``emotion`` /
        ``voiceprint`` / ``asr`` / ``vad`` / ``tts`` / ``memory_engine`` (full table in
        ``supermem/utils/defaults.py``). A value can be a factory or an already-built object.
    """

    def __init__(
        self,
        api_key: str | None = None,
        mode: str = "text_mode",
        memory_root: Path | str | None = None,
        space: str | None = None,
        user_id: str = "voice_user",
        base_url: str | None = None,
        enable_scene: bool | None = None,
        enable_music: bool | None = None,
        enable_abnormal_sound: bool | None = None,
        enable_voiceprint: bool | None = None,
        enable_emotion: bool | None = None,
        embedder: Any = None,
        vector_store: Any = None,
        classifier: Any = None,
        **util_overrides,
    ) -> None:
        if mode not in _NEED:
            raise ValueError("mode must be one of " + " / ".join(_NEED))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.mode = mode
        # embedder / vector_store / classifier are old parameter names for the same three capabilities; they are taken in and
        # normalised together (see _ALIASES): they accept objects, the capability-name path accepts factories, Utils.get handles both,
        # so there is now a single path instead of two sets of code going their own way.
        overrides = _canon({**util_overrides, "embedder": embedder,
                            "vector_store": vector_store, "classifier": classifier})
        overrides = {k: v for k, v in overrides.items() if v is not None}
        self.utils = Utils(mode, base_url, memory_root, overrides)

        audio = mode == "multi_modal"
        # Only capabilities overridden by the user are injected into components; otherwise components use their own defaults.
        pick = lambda n: self.utils.get(n) if n in overrides else None

        # 5 audio capability switches: explicit values win, otherwise derived from mode
        # (multi_modal turns all on, other modes turn audio off; emotion is on for anything other than left_brain_single).
        if enable_scene is None:          enable_scene = audio
        if enable_music is None:          enable_music = audio
        if enable_abnormal_sound is None: enable_abnormal_sound = audio
        if enable_voiceprint is None:     enable_voiceprint = audio
        if enable_emotion is None:        enable_emotion = mode != "left_brain_single"
        embedder     = pick("embedding")
        vector_store = pick("memory_engine")
        classifier   = pick("slots")

        self._vector_store = vector_store   # injected memory engine (default None -> mem0)
        # Defaults to the **current working directory**, not where the package is installed.
        #
        # The old default was <package dir>/results/voice_memory -- for pip users, memories got written into
        # site-packages/results/: gone on package upgrade, often read-only for system Python, and shared by every
        # project. Data should follow the project, not where the program is installed (git/docker/npm
        # all work this way). Overridable via SUPERMEM_MEMORY_ROOT.
        # Without memory_root, the space determines the location: supermem_memoryspace/<space>/,
        # and the default space is called demo. See utils/common/space.py.
        from supermem.utils.common.space import MemorySpace
        if memory_root or os.environ.get("SUPERMEM_MEMORY_ROOT"):
            self._memory_root = Path(memory_root or os.environ["SUPERMEM_MEMORY_ROOT"])
            self._memory_root.mkdir(parents=True, exist_ok=True)
            self._space = None
        else:
            self._space = MemorySpace(space)
            self._memory_root = self._space.dir
        # One sqlite per space: there used to be nine databases each with its own connection; splitting only turns "copy a
        # space" into "don't forget any file". Table names don't overlap.
        self._db_path = _space.db(self._memory_root)
        self._multi_modal = _space.mm(self._memory_root)
        _space.describe(self._memory_root, user_id=user_id, mode=mode)
        self._multi_modal.mkdir(parents=True, exist_ok=True)
        self._cognitive_db = self._db_path
        self._user_id = user_id
        self._base_url = resolve_base_url(base_url)
        # Official/default is OpenAI embeddings (OpenAILocalEmbedder, built
        # lazily in _get_repo() below); pass a different TextEmbedder-
        # conforming object here to use something else for the left-brain
        # store's raw-fact embedding (used for both ingest and Rank()'s
        # search-time ranking). openai_voice_demo uses this to swap in a
        # local model for speed -- see that demo's local_embedder.py.
        self._embedder = embedder
        # The query->slots+entities classifier (used by Classify). Default None -> built-in
        # QuerySlotClassifier (single LLM call). Pass any implementation of .classify(query)->QueryClassification
        # to switch to a local model; child-slot drill-down only happens if the optional .classify_child(...) exists.
        self._classifier = classifier

        # 5 audio capability switches: decided by mode (multi_modal all on, otherwise all off).
        self._enable_scene = enable_scene
        self._enable_music = enable_music
        self._enable_abnormal_sound = enable_abnormal_sound
        self._enable_voiceprint = enable_voiceprint
        self._enable_emotion = enable_emotion

        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        # _lock only guards building the lazy singletons in _cache, and _finish_ingest takes it
        # itself -- never hold it around an ingest (deadlock). _write_lock serialises the slow
        # write path per instance: the voice loop's background ingest threads, the /memories
        # job worker, and the page's edit/delete all go through it.
        self._write_lock = threading.RLock()
        self._retired = False      # set by retire(): this brain's files were wiped
        self._ingest_count = 0

        # Conversation exchanges: storing only the user's half leaves "let's do it your way" dangling. The reply layer calls
        # remember_reply() after every line; Ingest takes this turn's (left-brain disambiguation), Search and the right brain take
        # the previous turn's (emotion attribution), so keeping two turns is enough.
        # Written from the reply coroutine, read from the Search thread pool; the reader copies a snapshot (see last_agent_reply).
        self._exchanges: deque[tuple[str, str]] = deque(maxlen=2)

        # ── left-brain component (composition: owns its left-brain parts + explicit injection of cross-domain/runtime deps) ──
        # The whole left-brain block (slot filtering/entity narrowing/time-based candidate widening/vector ranking/query classification/LLM tagging/
        # slot->entity graph layer/subgraph bookkeeping and checkpoint/schema description refresh/cold memory archiving) moved into
        # LeftBrain. It owns 5 left-brain lazy singletons (repo/extractor/dynamic_slot_store/
        # graph_entity_store/subgraph_manager, sharing the same _cache/_lock with the host); anywhere that needs
        # text embedding / LLM(JSON) / LLM(text) / an injectable classifier / the session tracker, i.e.
        # cross-domain or runtime capabilities, gets them injected explicitly here as getters/function references (lazy semantics unchanged).
        # Constructed before _audio/_right: this class's _get_repo etc. forward to self._left, and the repo=self._get_repo
        # injected into _audio/_right goes through that forwarding to the same left-brain singleton here.
        self._left = LeftBrain(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            cognitive_db=self._cognitive_db,
            embedder=self._embedder,
            vector_store=self._vector_store,
            embed=self._embed_text,
            llm_json=self._llm_json,
            llm_text=self._llm_text,
            classifier=self._classifier,
            tracker=self._get_session_tracker,
            cache=self._cache,
            lock=self._lock,
        )

        # ── audio perception component (composition: owns its audio parts + explicit injection of left-brain deps) ──
        # The whole audio block (scene/voiceprint/emotion/ambient sound/audiomem tags/playback) moved into
        # AudioPerceiver. It owns 10 audio-side lazy singletons (env/clap/speaker/vp/
        # emotion/music/routine/place/trigger/audio_archive) with their caches/locks and
        # speaker binding state (_session_person_pin / _person_origin_session); anywhere that needs
        # left-brain storage / the extractor / voiceprint name mapping / tagging / fact appending / semantic ranking, i.e.
        # non-audio capabilities, gets them injected explicitly here as getters/function references (lazy semantics unchanged).
        self._audio = AudioPerceiver(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            enable_scene=self._enable_scene,
            enable_music=self._enable_music,
            enable_abnormal_sound=self._enable_abnormal_sound,
            enable_voiceprint=self._enable_voiceprint,
            enable_emotion=self._enable_emotion,
            repo=self._get_repo,
            extractor=self._get_extractor,
            registry=self._get_registry,
            tag=self._tag_memories,
            extract_and_append=self._extract_and_append,
            rank=self.Rank,
            ingest_env=lambda: self.IngestEnv,
            cache=self._cache,
            lock=self._lock,
        )

        # ── right-brain component (composition: owns its right-brain parts + explicit injection of cross-domain deps) ──
        # The whole right-brain block (heartnote emotional writes/inner monologue/graph layer/search/LLM cleanup) moved into
        # RightBrain. It owns 3 right-brain lazy singletons (rb_repo/rb_graph_store/
        # attribution_manager, sharing the same _cache/_lock with the host); anywhere that needs text
        # embedding / LLM(JSON) / LLM(text) / the session tracker / the left-brain repo / inner-monologue generation /
        # trait extraction, i.e. non-right-brain capabilities, gets them injected explicitly here as getters/function references
        # (lazy semantics unchanged; generate_inner_os / extract_rb_traits are resolved lazily so tests can patch them).
        self._right = RightBrain(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            cognitive_db=self._cognitive_db,
            embed=self._embed_text,
            llm_json=self._llm_json,
            llm_text=self._llm_text,
            tracker=self._get_session_tracker,
            repo=self._get_repo,
            generate_inner_os=lambda text, emotion, entities, agent_reply="": (
                self._generate_inner_os(text, emotion, entities, agent_reply)),
            extract_rb_traits=lambda text, emotion: self._extract_rb_traits(text, emotion),
            cache=self._cache,
            lock=self._lock,
        )

        # Public left/right brain handles: point directly at the real components (they already have search/write etc.).
        self.left_brain = self._left
        self.right_brain = self._right

    # ── lazy singletons ───────────────────────────────────────────────────────

    def _get_repo(self):
        # The left-brain singleton moved into the LeftBrain component with the left-brain block; forward to keep existing call sites/tests
        # that access it directly on the SuperMem instance working, as well as the repo=self._get_repo injected into _audio/_right
        # (they read/write the same "repo" in the shared _cache).
        return self._left._get_repo()

    def _get_rb_repo(self):
        # The right-brain singleton moved into the RightBrain component with the right-brain block; forward to keep existing call sites/tests
        # that access it directly on the SuperMem instance working (they read/write the same "rb_repo" in the shared _cache).
        return self._right._rb_repo()

    def _get_extractor(self):
        # Left-brain singleton moved into the LeftBrain component; forwarded (same "extractor" in the shared _cache).
        return self._left._get_extractor()

    def _get_registry(self):
        with self._lock:
            if "registry" not in self._cache:
                from supermem.utils.common.voice_input import VoiceprintRegistry
                # The voiceprint registry is voiceprint data, so per the space layout it goes under multi_modal/
                self._cache["registry"] = VoiceprintRegistry(
                    _space.mm(self._memory_root, "voiceprint_registry.json"),
                    entity_resolver=self._person_entity_id,
                )
        return self._cache["registry"]

    def _person_entity_id(self, name: str) -> str:
        """Person name -> id of the person entity in the cognitive graph; "" if not found (read-only, never creates entities).

        Voiceprint recognising "who this is" and the cognitive graph remembering "things about this person" used two separate id sets; this connects them:
        only once connected is speaker_entity_map non-empty, giving search(speaker_filter=...) an edge to follow.
        """
        try:
            from supermem.leftbrain.cognitive_graph.store import normalize_name
            store = self._get_repo()._cognitive_store
            for e in store.find_entities(self._user_id, name_norm=normalize_name(name)):
                # Note: use .value -- although EntityType is a str enum, on 3.11+ str() gives
                # "EntityType.PERSON" rather than "person".
                if getattr(e.entity_type, "value", e.entity_type) in ("person", "user"):
                    return e.id
        except Exception:
            pass
        return ""

    # ── audiomem: scene + voiceprint lazy singletons ─────────────────────────────

    def _get_env_detector(self):
        return self._audio._env_detector()

    def _clap_memory_enabled(self) -> bool:
        # AST always supplies the immediate hint. Once a CLAP checkpoint is
        # configured, the 4s-segmented CLAP pass takes over the background-sound
        # description memory write; set SUPERMEM_ENVIRONMENT_MEMORY_BACKEND=ast
        # to opt back out.
        return (
            os.environ.get("SUPERMEM_ENVIRONMENT_MEMORY_BACKEND", "clap").lower() == "clap"
            and bool(os.environ.get("SUPERMEM_CLAP_CHECKPOINT"))
        )

    def _get_clap_env_detector(self):
        return self._audio._clap_env_detector()

    def _finish_clap_environment(self, *a, **k) -> None:
        return self._audio._finish_clap_environment(*a, **k)

    def _get_trigger_store(self):
        return self._audio._trigger_store()

    def _get_audio_archive(self):
        return self._audio._audio_archive()

    def _get_speaker_encoder(self):
        return self._audio._speaker_encoder()

    def _get_vp_store(self):
        return self._audio._vp_store()

    def _get_emotion_detector(self):
        return self._audio._emotion_detector()

    def _get_music_store(self):
        return self._audio._music_store()

    def _get_routine_store(self):
        return self._audio._routine_store()

    def _get_place_store(self):
        return self._audio._place_store()

    # Speaker binding state and voiceprint reclaiming: state and logic both live with the audio component; forwarding kept here so
    # existing call sites/tests that access them directly on the SuperMem instance keep working (they read/write the same underlying dict).
    @property
    def _session_person_pin(self) -> dict[str, str]:
        return self._audio._session_person_pin

    @property
    def _person_origin_session(self) -> dict[str, str]:
        return self._audio._person_origin_session

    def _claimed_by_other_identity(self, *a, **k) -> bool:
        return self._audio._claimed_by_other_identity(*a, **k)

    def _reconcile_speaker_candidates(self, *a, **k) -> tuple[str, str]:
        return self._audio._reconcile_speaker_candidates(*a, **k)

    # ── audiomem: scene-triggered reminders ──────────────────────────────────────

    def CreateSceneTrigger(self, *a, **k) -> dict:
        return self._audio.CreateSceneTrigger(*a, **k)

    def GetOriginalAudio(self, *a, **k) -> dict:
        return self._audio.GetOriginalAudio(*a, **k)

    def TryPlayback(self, *a, **k) -> dict | None:
        return self._audio.TryPlayback(*a, **k)

    # ── Dynamic slot (new slots that emerge from the subgraph mechanism) ──────────

    def _get_dynamic_slot_store(self):
        # Left-brain singleton moved into the LeftBrain component; forwarded (same "dynamic_slot_store" in the shared _cache).
        return self._left._get_dynamic_slot_store()

    def _get_dynamic_slots(self) -> list[tuple[str, str]]:
        """Return the dynamic slots that have emerged for this user [(name, description), ...], forwarded to LeftBrain."""
        return self._left._get_dynamic_slots()

    # ── slot->entity graph layer (left brain: under SlotV2; right brain: 5 affective slots) ──

    def _get_graph_entity_store(self):
        # Left-brain singleton moved into the LeftBrain component; forwarded (same "graph_entity_store" in the shared _cache).
        return self._left._get_graph_entity_store()

    def _get_rb_graph_store(self):
        # Right-brain singleton moved into the RightBrain component; forwarded (same "rb_graph_store" in the shared _cache).
        return self._right._rb_graph_store()

    def _get_session_tracker(self):
        with self._lock:
            if "session_tracker" not in self._cache:
                from supermem.utils.common.session_tracker import SessionTracker
                self._cache["session_tracker"] = SessionTracker(
                    _space.db(self._memory_root)
                )
        return self._cache["session_tracker"]

    def _get_subgraph_manager(self):
        # Left-brain singleton moved into the LeftBrain component; forwarded (same "subgraph_manager" in the shared _cache).
        return self._left._get_subgraph_manager()

    def _get_attribution_manager(self):
        # Right-brain singleton moved into the RightBrain component; forwarded (same "attribution_manager" in the shared _cache).
        return self._right._attribution_manager()

    def _extract_rb_traits(self, text: str, emotion: str) -> list[tuple[str, str]]:
        """LLM judges whether this utterance reveals "likes_dislikes/expression_style/thinking_pattern/coping_style",
        and if so distils a short label. Returns [(slot_name, label), ...], possibly an empty list.

        The extraction step has already computed this as a by-product (see leftbrain/merged_extraction.py); on a hit it is used
        directly, saving this LLM round trip; only on a miss (merging disabled, or the model didn't follow the format) do we call it ourselves.
        """
        import json as _json

        from supermem.lang import is_zh as _is_zh, label_rule as _label_rule

        def _looks_cjk(t: str) -> bool:
            return any("\u4e00" <= ch <= "\u9fff" for ch in (t or ""))
        def _keep(items):
            """Language guard: drop labels written in a different script from the utterance.

            The prompt already says "follow the speaker's language" and the examples were switched per language, but at
            temperature=0 the model still occasionally outputs Chinese labels (measured: about one in two runs). Storing them
            is far worse than dropping them: this profile is spliced into the system prompt every turn, so Chinese would
            suddenly appear in an English conversation; dropping only costs one trait this turn, and it will be extracted again next turn.
            Same trade-off as attribution_manager keeping the original sentence when refinement changes the language.
            """
            want_cjk = _is_zh()
            out = []
            for slot, label in items:
                if _looks_cjk(label) != want_cjk:
                    print(f"[RBTrait] language mismatch, dropped: {slot} ← {label}", flush=True)
                    continue
                out.append((slot, label))
            return out

        from supermem.leftbrain import merged_extraction
        if merged_extraction.enabled():
            cached = merged_extraction.take_traits(text)
            if cached is not None:
                valid = {"likes_dislikes", "expression_style", "thinking_pattern", "coping_style", "emotion"}
                return _keep([(s, l) for s, l in cached if s in valid and l])

        # English prompt only.
        #
        # There used to be only a Chinese prompt: English users came in, facts were English but traits were all
        # Chinese -- and the model was **copying the examples** (what got stored were exactly the example texts).
        # Merely adding a "follow the input language" rule didn't help; the examples pull harder, so the whole
        # prompt had to change. Slot names are internal keys: search/quota/brain map all key on them.
        prompt = (
            f"The user said this (current emotion: {emotion or 'unknown'}):\n"
            f"\"{text[:300]}\"\n\n"
            "Does it reveal any of these subjective things about the speaker? "
            "At most ONE short label per category (3-8 words):\n"
            "- likes_dislikes: gut likes / dislikes / preferences\n"
            "- expression_style: habits of speaking and communicating\n"
            "- thinking_pattern: how they think, weigh things, decide\n"
            "- coping_style: what they do to cope with stress or bad feelings\n"
            "- emotion: WHEN they feel WHAT. **A pattern, never a bare feeling "
            "word**: \"tense before design reviews\", \"annoyed when "
            "interrupted\", \"calm when alone\" — NOT \"anxious\" / \"happy\". "
            "It becomes the title of a node on a graph; a bare word says nothing.\n\n"
            "Skip any category the utterance does not clearly show.\n"
            "**How to write a label**: a short pattern, no subject, no full stop:\n"
            "  good: hates being interrupted / wants comfort under stress / "
            "conclusion first\n"
            "  bad: The user tends to plan in detail. (a full sentence with a subject)\n"
            "  bad: I major in computer science (copying the utterance / a plain fact)\n"
            "The slot names above are internal keys — keep them exactly as written. "
            "Only the label follows the language rule below.\n"
            f"{_label_rule()}\n"
            'Output JSON only: {"items": [{"slot": "likes_dislikes", '
            '"label": "hates being interrupted"}, ...]} (items may be [])'
        )
        raw = self._llm_json(prompt)
        if not raw:
            return []
        try:
            items = _json.loads(raw).get("items", [])
        except Exception:
            return []
        valid_slots = {"likes_dislikes", "expression_style", "thinking_pattern", "coping_style", "emotion"}
        result = []
        for it in items:
            slot = str(it.get("slot", "")).strip()
            label = str(it.get("label", "")).strip()
            if slot in valid_slots and label:
                result.append((slot, label))
        return _keep(result)

    def _embed_text(self, text: str) -> list[float]:
        """Embedding used for graph-layer entities / slot anchors / the right-brain trait table.

        **If an embedder was injected, use the injected one** -- this used to hard-code OpenAI, bypassing
        ``SuperMem(embedding=...)``, so configuring a local model only half worked: memory vectors went local,
        but graph-layer entities and the trait table still went remote. The same store thus held two dimensions (384 / 1536),
        and the "share one cache with the left brain" below never actually happened -- the cache is keyed by model name,
        so with different models on the two channels it never hit once.

        This is also on the **query hot path**: right-brain search calls it every turn
        (traits_store.search_scored). Measured: local E5 query embedding 10ms,
        OpenAI 178ms; unifying saves that hop every turn and removes a single point of failure for network loss/rate limits.

        Without injection it stays as before (the default is OpenAI), so default-config behaviour and existing vectors are unchanged.
        """
        if self._embedder is not None:
            return self._embedder.embed_query_text(text) if hasattr(
                self._embedder, "embed_query_text") else self._embedder.embed_texts([text])[0]
        # Share one cache with the left brain: the entities needed here ('nuts'/'vegetarian'/'user') were just
        # embedded by the left brain; there's no need to send the exact same strings again.
        from supermem.utils.common import embed_cache
        model = resolve_model(role="embedding")
        return embed_cache.resolve(model, [text], self._embed_uncached)[0]

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        client = OpenAI(
            api_key=resolve_api_key(),
            base_url=self._base_url,
            timeout=15.0,
        )
        _kw = {
            "model": resolve_model(role="embedding"),
            "input": texts,
            "encoding_format": "float",   # some compatible backends don't support base64
        }
        if "openrouter" in str(resolve_base_url(self._base_url) or "").lower():
            _kw["extra_body"] = {"provider": {"order": ["OpenAI"], "allow_fallbacks": False}}
        resp = client.embeddings.create(**_kw)
        _exp = int(os.environ.get("SUPERMEM_EMBED_DIM", "1536"))
        if len(resp.data[0].embedding) != _exp:
            raise RuntimeError(f"embedding dimension {len(resp.data[0].embedding)} != {_exp}, the provider was swapped")
        data = resp.data
        if all(d.index is not None for d in data):
            data = sorted(data, key=lambda d: d.index)
        return [list(map(float, row.embedding)) for row in data]

    def _llm_json(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
                timeout=15.0,
            )
            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512,
            )
            from supermem.utils.common.cost_log import log_usage
            log_usage("llm_json", resp.model, getattr(resp, "usage", None))
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[SplitMgr] LLM failed: {e}")
            return ""

    def _llm_text(self, prompt: str, max_tokens: int = 300) -> str:
        """Unlike _llm_json: doesn't force JSON output; for plain-text use cases like attribution summaries."""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
                timeout=15.0,
            )
            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            from supermem.utils.common.cost_log import log_usage
            log_usage("llm_text", resp.model, getattr(resp, "usage", None))
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[Attribution] LLM failed: {e}")
            return ""

    # ── left-brain search steps (forwarded to LeftBrain: SearchCogGraph/SearchData/Rank etc.) ──

    def SearchCogGraph(self, *a, **k) -> tuple[set[str], QueryClassification]:
        """Slot filtering, forwarded to LeftBrain.SearchCogGraph."""
        return self._left.SearchCogGraph(*a, **k)

    def SearchData(self, *a, **k) -> set[str]:
        """Entity narrowing, forwarded to LeftBrain.SearchData."""
        return self._left.SearchData(*a, **k)

    def _search_data_impl(self, *a, **k) -> tuple[set[str], list[str]]:
        """The real SearchData implementation (also returns activated_names), forwarded to LeftBrain."""
        return self._left._search_data_impl(*a, **k)

    def _widen_for_time_question(self, *a, **k) -> set[str]:
        """Widen candidates for time questions, forwarded to LeftBrain."""
        return self._left._widen_for_time_question(*a, **k)

    def Rank(self, *a, **k) -> list[MemorySearchHit]:
        """Vector similarity ranking, forwarded to LeftBrain.Rank."""
        return self._left.Rank(*a, **k)

    # ── v5: LLM tagging (forwarded to LeftBrain) ─────────────────────────────────

    def _get_slot_base_embeddings(self, *a, **k) -> dict[str, list[float]]:
        return self._left._get_slot_base_embeddings(*a, **k)

    def _get_slot_dyn_embeddings(self, *a, **k) -> dict[str, list[float]]:
        return self._left._get_slot_dyn_embeddings(*a, **k)

    def _normalize_slot_name(self, *a, **k) -> str:
        return self._left._normalize_slot_name(*a, **k)

    def _llm_tag_memories(self, *a, **k) -> list[str]:
        return self._left._llm_tag_memories(*a, **k)

    # ── query classification (including dynamic slots) ──────────────────────────

    def Classify(self, *a, **k) -> QueryClassification:
        """LLM classifies query -> slots + entities, forwarded to LeftBrain.Classify."""
        return self._left.Classify(*a, **k)

    def PrimeSubgraphFromQuery(self, query: str, top_k: int = 10) -> dict:
        """Convenience wrapper for Classify()+Search(); returns how many entries this search booked.

        Subgraph decisions happen in two layers: after every search, Search() automatically records the retrieved memory_ids in an accumulated list
        (cheap, no LLM call, see _record_subgraph_activation); the real "build graph -> compute density ->
        decide" step is expensive and only runs once a batch has accumulated in RunSubgraphCheckpoint().

        Bookkeeping is an automatic side effect of Search(), so calling Search() directly has the same effect; this method only exists
        for compatibility with callers that want "Classify+Search+booked count in one go".
        """
        classification = self.Classify(query)
        result = self.Search(
            query=query, slots=classification.slots, entities=classification.entities,
            top_k=top_k,
        )
        return {"status": "recorded", "count": len({h.memory_id for h in result.hits})}

    def _record_subgraph_activation(self, *a, **k) -> None:
        """Book search results, forwarded to LeftBrain._record_subgraph_activation."""
        return self._left._record_subgraph_activation(*a, **k)

    def RunSubgraphCheckpoint(self, *a, **k) -> dict:
        """Subgraph checkpoint (build graph -> decide), forwarded to LeftBrain.RunSubgraphCheckpoint."""
        return self._left.RunSubgraphCheckpoint(*a, **k)

    def ArchiveColdMemories(self, *a, **k) -> dict:
        """Cold memory archiving, forwarded to LeftBrain.ArchiveColdMemories."""
        return self._left.ArchiveColdMemories(*a, **k)

    # ── full pipeline ─────────────────────────────────────────────────────────

    def Search(
        self,
        query: str,
        slots: list[str] | None = None,
        entities: list[str] | None = None,
        emotion: str | None = None,
        top_k: int = 5,
        scene_filter: str | None = None,
        speaker_filter: str | None = None,
    ) -> SearchResult:
        """Full search pipeline: SearchCogGraph -> SearchData -> Rank -> right brain -> summaries.

        Parameters
        ----------
        query:
            The user utterance, used for vector ranking.
        slots:
            Slot list provided by the voice module, e.g. ``["work"]``. If empty, degrades to a whole-store search.
        entities:
            Entity list provided by the voice module, e.g. ``["Alibaba"]``. May be empty.
        top_k:
            Maximum number of memories to return.
        scene_filter:
            Optional scene filter (audiomem), e.g. ``"office"``.
        speaker_filter:
            Optional speaker filter (audiomem), pass a person_id.
        """
        import time
        import concurrent.futures

        # Relative time words like "next week"/"tomorrow" are dead in vector space: extraction normalises dates to absolute dates
        # ("Wednesday, August 26, 2026"), while the question contains no absolute date, so it can't reach them. Measured:
        # "what do I have next week" retrieved none of the three next-week events, while "what am I doing on August 26"
        # hit all three -- the only difference was the phrasing. So relative time words are expanded in place into dates appended to the query;
        # this only affects the text used for search, it doesn't change what the user said and isn't written into memory.
        from supermem.leftbrain.time_expand import expand_relative_dates
        query = expand_relative_dates(query)

        # Scene-bound memory: if the caller didn't pass scene_filter explicitly, first infer scene intent from the query text.
        if scene_filter is None:
            from supermem.utils.audio.environment.scene_classifier import infer_scene_from_text
            inferred_scene = infer_scene_from_text(query)
            if inferred_scene is not None:
                scene_filter = inferred_scene.value

        # If the query doesn't mention a scene either, use the current/most recently detected scene as a soft preference (if narrowing
        # yields nothing it is automatically reverted, so memories from other scenes are never actually filtered out).
        if scene_filter is None:
            try:
                current_scene = self._get_trigger_store().get_last_scene(self._user_id)
                # "unknown" means "don't know where", it's not a scene -- filtering by it would shrink search down to
                # exactly the few entries tagged scene:unknown. Measured: only 7 entries in the whole store had that
                # tag, so **every subsequent search** picked only from those 7; asking "what am I allergic to"
                # returned macarons and strawberries, while the allergy entry (ranked first by pure vector, 0.935) never even became a candidate.
                if current_scene and current_scene != "unknown":
                    scene_filter = current_scene
            except Exception:
                pass

        # (1)(2)+summaries: the left-brain search section (SearchCogGraph->SearchData->time-based widening->related slot summaries)
        # is extracted wholesale into LeftBrain.search; entity narrowing finishes before the right brain runs (the right brain depends on the left brain's "activated"
        # entity set); t0/t1/t2 are returned by the component, timing semantics unchanged.
        left = self._left.search(
            query, slots, entities, scene_filter, speaker_filter,
        )
        slot_mem_ids      = left["slot_mem_ids"]
        final_ids         = left["final_ids"]
        activated_names   = left["activated_names"]
        classification    = left["classification"]
        related_summaries = left["related_summaries"]
        t0, t1, t2 = left["t0"], left["t1"], left["t2"]

        # (3) Right brain (depends on activated_names) and Rank (vector ranking, depends on final_ids) run concurrently --
        # neither depends on the other's output, so they can run in parallel.
        rb_hits: list[RightBrainHit] = []
        rb_directive = ""
        rb_duration  = 0.0

        # The right-brain search section is extracted into RightBrain.search and vector ranking into LeftBrain.rank; only the
        # ThreadPoolExecutor structure running Rank || right brain concurrently remains here, both halves now component calls.
        # This turn the user is responding to the agent's previous line; the right brain needs to see it (reaction signals + context anchors)
        agent_reply = self.last_agent_reply()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            rb_future = pool.submit(
                self._right.search, query, activated_names, emotion, top_k, agent_reply,
            )                                            # right brain starts running concurrently

            hits = self._left.rank(query, final_ids, top_k, speaker_filter=speaker_filter)
            t3 = time.time()

            rb_hits, rb_directive = rb_future.result()   # wait for the right brain (usually already done)
            t4 = time.time()
            rb_duration = t4 - t2                        # total right-brain time (from when activated_names is ready)

        # Low-confidence abstention hint: when neither brain has specific evidence for this question (left brain no hits/entity
        # doesn't exist, right brain only generic fallback), tell the responder explicitly "if evidence is insufficient, say you don't know".
        _specific_rb = {"response_experience", "situation_pattern", "relation"}
        rb_specific = any(h.source in _specific_rb for h in rb_hits)
        left_weak = (not hits) or (not activated_names)
        if left_weak and not rb_specific:
            hint = (
                "Note: the memory system found no specific evidence for this query "
                "(only generic profile context). If the retrieved content does not "
                "actually answer the question, say you don't know instead of guessing."
            )
            rb_directive = f"{rb_directive}\n{hint}".strip()

        # Scene-adaptive reply style (audiomem): read the user's current scene and generate a directive
        scene_directive = ""
        current_scene = ""
        try:
            from supermem.utils.audio.environment.scene_classifier import SceneTag, scene_to_response_directive
            last_scene = self._get_trigger_store().get_last_scene(self._user_id)
            if last_scene:
                current_scene = last_scene
                try:
                    scene_directive = scene_to_response_directive(SceneTag(last_scene))
                except ValueError:
                    pass
        except Exception:
            pass

        # Every real search is booked automatically (needed for subgraph cluster emergence), see _record_subgraph_activation.
        self._record_subgraph_activation(hits)

        return SearchResult(
            hits=hits,
            classification=classification,
            related_summaries=related_summaries,
            slot_mem_ids=slot_mem_ids,
            final_candidate_ids=final_ids,
            search_mode=_search_mode(slot_mem_ids, final_ids),
            rb_directive=rb_directive,
            rb_hits=rb_hits,
            scene_directive=scene_directive,
            current_scene=current_scene,
            timing={
                "slot_filter_ms":    round((t1 - t0) * 1000, 1),
                "entity_narrow_ms":  round((t2 - t1) * 1000, 1),
                "rank_ms":           round((t3 - t2) * 1000, 1),
                "rb_ms":             round(rb_duration * 1000, 1),
                "total_ms":          round((t4 - t0) * 1000, 1),
            },
        )

    # ── write ─────────────────────────────────────────────────────────────────

    def _get_user_name(self) -> str | None:
        """Extract the user's name from left-brain memories, forwarded to LeftBrain ("user_name" in the shared _cache)."""
        return self._left._get_user_name()

    @staticmethod
    def _is_english(text: str) -> bool:
        """True if text is predominantly English (ASCII letters dominate over CJK)."""
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff")
        alpha = sum(1 for c in text if c.isalpha())
        return alpha > 0 and cjk / max(alpha, 1) < 0.3

    def _generate_inner_os(self, text: str, emotion: str, entities: list[str],
                           agent_reply: str = "") -> str:
        """Use an LLM to turn the original sentence into an AI third-person inner-monologue style, with an emotion tag; returns "" on failure.

        ``agent_reply``: the line the agent said before the user said this. Emotion attribution depends on it -- the same
        "fine, whatever" after empathy and after being handed a solution are two different emotions; without the previous line the model can only
        guess cause and effect from the user's half.
        """
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
                timeout=10.0,
            )
            user_name = self._get_user_name()
            entity_hint = f", involving: {', '.join(entities)}" if entities else ""
            # Follows the **store language**, not whatever language this user utterance is in.
            #
            # It used to be decided by whether this utterance contained Chinese characters: in an English store, an occasional
            # Chinese line from the user made inner_os Chinese, out of step with the other fields of the same memory. Language is a store property,
            # see supermem/lang.py.
            pronoun = user_name if user_name else "they"
            system_prompt = (
                "You are an empathetic AI assistant recording your inner observations about the user's emotional state. "
                "Based on what the user said, write your (the AI's) internal reaction — "
                "as if you quietly sensed their emotion and were moved by it. "
                f"Requirements: third person (refer to the user as '{pronoun}'), "
                "conversational, warm, 15-25 words, start with [emotion word] in brackets. "
                f"Examples:\n"
                f"Input: Got yelled at by my boss today, emotion: sad\n"
                f"Output: [heartache] {pronoun} is holding it together on the outside, but being called out like that must really sting.\n"
                f"Input: My best friend is moving away, emotion: longing\n"
                f"Output: [worried] {pronoun} is losing someone close — once they're gone, who do they call on a hard day?\n"
                "Output only that one sentence, nothing else."
            )
            reply_line = (agent_reply or "").strip()
            if reply_line:
                prior = "What you (the AI) just said"
                user_content = (f"{prior}: {reply_line[:200]}\n"
                                f"What the user said: {text}\nEmotion: {emotion}{entity_hint}")
            else:
                user_content = f"What the user said: {text}\nEmotion: {emotion}{entity_hint}"

            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=80,
                temperature=0.7,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _detect_scene(self, *a, **k):
        return self._audio._detect_scene(*a, **k)

    def _detect_speaker(self, *a, **k):
        return self._audio._detect_speaker(*a, **k)

    def _bind_self_identity(self, *a, **k):
        return self._audio._bind_self_identity(*a, **k)

    def preprocess(self, *a, **k) -> "AudioPerception":
        """Streaming preprocessing (audio perception), forwarded to AudioPerceiver.preprocess."""
        return self._audio.preprocess(*a, **k)

    # ── what the agent said (the other half of the conversation) ─────────────────

    def remember_reply(self, user_text: str, reply: str) -> None:
        """Register what the agent just said. ``vm.reply()`` calls this automatically when done (see core.py);
        replies that don't go through it (the web demo's llm_stream / Realtime event stream) call it once themselves,
        or pass it in ``ingest(text, agent_reply=...)``, which registers it along the way."""
        reply = (reply or "").strip()
        if not reply:
            return
        pair = ((user_text or "").strip(), reply)
        if self._exchanges and self._exchanges[-1] == pair:
            return                      # don't register the same turn twice (reply() already recorded it, and ingest passed it again explicitly)
        self._exchanges.append(pair)

    def last_agent_reply(self, before_text: str | None = None) -> str:
        """The agent's most recent line. When ``before_text`` is given, skip the reply to it and return the line
        before that -- the one the user is responding to."""
        want = (before_text or "").strip()
        for user_text, reply in reversed(list(self._exchanges)):
            if want and user_text == want:
                continue                # this is the reply to before_text, not the line before it
            return reply
        return ""

    def _reply_to(self, text: str) -> str:
        """The agent's reply to "this user utterance" (only exists if this turn was already answered), for left-brain extraction disambiguation."""
        want = (text or "").strip()
        for user_text, reply in reversed(list(self._exchanges)):
            if user_text == want:
                return reply
        return ""

    def Ingest(
        self,
        text: str,
        speaker: str = "Speaker 0",
        emotion: str = "",
        entities: list[str] | None = None,
        session_id: int | str | None = None,
        audio_path: str | None = None,
        observed_at: str | None = None,
        async_facts: bool = False,
        agent_reply: str | None = None,
        on_complete=None,
    ) -> dict:
        """Store one voice input in the memory store.

        This is the main entry point joining supermem with the voice layer; internally it is a three-step pipeline::

            (1) preprocess()      text (+optional audio_path) -> AudioPerception
                                  (streaming preprocessing: all acoustic analysis such as scene/voiceprint/emotion)
            (2) assemble ctx      pack the perception result + text/timestamp
            (3) _finish_ingest()  fact extraction + left/right brain writes + audiomem tagging

        The voice side only needs to provide ``text`` (for structured input like multiple speakers/emotion see
        ``voice_input.ingest_voice_input``); all audio perception happens in (1).

        Parameters
        ----------
        observed_at:
            When this utterance actually happened (e.g. when backfilling historical conversations, pass the real date "2023-05-08" or an ISO
            string). Defaults to now; backfilled historical data must pass it explicitly, otherwise temporal reasoning and
            time-based sorting will be distorted.
        agent_reply:
            The agent's reply to this utterance. If omitted, the line registered by ``vm.reply()`` is used automatically, so
            the standard "reply first, then store" flow needs no change; replies not going through ``vm.reply()`` pass one explicitly.
            The left brain uses it to disambiguate the user's line; the right brain uses the **previous turn's** reply for emotion attribution.
        on_complete:
            Optional callback that receives the final result dict once facts and right-brain writes are done. With
            ``async_facts=True`` the callback runs on the background thread.

        Returns
        -------
        dict
            ``{facts_count, memory_ids, affect}``
        """
        import time

        self._check_live()        # the fast path below writes to sqlite too
        ts = observed_at or time.strftime("%H:%M:%S")

        if agent_reply is None:
            agent_reply = self._reply_to(text)       # this turn's reply, for left-brain disambiguation
        else:
            self.remember_reply(text, agent_reply)   # caller generated its own reply; register it along the way
        prior_reply = self.last_agent_reply(before_text=text)   # previous turn's, for right-brain attribution

        # (1) Streaming preprocessing: all acoustic analysis such as scene/voiceprint/emotion happens in this step (see preprocess)
        p = self.preprocess(text, speaker, emotion, session_id, audio_path)
        speaker          = p.speaker
        emotion          = p.emotion
        environment      = p.environment
        environment_hint = p.environment_hint
        scene_tag        = p.scene_tag
        scene_raw_labels = p.scene_raw_labels
        person_id        = p.person_id
        tune_result      = p.tune_result
        abnormal_hits    = p.abnormal_hits
        detection        = p.detection
        place_result     = None   # filled in by _finish_ingest's scene clustering stage
        new_routine      = None   # filled in by _finish_ingest's routine detection stage

        ctx = {
            "text": text, "speaker": speaker, "emotion": emotion, "entities": entities,
            "session_id": session_id, "audio_path": audio_path, "observed_at": observed_at,
            "ts": ts,
            # AST remains the immediate hint. When CLAP final-memory mode
            # is enabled, don't put that provisional text in the utterance memory.
            "environment": "" if self._clap_memory_enabled() else environment,
            "environment_hint": environment_hint,
            "scene_tag": scene_tag,
            "scene_raw_labels": scene_raw_labels, "person_id": person_id,
            "tune_result": tune_result, "abnormal_hits": abnormal_hits,
            "place_result": place_result, "new_routine": new_routine, "detection": detection,
            # fixed here so that with async_facts=True the background thread doesn't get mixed up with later turns
            "agent_reply": agent_reply or "", "prior_agent_reply": prior_reply or "",
        }

        if self._clap_memory_enabled() and audio_path is not None:
            threading.Thread(
                target=self._finish_clap_environment,
                args=(audio_path, text, session_id, environment_hint),
                daemon=True,
            ).start()

        def _notify_complete(result: dict) -> None:
            if on_complete is None:
                return
            try:
                on_complete(result)
            except Exception as e:
                print(f"[ingest] on_complete callback failed: {type(e).__name__}: {e}", flush=True)

        # async_facts=True: fact extraction + graph writes (the slow part) go to a background thread,
        # and Ingest() returns immediately with the synchronously computed audiomem fields. Default False.
        if async_facts:
            def _bg() -> None:
                """Background storage. Exceptions must be printed here.

                An exception raised in the thread is awaited by nobody, so the turn just **vanishes** --
                the log doesn't even say "stored 0 entries", nothing is in the store, and it looks like the LLM extracted nothing.
                Measured: two of six consecutive turns were lost exactly this way, and it took a long time to track down.
                """
                try:
                    _notify_complete(self._finish_ingest(ctx))
                except Exception as e:
                    import traceback
                    print(f"[ingest] this turn could not be stored ({type(e).__name__}: {e})\n"
                          f"{traceback.format_exc()}", flush=True)
                    _notify_complete({"error": str(e), "persistent_memory_created": False})

            threading.Thread(target=_bg, daemon=True).start()
            return {
                "facts_count":         None,
                "memory_ids":          [],
                "affect":              None,
                "error":               None,
                "triggered_reminders": [],
                "proactive_memories":  [],
                "current_scene":       scene_tag or "",
                "environment_hint":    environment_hint,
                "speaker_id":          person_id or "",
                "recognized_tune":     (
                    {"tune_id": tune_result.tune_id, "action": tune_result.action,
                     "heard_count": tune_result.heard_count}
                    if tune_result is not None else None
                ),
                "abnormal_sounds":     [l for l, _ in abnormal_hits],
                "recognized_place":    (
                    {"place_id": place_result.place_id, "action": place_result.action,
                     "visit_count": place_result.visit_count,
                     "previous_visit_at": place_result.previous_visit_at}
                    if place_result is not None else None
                ),
                "familiar_place_prompt": None,
                "new_routine":         None,
            }

        result = self._finish_ingest(ctx)
        _notify_complete(result)
        return result

    def _tag_memories(self, memory_ids, tags) -> None:
        """Write memory_tags for a batch of memories; tags=[(name, conf),...]. Skipped if the cog store doesn't support it."""
        cog_store = self._get_repo()._cognitive_store
        if cog_store and hasattr(cog_store, "upsert_memory_tags"):
            for mid in memory_ids:
                cog_store.upsert_memory_tags(mid, self._user_id, tags)

    def _extract_and_append(self, messages, instructions, ts, extra_metadata):
        """Synthesise messages -> extract atomic facts -> append to the store, returning the new memory_ids (empty if nothing extracted).
        Shared by the synthetic memories in audiomem for routine/music/abnormal/ambient sound."""
        extracted = self._get_extractor().extract(
            new_messages=messages, custom_instructions=instructions,
            observation_date=ts, current_date=ts,
        )
        if not extracted:
            return []
        return self._get_repo().append_extracted(
            extracted, user_id=self._user_id, extra_metadata=extra_metadata)

    _retired = False               # class default, so instances built with __new__ have it too

    def retire(self) -> None:
        """Refuse every later write: the caller is about to delete this brain's files.

        Waits for the write in flight (it holds _write_lock), then flips the flag, so a
        background ingest still queued on the lock cannot write into a wiped or re-created
        directory through this stale instance.
        """
        with self._write_lock:
            self._retired = True

    def _check_live(self) -> None:
        if self._retired:
            raise RuntimeError("This brain was cleared or deleted.")

    def _finish_ingest(self, ctx: dict) -> dict:
        """Serialised entry to _finish_ingest_locked (see _write_lock in __init__)."""
        with self._write_lock:
            self._check_live()
            return self._finish_ingest_locked(ctx)

    def _finish_ingest_locked(self, ctx: dict) -> dict:
        """The fact extraction + graph write (left/right brain) part of Ingest(), split out so that
        with async_facts=True it can run on a background thread."""
        text = ctx["text"]; speaker = ctx["speaker"]; emotion = ctx["emotion"]
        entities = ctx["entities"]; session_id = ctx["session_id"]
        audio_path = ctx["audio_path"]; observed_at = ctx["observed_at"]
        ts = ctx["ts"]; environment = ctx["environment"]; scene_tag = ctx["scene_tag"]
        scene_raw_labels = ctx["scene_raw_labels"]; person_id = ctx["person_id"]
        tune_result = ctx["tune_result"]; abnormal_hits = ctx["abnormal_hits"]
        place_result = ctx["place_result"]; new_routine = ctx["new_routine"]
        detection = ctx["detection"]
        environment_hint = ctx.get("environment_hint", "")
        agent_reply = ctx.get("agent_reply", "")
        prior_reply = ctx.get("prior_agent_reply", "")

        import uuid
        from supermem.utils.common.voice_input import VoiceInput, VoiceContent

        vi = VoiceInput(
            id=f"utt_{uuid.uuid4().hex[:8]}",
            time_stamp={"begin": ts, "end": ts},
            slots=[],
            contents=[VoiceContent(
                sub_id="0", time_start=ts, time_end=ts,
                sentence=text, voiceprint_id=speaker, emotion=emotion,
            )],
            environment=environment,
            agent_reply=agent_reply,       # goes into fact extraction together as the assistant message
        )

        # Left-brain fact extraction + storage (ingest_voice_input) extracted into LeftBrain.ingest_facts;
        # registry is the audio-side voiceprint name mapping (cross-domain), injected by this class during orchestration.
        result = self._left.ingest_facts(
            vi,
            registry=self._get_registry(),
            session_id=session_id,
            extra_metadata={"created_at": observed_at} if observed_at else None,
        )

        # The assistant's just-spoken line: store it **verbatim** as one entry, no fact extraction.
        #
        # Why not extract: what comes out is like "the assistant recommended stir-fried asparagus and salt-and-pepper mushrooms" -- it records
        # what the assistant did, not the user as a person, and its wording overlaps heavily with the current conversation, so the next question
        # scores it high and pushes genuinely relevant memories out of the top-k.
        # Why store it anyway: so that "what did you tell me before" can be answered. Search excludes role=
        # assistant by default (see mem0_backend_store.search), and only lets it in when the question asks what the assistant
        # itself said (asks_about_assistant).
        if agent_reply.strip():
            try:
                self._get_repo()._vector_store.add_text(
                    self._user_id, agent_reply.strip(), attributed_to="assistant",
                    metadata={"source": "agent_reply", "turn_id": vi.id,
                              **({"time_start": observed_at} if observed_at else {})},
                )
            except Exception as e:
                print(f"[ingest] failed to store the assistant's verbatim reply: {e}", flush=True)

        # No fact could be extracted this turn, but it really is "played some music for me": store one entry directly, skipping extraction.
        # Extraction judges "does this utterance contain facts about the user", but this turn's value isn't in the words,
        # it's in the audio. Without a memory row the tune_id has nowhere to attach, and later asking "replay that song
        # from just now" finds nothing.
        #
        # Two cases count: "acoustically recognised as music", or "**the person said** it's music". The latter is essential:
        # recognition relies on acoustic similarity, which misses phone speakers, noisy rooms and short clips, whereas "let me play you a song"
        # is more certain than any acoustic feature -- yet it also yields no fact (it's an action, not
        # a fact about the user), so both sides miss and the recording never enters the playback candidate pool.
        from supermem.utils.audio.perceiver import said_music as _is_said_music
        _said_music = _is_said_music(text)
        _tune_id = getattr(tune_result, "tune_id", None) if tune_result else None

        # "Say it, then play it" is the most natural order, and it gets split into **two turns**: a turn's recording runs from
        # detected voice onset until VAD decides the speaker is done, so the turn is saved when "let me play you a song" ends,
        # and the music lands in the next turn.
        #
        # If the next turn has only sound and no words, it's the song just mentioned -- even if acoustics didn't recognise it
        # (speaker playback, noise, short clips all get missed). Without recognising this, the pool only holds the turn where he
        # spoke, and playback plays the user's own voice, which sounds like "the music got cut off".
        # Sound-only turn: there's a recording but not a single word. That alone is worth storing -- later when asked about "that sound
        # just now" / "that song", it's the most likely answer. It may not be music (could be ambient sound),
        # so only sound_only is tagged; tune: is reserved for "actually recognised / he said so / inherited from the previous turn".
        from supermem.stream import SOUND_ONLY_TEXT
        _sound_only = bool(audio_path) and (
            not (text or "").strip() or (text or "").strip() == SOUND_ONLY_TEXT)

        _expect = getattr(self, "_expect_music_until", 0.0)
        _inherited = _sound_only and time.monotonic() < _expect
        if _said_music:
            self._expect_music_until = time.monotonic() + EXPECT_MUSIC_S
        elif (text or "").strip():
            self._expect_music_until = 0.0      # said something else, intent cancelled
        if _inherited:
            print("  [music] previous turn said music was coming, this turn is sound only -> that's the song", flush=True)

        if not result.memory_ids and (_tune_id or _said_music or _inherited or _sound_only):
            try:
                heard = getattr(tune_result, "heard_count", 0) or 0
                again = " (heard it before)" if heard > 1 else ""
                # Only call it "music" if recognised as music (or he said so himself), otherwise honestly say "some sound" --
                # ambient sound and noise also end up here, and calling it music would be making things up.
                what = "music" if (_tune_id or _said_music or _inherited) else "sound"
                _content = f"The user played some {what} for me to listen to{again}."
                mid = self._get_repo()._vector_store.add_text(
                    self._user_id, _content,
                    metadata={"source": "sound_only", "turn_id": vi.id,
                              **({"tune_id": _tune_id} if _tune_id else {}),
                              **({"time_start": observed_at} if observed_at else {})},
                )
                if mid:
                    # add_text above only wrote the **vector store**. memory_tags.memory_id has a
                    # foreign key to sqlite's memories table; without this row, the tune:/scene:/speaker: tags
                    # written right after would hit FOREIGN KEY constraint failed,
                    # and that exception is caught downstream as a single log line -- the symptom is "the music was clearly recognised,
                    # but no tune tag can be found in the store", and playback falls back to guessing by time and text.
                    # Normal turns never hit this: their memories are written by append_extracted, which writes both sides.
                    try:
                        from supermem.leftbrain.cognitive_graph.slot_v2 import SlotV2
                        self._get_repo()._cognitive_store.upsert_memory_record(
                            self._user_id, mid, SlotV2.DAILY_LIFE, _content,
                        )
                        # Mark "this turn's recording has **only sound, no speech**".
                        #
                        # A turn's recording runs from detected voice onset until VAD decides the speaker is done. So
                        # "let me play you a song" and the music after it are **two turns**: the first stores
                        # his own words (music may already be in the background, so it also gets a tune tag),
                        # and only the second is the music itself. Playback must pick the latter -- picking wrong plays
                        # the user's own voice, which sounds like "the music got cut off".
                        # sound_only: this turn's recording has only sound, no speech.
                        # tune:*: this turn is "music" -- the playback candidate pool queries tune:%,
                        #   so without it this entry never enters the pool. When acoustics recognise which song,
                        #   the perceiver adds a real tune_id as well, no conflict; if not recognised
                        #   there's only unidentified -- we genuinely don't know which song, so don't pretend to.
                        tags = [("sound_only", 0.95)]
                        if _tune_id:
                            tags.append((f"tune:{_tune_id}", 0.9))
                        elif _said_music or _inherited:
                            # we genuinely don't know which song, so don't pretend to; but it is definitely "music".
                            tags.append(("tune:unidentified", 0.9))
                        self._tag_memories([mid], tags)
                    except Exception as e:
                        print(f"[ingest] failed to backfill memories for the music turn (tags won't attach): {e}",
                              flush=True)
                    result.memory_ids = list(result.memory_ids or []) + [mid]
            except Exception as e:
                print(f"[ingest] failed to store the music turn: {e}", flush=True)

        # ── audiomem: scene/voiceprint tag writes + triggered reminders + recording archive + proactive push ──
        audiomem = self._write_audiomem_tags(
            result, scene_tag, scene_raw_labels, detection, audio_path,
            person_id, tune_result, abnormal_hits, ts, session_id, text,
        )
        triggered_reminders = audiomem["triggered_reminders"]
        proactive_memories = audiomem["proactive_memories"]
        familiar_place_prompt = audiomem["familiar_place_prompt"]
        place_result = audiomem["place_result"]
        new_routine = audiomem["new_routine"]

        self._write_left_brain(result, text)
        heartnote_id = self._write_right_brain(
            emotion, result, text, entities, observed_at, prior_reply)
        # Emotion attribution: record a response experience only when warranted. Called separately from heartnote -- it shouldn't be
        # blocked by "was an emotion detected"; in text_mode emotion is often empty.
        self._right.learn_from_reaction(
            text, emotion, entities, prior_reply,
            memory_id=(result.memory_ids[0] if result.memory_ids else None),
            observed_at=observed_at, heartnote_id=heartnote_id,
        )

        # async cleanup: triggered once every 50 new heartnotes
        threading.Thread(target=self._check_and_cleanup, daemon=True).start()
        # async cleanup: periodic archiving of original audio, at most once a day, deleting WAV files older than 30 days
        threading.Thread(target=self._check_and_cleanup_audio, daemon=True).start()

        # ── short/long-term attribution triggers (run once a batch accumulates / at session boundaries) ──
        turn_info = self._get_session_tracker().record_turn(self._user_id, session_id)

        # Short-term attribution isn't extraction, it's **consolidation**: re-read the existing memories under an entity and rewrite
        # its one-line description. It reads accumulated state, unrelated to what was just said. It used to run every turn;
        # measured, it alone took 7.0s of an 18.4s ingest -- and re-summarising the whole entity after a single new memory
        # had nothing new to summarise anyway.
        # Now triggered by the **number of new memories**: consolidate once left-brain facts + right-brain heartnotes together reach
        # SHORT_TERM_MIN_MEMORIES. Counting memories rather than entities because
        # "how much new there is to summarise" depends on the volume of new memories -- an entity touched ten times doesn't necessarily
        # have ten more items of content. Below the threshold it keeps accumulating (touch is INSERT OR IGNORE, so the same
        # memory is never counted twice).
        try:
            tracker = self._get_session_tracker()
            for mid in list(getattr(result, "memory_ids", None) or []):
                tracker.touch(self._user_id, "rb_pending_memories", str(mid))
            if heartnote_id:
                tracker.touch(self._user_id, "rb_pending_memories", str(heartnote_id))
            n = tracker.count_touched(self._user_id, "rb_pending_memories")
            if n >= SHORT_TERM_MIN_MEMORIES or turn_info["session_changed"]:
                tracker.pop_touched(self._user_id, "rb_pending_memories")   # reset the counter
                touched = tracker.pop_touched(self._user_id, "rb_entity_short")
                if touched:
                    self._get_attribution_manager().run_short_term(self._user_id, touched)
        except Exception as e:
            print(f"[Attribution] short-term attribution failed: {e}")

        if turn_info["session_changed"]:
            self._run_session_boundary_batch()

        return {
            "facts_count":         result.facts_count,
            "memory_ids":          result.memory_ids,
            "persistent_memory_created": bool(result.memory_ids or heartnote_id),
            "affect":              result.affect,
            # Extraction swallows its failures (missing key, network) and returns 0 facts;
            # surface the reason so callers like the /memories job can tell "nothing to store"
            # from "could not store".
            "error":               getattr(result, "error", None),
            "triggered_reminders": triggered_reminders,
            "proactive_memories":  proactive_memories,
            "current_scene":       scene_tag or "",
            "environment_hint":    environment_hint,
            "speaker_id":          person_id or "",
            "speaker_name":        (
                self._get_registry().display_name(person_id) if person_id else speaker
            ),
            "recognized_tune":     (
                {"tune_id": tune_result.tune_id, "action": tune_result.action,
                 "heard_count": tune_result.heard_count}
                if tune_result is not None else None
            ),
            "abnormal_sounds":     [l for l, _ in abnormal_hits],
            "recognized_place":    (
                {"place_id": place_result.place_id, "action": place_result.action,
                 "visit_count": place_result.visit_count,
                 "previous_visit_at": place_result.previous_visit_at}
                if place_result is not None else None
            ),
            "familiar_place_prompt": familiar_place_prompt,
            "new_routine":         new_routine,
        }

    def _write_audiomem_tags(self, *a, **k) -> dict:
        """audiomem write section, forwarded to AudioPerceiver._write_audiomem_tags."""
        return self._audio._write_audiomem_tags(*a, **k)

    def _write_left_brain(self, result, text) -> None:
        """Left-brain write section (LLM slot tagging + slot->entity graph layer), forwarded to LeftBrain.write."""
        return self._left.write(result, text)

    def _write_right_brain(self, emotion, result, text, entities, observed_at,
                           agent_reply: str = "") -> str | None:
        """Right-brain write section, forwarded to RightBrain.write. ``agent_reply`` is the line the agent said
        before this user utterance (context for emotion attribution). Returns this turn's heartnote id."""
        return self._right.write(emotion, result, text, entities, observed_at, agent_reply)

    def _run_session_boundary_batch(self) -> None:
        """Session boundary batch: left-brain subgraph decisions + right-brain long-term attribution.

        The left-brain part takes the search bookkeeping accumulated during the session and decides once
        (RunSubgraphCheckpoint); for pure ingest (no interleaved searches) it is naturally a no-op.

        Called automatically by Ingest() when it detects a session_id change. session_changed is inferred backwards from "seeing
        the first ingest of the next session", so the last session has no next entry to trigger it;
        after ingesting, callers must therefore call Flush() once explicitly to run the last session.
        """
        try:
            self.RunSubgraphCheckpoint()
        except Exception as e:
            print(f"[Subgraph] session boundary check failed: {e}")

        # Schema description refresh: for slots that gained memories this session, rewrite a combined description of <=40 words;
        # it is attached to the prompt at search time, giving cross-memory aggregate information no single fact can.
        try:
            self._refresh_schema_descriptions()
        except Exception as e:
            print(f"[SchemaDesc] refresh failed: {e}")

        try:
            touched_slots = self._get_session_tracker().pop_touched(self._user_id, "rb_slot_long")
            if touched_slots:
                self._get_attribution_manager().run_long_term(self._user_id, touched_slots)
        except Exception as e:
            print(f"[Attribution] long-term attribution failed: {e}")

    def _refresh_schema_descriptions(self) -> None:
        """Rewrite a combined description for slots whose memory count changed, forwarded to LeftBrain."""
        return self._left._refresh_schema_descriptions()

    def Flush(self) -> None:
        """Call once when the conversation/session formally ends, to run the batch processing the last session missed
        (subgraph decisions + right-brain long-term attribution, see _run_session_boundary_batch).
        Idempotent: a no-op when there are no new touched refs.
        """
        self._run_session_boundary_batch()
        try:
            touched = self._get_session_tracker().pop_touched(self._user_id, "rb_entity_short")
            if touched:
                self._get_attribution_manager().run_short_term(self._user_id, touched)
        except Exception as e:
            print(f"[Attribution] short-term attribution failed: {e}")

    def IngestEnv(self, *a, **k) -> dict:
        """Store an ambient sound event in the memory store, forwarded to AudioPerceiver.IngestEnv."""
        return self._audio.IngestEnv(*a, **k)

    def _check_and_cleanup(self) -> None:
        """Trigger a right-brain cleanup every 50 new heartnotes, forwarded to RightBrain.check_and_cleanup."""
        return self._right.check_and_cleanup()

    def _check_and_cleanup_audio(self, retention_days: int = 30) -> None:
        """Periodic archiving of original audio: at most once a day, deletes WAV files older than the retention period.
        Time-triggered and throttled with a separate state file, to avoid scanning the DB on every Ingest.
        """
        try:
            import json as _json
            from datetime import datetime, timezone
            last_run = _space.kv_get(self._memory_root, "audio_cleanup_last_run", "")
            now = datetime.now(timezone.utc)
            if last_run:
                try:
                    elapsed_hours = (now - datetime.fromisoformat(last_run)).total_seconds() / 3600
                except ValueError:
                    elapsed_hours = 999
            else:
                elapsed_hours = 999

            if elapsed_hours < 24:
                return

            _space.kv_set(self._memory_root, "audio_cleanup_last_run", now.isoformat())
            self._get_audio_archive().cleanup_expired(retention_days=retention_days)
        except Exception as e:
            print(f"[Cleanup] audio check error: {e}")

    def _run_cleanup(self) -> None:
        """Clean right-brain heartnotes with an LLM, forwarded to RightBrain.run_cleanup."""
        return self._right.run_cleanup()


__all__ = ["Orchestrator", "SearchResult", "Utils"]
