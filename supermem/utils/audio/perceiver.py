"""Audio perception component AudioPerceiver.

The **whole audio-perception block** extracted from the SuperMem god class -- acoustic scene /
background music / abnormal environmental sounds / speaker voiceprint / self-introduction binding /
prosodic emotion, plus the things built around them: scene-triggered reminders, original-audio
archiving and playback, environmental-sound ingestion, and audiomem tag writing.

Follows mem0's composition pattern:
  * **The component owns its parts** -- the 10 audio-side lazy singletons (env/CLAP/speaker/vp/emotion/
    music/routine/place/trigger/audio_archive), together with their own caches and locks, live inside
    this component; the engine no longer holds them.
  * **Dependencies are injected explicitly** -- anywhere that needs **non-audio** capabilities such as
    "left-brain storage / extractor / voiceprint-to-name mapping / tagging / fact appending / semantic
    ranking" receives them in __init__ as getters/function references (lazy-loading semantics
    unchanged), and the component calls them via self._dep().

Logic is unchanged word for word: method bodies were moved as is; only "how dependencies are obtained" changed.
"""

from __future__ import annotations

from supermem.utils.common import space as _space

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

#: Turns shorter than this many seconds skip voiceprint identification -- see the note in _detect_speaker.
SPEAKER_MIN_S = float(os.environ.get("SUPERMEM_SPEAKER_MIN_S", "2.5"))


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class AudioPerception:
    """Structured output of preprocess(): the summary after one audio clip has gone through every
    acoustic perception module. speaker/emotion may be overridden by perception results (voiceprint
    person_id, prosodic emotion); the other fields are used by Ingest to build ctx and write audiomem tags."""
    speaker: str
    emotion: str
    environment: str = ""              # AST/CLAP background-sound description
    environment_hint: str = ""         # immediate scene hint (always from AST)
    scene_tag: str | None = None       # classified scene tag, e.g. "transit"
    scene_raw_labels: list[str] = field(default_factory=list)
    person_id: str | None = None       # speaker id identified by voiceprint
    tune_result: Any = None            # background music / humming recognition result
    abnormal_hits: list = field(default_factory=list)  # abnormal environmental sounds
    detection: dict = field(default_factory=dict)      # raw result of the single AST inference


# ── AudioPerceiver component ───────────────────────────────────────────────────

# Query used for proactive retrieval on a scene switch -- the memory topics most often associated with that scene (audiomem)
_SCENE_RECALL_QUERY: dict[str, str] = {
    "office":  "work tasks deadlines meetings",
    "transit": "to-do items on the commute",
    "home":    "things to do at home shopping list",
    "café":    "ideas inspiration brainstorming",
    "meeting": "meeting agenda project progress",
    "outdoor": "exercise health goals",
    "quiet":   "study plan focused tasks",
}

# "Play back the original audio" intent keywords: any hit means the user is asking for playback, and the whole sentence goes to semantic search.
_PLAYBACK_PATTERNS = ["replay", "play back", "play it back", "play that back",
                      "play it again", "play that again", "let me hear"]


#: Phrasings where the user says out loud "I'm going to play you some music". Acoustic recognition can
#: miss it (speaker playback, noisy surroundings, short clips), but when the person says so outright there
#: is nothing to doubt -- see the tune:unidentified part of _write_audiomem_tags.
#:
#: Only phrasings where **the user is the one providing** count. "Replay the first song" / "play that song
#: for me" are **asking for** music -- that turn recorded their question, not a song; an earlier regex was
#: broad enough to count those too, so the question turn entered the playback candidate pool as the newest
#: entry, and asking would play back what the user had just said.
_ASKING_PLAYBACK = re.compile(
    r"\breplay\b|play (it|that|this) (again|back)|hear (it|that|this) again|play ?back"
    r"|\b(can|could|would|will) you\b"
    r"|\b(play|find|put on) (me|for me)\b|\bplay\b.*\bfor me\b"
    r"|\blet me (hear|listen)\b",
    re.IGNORECASE,
)
_SAID_MUSIC = re.compile(
    r"(for|to) you to (hear|listen)|play (it|this) for you|play you (a|some|this)"
    r"|\bI(\s*'m| am)?\s*(going to|gonna|want to|wanna|will|'ll)?\s*(play|hum|sing)\b(\s+you)?\s*(a|some|this)?\s*(music|song|tune|melody|piano|guitar)"
    r"|\b(this|that)( is|'s)?( a| the)?\s*(music|song|tune|melody)\s*(sounds (good|nice)|is (good|nice)|what do you think|listen)"
    r"|\bhum (a|some|this)( song| tune)? (for|to) you\b",
    re.IGNORECASE,
)


def said_music(text: str) -> bool:
    """Is this sentence "I'm going to play you some music"? Always false on a sentence that is **asking for** playback."""
    t = text or ""
    return bool(_SAID_MUSIC.search(t)) and not _ASKING_PLAYBACK.search(t)


class AudioPerceiver:
    """Audio perception component that owns its parts and receives dependencies by explicit injection.

    Constructor arguments come in two kinds:

    **Runtime parameters** owned by the component (audio-side behaviour switches and path/identity)::

        memory_root, user_id, base_url,
        enable_scene, enable_music, enable_abnormal_sound,
        enable_voiceprint, enable_emotion

    Explicitly **injected non-audio dependencies** (all passed as getters/function references, lazy-loading semantics unchanged)::

        repo               -> self._get_repo         left-brain repository (cognitive_store etc.)
        extractor          -> self._get_extractor    fact extractor
        registry           -> self._get_registry     voiceprint-to-name mapping
        tag                -> self._tag_memories      writes memory_tags onto memories
        extract_and_append -> self._extract_and_append synthetic message -> extraction -> storage
        rank               -> self.Rank               semantic ranking (for playback / proactive recall)

    The 10 audio-side lazy singletons (env/clap/speaker/vp/emotion/music/routine/place/
    trigger/audio_archive), with their caches and locks, are owned by this component; see self._*_detector etc. below.

    Speaker binding state (_session_person_pin / _person_origin_session) also moves with the audio component.
    """

    def __init__(
        self,
        *,
        memory_root: Path,
        user_id: str,
        base_url: str | None,
        enable_scene: bool,
        enable_music: bool,
        enable_abnormal_sound: bool,
        enable_voiceprint: bool,
        enable_emotion: bool,
        repo: Callable[[], Any],
        extractor: Callable[[], Any],
        registry: Callable[[], Any],
        tag: Callable[..., Any],
        extract_and_append: Callable[..., Any],
        rank: Callable[..., Any],
        ingest_env: Callable[[], Callable[..., Any]],
        cache: dict[str, Any] | None = None,
        lock: Any = None,
    ) -> None:
        # ── Runtime parameters ──
        self._memory_root = memory_root
        self._user_id = user_id
        self._base_url = base_url
        self._enable_scene = enable_scene
        self._enable_music = enable_music
        self._enable_abnormal_sound = enable_abnormal_sound
        self._enable_voiceprint = enable_voiceprint
        self._enable_emotion = enable_emotion

        # ── Injected non-audio dependencies (getters/function references) ──
        self._repo = repo
        self._extractor = extractor
        self._registry = registry
        self._tag = tag
        self._extract_and_append_fn = extract_and_append
        self._rank = rank
        # ingest_env is a getter for "the current IngestEnv entry point" (resolved at call time),
        # so _finish_clap_environment goes through the entry point the host exposes; patching the host intercepts it.
        self._ingest_env_getter = ingest_env

        # ── Audio part cache owned by the component ──
        # The host may share the same cache/lock (audio-side lazy singletons and the host's _get_* land in the
        # same dict, so existing call sites/tests that read and write the host's _cache see the same view as this component).
        self._cache: dict[str, Any] = cache if cache is not None else {}
        self._lock = lock if lock is not None else threading.Lock()

        # Speaker binding state (formerly engine instance state, moves with the audio component)
        # session_id -> person_id: once a session has confirmed who is speaking via self-introduction, every later
        # sentence in the same session trusts that person first, rather than voting sentence by sentence on raw voiceprint scores.
        self._session_person_pin: dict[str, str] = {}
        # person_id -> session_id: the session in which this id was first created. Combined with
        # _session_person_pin, it is the second independent signal in _reconcile_speaker_candidates besides the
        # acoustic cross_score: an id whose birth session was later pinned to another person is most likely a
        # misidentified fragment of one of that person's noisy sentences, and must not be merged into an acoustically matched third-party identity.
        self._person_origin_session: dict[str, str] = {}

    # ── audiomem: scene + voiceprint lazy singletons ─────────────────────────────

    def _env_detector(self):
        with self._lock:
            if "env_detector" not in self._cache:
                from supermem.utils.audio.environment.environment_detector_ast import ASTEnvironmentDetector
                self._cache["env_detector"] = ASTEnvironmentDetector()
        return self._cache["env_detector"]

    def _clap_env_detector(self):
        with self._lock:
            if "clap_env_detector" not in self._cache:
                from supermem.utils.audio.environment.environment_detector_clap import CLAPEnvironmentDetector
                self._cache["clap_env_detector"] = CLAPEnvironmentDetector(
                    checkpoint=os.environ["SUPERMEM_CLAP_CHECKPOINT"]
                )
        return self._cache["clap_env_detector"]

    def _finish_clap_environment(self, audio_path, text, session_id, environment_hint="") -> None:
        """Re-check the environmental sound in the background with a long-window CLAP pass and write a separate environment memory.

        CLAP's candidate vocabulary is a fixed set; when the real sound class is not in it, pairs is empty,
        and we fall back to the AST hint (CLAP mode has already cleared ctx["environment"]).
        """
        try:
            _ingest_env = self._ingest_env_getter()
            detection = self._clap_env_detector().detect_full(Path(audio_path))
            pairs = detection.get("pairs") or []
            if not pairs:
                if environment_hint:
                    _ingest_env(
                        audio_path,
                        recent_context=[{"role": "user", "content": text}] if text else None,
                        session_id=session_id,
                        environment_override=environment_hint,
                    )
                return
            env_str = "background sounds: " + ", ".join(
                f"{label}({score:.2f})" for label, score in pairs
            )
            _ingest_env(
                audio_path,
                recent_context=[{"role": "user", "content": text}] if text else None,
                session_id=session_id,
                environment_override=env_str,
            )
        except Exception as exc:
            print(f"  [clap-env] background write skipped: {exc}", flush=True)

    def _trigger_store(self):
        with self._lock:
            if "trigger_store" not in self._cache:
                from supermem.utils.audio.environment.scene_trigger import SceneTriggerStore
                self._cache["trigger_store"] = SceneTriggerStore(
                    _space.db(self._memory_root)
                )
        return self._cache["trigger_store"]

    def _audio_archive(self):
        with self._lock:
            if "audio_archive" not in self._cache:
                from supermem.utils.audio.audio_archive import AudioArchive
                self._cache["audio_archive"] = AudioArchive(
                    _space.db(self._memory_root)
                )
        return self._cache["audio_archive"]

    def _speaker_encoder(self):
        with self._lock:
            if "speaker_encoder" not in self._cache:
                from supermem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
                self._cache["speaker_encoder"] = SpeakerEncoder()
        return self._cache["speaker_encoder"]

    def _vp_store(self):
        with self._lock:
            if "vp_store" not in self._cache:
                from supermem.utils.audio.voiceprint.voiceprint_store import VoiceprintStore
                from supermem.utils.common.voice_config import VoiceStoreConfig
                voice_cfg = VoiceStoreConfig.from_env()
                self._cache["vp_store"] = VoiceprintStore(
                    _space.mm(self._memory_root, "voiceprints"),
                    match_threshold=voice_cfg.match_threshold,
                    candidate_threshold=voice_cfg.candidate_threshold,
                    merge_threshold=voice_cfg.merge_threshold,
                )
        return self._cache["vp_store"]

    def _claimed_by_other_identity(self, candidate_pid: str, other_pid: str) -> bool:
        """Whether ``candidate_pid``'s birth session has already been confirmed, via self-introduction, as a
        third-party identity unrelated to both ids in this proposed merge (``candidate_pid`` and ``other_pid``).

        If so, ``candidate_pid`` is most likely a misidentified fragment of one noisy sentence from that third
        party and must not be merged into ``other_pid``. A birth session pinned to ``other_pid`` itself is the normal case and is not blocked.
        """
        origin_session = self._person_origin_session.get(candidate_pid)
        if origin_session is None:
            return False
        pinned = self._session_person_pin.get(origin_session)
        return pinned is not None and pinned not in (candidate_pid, other_pid)

    def _reconcile_speaker_candidates(self, person_id: str, speaker: str) -> tuple[str, str]:
        """After a real match, check whether any candidate voiceprint can now be confirmed and merged back.

        Only the acoustic side is handled here: ``VoiceprintStore`` knows nothing about names and judges "whether
        the profiles accumulated on both sides really are the same voice"; the name-conflict guard lives here --
        two person_ids that have explicitly reported different names are never force-merged just because the
        acoustic score is high enough. Returns a possibly updated ``(person_id, speaker)`` (if the current one
        happens to be the one merged away).
        """
        vp_store = self._vp_store()
        registry = self._registry()
        for _cid, absorb_pid, into_pid, cross in vp_store.find_resolvable_candidates(person_id):
            absorb_name = registry.display_name(absorb_pid)
            into_name = registry.display_name(into_pid)
            absorb_named = absorb_name != absorb_pid
            into_named = into_name != into_pid
            if absorb_named and into_named and absorb_name != into_name:
                continue
            # Second independent signal besides the acoustic cross_score: if either side's birth session has been
            # pinned to a third party via self-introduction, this id belongs to someone else and must not be merged just because the acoustic score is high.
            if self._claimed_by_other_identity(absorb_pid, into_pid) or \
                    self._claimed_by_other_identity(into_pid, absorb_pid):
                continue
            vp_store.merge_persons(absorb_pid, into_pid)
            if absorb_named and not into_named:
                registry.bind(into_pid, name=absorb_name)
            # merge_persons() only merges voiceprint profiles and does not touch memory_tags; backfill the old
            # speaker:{absorb_pid} tags to into_pid, otherwise per-person retrieval filtering misses them.
            try:
                cog_store = self._repo()._cognitive_store
                if cog_store and hasattr(cog_store, "rename_tag_value"):
                    cog_store.rename_tag_value(
                        self._user_id, f"speaker:{absorb_pid}", f"speaker:{into_pid}"
                    )
            except Exception as _e:
                print(f"  [speaker_identity] tag backfill failed: {_e}", flush=True)
            if person_id == absorb_pid:
                person_id = into_pid
                if speaker == absorb_pid:
                    speaker = into_pid
        return person_id, speaker

    def _emotion_detector(self):
        with self._lock:
            if "emotion_detector" not in self._cache:
                # Prosodic VAD + Qwen2.5-Omni attribution on negatively salient turns; interface detect(audio_path)->str.
                from supermem.utils.audio.emotion.paper_emotion_detector import PaperAlignedEmotionDetector
                self._cache["emotion_detector"] = PaperAlignedEmotionDetector()
        return self._cache["emotion_detector"]

    def _music_store(self):
        with self._lock:
            if "music_store" not in self._cache:
                from supermem.utils.audio.environment.music_memory import MusicMemoryStore
                self._cache["music_store"] = MusicMemoryStore(
                    _space.mm(self._memory_root, "music_profiles")
                )
        return self._cache["music_store"]

    def _routine_store(self):
        with self._lock:
            if "routine_store" not in self._cache:
                from supermem.utils.audio.environment.routine_memory import RoutineStore
                self._cache["routine_store"] = RoutineStore(
                    _space.db(self._memory_root)
                )
        return self._cache["routine_store"]

    def _place_store(self):
        with self._lock:
            if "place_store" not in self._cache:
                from supermem.utils.audio.environment.place_memory import PlaceMemoryStore
                self._cache["place_store"] = PlaceMemoryStore(
                    _space.mm(self._memory_root, "place_profiles")
                )
        return self._cache["place_store"]

    # ── audiomem: scene-triggered reminders ──────────────────────────────────────

    def CreateSceneTrigger(self, text: str) -> dict:
        """Parse a scene-trigger intent from the user's sentence and store the reminder.

        Parameters
        ----------
        text:
            The user's speech transcript, e.g. "remind me to call mom when I get on the bus".

        Returns
        -------
        dict
            ``{created: bool, scene: str, message: str}``
        """
        from supermem.utils.audio.environment.scene_trigger import parse_trigger_intent
        scene, message, required_label = parse_trigger_intent(text)
        if scene is None:
            return {"created": False, "scene": "", "message": ""}

        store = self._trigger_store()
        trigger = store.create(self._user_id, scene.value, message, required_label=required_label)
        return {"created": True, "scene": scene.value, "message": message, "id": trigger.id}

    def GetOriginalAudio(self, memory_id: str) -> dict:
        """Given a memory_id, return the original recording info usable for "playback confirmation".

        The original audio may have passed its retention period and been cleaned up (see
        audio_archive.cleanup_expired), so besides the path we also explicitly check that the file still exists.

        Returns
        -------
        dict
            ``{found: bool, audio_path: str|None}`` -- found=False means it was never
            archived, or the original audio exceeded its retention period and was cleaned up.
        """
        path = self._audio_archive().get_audio_path(memory_id, self._user_id)
        if not path:
            return {"found": False, "audio_path": None}
        p = Path(path)
        if not p.exists():
            return {"found": False, "audio_path": None}
        return {"found": True, "audio_path": str(p)}

    def TryPlayback(self, text: str) -> dict | None:
        """Detect whether this sentence asks to "play back the original audio"; on a hit, find the most relevant memory and fetch its original recording.

        Unlike GetOriginalAudio(memory_id) -- which requires the caller to already know which
        memory it is -- this starts from natural language like "replay the part where we discussed the price",
        first guesses the memory via semantic search, then looks up its original audio.

        Returns
        -------
        dict | None
            Returns None when there is no playback intent, no relevant memory is found, or the original audio
            has passed its retention period; on a hit returns ``{"memory_id", "memory_text", "audio_path"}``.
        """
        if not any(p in text.lower() for p in _PLAYBACK_PATTERNS):
            return None
        hits = self._rank(text, set(), top_k=3)
        for h in hits:
            audio = self.GetOriginalAudio(h.memory_id)
            if audio["found"]:
                return {
                    "memory_id": h.memory_id,
                    "memory_text": h.text,
                    "audio_path": audio["audio_path"],
                }
        return None

    # ── Acoustic perception: scene / voiceprint / self-introduction binding ──────

    def _detect_scene(self, _apath):
        """Acoustic scene detection (AST): one detect_full() inference yields scene label pairs, music/humming,
        abnormal environmental sounds and the raw embedding; runs if any of the three switches is on, and each
        post-processing step checks its own switch. Returns (environment, environment_hint, scene_tag,
        scene_raw_labels, tune_result, abnormal_hits, detection)."""
        environment = ""
        environment_hint = ""
        scene_tag: str | None = None
        scene_raw_labels: list[str] = []
        tune_result = None
        abnormal_hits: list[tuple[str, float]] = []
        detection = {"pairs": [], "music": None, "abnormal": [], "embedding": None}
        try:
            from supermem.utils.audio.environment.scene_classifier import classify_scene
            detector = self._env_detector()
            detection = detector.detect_full(_apath)

            if self._enable_scene:
                pairs = detection["pairs"]
                if pairs:
                    parts = ", ".join(f"{l}({s:.2f})" for l, s in pairs)
                    environment = f"background sounds: {parts}"
                    environment_hint = environment
                    scene_result = classify_scene(pairs)
                    scene_tag = scene_result.tag.value
                    scene_raw_labels = [l for l, _ in scene_result.raw_matches]
        except Exception as _e:
            print(f"  [env] detection skipped: {_e}", flush=True)

        # ── Background music / humming recognition memory (AST embedding) ──
        if self._enable_music:
            try:
                music = detection.get("music")
                if music is not None:
                    tune_result = self._music_store().identify(
                        music["embedding"], labels=[l for l, _ in music["labels"]]
                    )
            except Exception as _e:
                print(f"  [music] detection skipped: {_e}", flush=True)

        # ── Abnormal environmental sound memory (breaking / alarm / screaming) ──
        if self._enable_abnormal_sound:
            try:
                abnormal_hits = detection.get("abnormal") or []
            except Exception as _e:
                print(f"  [abnormal] detection skipped: {_e}", flush=True)

        return (environment, environment_hint, scene_tag, scene_raw_labels,
                tune_result, abnormal_hits, detection)

    @staticmethod
    def _long_enough(_apath) -> bool:
        """Is this audio long enough to be worth identifying the speaker? If the duration cannot be read, let it through (a probe must not block the main flow)."""
        try:
            import soundfile as sf
            return sf.info(str(_apath)).duration >= SPEAKER_MIN_S
        except Exception:
            return True

    def _detect_speaker(self, speaker, session_id, text, _apath):
        """Voiceprint identification (3D-Speaker ERes2Net): compute the voiceprint vector, identify person_id, and
        reclaim candidate voiceprints when needed. Returns (person_id, speaker, stable_voiceprint, vec).

        Audio that is too short skips identification entirely. 1-2 seconds is only enough for one window (see
        campplus_worker's SUPERMEM_SPEAKER_WINDOW=3.0); the vector is very noisy and the score often lands in the
        candidate range (0.40-0.50), and identify()'s candidate branch forks a new person every time -- so the
        same person gets split into a pile of person_* (measured: 12 in one demo store, 7 of them with
        obs_count=2). Once split, "is this still the same speaker" can no longer be judged reliably, and the web
        demo's stranger gate suddenly turns around and says "I don't know you".
        Better not to know who it is this turn than to stuff noise into the voiceprint store. Threshold:
        SUPERMEM_SPEAKER_MIN_S (default 2.5 seconds).
        """
        person_id: str | None = None
        stable_voiceprint = False
        vec = None
        if not self._long_enough(_apath):
            return person_id, speaker, stable_voiceprint, vec
        try:
            vec = self._speaker_encoder().embed(_apath)
            if vec is not None:
                pinned_pid = (
                    self._session_person_pin.get(session_id)
                    if session_id is not None else None
                )
                id_result = self._vp_store().identify(
                    vec, context=text[:80], pinned_person_id=pinned_pid,
                )
                person_id = id_result.person_id
                if id_result.action == "new" and session_id is not None:
                    self._person_origin_session.setdefault(person_id, session_id)
                # A candidate is an unconfirmed record that "may belong to an existing voiceprint"; it must not be
                # used to build a name mapping, otherwise one misidentification permanently pollutes the identity profile.
                stable_voiceprint = id_result.action in {"new", "match"}
                # when the caller passed the default placeholder speaker, override it with the identification result
                if speaker == "Speaker 0":
                    speaker = person_id

                # ── Automatic candidate reclaim: this match made the profile richer, so take the chance to
                # re-check; candidates that converge on the same person are merged back, so memories are not
                # scattered across two person_ids and speaker_filter does not miss the other half.
                if id_result.action == "match":
                    person_id, speaker = self._reconcile_speaker_candidates(person_id, speaker)
        except Exception as _e:
            print(f"  [speaker] identification skipped: {_e}", flush=True)
        return person_id, speaker, stable_voiceprint, vec

    def _bind_self_identity(self, person_id, speaker, stable_voiceprint, vec, text, session_id):
        """Self-introduction binding ("I'm annie" -> bind the current voiceprint to a name). After binding,
        voice_input_to_messages() automatically replaces Speaker N with the real name.
        Returns (person_id, speaker, stable_voiceprint)."""
        try:
            from supermem.utils.audio.voiceprint.speaker_identity import parse_self_identification
            self_name = parse_self_identification(text)
            if self_name:
                registry = self._registry()
                existing_name = registry.display_name(person_id)
                # Automatic extraction only fills in empty names and never overwrites a confirmed name (name
                # corrections should go through the explicit interface). "Unbound" does not mean "claimable": the
                # voiceprint threshold is unstable on short sentences, and if a voiceprint that has accumulated real
                # observations is mismatched with someone else's self-reported name, the earlier person's words get
                # attributed to the wrong person. The threshold is obs_count<=2 (this one + at most 1 adjacent
                # history entry), not <=1: the self-introduction is often not the first sentence, and <=1 would
                # misjudge "the same person giving their name right after" as a split.
                obs_count = self._vp_store().get_meta(person_id).get("obs_count", 0)
                fresh_unbound = existing_name == person_id and obs_count <= 2
                if existing_name == self_name or fresh_unbound:
                    registry.bind(person_id, name=self_name)
                    if session_id is not None:
                        self._session_person_pin[session_id] = person_id
                else:
                    # A self-reported name is strong evidence: when the current voiceprint is already bound to
                    # someone else's name, or is unbound but has historical observations (a suspected mismatch),
                    # split off a separate voiceprint and bind this sentence and later memories to the self-reported name.
                    if vec is not None:
                        split_result = self._vp_store().create_person(
                            vec, context=text
                        )
                        person_id = split_result.person_id
                        if session_id is not None:
                            self._person_origin_session.setdefault(person_id, session_id)
                        stable_voiceprint = True
                        if speaker == "Speaker 0":
                            speaker = person_id
                        registry.bind(person_id, name=self_name)
                        if session_id is not None:
                            self._session_person_pin[session_id] = person_id
                    else:
                        print(
                            f"  [speaker_identity] ignored conflicting auto-name "
                            f"{self_name!r}; {person_id} is already {existing_name!r} "
                            f"(obs_count={obs_count})",
                            flush=True,
                        )
        except Exception as _e:
            print(f"  [speaker_identity] bind failed: {_e}", flush=True)
        return person_id, speaker, stable_voiceprint

    def preprocess(
        self,
        text: str,
        speaker: str = "Speaker 0",
        emotion: str = "",
        session_id: int | str | None = None,
        audio_path: str | None = None,
    ) -> "AudioPerception":
        """Streaming preprocessing: step (1) of the Ingest() inference path, a public seam that can be called on its own.

        Runs one turn of audio through every acoustic perception module -- acoustic scene / background music /
        abnormal environmental sound / speaker voiceprint / self-introduction binding / prosodic emotion, plus
        self-introduction binding and emotion fallback in text-only mode -- and summarises it into one
        AudioPerception. The voice layer only needs to supply text (+ optional audio_path); all perception
        happens here; afterwards Ingest() assembles ctx and writes memories.

        It is public on its own to decouple "preprocessing" from "writing memories": you can get scene/speaker/
        emotion signals first for a real-time reply, then decide whether to Ingest. It writes no memories, but does
        update shared state such as the voiceprint registry in place.
        """
        environment = ""
        environment_hint = ""
        scene_tag: str | None = None
        scene_raw_labels: list[str] = []
        person_id: str | None = None
        stable_voiceprint = False
        vec = None
        tune_result = None   # supermem.utils.audio.environment.music_memory.TuneIdentifyResult | None
        abnormal_hits: list[tuple[str, float]] = []
        detection = {"pairs": [], "music": None, "abnormal": [], "embedding": None}
        if audio_path is not None:
            _apath = Path(audio_path)

            if self._enable_scene or self._enable_music or self._enable_abnormal_sound:
                (environment, environment_hint, scene_tag, scene_raw_labels,
                 tune_result, abnormal_hits, detection) = self._detect_scene(_apath)

            if self._enable_voiceprint:
                person_id, speaker, stable_voiceprint, vec = self._detect_speaker(
                    speaker, session_id, text, _apath,
                )
                if person_id and stable_voiceprint:
                    person_id, speaker, stable_voiceprint = self._bind_self_identity(
                        person_id, speaker, stable_voiceprint, vec, text, session_id,
                    )

            # ── Emotion recognition (prosodic VAD + Qwen2.5-Omni attribution on negatively salient turns) ──
            # Only auto-detect and fill in when the caller did not pass emotion explicitly.
            if self._enable_emotion and not emotion:
                try:
                    emotion = self._emotion_detector().detect(_apath)
                except Exception as _e:
                    print(f"  [emotion] detection skipped: {_e}", flush=True)

        # ── Self-introduction binding, text mode (no audio / no voiceprint) ──
        # Text-only calls never reach the voiceprint branch's binding. In the text channel the speaker id is a
        # stable identifier given by the caller with no mismatch risk, so we only need "fill empty names, never
        # overwrite confirmed ones" and no obs_count guard.
        if person_id is None and speaker and speaker != "Speaker 0":
            try:
                from supermem.utils.audio.voiceprint.speaker_identity import parse_self_identification
                self_name = parse_self_identification(text)
                if self_name:
                    registry = self._registry()
                    if registry.display_name(speaker) == speaker:  # only fill in empty names
                        registry.bind(speaker, name=self_name)
            except Exception as _e:
                print(f"  [speaker_identity] text-mode bind failed: {_e}", flush=True)

        # ── Text emotion fallback (no audio / prosodic detection gave no result) ──
        # Emotion detection only hangs off the audio branch, so for text-only calls emotion is always empty and
        # the right brain gets nothing. Here we match text at zero cost against anchor_router's emotion keyword
        # table: it only counts on an explicit emotion word; if nothing is recognised it returns None and no
        # heartnote is written. SUPERMEM_TEXT_EMOTION=0 turns it off.
        if (
            self._enable_emotion and not emotion and text
            and os.environ.get("SUPERMEM_TEXT_EMOTION", "1") != "0"
        ):
            from supermem.rightbrain.anchor_router import normalize_emotion_strict
            detected = normalize_emotion_strict(text)
            if detected:
                emotion = detected

        return AudioPerception(
            speaker=speaker,
            emotion=emotion,
            environment=environment,
            environment_hint=environment_hint,
            scene_tag=scene_tag,
            scene_raw_labels=scene_raw_labels,
            person_id=person_id,
            tune_result=tune_result,
            abnormal_hits=abnormal_hits,
            detection=detection,
        )

    # ── audiomem write section ─────────────────────────────────────────────────

    def _write_audiomem_tags(
        self, result, scene_tag, scene_raw_labels, detection, audio_path,
        person_id, tune_result, abnormal_hits, ts, session_id, text,
    ) -> dict:
        """audiomem write section: scene/voiceprint/music/abnormal-sound/place tags + triggered reminders + recording
        archive + proactive push on scene switch. Returns the fields the later return needs."""
        triggered_reminders: list[dict] = []
        proactive_memories: list[dict] = []
        familiar_place_prompt: dict | None = None
        place_result = None
        new_routine = None
        _fire_result = None
        if scene_tag and result.memory_ids:
            try:
                self._tag(result.memory_ids, [(f"scene:{scene_tag}", 0.95)])
            except Exception as _e:
                print(f"  [scene] tag write failed: {_e}", flush=True)

            # scene change -> fire matching reminders (before update_scene, preserving the scene_changed info)
            try:
                from supermem.utils.audio.environment.scene_trigger import check_and_fire
                _fire_result = check_and_fire(
                    self._trigger_store(), self._user_id, scene_tag,
                    raw_labels=scene_raw_labels,
                )
                triggered_reminders = [
                    {"id": t.id, "message": t.message, "scene": t.trigger_scene}
                    for t in _fire_result.fired
                ]
            except Exception as _e:
                print(f"  [scene_trigger] check failed: {_e}", flush=True)

            # Daily sound routines: build routine memories automatically -- record one observation only when
            # actually "entering" a scene (scene_changed), and once enough days accumulate, generate the memory only on the turn that crosses the threshold.
            if _fire_result is not None and _fire_result.scene_changed:
                try:
                    from datetime import datetime as _datetime
                    from supermem.utils.audio.environment.routine_memory import bucket_label
                    routine_check = self._routine_store().observe(
                        self._user_id, scene_tag, _datetime.now()
                    )
                    if routine_check["is_new_routine"]:
                        blabel = bucket_label(routine_check["bucket"])
                        new_routine = {
                            "scene": scene_tag,
                            "bucket": routine_check["bucket"],
                            "bucket_label": blabel,
                            "distinct_days": routine_check["distinct_days"],
                        }
                        synthetic_message = [{"role": "user", "content": (
                            f"[Routine pattern detected: user is regularly in "
                            f"'{scene_tag}' scene during {blabel}, "
                            f"observed on {routine_check['distinct_days']} different days]"
                        )}]
                        custom_instructions = (
                            "This is an automatically detected behavioral routine based on "
                            "recurring acoustic scene observations over multiple days. "
                            "Extract a concise memory fact describing this habitual pattern "
                            "(e.g. 'User usually commutes around 7-9am'). "
                            "Do NOT invent specifics beyond the scene and time window given."
                        )
                        self._extract_and_append_fn(synthetic_message, custom_instructions, ts, {
                            "source": "routine",
                            "routine_scene": scene_tag,
                            "routine_bucket": routine_check["bucket"],
                            "routine_distinct_days": routine_check["distinct_days"],
                            **({"session_id": session_id} if session_id is not None else {}),
                        })
                except Exception as _e:
                    print(f"  [routine] check failed: {_e}", flush=True)

            # Automatic clustering of familiar places: likewise identified only once on "entering" a scene
            # (scene_changed), using the whole recording's raw AST embedding to capture this specific place's acoustic fingerprint.
            if (
                _fire_result is not None and _fire_result.scene_changed
                and detection.get("embedding") is not None
            ):
                try:
                    from datetime import datetime as _datetime
                    place_result = self._place_store().identify(
                        detection["embedding"], scene=scene_tag, when=_datetime.now()
                    )
                except Exception as _e:
                    print(f"  [place] identification skipped: {_e}", flush=True)

        # WAV archive: record the audio_path -> memory_id mapping
        if audio_path and result.memory_ids:
            try:
                self._audio_archive().record(
                    result.memory_ids, self._user_id, str(audio_path)
                )
            except Exception as _e:
                print(f"  [audio_archive] record failed: {_e}", flush=True)

        # voiceprint tag: write person_id into memory_tags for later per-speaker retrieval
        if person_id and result.memory_ids:
            try:
                self._tag(result.memory_ids, [(f"speaker:{person_id}", 1.0)])
            except Exception as _e:
                print(f"  [speaker] tag write failed: {_e}", flush=True)

        # Music/humming tag: write tune_id into memory_tags for "same song/tune" retrieval;
        # when heard_count>=2, also generate a "heard a familiar tune again" memory fact.
        # "Is there music in this recording" and "which song is it" are two different things; the tag should follow the former.
        #
        # The AST environmental-sound classifier already judges the former (_MUSIC_KEYWORDS: music / singing /
        # humming / whistling / musical instrument...); a non-empty detection["music"] means it says yes. tune_result,
        # however, comes from music_store.identify() -- that is **identity matching**, answering "which stored
        # song does this resemble"; returning None only means it cannot tell which song, not that there is no
        # music. The tag used to be bound to tune_result, so every turn where music was "heard but not
        # recognised" was kept out of the playback candidate pool.
        ast_music = (detection or {}).get("music") or None
        if (ast_music or tune_result is not None) and result.memory_ids:
            tune_id = getattr(tune_result, "tune_id", None) if tune_result else None
            try:
                self._tag(result.memory_ids,
                          [(f"tune:{tune_id}" if tune_id else "tune:unidentified", 0.9)])
                # detection["music"] is {"labels": [(label, score), ...], "embedding": ...}
                _labels = (ast_music or {}).get("labels") or []
                top = f", strongest {_labels[0][0]} {_labels[0][1]:.2f}" if _labels else ""
                which = (f"{tune_result.action}, time #{tune_result.heard_count}"
                         if tune_result else "could not tell which song")
                print(f"  [music] tag {tune_id or 'unidentified'} → "
                      f"{len(result.memory_ids)} memories ({which}{top})", flush=True)
            except Exception as _e:
                print(f"  [music] tag write failed: {_e}", flush=True)

            if tune_result.action == "match" and tune_result.heard_count >= 2:
                try:
                    tune_labels = self._music_store().get_meta(tune_result.tune_id).get("labels", [])
                    synthetic_message = [{"role": "user", "content": (
                        f"[Recognized recurring background music/humming, heard "
                        f"{tune_result.heard_count} times before: {', '.join(tune_labels) or 'unknown tune'}]"
                    )}]
                    custom_instructions = (
                        "This is a recognition event for a recurring background tune (music or humming) "
                        "that has been heard multiple times before, detected via acoustic similarity. "
                        "Extract a concise memory fact noting that this familiar tune came up again. "
                        "Do NOT invent song titles or lyrics you don't actually know."
                    )
                    self._extract_and_append_fn(synthetic_message, custom_instructions, ts, {
                        "source": "music_recognition",
                        "tune_id": tune_result.tune_id,
                        "heard_count": tune_result.heard_count,
                        **({"session_id": session_id} if session_id is not None else {}),
                    })
                except Exception as _e:
                    print(f"  [music] recognition fact skipped: {_e}", flush=True)

        # Acoustics did not recognise it, but **the person said outright that this is music**: tag it anyway.
        #
        # Recognition is voiceprint-style similarity matching, and phone speaker playback, noisy surroundings or
        # clips that are too short all make it miss -- measured: a song played on a real device showed up in the
        # log as a single "music recognition missed" line. But the user said "let me play you a song", which is
        # more certain than any acoustic feature. Miss this and that recording never enters the playback candidate
        # pool, and later asking for "that song from last Wednesday" finds nothing.
        #
        # Use tune:unidentified rather than inventing a tune_id: we really do not know which song it is, so don't pretend.
        # The candidate pool queries `tune:%` (see _tune_memories in web/run.py), which matches this.
        # Don't name the variable the same as the module-level said_music -- that would shadow the function as a
        # local variable, and referencing it before assignment is an UnboundLocalError.
        _said = said_music(text)
        if (tune_result is None and not ast_music
                and _said and audio_path and result.memory_ids):
            try:
                self._tag(result.memory_ids, [("tune:unidentified", 0.6)])
                print(f"  [music] not recognised acoustically, but the user said it is music → tagged anyway"
                      f" ({len(result.memory_ids)} memories)", flush=True)
            except Exception as _e:
                print(f"  [music] tag write failed: {_e}", flush=True)

        # Log the untagged cases separately. The tag is the only direct evidence that "this recording contains
        # music", and playback relies on it to narrow candidates to real music; without it we can only fall back
        # to guessing by time and text. On a real device we have seen "music clearly heard but no tag", and looking
        # at the store afterwards cannot tell which case it was.
        #
        # Keep this as a separate if, not an elif of the one above -- the "generate a memory after hearing it
        # twice" part is nested inside that if, and adding an elif would push it out, so it would read attributes
        # of tune_result when it is None (AttributeError: 'NoneType' object has no attribute 'action').
        elif audio_path and not result.memory_ids:
            why = "no facts were stored this turn"
            print(f"  [music] no tune tag: {why}", flush=True)

        # Abnormal environmental sound memory: breaking / alarm / screaming -- worth recording from the first
        # occurrence; every detection gets a tag + a memory fact. Writing the standalone alert fact does not depend
        # on result.memory_ids (whether an abnormal sound is worth remembering is unrelated to whether this sentence has standalone fact value).
        if abnormal_hits:
            alert_memory_ids: list[str] = []
            try:
                labels_str = ", ".join(f"{l}({s:.2f})" for l, s in abnormal_hits)
                synthetic_message = [{"role": "user", "content": (
                    f"[Abnormal environmental sound event detected: {labels_str}]"
                )}]
                custom_instructions = (
                    "This is an unusual/alerting non-speech environmental sound event "
                    "(e.g. breaking glass, alarm, siren, or screaming) captured during the "
                    "conversation. Extract a concise memory fact describing this notable event. "
                    "Do NOT extract facts about the conversation topic itself."
                )
                alert_memory_ids = self._extract_and_append_fn(synthetic_message, custom_instructions, ts, {
                    "source": "abnormal_sound",
                    "abnormal_labels": [l for l, _ in abnormal_hits],
                    **({"session_id": session_id} if session_id is not None else {}),
                })
            except Exception as _e:
                print(f"  [abnormal] alert fact skipped: {_e}", flush=True)

            # Tagging: tag both the original turn's memories and the standalone alert facts created above, so that
            # at least one retrievable memory carries the abnormal:<label> tag.
            try:
                cog_store = self._repo()._cognitive_store
                tag_target_ids = list(result.memory_ids) + alert_memory_ids
                if cog_store and hasattr(cog_store, "upsert_memory_tags") and tag_target_ids:
                    tags = [
                        (f"abnormal:{label.lower().replace(' ', '_').replace(',', '')}", score)
                        for label, score in abnormal_hits
                    ]
                    for mid in tag_target_ids:
                        cog_store.upsert_memory_tags(mid, self._user_id, tags)
            except Exception as _e:
                print(f"  [abnormal] tag write failed: {_e}", flush=True)

        # Familiar-place tag: write place_id into memory_tags for later "same specific place" retrieval.
        if place_result is not None and result.memory_ids:
            try:
                self._tag(result.memory_ids, [(f"place:{place_result.place_id}", 0.9)])
            except Exception as _e:
                print(f"  [place] tag write failed: {_e}", flush=True)

        # Proactive "last time you were here" prompt for familiar surroundings: only a match has a "last time" to
        # mention. From memories previously tagged place:<id> at this place, pick a few relevant to the current topic, with visit info attached.
        if place_result is not None and place_result.action == "match" and result.memory_ids:
            try:
                cog_store = self._repo()._cognitive_store
                place_memories: list[dict] = []
                if cog_store and hasattr(cog_store, "memory_ids_for_slots_v2"):
                    place_mem_ids = set(
                        cog_store.memory_ids_for_slots_v2(
                            self._user_id, [f"place:{place_result.place_id}"]
                        )
                    ) - set(result.memory_ids)
                    if place_mem_ids:
                        _hits = self._rank(text, place_mem_ids, top_k=3)
                        place_memories = [
                            {"memory_id": h.memory_id, "content": h.text, "score": round(h.score, 3)}
                            for h in _hits
                        ]
                familiar_place_prompt = {
                    "place_id": place_result.place_id,
                    "visit_count": place_result.visit_count,
                    "previous_visit_at": place_result.previous_visit_at,
                    "memories": place_memories,
                }
            except Exception as _e:
                print(f"  [place] proactive recall failed: {_e}", flush=True)

        # Proactive push on scene switch: retrieve scene-related memories when entering a new scene (scene_changed avoids repeated firing)
        if scene_tag and _fire_result is not None and _fire_result.scene_changed:
            try:
                _query = _SCENE_RECALL_QUERY.get(scene_tag, "")
                if _query:
                    _hits = self._rank(_query, set(), top_k=3)
                    proactive_memories = [
                        {"memory_id": h.memory_id, "content": h.text, "score": round(h.score, 3)}
                        for h in _hits
                    ]
            except Exception as _e:
                print(f"  [proactive] failed: {_e}", flush=True)

        return {
            "triggered_reminders": triggered_reminders,
            "proactive_memories": proactive_memories,
            "familiar_place_prompt": familiar_place_prompt,
            "place_result": place_result,
            "new_routine": new_routine,
        }

    # ── Standalone environmental-sound ingestion ───────────────────────────────

    def IngestEnv(
        self,
        audio_path,
        recent_context: list[dict] | None = None,
        session_id: int | str | None = None,
        environment_override: str | None = None,
    ) -> dict:
        """Store an environmental sound event in the memory store.

        Parameters
        ----------
        audio_path:
            Path to the environmental-sound wav file.
        recent_context:
            Text of the last few conversation turns, as [{"role": "user"/"assistant", "content": "..."}].
            Lets the LLM infer what the user was doing at the time.
        session_id:
            Current session ID, used for temporal ordering.

        Returns
        -------
        dict
            ``{facts_count, memory_ids}``
        """
        import time
        from pathlib import Path as _Path

        # ── Step 1: detect background sounds; when CLAP re-checks in the background, use its result directly ──
        if environment_override:
            env_str = environment_override
        else:
            try:
                pairs = self._env_detector().detect_full(_Path(audio_path)).get("pairs") or []
                env_str = (
                    "background sounds: " + ", ".join(f"{label}({score:.2f})" for label, score in pairs)
                    if pairs else ""
                )
            except Exception as e:
                print(f"  [IngestEnv] detection failed: {e}", flush=True)
                return {"facts_count": 0, "memory_ids": []}

        if not env_str:
            return {"facts_count": 0, "memory_ids": []}

        # ── Step 2: build custom_instructions (environmental sound + conversation context) ──
        ctx_lines = ""
        if recent_context:
            ctx_lines = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in recent_context[-6:]
            )

        custom_instructions = (
            "This is a non-speech environmental sound event captured during the conversation. "
            "Based on the detected background sounds and the recent conversation context below, "
            "extract a concise memory fact describing what the user was likely doing or experiencing at this moment. "
            "Do NOT extract facts about the conversation topic itself — focus only on the environmental activity.\n"
            + (f"Recent conversation context:\n{ctx_lines}" if ctx_lines else "")
        )

        # ── Step 3: generate the fact with the extractor ─────────────────────────
        synthetic_message = [{"role": "user", "content": f"[Environmental sound event: {env_str}]"}]
        try:
            extracted = self._extractor().extract(
                new_messages=synthetic_message,
                custom_instructions=custom_instructions,
                observation_date=time.strftime("%H:%M:%S"),
                current_date=time.strftime("%H:%M:%S"),
            )
        except Exception as e:
            print(f"  [IngestEnv] extraction failed: {e}", flush=True)
            return {"facts_count": 0, "memory_ids": []}

        if not extracted:
            return {"facts_count": 0, "memory_ids": []}

        # ── Step 4: store in the left brain ─────────────────────────────────────
        meta = {
            "source":           "environment",
            "background_sounds": env_str,
            **({"session_id": session_id} if session_id is not None else {}),
        }
        memory_ids = self._repo().append_extracted(
            extracted, user_id=self._user_id, extra_metadata=meta
        )

        return {"facts_count": len(extracted), "memory_ids": memory_ids or []}
