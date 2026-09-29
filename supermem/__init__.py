"""Supermem: a memory framework with separate left and right brains, plus an audio-native perception layer.

This file is the "component directory" -- every supermem component is exposed here,
available via `from supermem import X`. Audio components (SpeakerEncoder /
ASTEnvironmentDetector etc.) depend on torch / sherpa-onnx, so they are loaded lazily
(PEP 562 `__getattr__`): a module is imported only when one of its names is actually accessed,
and `import supermem` itself never pulls in these heavy dependencies -- a text-only install can still use the core.

Groups at a glance:
  · Core             SuperMem / SearchResult / RightBrainHit / AudioPerception
  · Reply (output)   openai_reply / normalize_reply (see supermem/reply.py)
  · Voice adapters   VoiceInput / VoiceContent / VoiceprintRegistry / ingest_voice_input …
  · Audio perception SpeakerEncoder / *EnvironmentDetector / Scene* / Music/Place/Routine …
  · Right-brain emotion layer  EmotionLayer / EmotionLayerConfig / EmotionLayerResult
  · Convenience memory API    Memory / inject / recall / remember
  · Subpackages      leftbrain / rightbrain / utils (audio·common·fusion all live under utils)
"""

from __future__ import annotations

# ── INFO noise from third-party libraries ───────────────────────────────────
# Running the basic usage once, the openai SDK logs a line "HTTP Request: POST ... 200 OK" per request
# (twenty or thirty lines), funasr prints an rtf progress bar per transcribed chunk, mem0 logs a line per stored item
# "Updating memory with data=...". The real output is buried in between, and first-time users think something went wrong.
#
# Use a filter rather than setLevel: a later basicConfig by the caller cannot override this.
# To see everything: SUPERMEM_VERBOSE=1.
def _quiet_third_party_logs() -> None:
    import logging
    import os

    if os.environ.get("SUPERMEM_VERBOSE", "0") != "0":
        return

    os.environ.setdefault("TQDM_DISABLE", "1")          # progress bars from funasr / transformers

    # Use setLevel rather than addFilter: a filter only applies to the logger it is attached to and does not
    # propagate to child loggers (openai._base_client, funasr.xxx are child loggers) -- in practice it does not block them.
    # Levels, however, are inherited: setting "openai" to WARNING also applies to "openai._base_client".
    # httpx2 / httpcore2 are forked versions pulled in by mem0's dependency chain; their logger names carry the 2 as well.
    # Listing only "httpx" does not block them -- every embedding / chat call would print a line
    # "HTTP Request: POST https://api.openai.com/v1/embeddings ...",
    # a dozen or more lines per ingest.
    for name in ("openai", "httpx", "httpx2", "httpcore", "httpcore2",
                 "mem0", "funasr", "modelscope",
                 "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # transformers has a class of notices that go through its own warning system, which logging levels do not control
    # ("Using a slow image processor as use_fast is unset..."). Use its own switch.
    # tokenizers prints a "The current process just got forked..." warning after a fork.
    # We do not rely on its parallelism anyway (the heavy work is in ASR/embedding), so turn it off.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


_quiet_third_party_logs()



try:                       # only an installed package has metadata; running straight from the source tree does not
    from importlib.metadata import PackageNotFoundError, version as _v
    __version__ = _v("supermem")
except Exception:
    __version__ = "0.0.0.dev"

import importlib
from typing import TYPE_CHECKING

# ── name -> source module lazy mapping ("module.path:attribute") ─────────────────────
# Every component is registered here; __getattr__ imports it on first access and caches it in this module's
# namespace, after which it is a plain attribute access with no extra overhead.
_LAZY: dict[str, str] = {
    # ── Core (the SuperMem facade is in core.py; orchestration/data structures are in orchestrator.py;
    #    LeftBrain/RightBrain are real components)──
    "SuperMem":                 "supermem.core:SuperMem",
    "Utils":                    "supermem.orchestrator:Utils",
    "LeftBrain":                "supermem.leftbrain.brain:LeftBrain",
    "RightBrain":               "supermem.rightbrain.brain:RightBrain",
    "SearchResult":             "supermem.orchestrator:SearchResult",
    "RightBrainHit":            "supermem.rightbrain.brain:RightBrainHit",
    "AudioPerception":          "supermem.utils.audio.perceiver:AudioPerception",
    "VoiceStream":              "supermem.stream:VoiceStream",
    "openai_reply":             "supermem.reply:openai_reply",
    "normalize_reply":          "supermem.reply:normalize",
    "Turn":                     "supermem.stream:Turn",
    "StreamState":              "supermem.stream:StreamState",

    # ── Voice input adapter layer (structured output of the upstream voice module -> left-brain injection)──
    "VoiceInput":               "supermem.utils.common.voice_input:VoiceInput",
    "VoiceContent":             "supermem.utils.common.voice_input:VoiceContent",
    "VoiceIngestResult":        "supermem.utils.common.voice_input:VoiceIngestResult",
    "VoiceprintRegistry":       "supermem.utils.common.voice_input:VoiceprintRegistry",
    "VoiceprintEntry":          "supermem.utils.common.voice_input:VoiceprintEntry",
    "ingest_voice_input":       "supermem.utils.common.voice_input:ingest_voice_input",
    "voice_input_to_messages":  "supermem.utils.common.voice_input:voice_input_to_messages",
    "emotion_to_affect":        "supermem.utils.common.voice_input:emotion_to_affect",
    "map_voice_slots_to_slotv2":"supermem.utils.common.voice_input:map_voice_slots_to_slotv2",

    # ── Audio-native perception: speaker voiceprint ──
    "SpeakerEncoder":           "supermem.utils.audio.voiceprint.speaker_encoder:SpeakerEncoder",
    "VoiceprintStore":          "supermem.utils.audio.voiceprint.voiceprint_store:VoiceprintStore",
    "IdentifyResult":           "supermem.utils.audio.voiceprint.voiceprint_store:IdentifyResult",
    "parse_self_identification":"supermem.utils.audio.voiceprint.speaker_identity:parse_self_identification",

    # ── Audio-native perception: acoustic environment / scene ──
    "ASTEnvironmentDetector":   "supermem.utils.audio.environment.environment_detector_ast:ASTEnvironmentDetector",
    "CLAPEnvironmentDetector":  "supermem.utils.audio.environment.environment_detector_clap:CLAPEnvironmentDetector",
    "SceneTag":                 "supermem.utils.audio.environment.scene_classifier:SceneTag",
    "SceneResult":              "supermem.utils.audio.environment.scene_classifier:SceneResult",
    "classify_scene":           "supermem.utils.audio.environment.scene_classifier:classify_scene",
    "scene_to_response_directive":"supermem.utils.audio.environment.scene_classifier:scene_to_response_directive",
    "infer_scene_from_text":    "supermem.utils.audio.environment.scene_classifier:infer_scene_from_text",

    # ── Audio-native perception: scene reminder triggers ──
    "SceneTrigger":             "supermem.utils.audio.environment.scene_trigger:SceneTrigger",
    "SceneTriggerStore":        "supermem.utils.audio.environment.scene_trigger:SceneTriggerStore",
    "TriggerFireResult":        "supermem.utils.audio.environment.scene_trigger:TriggerFireResult",
    "parse_trigger_intent":     "supermem.utils.audio.environment.scene_trigger:parse_trigger_intent",
    "check_and_fire":           "supermem.utils.audio.environment.scene_trigger:check_and_fire",

    # ── Audio-native perception: music / place / routine memory ──
    "MusicMemoryStore":         "supermem.utils.audio.environment.music_memory:MusicMemoryStore",
    "TuneIdentifyResult":       "supermem.utils.audio.environment.music_memory:TuneIdentifyResult",
    "PlaceMemoryStore":         "supermem.utils.audio.environment.place_memory:PlaceMemoryStore",
    "PlaceIdentifyResult":      "supermem.utils.audio.environment.place_memory:PlaceIdentifyResult",
    "RoutineStore":             "supermem.utils.audio.environment.routine_memory:RoutineStore",
    "bucket_label":             "supermem.utils.audio.environment.routine_memory:bucket_label",

    # ── Recording archive / session / config ──
    "AudioArchive":             "supermem.utils.audio.audio_archive:AudioArchive",
    "SessionTracker":           "supermem.utils.common.session_tracker:SessionTracker",
    "VoiceStoreConfig":         "supermem.utils.common.voice_config:VoiceStoreConfig",

    # ── Right-brain emotion layer ──
    "EmotionLayer":             "supermem.utils.audio.emotion:EmotionLayer",
    "EmotionLayerConfig":       "supermem.utils.audio.emotion:EmotionLayerConfig",
    "EmotionLayerResult":       "supermem.utils.audio.emotion:EmotionLayerResult",

    # ── Startup self-check (per-component timing + gated startup)──
    "run_startup_check":        "supermem.startup_check:run_startup_check",
    "check_and_gate":           "supermem.startup_check:check_and_gate",
    "StartupReport":            "supermem.startup_check:StartupReport",

    # ── Convenience memory API ──
    "Memory":                   "supermem.memory_api:Memory",
    "build_memory_context":     "supermem.memory_api:build_memory_context",
    "inject":                   "supermem.memory_api:inject",
    "recall":                   "supermem.memory_api:recall",
    "remember":                 "supermem.memory_api:remember",
}

# Subpackages are exposed as components too: `from supermem import emotion` / `supermem.leftbrain` …
# Components inside each subpackage are handled by its own __init__.__all__; only the packages themselves are listed here.
_SUBPACKAGES: tuple[str, ...] = (
    "leftbrain", "rightbrain", "utils",
)

__all__ = sorted([*_LAZY, *_SUBPACKAGES])


def __getattr__(name: str):
    """PEP 562 lazy attribute access: import components on demand so `import supermem` does not trigger heavy dependencies."""
    target = _LAZY.get(name)
    if target is not None:
        module_path, _, attr = target.partition(":")
        value = getattr(importlib.import_module(module_path), attr)
        globals()[name] = value          # cache it; later accesses are plain attribute lookups
        return value
    if name in _SUBPACKAGES:
        module = importlib.import_module(f"supermem.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])


# Let static type checkers / IDEs see these names too (not executed at runtime, no heavy dependencies).
if TYPE_CHECKING:  # pragma: no cover
    from supermem.core import SuperMem
    from supermem.orchestrator import SearchResult
    from supermem.rightbrain.brain import RightBrainHit
    from supermem.utils.audio.perceiver import AudioPerception
    from supermem.utils.audio.emotion import (
        EmotionLayer, EmotionLayerConfig, EmotionLayerResult,
    )

def sample_audio(path):
    """Resolve a sample-audio path into one that can actually be opened.

    README and examples use ``assets/input.wav`` -- a path relative to **the repo**.
    People who ``pip install supermem`` do not have that directory, so running the first README example gives
    a LibsndfileError, and that is exactly their first impression of this project.

    So: if the given path exists, use it as is (people who cloned the repo are unaffected); if it does not, and the package ships
    a sample audio file of the same name, fall back to the bundled one. Other paths behave as before -- not found is still not found,
    and the original error is raised as usual, rather than quietly turning "user typo in the path" into "played a sample clip".
    """
    import os
    from pathlib import Path

    if path is None:
        return None
    p = Path(str(path))
    if p.exists():
        return str(path)
    packaged = Path(__file__).resolve().parent / "assets" / p.name
    if packaged.is_file():
        return str(packaged)
    return str(path)
