"""Voiceprint and raw-audio storage configuration.

Pick a scene via environment variable:
    VOICE_SCENE=medical   -> keep both voiceprint and raw audio
    VOICE_SCENE=companion -> keep voiceprint only
    VOICE_SCENE=diary     -> keep neither
    VOICE_SCENE=default   -> keep voiceprint only (default)

Individual overrides are also available:
    VOICE_ENABLE_VOICEPRINT=true/false
    VOICE_RETAIN_RAW_AUDIO=true/false
    VOICE_RAW_AUDIO_DIR=/path/to/dir
    VOICE_MATCH_THR=0.50
    VOICE_CAND_THR=0.40
    VOICE_MERGE_THR=0.65
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VoiceStoreConfig:
    retain_voiceprint: bool = True
    retain_raw_audio: bool = False
    raw_audio_dir: Optional[Path] = None
    match_threshold: float = 0.50
    candidate_threshold: float = 0.40
    # Threshold for permanently merging (and renaming) two person_ids' candidate voiceprints,
    # deliberately higher than match_threshold: match_threshold only asks "does this utterance
    # sound alike", and a wrong call merely softly pollutes a profile a little (later real
    # observations can pull the centroid back); merging is a hard, nearly irreversible operation --
    # two profiles fused into one, names overwriting each other. In testing, 0.542 (just above
    # match_threshold) was enough to permanently fuse two genuinely different people
    # (Nancy/Jennifer, whose voices are hard to tell apart) with no way to split them again,
    # so a clearly higher confidence is required.
    merge_threshold: float = 0.65

    _PRESETS: dict = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def for_scene(cls, scene: str) -> "VoiceStoreConfig":
        presets = {
            "medical":   cls(retain_voiceprint=True,  retain_raw_audio=True),
            "legal":     cls(retain_voiceprint=True,  retain_raw_audio=True),
            "companion": cls(retain_voiceprint=True,  retain_raw_audio=False),
            "diary":     cls(retain_voiceprint=False, retain_raw_audio=False),
            "default":   cls(retain_voiceprint=True,  retain_raw_audio=False),
        }
        return presets.get(scene, cls())

    @classmethod
    def from_env(cls) -> "VoiceStoreConfig":
        scene = os.environ.get("VOICE_SCENE", "default")
        cfg = cls.for_scene(scene)

        # environment variables can override the scene preset field by field
        if "VOICE_ENABLE_VOICEPRINT" in os.environ:
            cfg.retain_voiceprint = os.environ["VOICE_ENABLE_VOICEPRINT"].lower() == "true"
        if "VOICE_RETAIN_RAW_AUDIO" in os.environ:
            cfg.retain_raw_audio = os.environ["VOICE_RETAIN_RAW_AUDIO"].lower() == "true"
        if "VOICE_RAW_AUDIO_DIR" in os.environ:
            cfg.raw_audio_dir = Path(os.environ["VOICE_RAW_AUDIO_DIR"])
        if "VOICE_MATCH_THR" in os.environ:
            cfg.match_threshold = float(os.environ["VOICE_MATCH_THR"])
        if "VOICE_CAND_THR" in os.environ:
            cfg.candidate_threshold = float(os.environ["VOICE_CAND_THR"])
        if "VOICE_MERGE_THR" in os.environ:
            cfg.merge_threshold = float(os.environ["VOICE_MERGE_THR"])

        return cfg
