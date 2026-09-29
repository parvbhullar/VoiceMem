"""VAD negative-significance check (triggers in-segment fusion; EMA not involved)."""

from __future__ import annotations

from typing import Protocol

from supermem.utils.audio.emotion.types import VAD


class NegativeVadTriggerConfig(Protocol):
    valence_negative_threshold: float
    min_arousal_for_anomaly: float | None


def is_negative_vad_significant(vad: VAD, config: NegativeVadTriggerConfig) -> bool:
    """Whether this turn's VAD is "negatively significant": valence below the threshold, with an optional arousal floor."""
    if vad.valence > float(config.valence_negative_threshold):
        return False
    min_ar = config.min_arousal_for_anomaly
    if min_ar is not None and vad.arousal < float(min_ar):
        return False
    return True
