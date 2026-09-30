"""Paper-aligned replacement implementation for EmotionDetector.

In the paper, phi(x_t) means "compute continuous V/A (valence/arousal) from raw
audio, then run multimodal attribution only on negatively significant turns".
That is different from the emotion2vec+ used by ``emotion_detector.py``
(a standalone nine-class audio emotion classifier whose behaviour does not match
the description of phi(x_t)): emotion2vec answers "which fixed emotion class does
this audio belong to", while phi(x_t) answers "what is the acoustic state of this
audio, and only turns worth attention deserve the expensive attribution".

The real implementation has two layers, mapped onto components that already exist
in ``supermem/emotion/`` but were never referenced by ``core.py`` before:
  1. Prosodic VAD (``vad_audio.HeuristicWavVADEstimator``) -- purely acoustic
     features, run on every turn, cheap.
  2. Qwen2.5-Omni multimodal attribution (``attribution_qwen_omni.QwenOmniEmotionAttributor``)
     -- only loaded/called for turns that VAD flags as "negatively significant"
     (``vad_trigger.is_negative_vad_significant``), so not every turn pays the
     multimodal LLM inference cost. The model is lazy-loaded; if loading fails
     (no GPU / weights not downloaded / not enough VRAM) it gracefully falls back
     to a coarse VAD-only classification and does not crash Ingest().

The return signature stays exactly the same as ``EmotionDetector.detect() -> str``
(it returns an emotion label string), so the dozen-odd downstream consumers of the
``emotion`` string in ``core.py::Ingest()`` (right-brain heartnote emotion anchors,
inner OS generation, emotional trait extraction, ...) need no changes; the string is
just now truly computed from the raw acoustic signal + (when needed) a multimodal
LLM, instead of a nine-class label unrelated to the audio waveform.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from supermem.utils.audio.emotion.layer import EmotionLayerConfig
from supermem.utils.audio.emotion.vad_audio import HeuristicWavVADEstimator, VADEstimator
from supermem.utils.audio.emotion.vad_trigger import is_negative_vad_significant


def _vad_to_label(valence: float, arousal: float) -> str:
    """VAD quadrant -> coarse emotion label, used as a fallback when the turn is
    "not significant" or Qwen-Omni is unavailable.

    The label vocabulary is aligned with ``anchor_router._CANONICAL_EMOTIONS`` (the
    fixed 8 classes used for right-brain emotion anchors), so labels from the fallback
    path are also recognised/normalised correctly downstream, instead of always
    returning the same value whenever the model is not triggered, like emotion2vec's
    "neutral" fallback does.

    The output is written via ``supermem.lang`` (``display_emotion``): this label is
    stored in memory and also shown to the user, and it normalises back to the same
    internal value when read again.
    """
    from supermem.lang import display_emotion
    if valence >= 0.15:
        canon = "happy" if arousal >= 0.4 else "calm"
    elif valence <= -0.15:
        if arousal >= 0.55:
            canon = "anxious"
        elif arousal >= 0.35:
            canon = "wronged"
        else:
            canon = "sad"
    else:
        canon = "calm" if arousal < 0.4 else "conflicted"
    return display_emotion(canon)


def _load_omni(model_path: str, *, device_map: str) -> tuple[Any, Any, Any]:
    """Load the Qwen2.5-Omni processor/tokenizer/model trio (bf16; only text
    output is needed, so the Thinker sub-model saves VRAM versus the full Omni).
    Same logic as ``examples/load_qwen_omni_attributor.py::load_omni``, inlined
    here instead of importing that script -- ``examples/`` is not part of the
    package and runtime code should not depend on it."""
    import torch
    from transformers import AutoTokenizer, Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    try:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True,
        )
    except TypeError:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True,
        )
    model.eval()
    return processor, tokenizer, model


class PaperAlignedEmotionDetector:
    """Always-on VAD estimator + lazy-loaded Qwen2.5-Omni attributor, with an
    interface compatible with ``EmotionDetector`` (``detect(audio_path) -> str``);
    ``core.py::_get_emotion_detector()`` can swap in this class without changing callers.
    """

    def __init__(
        self,
        *,
        vad_estimator: VADEstimator | None = None,
        layer_config: EmotionLayerConfig | None = None,
        omni_model_path: str | None = None,
        omni_device_map: str | None = None,
    ) -> None:
        self._vad = vad_estimator or HeuristicWavVADEstimator()
        self._config = layer_config or EmotionLayerConfig()
        # Env var takes priority; defaults to 3B when unset (less VRAM than 7B; both
        # are cached on this machine, see Phase 5 isolation tests) -- real deployments
        # should configure this according to their GPU.
        self._omni_model_path = omni_model_path or os.environ.get("SUPERMEM_OMNI_MODEL", "Qwen/Qwen2.5-Omni-3B")
        # On a shared multi-GPU machine where other processes fill VRAM, "auto" may
        # shard the model onto busy cards and OOM (reproduced in Phase 5 isolation
        # tests) -- real deployments should set SUPERMEM_OMNI_DEVICE to a card with
        # free VRAM; "auto" here just avoids imposing assumptions and is not a safe
        # default for every machine.
        self._omni_device_map = omni_device_map or os.environ.get("SUPERMEM_OMNI_DEVICE", "auto")
        self._attributor: Any = None       # lazy-loaded; None = not tried yet, False = load failed before (no retry)
        self._attributor_failed = False

    def _ensure_attributor(self):
        if self._attributor is not None:
            return self._attributor
        if self._attributor_failed:
            return None
        if os.environ.get("SUPERMEM_OMNI_ATTRIBUTION", "1") == "0":
            # Off by switch: same fallback as a failed load. On a CPU-only box the 3B Omni
            # model loads offloaded to disk and one attribution runs for minutes, holding
            # ~2 cores the whole time -- every STT call and reply in the demo slowed with it.
            self._attributor_failed = True
            return None
        try:
            from supermem.utils.audio.emotion.attribution_qwen_omni import QwenOmniEmotionAttributor

            processor, tokenizer, model = _load_omni(self._omni_model_path, device_map=self._omni_device_map)
            self._attributor = QwenOmniEmotionAttributor(processor=processor, model=model, tokenizer=tokenizer)
        except Exception as e:
            # After falling back to pure acoustic quadrant classification, emotion only
            # looks at the voice's valence/arousal and ignores content entirely -- saying
            # "I like strawberries" in a flat tone lands in the low-valence quadrant and
            # is judged sad. All right-brain emotion records are built on this label, so
            # once it limps, the whole right-brain half becomes untrustworthy.
            # Missing torchvision is the most common cause (transformers needs
            # Qwen2VLVideoProcessor to load Qwen2.5-Omni, which imports torchvision),
            # and nobody would think of it. So this message must be prominent and give
            # the fix directly, not just print a one-line exception.
            hint = ""
            if "torchvision" in str(e).lower() or "Torchvision" in str(e):
                hint = ("\n  [emotion]   → torchvision is missing. Just install it: pip install torchvision"
                        "\n  [emotion]     (without it, emotion relies only on acoustic quadrants and right-brain emotion memory is unreliable)")
            print(f"  [emotion] ⚠ Qwen-Omni attributor failed to load, emotion falls back to coarse acoustic classification: {e}{hint}",
                  flush=True)
            self._attributor_failed = True
            return None
        return self._attributor

    def detect(self, audio_path: Path) -> str:
        """Return an emotion label string (compatible with ``EmotionDetector.detect()``)."""
        try:
            vad = self._vad.estimate(str(audio_path))
        except Exception as e:
            print(f"  [emotion] VAD estimation failed: {e}", flush=True)
            return "unknown"

        if not is_negative_vad_significant(vad, self._config):
            return _vad_to_label(vad.valence, vad.arousal)

        attributor = self._ensure_attributor()
        if attributor is None:
            return _vad_to_label(vad.valence, vad.arousal)

        try:
            import uuid as _uuid

            from supermem.utils.audio.emotion.types import TurnEmotionRecord

            turn = TurnEmotionRecord(turn_id=_uuid.uuid4().hex, session_id="ingest", vad=vad)
            result = attributor.analyze_turn_with_audio(
                audio_path=str(audio_path), asr_text=None,
                left_memory_block="", emotion_graph_context=None, turn=turn,
            )
            label = (result.emotion.label or "").strip()
            print(f"  [emotion] Qwen-Omni attribution → {label!r} (VAD={vad})", flush=True)
            return label or _vad_to_label(vad.valence, vad.arousal)
        except Exception as e:
            print(f"  [emotion] Qwen-Omni attribution failed, falling back to coarse VAD classification: {e}", flush=True)
            return _vad_to_label(vad.valence, vad.arousal)
