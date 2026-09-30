"""Factories for supermem's built-in default implementations of each capability (util name -> no-arg factory).

core.py's Utils uses this to build defaults; passing a function to SuperMem(embedding=..., slots=...) overrides that entry.
Nine slots: embedding / schema / entity / emotion / voiceprint / asr / vad /
memory_engine / tts. The first eight are on the core path (which ones load depends on mode via _NEED);
tts is not -- the memory system stops at text, speech output is an optional layer.
Kept here rather than in core.py so the top-level facade only describes the "system skeleton" and isn't bloated by these imports.
"""
from __future__ import annotations

import os


def default_utils(base_url, memory_root):
    def embedding():
        from supermem.leftbrain.local_memory_store import OpenAILocalEmbedder, OpenAILocalEmbedderConfig
        return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(base_url=base_url))
    def slots():
        # Default local E5 classifier: 0 LLM, 0 network -- the 0-300ms speculative-prefetch budget
        # can't afford a network call, and Classify is on that path (_speculate in supermem/stream.py).
        # sentence-transformers isn't a base dependency (installed with the [demo] extra); if missing,
        # fall back to the LLM version and print a note -- a silent fallback means quietly spending money.
        # SUPERMEM_SLOTS=openai forces the LLM version (when entity extraction / sub-slot drill-down is needed).
        if os.environ.get("SUPERMEM_SLOTS", "local").lower() != "openai":
            try:
                from supermem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
                from supermem.leftbrain.local_e5_embedder import shared_e5
                return LocalQueryClassifier(model=shared_e5())   # shares one E5 instance with the local embedder
            except ImportError as e:
                print(f"[slots] local classifier unavailable ({e}) -> falling back to LLM QuerySlotClassifier. "
                      "Install sentence-transformers (or pip install -e '.[demo]') to use the local one.",
                      flush=True)
        from supermem.leftbrain.cognitive_graph.query_slot_classifier import QuerySlotClassifier
        return QuerySlotClassifier()
    def entity():
        from supermem.leftbrain.cognitive_graph.annotator import CognitiveAnnotator, CognitiveAnnotatorConfig
        return CognitiveAnnotator(CognitiveAnnotatorConfig(base_url=base_url))
    def emotion():
        from supermem.utils.audio.emotion.paper_emotion_detector import PaperAlignedEmotionDetector
        return PaperAlignedEmotionDetector()
    def voiceprint():
        from supermem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
        return SpeakerEncoder(device="cpu")
    def asr():
        # SUPERMEM_ASR=openai uses the transcription API: no model download and
        # multilingual, Hindi included. sherpa uses the sherpa-onnx streaming
        # zipformer (bilingual zh-en, pure onnx, no torch). The default is FunASR
        # paraformer-zh-streaming, which is **Chinese only** -- speak English at
        # it and the words come back forced into Chinese characters.
        _pick = os.environ.get("SUPERMEM_ASR", "funasr").lower()
        if _pick == "openai":
            from supermem.utils.audio.asr import OpenAIStreamingASR
            return OpenAIStreamingASR()
        if _pick == "elevenlabs":
            # ElevenLabs Scribe: keeps the spoken language (Hindi stays Hindi), see ElevenLabsScribeASR.
            from supermem.utils.audio.asr import ElevenLabsScribeASR
            return ElevenLabsScribeASR()
        if _pick == "sherpa":
            from supermem.utils.audio.asr import StreamingASR
            from supermem.utils.common.paths import model_path
            return StreamingASR(str(model_path(
                "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20", kind="asr")))
        from supermem.utils.audio.asr import FunASRStreamingASR
        return FunASRStreamingASR()
    def vad():
        # VAD that decides "finished speaking". Built-in silero by default; to use your own, pass an
        # object with is_speech(frame)->bool (SuperMem(vad=lambda: MyVad()) or the vad section of config).
        from supermem.utils.audio.stream_io import make_vad
        return make_vad()
    def tts():
        # The ninth replaceable slot. The core path doesn't use it -- the memory system stops at text,
        # speech output is the caller's business -- so tts isn't in _NEED (warmup won't load it);
        # whoever needs speech calls utils.get("tts").
        # base_url is not passed down: it usually points at a self-hosted LLM/embedding service that
        # most likely has no /audio/speech, so following it would only blow up at speak time. Use OPENAI_TTS_BASE_URL to change the endpoint.
        from supermem.tts import make_tts
        return make_tts()
    def memory_engine():
        from pathlib import Path
        from supermem.leftbrain.mem0_backend_store import Mem0BackendStore
        # memory_root is passed down by the Orchestrator (defaults already resolved); this fallback only
        # matters when default_utils is constructed directly, and keeps the same default as above.
        return Mem0BackendStore(embedding(),
                                memory_root=Path(memory_root or Path.cwd() / "supermem_memory"))
    return {"embedding": embedding, "slots": slots, "entity": entity, "emotion": emotion,
            "voiceprint": voiceprint, "asr": asr, "vad": vad, "memory_engine": memory_engine,
            "tts": tts}
