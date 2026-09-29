"""Small audio helpers for streaming input: resampling + silero VAD.

Moved as-is from web/utils.py (the mind-map html sends 24k, streaming ASR wants 16k;
silero VAD decides when the speaker is done) and promoted to a core capability, shared
by VoiceStream in supermem/stream.py and the web demo (removing duplication).

VAD is now an injectable capability (``SuperMem(vad=...)`` / the ``vad`` section of the
config); ``make_vad`` is just the built-in silero implementation. To swap in your own,
pass any object with ``is_speech(frame) -> bool``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from supermem.utils.common.paths import model_path, require


def resample(f32, src=24000, dst=16000):        # mind-map html sends 24k, streaming ASR wants 16k
    n = int(len(f32) * dst / src)
    return np.interp(np.arange(n) * src / dst, np.arange(len(f32)), f32).astype(np.float32)


def read_wav(path) -> tuple[np.ndarray, int]:
    """Read an audio file -> (float32 mono, sample rate). Uses soundfile if available
    (supports all formats), otherwise falls back to the stdlib wave module (PCM wav
    only, but no extra dependency)."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
        return (audio[:, 0] if audio.ndim > 1 else audio), sr
    except ImportError:
        import wave
        with wave.open(str(path), "rb") as w:
            sr, n_ch = w.getframerate(), w.getnchannels()
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        return (pcm[::n_ch] if n_ch > 1 else pcm), sr


def transcribe_file(asr, path, chunk_s: float = 0.6) -> str:
    """Transcribe a whole file into one text using the streaming ASR: feed it in
    chunks, then flush to finish.

    This just wraps the "streaming interface" as a "whole-utterance interface" without
    introducing a second ASR; one-off calls like ingest(audio=...) don't need to load a
    separate non-streaming model for this.
    """
    audio, sr = read_wav(path)
    if sr != 16000:
        audio = resample(audio, src=sr)
    asr.reset()
    step, text = max(1, int(16000 * chunk_s)), ""
    for i in range(0, len(audio), step):
        text = asr.feed(audio[i:i + step]) or text
    flush = getattr(asr, "flush", None)          # chunked ASR zero-pads and emits the partial trailing chunk
    if flush is not None:
        text = flush() or text
    return (text or "").strip()


def make_vad(model: str | None = None, threshold: float = 0.5):
    """Built-in VAD: silero (via sherpa-onnx). Returns a small object exposing only
    ``is_speech(frame)``.

    If ``model`` is not given, uses ``SUPERMEM_SILERO_VAD`` / ``SUPERMEM_MODELS_DIR/silero_vad.onnx``.
    This .onnx has no auto-download fallback, so a missing file is reported explicitly
    (instead of letting sherpa raise an obscure error).
    """
    import sherpa_onnx
    path = require(
        Path(model) if model else model_path("silero_vad.onnx", "vad", kind="vad"),
        "silero VAD model silero_vad.onnx",
    )
    v = sherpa_onnx.VoiceActivityDetector(sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(model=str(path), threshold=threshold),
        sample_rate=16000), buffer_size_in_seconds=30)

    class _V:
        def is_speech(self, frame): v.accept_waveform(frame); return v.is_speech_detected()
    return _V()
