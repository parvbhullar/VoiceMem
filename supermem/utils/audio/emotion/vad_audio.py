from __future__ import annotations

import math
import wave
from typing import Protocol, runtime_checkable

import numpy as np

from supermem.utils.audio.emotion.types import VAD


def load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV as float32 mono in [-1, 1], preferring soundfile."""
    try:
        import soundfile as sf  # type: ignore[import-untyped]

        audio, sr = sf.read(path, always_2d=False)
    except ImportError:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sw == 2:
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sw}")

        if nch > 1:
            x = x.reshape(-1, nch).mean(axis=1)
        audio = x
    else:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = np.asarray(audio, dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.5:
            audio = np.clip(audio / 32768.0, -1.0, 1.0)

    return audio, int(sr)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _zero_crossing_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    s = np.sign(x)
    s[s == 0] = 1
    return float(np.mean(np.abs(np.diff(s)) > 0))


def _frame_prosody_features(x: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Extract frame-level mean RMS, dynamic range and ZCR from the raw waveform (no pre-emphasis)."""
    frame = max(1, int(sr * 0.025))
    hop = max(1, frame // 2)
    rms_list: list[float] = []
    zcr_list: list[float] = []
    for start in range(0, len(x), hop):
        chunk = x[start : start + frame]
        if chunk.size < frame // 2:
            break
        rms_list.append(_rms(chunk))
        zcr_list.append(_zero_crossing_rate(chunk))

    if not rms_list:
        return _rms(x), 0.0, _zero_crossing_rate(x)

    rms_arr = np.asarray(rms_list, dtype=np.float64)
    r_mean = float(np.mean(rms_arr))
    r_dyn = float(np.percentile(rms_arr, 90) - np.percentile(rms_arr, 10))
    z_mean = float(np.mean(zcr_list))
    return r_mean, r_dyn, z_mean


def _prosody_to_vad(r_mean: float, r_dyn: float, z_mean: float) -> VAD:
    """RMS/ZCR/dynamic range → V/A.

    Calibrated on Fish TTS speech: upbeat sentences have low ZCR and slightly higher dynamics; tired/frustrated sentences have high ZCR.
    """
    # Valence: low ZCR leans positive; high ZCR (breathy/tense) leans negative; moderate dynamic range leans slightly positive
    valence = (
        0.55 * math.tanh((0.11 - z_mean) * 12.0)
        + 0.35 * math.tanh((r_dyn - 0.16) * 10.0)
        - 0.20
    )
    valence = float(max(-1.0, min(1.0, valence)))

    # Arousal: higher ZCR and dynamic range → higher arousal (both frustration and excitement can sound "rougher")
    arousal = (
        0.45 * min(1.0, z_mean / 0.20)
        + 0.35 * min(1.0, r_dyn / 0.24)
        + 0.20 * min(1.0, r_mean / 0.10)
    )
    arousal = float(max(0.0, min(1.0, arousal)))

    return VAD(valence=valence, arousal=arousal)


@runtime_checkable
class VADEstimator(Protocol):
    def estimate(self, audio_path: str) -> VAD:
        ...


class HeuristicWavVADEstimator:
    """Purely acoustic V/A: frame-level RMS + dynamic range + ZCR (calibrated for TTS/dialogue wavs).

    The old pre-emphasis + fixed rms_ref squashed Fish TTS clips into the same V/A bucket; replaced by this implementation.
    """

    def estimate(self, audio_path: str) -> VAD:
        x, sr = load_wav_mono(audio_path)
        if x.size == 0:
            return VAD(valence=0.0, arousal=0.0)
        r_mean, r_dyn, z_mean = _frame_prosody_features(x, sr)
        return _prosody_to_vad(r_mean, r_dyn, z_mean)
