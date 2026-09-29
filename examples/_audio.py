"""One full-duplex audio device + echo cancellation, shared by 04 / 05.

With speakers on, the mic records "your voice + the assistant's voice from the speaker". AEC takes the signal the
speaker is playing (far-end) as reference and subtracts it from the mic signal; what remains is the real human voice.
Without this step, transcription stores the assistant's words in memory as if you said them, and VAD keeps thinking
someone is talking -- the assistant cuts itself off as soon as it starts speaking.

So playback must go through the **same** `sd.Stream` as recording: only within the same callback can you get a
far-end that is exactly aligned with this mic frame.

    audio = AudioIO(loop, on_barge_in=stop.set)
    audio.start()
    pcm = await audio.mic.get()      # echo-cancelled 16k PCM16
    audio.play(pcm24k)               # 24k TTS/realtime audio, downsampled to 16k internally

(examples/03 inlines a copy of the same thing; that example deliberately stays a self-contained single file.)
"""
import asyncio
import threading

import numpy as np
import sounddevice as sd
from pywebrtc_audio import AudioProcessor

from supermem.utils.audio.stream_io import resample

SR = 16000        # WebRTC APM only takes 8/16/32/48k; supermem is 16k internally too, a perfect fit
BLOCK = 160       # 10ms, APM's native frame length
PLAY_SR = 24000   # both TTS and realtime output 24k


class AudioIO:
    """Recording (echo-cancelled) + playback + barge-in detection, all on one device.

    ``on_barge_in``: when human voice is heard for ``barge_s`` while the assistant is speaking, it is called once on
    the event loop (after the playback buffer has already been cleared). ``grace_s`` is a grace period right after
    the assistant starts -- then the mic holds almost only the assistant's own voice and AEC has not converged yet,
    so it would easily cut itself off at the first sound.
    """

    def __init__(self, loop, on_barge_in=None, *, threshold=0.3,
                 barge_s=0.5, grace_s=0.6, mic_backlog=100):
        self.loop = loop
        self.mic: asyncio.Queue = asyncio.Queue(maxsize=mic_backlog)
        self.on_barge_in = on_barge_in
        self.threshold, self.barge_s, self.grace_s = threshold, barge_s, grace_s

        self._buf = bytearray()                 # playback buffer, 16k int16
        self._lock = threading.Lock()
        self._active = threading.Event()        # assistant is speaking
        self._spoke_s = 0.0                     # how long human voice has been heard continuously
        self._said_s = 0.0                      # how long the assistant has spoken this turn (for the grace period)

        self.aec = AudioProcessor(sample_rate=SR, echo_cancellation=True,
                                  noise_suppression=True, auto_gain_control=False,
                                  stream_delay_ms=0)
        self.stream = sd.Stream(samplerate=SR, blocksize=BLOCK, channels=1,
                                dtype="float32", callback=self._cb)

    # ── Audio callback (on the audio thread: no blocking and no network) ────────
    def _pull(self, n) -> np.ndarray:
        with self._lock:
            take = bytes(self._buf[:n * 2])
            del self._buf[:n * 2]
        out = np.zeros(n, dtype=np.float32)
        f = np.frombuffer(take, np.int16).astype(np.float32) / 32768.0
        out[:len(f)] = f
        return out

    def _cb(self, indata, outdata, frames, _time, _status):
        far = self._pull(frames)                       # what the speaker plays for this frame
        outdata[:, 0] = far
        clean = self.aec.process(indata[:, 0].copy(), far)   # subtract it from the mic

        if self._active.is_set():
            self._said_s += frames / SR
            self._spoke_s = (self._spoke_s + frames / SR
                             if self.aec.speech_probability >= self.threshold else 0.0)
            if self._spoke_s >= self.barge_s and self._said_s >= self.grace_s:
                self._active.clear()
                self._spoke_s = 0.0
                self.stop_playing()                    # stop playback first: a local operation, takes effect immediately
                if self.on_barge_in:
                    self.loop.call_soon_threadsafe(self.on_barge_in)
        else:
            self._spoke_s = 0.0

        pcm = (np.clip(clean, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        # drop when full: nobody consumes during the seconds the assistant talks; keeping it only makes the next turn's ASR lag far behind
        self.loop.call_soon_threadsafe(
            lambda: None if self.mic.full() else self.mic.put_nowait(pcm))

    # ── Playback side ─────────────────────────────────────────────────────
    def play(self, pcm24: bytes):
        """Put 24k PCM16 into the playback buffer (downsampled to 16k internally). This is also exactly the AEC reference signal."""
        f = np.frombuffer(pcm24, np.int16).astype(np.float32) / 32768.0
        f16 = resample(f, src=PLAY_SR, dst=SR)
        with self._lock:
            self._buf += (np.clip(f16, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    def stop_playing(self):
        with self._lock:
            self._buf.clear()

    def busy(self) -> bool:
        with self._lock:
            return bool(self._buf)

    # ── Start/end of the assistant's turn (barge-in detection needs to know if it is speaking) ──
    def assistant_started(self):
        self._spoke_s = self._said_s = 0.0
        self._active.set()

    def assistant_done(self):
        self._active.clear()

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()
