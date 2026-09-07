"""API-backed streaming ASR: single-flight partials, final flush.

The transcribe call is injected, so these run with no network and no models::

    python tests/test_openai_asr.py
"""
import sys
import threading
import time
import unittest
import wave
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicemem.utils.audio.asr import OpenAIStreamingASR, _wav_bytes  # noqa: E402

RATE = 16000


def _audio(seconds):
    return np.zeros(int(RATE * seconds), dtype=np.float32)


def _settle(asr, timeout=2.0):
    """Wait for the in-flight partial thread, so assertions are not racy."""
    end = time.time() + timeout
    while time.time() < end:
        t = asr._inflight
        if t is None or not t.is_alive():
            return
        time.sleep(0.01)


class WavEncodingTest(unittest.TestCase):
    def test_encodes_mono_16bit_at_the_stream_rate(self):
        blob = _wav_bytes(np.array([0.0, 1.0, -1.0], dtype=np.float32))

        with wave.open(BytesIO(blob)) as w:
            self.assertEqual(w.getnchannels(), 1)
            self.assertEqual(w.getsampwidth(), 2)
            self.assertEqual(w.getframerate(), RATE)
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        self.assertEqual(list(pcm), [0, 32767, -32767])

    def test_clips_instead_of_wrapping_around(self):
        """Without the clip, 1.5 wraps to a large negative sample — audible as
        a click, and it corrupts whatever the model hears."""
        blob = _wav_bytes(np.array([1.5, -1.5], dtype=np.float32))

        with wave.open(BytesIO(blob)) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        self.assertEqual(list(pcm), [32767, -32767])


class PartialTest(unittest.TestCase):
    def test_feed_returns_empty_until_the_first_partial_lands(self):
        asr = OpenAIStreamingASR(transcribe=lambda wav: "hello", partial_every_s=0.1)

        first = asr.feed(_audio(0.2))          # fires the request, returns at once

        self.assertEqual(first, "")
        _settle(asr)
        self.assertEqual(asr.feed(_audio(0.01)), "hello")

    def test_only_one_request_is_in_flight_at_a_time(self):
        calls = []
        gate = threading.Event()

        def slow(wav):
            calls.append(len(wav))
            gate.wait(1.0)
            return "text"

        asr = OpenAIStreamingASR(transcribe=slow, partial_every_s=0.01)
        for _ in range(6):
            asr.feed(_audio(0.1))              # would be 6 requests without single-flight

        self.assertEqual(len(calls), 1)
        gate.set()
        _settle(asr)

    def test_a_failing_partial_is_swallowed_and_does_not_wedge_the_stream(self):
        state = {"n": 0}

        def flaky(wav):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("network blip")
            return "recovered"

        asr = OpenAIStreamingASR(transcribe=flaky, partial_every_s=0.05)
        asr.feed(_audio(0.1))
        _settle(asr)
        asr.feed(_audio(0.1))
        _settle(asr)

        self.assertEqual(asr.feed(_audio(0.01)), "recovered")


class FlushTest(unittest.TestCase):
    def test_flush_transcribes_the_whole_utterance_and_wins(self):
        seen = []

        def fake(wav):
            seen.append(len(wav))
            return f"call{len(seen)}"

        asr = OpenAIStreamingASR(transcribe=fake, partial_every_s=99)
        asr.feed(_audio(1.0))
        asr.feed(_audio(1.0))

        self.assertEqual(asr.flush(), "call1")
        self.assertGreater(seen[-1], 2 * RATE)          # got both seconds

    def test_flush_keeps_the_last_partial_when_the_call_fails(self):
        state = {"n": 0}

        def fake(wav):
            state["n"] += 1
            if state["n"] == 1:
                return "partial text"
            raise RuntimeError("boom")

        asr = OpenAIStreamingASR(transcribe=fake, partial_every_s=0.05)
        asr.feed(_audio(0.2))
        _settle(asr)

        self.assertEqual(asr.flush(), "partial text")

    def test_a_too_short_utterance_is_not_sent(self):
        calls = []
        asr = OpenAIStreamingASR(transcribe=lambda w: calls.append(1) or "x",
                                 partial_every_s=99)
        asr.feed(_audio(0.05))

        self.assertEqual(asr.flush(), "")
        self.assertEqual(calls, [])

    def test_reset_clears_the_buffer_and_the_text(self):
        asr = OpenAIStreamingASR(transcribe=lambda w: "old", partial_every_s=0.05)
        asr.feed(_audio(0.2))
        _settle(asr)

        asr.reset()

        self.assertEqual(asr.feed(_audio(0.01)), "")
        self.assertEqual(asr.flush(), "")


class LanguageTest(unittest.TestCase):
    def test_auto_means_no_language_hint(self):
        self.assertEqual(OpenAIStreamingASR(language="auto", transcribe=lambda w: "").language, "")
        self.assertEqual(OpenAIStreamingASR(language="", transcribe=lambda w: "").language, "")

    def test_an_explicit_language_is_kept(self):
        self.assertEqual(OpenAIStreamingASR(language="hi", transcribe=lambda w: "").language, "hi")


if __name__ == "__main__":
    unittest.main(verbosity=2)
