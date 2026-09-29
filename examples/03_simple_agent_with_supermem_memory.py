"""Minimal voice agent: supermem streaming memory -> OpenAI -> small TTS played in real time, interrupted as soon as the user speaks."""

"""First run
pip install openai sounddevice scipy"""


import asyncio
import contextlib
import io
import os
import queue
import re
import threading

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from openai import AsyncOpenAI
from pywebrtc_audio import AudioProcessor
from supermem import SuperMem


# ── config ────────────────────────────────────────────────────────────
SR = 16000
BLOCK = 160                    # 10 ms: WebRTC AEC native frame size
BARGE_IN_S = 0.5
BARGE_THRESHOLD = 0.30

OPENAI_MODEL = "gpt-4o-mini"

TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
TTS_NATIVE_SR = 24000
TTS_MIN_CHARS = 6              # lower = faster first audio, but more fragmented

PROMPT = """You are a voice assistant. Combining the useful information for this turn from the left brain with the
episodic and personality memories accumulated in the right brain, give the user a warm, valuable answer that is
suitable to be spoken aloud directly. Do not describe or reveal the memories you see.

[Left brain · factual memory]
{left}

[Right brain · episodic memory]
{right}

[User just said]
{text}
"""


# ── tiny thread-safe playback buffer ─────────────────────────────────
class AudioBuffer:
    def __init__(self):
        self.q = queue.Queue()
        self.cur = np.empty(0, dtype=np.float32)

    def put(self, audio):
        if len(audio):
            self.q.put(np.asarray(audio, dtype=np.float32))

    def clear(self):
        self.cur = np.empty(0, dtype=np.float32)
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def pull(self, n):
        out = np.zeros(n, dtype=np.float32)
        pos = 0

        while pos < n:
            if not len(self.cur):
                try:
                    self.cur = self.q.get_nowait()
                except queue.Empty:
                    break

            k = min(n - pos, len(self.cur))
            out[pos:pos + k] = self.cur[:k]
            self.cur = self.cur[k:]
            pos += k

        return out


# ── one duplex device: speaker reference -> AEC -> SuperMem ──────────
class AudioIO:
    def __init__(self, loop):
        self.loop = loop
        self.mic = asyncio.Queue(maxsize=100)
        self.playback = AudioBuffer()

        self.assistant_active = threading.Event()
        self.stop_reply = threading.Event()

        self.spoke_s = 0.0

        self.aec = AudioProcessor(
            sample_rate=SR,
            echo_cancellation=True,
            noise_suppression=True,
            auto_gain_control=False,
            stream_delay_ms=0,
        )

        self.stream = sd.Stream(
            samplerate=SR,
            blocksize=BLOCK,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )

    def _push_mic(self, pcm):
        if not self.mic.full():
            self.mic.put_nowait(pcm)

    def _callback(self, indata, outdata, frames, _time, status):
        far = self.playback.pull(frames)
        outdata[:, 0] = far

        near = indata[:, 0].copy()
        clean = self.aec.process(near, far)

        # Only detect barge-in while assistant is speaking.
        if self.assistant_active.is_set():
            if self.aec.speech_probability >= BARGE_THRESHOLD:
                self.spoke_s += frames / SR
            else:
                self.spoke_s = 0.0

            if self.spoke_s >= BARGE_IN_S:
                self.stop_reply.set()
                self.playback.clear()
        else:
            self.spoke_s = 0.0

        pcm = (
            np.clip(clean, -1.0, 1.0) * 32767
        ).astype(np.int16).tobytes()

        self.loop.call_soon_threadsafe(self._push_mic, pcm)

    def start(self):
        self.stream.start()

    def close(self):
        self.stream.stop()
        self.stream.close()


# ── persistent Kokoro: load ONCE before "[ready]" ────────────────────
class TTS:
    def __init__(self, audio: AudioIO):
        from openai import OpenAI

        self.audio = audio
        self.text_q = queue.Queue()
        self.closed = False
        self.client = OpenAI()

        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def feed(self, text):
        if text:
            self.text_q.put(text)

    def reset(self):
        while True:
            try:
                self.text_q.get_nowait()
            except queue.Empty:
                break
        self.audio.playback.clear()

    def _worker(self):
        while not self.closed:
            text = self.text_q.get()

            if text is None:
                return

            if self.audio.stop_reply.is_set():
                continue

            try:
                # Stream PCM instead of waiting for the whole sentence to synthesise -- play as soon as the first chunk arrives.
                with self.client.audio.speech.with_streaming_response.create(
                    model=TTS_MODEL, voice=TTS_VOICE, input=text,
                    response_format="pcm",          # 24kHz mono PCM16
                ) as resp:
                    tail = b""
                    for chunk in resp.iter_bytes(4096):
                        if self.audio.stop_reply.is_set():
                            break
                        buf = tail + chunk
                        cut = len(buf) & ~1          # PCM16 is two bytes per sample, don't split one in half
                        tail = buf[cut:]
                        if not cut:
                            continue
                        wav = np.frombuffer(buf[:cut], np.int16).astype(np.float32) / 32768.0
                        # AEC and the speaker share the same 16k far-end signal
                        self.audio.playback.put(resample_poly(wav, SR, TTS_NATIVE_SR))
            except Exception as e:
                print(f"\n[tts] Synthesis failed: {type(e).__name__}: {e}", flush=True)


# ── LLM token stream -> terminal + short TTS chunks ─────────────────
def tts_chunks():
    buf = ""

    def push(delta):
        nonlocal buf
        buf += delta

        # punctuation gets spoken immediately;
        # otherwise don't hold more than a few Chinese chars.
        if re.search(r"[\u3002\uff01\uff1f!?\uff1b;\uff0c,\n]$", buf) or len(buf) >= TTS_MIN_CHARS:
            x, buf = buf, ""
            return x
        return None

    def flush():
        nonlocal buf
        x, buf = buf, ""
        return x

    return push, flush


async def answer(client, tts, audio, text, left, right):
    prompt = PROMPT.format(
        left="\n".join(left) or "(none)",
        right="\n".join(right) or "(none)",
        text=text,
    )

    push, flush = tts_chunks()
    full = []

    audio.stop_reply.clear()
    audio.assistant_active.set()

    try:
        stream = await client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            stream=True,
        )

        async for event in stream:
            if audio.stop_reply.is_set():
                break

            if event.type != "response.output_text.delta":
                continue

            delta = event.delta
            full.append(delta)

            print(delta, end="", flush=True)

            chunk = push(delta)
            if chunk:
                tts.feed(chunk)

        tail = flush()
        if tail and not audio.stop_reply.is_set():
            tts.feed(tail)

    finally:
        if audio.stop_reply.is_set():
            tts.reset()
            print("\n[barge-in]", flush=True)

        audio.assistant_active.clear()

    return "".join(full).strip()


# ── terminal formatting ──────────────────────────────────────────────
def show_turn(st):
    print(f"\n\nYou: {st.transcript}")

    print("\n┌─ Left brain ───────────────")
    for x in st.result_leftbrain:
        print(f"│ {x}")
    print("└────────────────────────────")

    print("┌─ Right brain ──────────────")
    for x in st.result_rightbrain:
        print(f"│ {x}")
    print("└────────────────────────────")

    print("\nAssistant: ", end="", flush=True)


# ── main: initialization first; ONLY THEN user starts talking ────────
async def main():
    key = os.environ["OPENAI_API_KEY"]

    loop = asyncio.get_running_loop()
    client = AsyncOpenAI(api_key=key)

    vm = SuperMem(openai_key=key)
    # Local models load lazily; without this the first utterance waits ~25s
    # for E5 / FunASR / silero / perception to load.
    print("[warmup] loading local models…", flush=True)
    await asyncio.to_thread(vm.warmup)
    stream = vm.stream(src_rate=SR, vad_threshold=BARGE_THRESHOLD)

    audio = AudioIO(loop)
    tts = await asyncio.to_thread(TTS, audio)   # TTS fully loaded here

    audio.start()
    print("[ready] All set, start talking.", flush=True)

    try:
        while True:
            pcm = await audio.mic.get()
            st = await stream.feed(pcm)

            # Terminal does NOT print intermediate ASR states.
            if st.state != "turn_over":
                continue

            show_turn(st)

            reply = await answer(
                client, tts, audio,
                st.transcript,
                st.result_leftbrain,
                st.result_rightbrain,
            )

            print()

            if reply:
                # agent_reply is a keyword argument: writing ingest(text, reply) would put reply
                # into audio, where it would be treated as an audio file path.
                vm.ingest(st.transcript, agent_reply=reply)

    finally:
        audio.close()


if __name__ == "__main__":
    asyncio.run(main())
