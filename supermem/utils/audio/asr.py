"""Speech to text: streaming recognition (live partials) + non-streaming accurate transcription (final text).

Two streaming implementations share one interface (``feed(samples) -> accumulated text`` / ``flush()`` / ``reset()``),
selected by the ``asr`` factory in ``utils/defaults.py`` according to ``SUPERMEM_ASR``:

  - ``FunASRStreamingASR``  FunASR paraformer-zh-streaming (**default**, more accurate for Chinese)
  - ``StreamingASR``        sherpa-onnx streaming zipformer (Chinese/English bilingual, pure onnx, no torch)
"""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import threading
import time

import numpy as np

SAMPLE_RATE = 16000

SENSEVOICE_EMOTION_MAP = {
    "NEUTRAL": "neutral",
    "HAPPY": "happy",
    "ANGRY": "angry",
    "SAD": "sad",
    "FEARFUL": "fear",
    "FEAR": "fear",
    "DISGUSTED": "disgust",
    "SURPRISED": "surprised",
}


def pick_device() -> str:
    """Pick the best available device automatically: cuda > mps (Apple M) > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class StreamingASR:
    """sherpa-onnx streaming zipformer, producing live partial text. Enabled when ``SUPERMEM_ASR=sherpa``."""

    def __init__(self, asr_dir: str) -> None:
        import sherpa_onnx          # lazy: sherpa is not loaded when the default FunASR path is used
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=f"{asr_dir}/tokens.txt",
            encoder=f"{asr_dir}/encoder-epoch-99-avg-1.onnx",
            decoder=f"{asr_dir}/decoder-epoch-99-avg-1.onnx",
            joiner=f"{asr_dir}/joiner-epoch-99-avg-1.onnx",
            num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
            decoding_method="greedy_search",
        )
        self.stream = self.rec.create_stream()

    def feed(self, samples):
        self.stream.accept_waveform(SAMPLE_RATE, samples)
        while self.rec.is_ready(self.stream):
            self.rec.decode_stream(self.stream)
        return self.rec.get_result(self.stream)

    def flush(self) -> str:
        """Matches the FunASRStreamingASR interface; sherpa emits everything frame by frame, so there is no tail to flush."""
        return self.rec.get_result(self.stream)

    def reset(self) -> None:
        self.stream = self.rec.create_stream()


# ── Default streaming ASR: FunASR paraformer-zh-streaming ──────────────────────

class FunASRStreamingASR:
    """FunASR ``paraformer-zh-streaming``, producing live partial text (the core default streaming ASR).

    paraformer infers **in chunks** (``chunk_size=[0,10,5]`` -> 600ms per chunk), while the
    frame length fed into ``VoiceStream.feed()`` is up to the caller (the web frontend sends 20ms
    frames), so internally we buffer until 600ms is available before running; ``feed()`` has the
    same semantics as the sherpa version and returns the **accumulated** text
    (``VoiceStream`` does ``self._text = asr.feed(frame)`` as an assignment, so it cannot return a delta).

    ``flush()``: called once by ``VoiceStream`` when VAD decides the speaker is done -- pads the
    partial-chunk tail with zeros and runs ``is_final=True`` once, pulling out the last few words
    still held in the decoder look-ahead. If more audio arrives after a flush (the speaker paused
    mid-sentence and resumed), a new sub-stream is started automatically and keeps accumulating,
    so this turn's text is not lost.
    """

    CHUNK_SIZE = [0, 10, 5]                    # paraformer-streaming standard setting
    STRIDE     = CHUNK_SIZE[1] * 960           # 9600 samples @16k = 600ms
    LOOK_BACK  = dict(encoder_chunk_look_back=4, decoder_chunk_look_back=1)

    #: Location inside the offline bundle. Sits next to the fallback under asr/ -- both are
    #: streaming ASRs; the only difference is default (FunASR, more accurate for Chinese) /
    #: fallback (sherpa, pure onnx, no torch dependency).
    LOCAL_DIR = "funasr-paraformer-zh-streaming"

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        import logging as _logging
        import os as _os

        # The `import funasr` line by itself raises the **root logger** from WARNING to INFO and
        # attaches a handler (measured: WARNING/0 handlers before import, INFO/1 after).
        # The fallout is not just its own spam -- afterwards openai/httpx INFO logs all show up too,
        # and one basic usage can print dozens of "HTTP Request: POST ... 200 OK" lines, burying the
        # real output. Record the state before import and restore it afterwards. SUPERMEM_VERBOSE=1 keeps it as is.
        _quiet = _os.environ.get("SUPERMEM_VERBOSE", "0") == "0"
        _root = _logging.getLogger()
        _lvl, _handlers = _root.level, list(_root.handlers)

        from funasr import AutoModel          # lazy: funasr is only loaded when streaming ASR is actually used

        if _quiet:
            _os.environ.setdefault("TQDM_DISABLE", "1")   # otherwise every transcribed chunk prints an rtf progress bar
            _root.setLevel(_lvl)
            for _h in list(_root.handlers):
                if _h not in _handlers:
                    _root.removeHandler(_h)
        # funasr's AutoModel calls logging.basicConfig(level=log_level), defaulting to
        # INFO -- it sets the **root**, so INFO logs from libraries like openai/httpx all show up
        # too (dozens of "HTTP Request: POST ... 200 OK" lines). Setting levels on individual
        # loggers in supermem/__init__ cannot block this, because it changes root. Pass the argument directly.
        if model is None:
            # Use the local copy if the offline bundle has it; otherwise let funasr download it by
            # model name (848M, which would stall the user's first sentence -- so the offline bundle
            # ships it, and it works out of the box for anyone who pulls it).
            from supermem.utils.common.paths import models_dir
            local = models_dir() / "asr" / self.LOCAL_DIR
            model = str(local) if (local / "config.yaml").exists() else "paraformer-zh-streaming"
        self.model = AutoModel(model=model, device=device or pick_device(),
                               disable_update=True,
                               log_level="ERROR" if _quiet else "INFO")
        self.reset()

    def _run(self, samples, is_final: bool) -> str:
        res = self.model.generate(input=samples, cache=self._cache, is_final=is_final,
                                  chunk_size=self.CHUNK_SIZE, **self.LOOK_BACK)
        if res and res[0].get("text"):
            self._text += res[0]["text"]
        return self._text

    def feed(self, samples) -> str:
        """Feed 16k float32 frames of any length; run one chunk per 600ms buffered. Returns the accumulated text."""
        if self._final:                        # audio after the previous flush -> start a new sub-stream and keep accumulating
            self._cache, self._final = {}, False
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        while len(self._buf) >= self.STRIDE:
            self._run(self._buf[:self.STRIDE], False)
            self._buf = self._buf[self.STRIDE:]
        return self._text

    def flush(self) -> str:
        """Called when VAD decides the speaker is done: zero-pad the tail and run is_final=True so the last few words are not lost. Idempotent."""
        if self._final:
            return self._text
        tail = self._buf
        self._buf = np.zeros(0, dtype=np.float32)
        tail = (np.pad(tail, (0, self.STRIDE - len(tail))) if len(tail)
                else np.zeros(self.STRIDE, dtype=np.float32))
        self._final = True
        return self._run(tail, True)

    def reset(self) -> None:
        self._cache: dict = {}
        self._buf = np.zeros(0, dtype=np.float32)
        self._text = ""
        self._final = False


class Transcriber:
    """SenseVoiceSmall produces the final text (zh/en). More accurate than the
    streaming ASR, so it is used once a turn is locked in.

    ``language`` defaults to ``auto`` (or set ``SUPERMEM_ASR_LANGUAGE``). It used
    to be hardcoded to ``zh``: SenseVoice is multilingual, and pinned to Chinese
    it forces English speech into Chinese characters -- saying "hello how are
    you" came back as a garbled mix of Chinese characters and "lohow areyou". That wrong transcript then goes
    into extraction and retrieval, so everything downstream is garbage.
    SenseVoice accepts auto / zh / en / yue / ja / ko / nospeech."""

    def __init__(self, device: str, language: str = "") -> None:
        self.language = (language
                         or os.environ.get("SUPERMEM_ASR_LANGUAGE", "")
                         or "auto")
        from funasr import AutoModel        # lazy import: funasr is only needed for non-streaming accurate transcription
        from supermem.utils.common.paths import hf_model
        _name = hf_model("emotion", "FunAudioLLM/SenseVoiceSmall", "asr")
        self.model = AutoModel(model=_name, hub="hf",
                               device=device, disable_update=True,
                               trust_remote_code=False)

    def _generate(self, audio) -> str:
        res = self.model.generate(input=audio, cache={}, language=self.language,
                                  use_itn=True, ban_emo_unk=True)
        if not res:
            return ""
        return res[0].get("text", "") or ""

    def run(self, audio) -> str:
        return re.sub(r"<\|[^|]*\|>", "", self._generate(audio)).strip()

    def run_with_emotion(self, audio) -> tuple[str, str]:
        """Get both the text and the acoustic emotion token from a single SenseVoice inference."""
        raw = self._generate(audio)
        tags = re.findall(r"<\|([^|]+)\|>", raw.upper())
        emotion = next((SENSEVOICE_EMOTION_MAP[tag] for tag in tags
                        if tag in SENSEVOICE_EMOTION_MAP), "neutral")
        return re.sub(r"<\|[^|]*\|>", "", raw).strip(), emotion


# ── API streaming ASR: no local model, an OpenAI-compatible transcription endpoint ──

def _wav_bytes(samples, rate: int = SAMPLE_RATE) -> bytes:
    """float32 [-1,1] -> a 16-bit WAV in memory. The transcription endpoint
    wants a file, not raw PCM."""
    import io
    import wave
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class OpenAIStreamingASR:
    """Transcription over the API, with no local model at all. Enabled by
    ``SUPERMEM_ASR=openai``.

    Why not "buffer the whole turn and transcribe once": the end-of-turn check
    in ``stream.py`` is gated on ``self._text.strip()``, so an ASR whose
    ``feed()`` keeps returning "" would never let a turn finish and ``flush()``
    would never be called. And the speculative prefetch eats the partial text --
    without partials the core of this demo (having the memory searched before
    you stop talking) is switched off.

    So partials go over the network too, but **only one request is ever in
    flight** (single-flight): ``feed()`` returns the text it already has and the
    new request runs on a background thread, visible to the next ``feed()``.
    While the speaker is still talking this path never blocks.

    ``flush()`` is the one synchronous wait, and its result is the turn's final
    text.

    The cost, plainly: one request per ``partial_every_s`` seconds (0.9s by
    default, and single-flight means it throttles itself when the network is
    slow), plus one at the end. In exchange it is multilingual with automatic
    detection -- Chinese, English, Hindi -- which is exactly what the two local
    models cannot do.
    """

    def __init__(self, model: str = "", language: str = "",
                 transcribe=None, partial_every_s: float = 0.9) -> None:
        self.model = model or os.environ.get("SUPERMEM_ASR_MODEL", "") or "gpt-4o-mini-transcribe"
        lang = (language or os.environ.get("SUPERMEM_ASR_LANGUAGE", "") or "auto").lower()
        self.language = "" if lang in ("", "auto") else lang
        self.partial_every_s = partial_every_s
        self._transcribe = transcribe or self._api_transcribe
        self._client = None
        self._lock = threading.Lock()
        self.reset()

    # ── the network half ─────────────────────────────────────────────────────
    def _api_transcribe(self, wav: bytes) -> str:
        if self._client is None:
            from openai import OpenAI
            from supermem.llm_config import resolve_api_key, resolve_base_url
            self._client = OpenAI(api_key=resolve_api_key(None),
                                  base_url=resolve_base_url(None))
        kw = {"language": self.language} if self.language else {}
        r = self._client.audio.transcriptions.create(
            model=self.model, file=("turn.wav", wav, "audio/wav"), **kw)
        return (getattr(r, "text", "") or "").strip()

    def _run(self, audio) -> None:
        """Background thread: transcribe the current buffer and update _text on
        success. Never raises -- losing one partial is fine, flush() will
        transcribe the whole thing again anyway."""
        try:
            text = self._transcribe(_wav_bytes(audio))
        except Exception as e:                       # noqa: BLE001
            print(f"[asr] partial transcription failed (ignored): {type(e).__name__}: {e}", flush=True)
            return
        with self._lock:
            if text:
                self._text = text

    # ── streaming interface (same shape as the other two ASRs) ───────────────
    def feed(self, samples) -> str:
        self._buf.append(np.asarray(samples, dtype=np.float32))
        self._n += len(samples)
        if self._inflight is None or not self._inflight.is_alive():
            since = (self._n - self._asked_at) / SAMPLE_RATE
            if since >= self.partial_every_s:
                self._asked_at = self._n
                audio = np.concatenate(self._buf)
                self._inflight = threading.Thread(target=self._run, args=(audio,), daemon=True)
                self._inflight.start()
        with self._lock:
            return self._text

    def flush(self) -> str:
        """The turn's final text. The one synchronous wait."""
        if not self._buf:
            return self._text
        audio = np.concatenate(self._buf)
        if len(audio) < int(0.2 * SAMPLE_RATE):      # too short; transcribing it would only yield noise
            return self._text
        try:
            text = self._transcribe(_wav_bytes(audio))
        except Exception as e:                       # noqa: BLE001
            print(f"[asr] final transcription failed: {type(e).__name__}: {e}", flush=True)
            return self._text
        if text:
            with self._lock:
                self._text = text
        return self._text

    def warmup(self) -> None:
        """Move the cost of the first network call to startup.

        Measured: the first transcription in a process takes ~2.0s against
        ~0.65s for every one after it. That second and a half lands on the
        user's **first sentence** -- the one that decides whether this demo
        feels fast. DNS, TLS and client construction are all in it, and sending
        a moment of silence pays for them.

        Never raises: warmup is an optimisation and must not stop the server
        from starting (no network at boot should still boot). And even if the
        server rejects the silence, the TLS handshake and connection are already
        established, which is the point.
        """
        try:
            self._transcribe(_wav_bytes(np.zeros(int(0.3 * SAMPLE_RATE),
                                                dtype=np.float32)))
        except Exception as e:  # noqa: BLE001
            print(f"[asr] warmup failed (ignored): {type(e).__name__}: {e}", flush=True)
        finally:
            # The warmup result must never stay in _text: otherwise there is
            # already text before the first turn begins, and stream.py reads
            # that as the user having spoken.
            self.reset()

    def reset(self) -> None:
        self._buf: list = []
        self._n = 0
        self._asked_at = 0
        self._inflight = None
        self._text = ""


class ElevenLabsScribeASR(OpenAIStreamingASR):
    """ElevenLabs Scribe over the batch API, same streaming contract as
    ``OpenAIStreamingASR`` (single-flight partials, one synchronous flush).
    Enabled by ``SUPERMEM_ASR=elevenlabs``.

    Why: with language "auto", gpt-4o-mini-transcribe sometimes answers a Hindi
    utterance with an English *translation* ("You tell me, who led ser..."),
    and the reply then follows the English. Scribe transcribes in the language
    that was spoken -- Hindi stays Devanagari, English stays Latin inside a
    Hinglish sentence -- and reports which one it heard (``last_language``).
    """

    URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, model: str = "", language: str = "", transcribe=None,
                 partial_every_s: float = 0.9, api_key: str = "", client=None) -> None:
        model = model or os.environ.get("SUPERMEM_ASR_MODEL_ELEVENLABS", "") or "scribe_v1"
        super().__init__(model=model, language=language, transcribe=transcribe,
                         partial_every_s=partial_every_s)
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self._http = client
        self.last_language = ""
        # Domain words Scribe should expect ("CIBIL" came back as "Sybil" without them).
        self.keyterms = [k.strip() for k in os.environ.get("SUPERMEM_ASR_KEYTERMS", "").split(",") if k.strip()]

    def _api_transcribe(self, wav: bytes) -> str:
        if self._http is None:
            import httpx
            self._http = httpx.Client(timeout=30.0)      # kept alive: partials reuse the TLS connection
        data = {"model_id": self.model, "tag_audio_events": "false"}
        if self.language:
            data["language_code"] = self.language
        if self.keyterms:
            data["keyterms"] = self.keyterms            # a list is sent as repeated form fields
        # Scribe mostly answers in 1-2 s but spiked to 17 s; a bounded wait plus one retry
        # keeps a slow call from holding the whole turn.
        for attempt in (0, 1):
            try:
                r = self._http.post(self.URL, headers={"xi-api-key": self.api_key}, data=data,
                                    files={"file": ("turn.wav", wav, "audio/wav")},
                                    timeout=float(os.environ.get("SUPERMEM_ASR_TIMEOUT_S", "8")))
                break
            except Exception:  # noqa: BLE001 -- httpx timeout / network: retry once, then give up
                if attempt:
                    raise
        if r.status_code != 200:
            raise RuntimeError(f"Scribe HTTP {r.status_code}: {r.text[:200]}")
        j = r.json()
        self.last_language = j.get("language_code") or ""
        return (j.get("text") or "").strip()


class ElevenLabsRealtimeASR:
    """ElevenLabs Scribe v2 realtime over one long-lived WebSocket. Enabled by
    ``SUPERMEM_ASR=elevenlabs_realtime``.

    Why: the batch Scribe path uploads the whole turn again at the end, and that
    final request was the slowest part of a voice turn (flush 1.5-1.7 s, measured).
    Here every frame is streamed as it is captured, partials come back while the
    user talks, and ``flush()`` only sends a manual commit for audio the server
    already holds -- the committed text is back in a few hundred ms.

    Threads: one sends (``_out`` queue), one receives. The socket is reopened
    after a drop; audio sent while it was down is lost, and ``flush()`` then falls
    back to the last partial rather than hang.
    """

    URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    SEND_EVERY_S = 0.1          # batch frames: ~10 messages a second instead of ~30
    IDLE_COMMIT_S = 20.0        # the mic streams between turns too; drop silence the server holds

    def __init__(self, language: str = "", api_key: str = "", keyterms=None,
                 connect=None, commit_timeout_s: float | None = None) -> None:
        lang = (language or os.environ.get("SUPERMEM_ASR_LANGUAGE", "") or "auto").lower()
        self.language = "" if lang in ("", "auto") else lang
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if keyterms is None:
            keyterms = [k.strip() for k in os.environ.get("SUPERMEM_ASR_KEYTERMS", "").split(",")
                        if k.strip()]
        self.keyterms = list(keyterms)
        self.commit_timeout_s = (commit_timeout_s if commit_timeout_s is not None
                                 else float(os.environ.get("SUPERMEM_ASR_COMMIT_TIMEOUT_S", "3")))
        self.last_language = ""
        self._connect = connect or self._ws_connect
        self._cond = threading.Condition()
        self._out: "queue.Queue[str | None]" = queue.Queue()
        self._thread = None
        self._closed = False
        self._ready = threading.Event()   # set once the server said session_started
        self._skip = 0                    # results still owed for turns already reset
        self._waiting = 0                 # commits of this turn not answered yet
        self.reset()

    # ── socket ───────────────────────────────────────────────────────────────
    def url(self) -> str:
        from urllib.parse import urlencode
        q = [("model_id", "scribe_v2_realtime"), ("audio_format", f"pcm_{SAMPLE_RATE}"),
             ("commit_strategy", "manual")]
        if self.language:
            q.append(("language_code", self.language))
        q += [("keyterms", k) for k in self.keyterms]
        return f"{self.URL}?{urlencode(q)}"

    def _ws_connect(self, url, headers):
        from websockets.sync.client import connect
        return connect(url, additional_headers=headers, open_timeout=10, max_size=None)

    def _ensure(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._closed = False
            self._thread = threading.Thread(target=self._run, name="scribe-realtime", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        backoff = 0.2
        while not self._closed:
            try:
                ws = self._connect(self.url(), {"xi-api-key": self.api_key})
            except Exception as e:  # noqa: BLE001
                print(f"[asr] realtime connect failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            opened = time.monotonic()
            reader = threading.Thread(target=self._read, args=(ws,), daemon=True)
            reader.start()
            try:
                while not self._closed and reader.is_alive():
                    try:
                        msg = self._out.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    if msg is None:
                        continue                 # wake-up: re-check closed / reader
                    ws.send(msg)
            except Exception as e:  # noqa: BLE001
                print(f"[asr] realtime send failed: {type(e).__name__}: {e}", flush=True)
            finally:
                try:
                    ws.close()
                except Exception:  # noqa: BLE001
                    pass
                reader.join(timeout=1.0)
                self._ready.clear()
                with self._cond:                 # a new session owes nothing for the old one
                    self._skip = self._waiting = 0
                    self._cond.notify_all()
            # A session the server drops at once (bad key, quota) must not become a hot loop.
            if time.monotonic() - opened < 2.0:
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
            else:
                backoff = 0.2

    def _read(self, ws) -> None:
        try:
            while True:
                self._on_message(json.loads(ws.recv()))
        except Exception as e:  # noqa: BLE001
            if not self._closed:
                print(f"[asr] realtime socket closed: {type(e).__name__}: {e}", flush=True)
        finally:
            self._out.put(None)                  # wake the sender so it reconnects

    def _on_message(self, m: dict) -> None:
        kind = m.get("message_type", "")
        with self._cond:
            if kind == "session_started":
                self._ready.set()
            elif kind == "partial_transcript":
                if not self._skip:               # while a reset turn is still owed, partials are its
                    self._partial = (m.get("text") or "").strip()
            elif kind.startswith("committed_transcript"):
                if kind != "committed_transcript":
                    return                       # the _with_timestamps twin of the same commit
                if self._skip:
                    self._skip -= 1
                    return
                text = (m.get("text") or "").strip()
                if text:
                    self._committed.append(text)
                self._partial = ""
                self._waiting = max(0, self._waiting - 1)
                self._cond.notify_all()
            elif "error" in m or kind.endswith("error"):
                print(f"[asr] realtime {kind}: {m.get('error') or m}", flush=True)
                if self._skip:
                    self._skip -= 1              # the stale commit got this error instead
                elif self._waiting:
                    self._waiting -= 1           # e.g. insufficient_audio_activity for our commit
                    self._cond.notify_all()

    def _send_audio(self, commit: bool = False) -> None:
        pcm = b""
        if self._pending:
            audio = np.concatenate(self._pending)
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            self._pending, self._pending_n = [], 0
        if not pcm and not commit:
            return
        self._out.put(json.dumps({"message_type": "input_audio_chunk",
                                  "audio_base_64": base64.b64encode(pcm).decode(),
                                  "commit": commit, "sample_rate": SAMPLE_RATE}))

    def _text(self) -> str:
        return " ".join(self._committed + ([self._partial] if self._partial else []))

    # ── streaming interface (same shape as the other ASRs) ───────────────────
    def feed(self, samples) -> str:
        self._ensure()
        samples = np.asarray(samples, dtype=np.float32)
        with self._cond:
            self._pending.append(samples)
            self._pending_n += len(samples)
            self._fed += len(samples)
            if self._pending_n >= self.SEND_EVERY_S * SAMPLE_RATE:
                self._send_audio()
            if (self._fed >= self.IDLE_COMMIT_S * SAMPLE_RATE and not self._committed
                    and not self._partial and not self._waiting):
                # Nobody has said anything for a while: commit and discard, so the
                # server never holds minutes of silence for the next turn.
                self._send_audio(commit=True)
                self._skip += 1
                self._fed = 0
            return self._text()

    def flush(self) -> str:
        """Commit what the server holds and wait for its text. The one blocking call."""
        with self._cond:
            if not self._fed:
                return self._text()
            self._send_audio(commit=True)
            self._fed = 0
            self._waiting += 1
            done = self._cond.wait_for(lambda: self._waiting == 0, timeout=self.commit_timeout_s)
            if not done:
                print(f"[asr] realtime commit not answered in {self.commit_timeout_s:.1f}s, "
                      "using the partial", flush=True)
                self._skip += self._waiting
                self._waiting = 0
            return self._text()

    def warmup(self) -> None:
        """Open the socket at startup, so the first turn does not pay for TLS + session."""
        self._ensure()
        if not self._ready.wait(timeout=10):
            print("[asr] realtime warmup: no session_started within 10s (ignored)", flush=True)
        self.reset()

    def reset(self) -> None:
        with self._cond:
            self._skip += self._waiting       # their answers still come; they belong to this turn
            self._waiting = 0
            self._committed: list = []
            self._partial = ""
            self._pending: list = []
            self._pending_n = 0
            self._fed = 0

    def close(self) -> None:
        self._closed = True
        self._out.put(None)
