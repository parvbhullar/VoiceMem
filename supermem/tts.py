"""Text -> speech. The reply layer only produces text (see ``supermem/reply.py``); making sound is this optional layer.

Four built-in backends, all emitting **24kHz mono PCM16**: the default is the OpenAI api; ``local`` / ``piper``
use offline piper; ``voxcpm`` uses VoxCPM2; ``breeze`` connects to the Breeze TTS 2 streaming service
(natural-language control of tone, see ``BreezeTTS``; non-commercial license, so it is not the default).

**This is the ninth swappable slot** (the first eight are in ``supermem/utils/defaults.py``). The contract is a single method::

    class MyTTS:
        async def stream(self, text: str):     # async-yields 24kHz mono PCM16 bytes
            ...

    vm = SuperMem(tts=lambda: MyTTS())                      # style A: injection
    vm = SuperMem.from_config({"tts": {"provider": "voxcpm"}})   # style B: declarative

The core pipeline never touches it -- the memory system stops at text; making sound is the caller's job. So ``tts`` is not in
``_NEED`` (warmup will not start it); whoever wants sound calls ``vm.utils.get("tts")``;
users without piper / voxcpm installed are unaffected.

Backends that support alignment info may yield ``TimedAudioChunk``; timestamps are sample
positions relative to the start of the segment. Existing plain-``bytes`` backends need no changes.

``speak_stream()`` is "synthesize while generating": as soon as a sentence is complete it is sent for synthesis, without waiting for the full text --
if we waited for the full text, the words would be done typing before the audio even started (measured TTS first frame alone is ~1.2s).
"""
from __future__ import annotations

import asyncio
import os

import numpy as np

from supermem.utils.audio.stream_io import resample
from supermem.audio_timing import TimedAudioChunk, TextTimestamp
from supermem.llm_config import resolve_model

#: Kept for the old name. The real resolution happens at OpenAITTS.__init__ time (see llm_config) --
#: this used to read env at import time, so setting env after import silently had no effect.
TTS_MODEL = resolve_model(role="tts")
TTS_BACKEND = os.environ.get("TTS_BACKEND", "openai")   # openai(api) | local | voxcpm
#: Voice. This used to be hard-coded in the synthesis function, so llm_tts users had to edit source to change it --
#: the built-in default implementation should be configurable too, otherwise "swappable" only means replacing the whole backend.
TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "alloy")
#: How to read it (pace, emphasis, pauses). gpt-4o-mini-tts supports instructions; this is where tone is controlled.
#: Leave empty to not pass the parameter -- older models (tts-1) do not accept it and would error.
TTS_INSTRUCTIONS = os.environ.get("OPENAI_TTS_INSTRUCTIONS", "")
#: Only meaningful when "generating candidate voices" (same seed + same text = same speaker, easy to pick from).
#: **It does not guarantee a consistent voice across sentences**; what actually locks the voice is ref_audio, see the BreezeTTS docs.
BREEZE_SEED = int(os.environ.get("SUPERMEM_BREEZE_SEED", "42"))
SAMPLE_RATE = 24000

# How we cut segments for TTS decides how soon the first sound comes out. Measured gpt-4o-mini-tts first-frame latency grows
# with text length: 8 chars 615ms / 25 chars 902ms / 100 chars 1318ms -- so **the first segment should be as short as possible** (sound early),
# later segments can be long (fewer calls, more coherent tone).
_SENT_END  = "\u3002\uff01\uff1f!?…\n"          # sentence end: the normal cut point
_SOFT_END  = "\uff0c,\u3001\uff1b;\uff1a: "          # mid-sentence pause: only used for the first segment, to get the first sound out sooner
#
# The numbers below used to be hard-coded, calibrated against gpt-4o-mini-tts first-frame latency. Switching backends means recalibrating:
# every segment is **synthesized independently**, and intonation does not carry across segments, so more segments means more audible seams --
# it sounds "stitched together sentence by sentence" rather than spoken continuously. Cutting finely trades fluency for first-frame latency;
# which side is worth more depends on how fast the backend is.
_FIRST_MIN = int(os.environ.get("SUPERMEM_TTS_FIRST_MIN", "6"))
_FIRST_MAX = int(os.environ.get("SUPERMEM_TTS_FIRST_MAX", "20"))
_SENT_MIN  = int(os.environ.get("SUPERMEM_TTS_SENT_MIN", "12"))
_SENT_MAX  = int(os.environ.get("SUPERMEM_TTS_SENT_MAX", "60"))
#: Whether the first segment may break at a comma. Allowed by default -- it gets the first sound out sooner. But this **splits one sentence into two
#: independent syntheses**, with the seam landing mid-sentence, which is the worst-sounding kind. When the backend's first frame is fast enough, set 0
#: so every segment ends at a sentence end and nothing breaks inside a sentence.
_FIRST_SOFT = os.environ.get("SUPERMEM_TTS_FIRST_SOFT", "1") != "0"


def cut_point(buf: str, first: bool) -> bool:
    """Is this segment ready to be sent for synthesis."""
    s = buf.strip()
    if not s:
        return False
    if first:                                  # get the first sound out: commas count too, otherwise cut by length
        ends = _SENT_END + _SOFT_END if _FIRST_SOFT else _SENT_END
        return (len(s) >= _FIRST_MIN and s[-1] in ends) or len(s) >= _FIRST_MAX
    return (len(s) >= _SENT_MIN and s[-1] in _SENT_END) or len(s) >= _SENT_MAX


# ── Built-in backends ─────────────────────────────────────────────────────────

class BaseTTS:
    """Common shell for built-in backends: subclasses only write ``_raw()``; sample alignment is done here.

    **Cut on sample boundaries**: the http stream is chunked by network packets; measured 62 of 69 chunks had an odd byte count, while
    a PCM16 sample is 2 bytes -- consumers (``Int16Array``/``np.frombuffer``) error out on an odd length
    and that whole chunk of audio is lost. Here the half sample spanning chunks is carried to the next chunk, so every chunk we emit
    contains whole samples. Your own backend need not inherit this, as long as ``stream()`` emits whole samples.
    """

    async def stream(self, text: str, instruction: str | None = None):
        """``instruction``: how to read **this turn**. The emotion the perception layer detects each turn must be able to reach the voice,
        and emotion changes turn by turn, so putting it on the instance would make it a global constant. Pass None to use the instance default.
        Backends that do not support it (piper / voxcpm) can just ignore it."""
        tail = b""
        async for chunk in self._raw(text, instruction):
            timed = chunk if isinstance(chunk, TimedAudioChunk) else None
            raw = timed.pcm if timed is not None else chunk
            buf = tail + raw
            cut = len(buf) & ~1                     # round down to even
            tail = buf[cut:]
            if cut:
                if timed is None:
                    yield buf[:cut]
                else:
                    yield TimedAudioChunk(
                        pcm=buf[:cut], timestamps=timed.timestamps,
                        sample_rate=timed.sample_rate)
        if tail:
            yield tail + b"\x00"                    # pad the trailing half sample

    def _raw(self, text: str, instruction: str | None = None):
        raise NotImplementedError


class OpenAITTS(BaseTTS):
    """Online api: OpenAI TTS (default gpt-4o-mini-tts); response_format=pcm is 24k PCM16.

    ``base_url`` by default does **not** follow ``SuperMem(base_url=...)``: that usually points to a self-hosted
    LLM / embedding service which most likely has no ``/audio/speech``, and following it would only fail once we try to speak.
    To change the endpoint, set it explicitly here (or ``OPENAI_TTS_BASE_URL``).
    """

    def __init__(self, model=None, voice=None, instructions=None,
                 api_key=None, base_url=None):
        self.model = resolve_model(model, "tts")
        self.voice = voice or TTS_VOICE
        self.instructions = TTS_INSTRUCTIONS if instructions is None else instructions
        self._key = api_key
        self._base = base_url or os.environ.get("OPENAI_TTS_BASE_URL") or None
        self._client = None

    def _cli(self):
        """The client is created on first use, so ``import supermem.tts`` does not require a key."""
        if self._client is None:
            from openai import AsyncOpenAI
            kw = {}
            if self._key:
                kw["api_key"] = self._key
            if self._base:
                kw["base_url"] = self._base
            self._client = AsyncOpenAI(**kw)
        return self._client

    async def _raw(self, text, instruction=None):
        kw = {"model": self.model, "voice": self.voice,
              "input": text, "response_format": "pcm"}
        ins = instruction or self.instructions
        if ins:
            kw["instructions"] = ins
        async with self._cli().audio.speech.with_streaming_response.create(**kw) as resp:
            async for chunk in resp.iter_bytes():
                yield chunk


class PiperTTS(BaseTTS):
    """Small offline model: piper (pure offline onnx, multilingual). Install: pip install piper-tts.

    ``model`` points to the voice .onnx (defaults to ``SUPERMEM_TTS_MODEL``). To use kokoro /
    edge-tts etc., write a class like this one -- the outside only cares about ``stream()``.
    """

    def __init__(self, model=None):
        self.model = resolve_model(model, "tts", default=None)
        self._voice = None

    def _load(self):
        if self._voice is None:
            if not self.model:
                raise ValueError(
                    "The piper backend needs a voice file: set SUPERMEM_TTS_MODEL to a .onnx path, "
                    'or give {"provider": "piper", "config": {"model": ".../x.onnx"}} in the config')
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.model)   # piper api varies by version, check its docs
        return self._voice

    async def _raw(self, text, instruction=None):
        v = self._load()                      # piper has no tone input; instruction is ignored
        sr = getattr(getattr(v, "config", None), "sample_rate", 22050)
        for raw in v.synthesize_stream_raw(text):       # sync generator, int16 bytes @ sr
            f = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
            out = resample(f, src=sr, dst=SAMPLE_RATE)  # normalize to 24k
            yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class VoxCPMTTS(BaseTTS):
    """Large offline model: VoxCPM2 (2B, multilingual). Install: pip install voxcpm.
    ``model`` may point to a local directory; defaults to openbmb/VoxCPM2 on HF (uses the local cache)."""

    def __init__(self, model=None):
        self.model = resolve_model(model, "tts", default=None) or "openbmb/VoxCPM2"
        self._m = None

    def _load(self):
        if self._m is None:
            from voxcpm import VoxCPM
            self._m = VoxCPM.from_pretrained(self.model, load_denoiser=False)
        return self._m

    async def _raw(self, text, instruction=None):
        m = self._load()                      # same as above for voxcpm
        sr = m.tts_model.sample_rate
        for f in m.generate_streaming(text=text):
            out = resample(np.asarray(f, np.float32).reshape(-1), src=sr, dst=SAMPLE_RATE)
            yield (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class BreezeTTS(BaseTTS):
    """Client for the Breeze TTS 2 (breezeblue-ai/breeze-tts) streaming service.

    It **is a service, not an importable library**: on a GPU machine start

        python -m breeze_infer.api <model_path> --host 0.0.0.0 --port 7860

    This side is just an http client, so SuperMem pulls in no heavy dependencies. It needs Linux +
    NVIDIA (about 7.7 GiB VRAM, 12GB recommended) and cannot run on macOS -- ``base_url`` usually points to
    another machine (same architecture as examples/04_all_local_l40s.py).

    **Not the default backend, and it should not be made the default**: the code is Apache 2.0, but the weights are under BreezeBlue's
    research / non-commercial license, and commercial use needs written permission from RESONIA, INC. Making it the default would push that restriction onto
    every SuperMem user.

    The path is the same as OpenAI's, ``/v1/audio/speech``, but it accepts form-data
    (``text`` / ``instruction`` / ``cfg_scale`` / ``ref_audio`` / ``ref_text`` /
    ``seed``), not JSON ``input`` / ``voice`` -- so you cannot just point OpenAITTS at it.

    ``instruction`` is where you direct the tone in natural language ("a bit slower, lower your voice when talking about sad things"),
    which is the reason to pick it: on the OpenAI realtime side you can only slip some text into the persona, and the model often ignores it.
    The body text can also contain vocal events: ``(sigh)``, ``[laugh]``.

    **Three modes, do not mix them up** (the server uses two different templates depending on whether ref_audio is given):

    ===============  ==================================  ==================
    Mode             Parameters                           Voice
    ===============  ==================================  ==================
    Voice Design     instruction                          **changes every time**
    Voice Clone      ref_audio + ref_text                 fixed
    Voice Direction  ref_audio + ref_text + instruction   fixed + tone can be directed
    ===============  ==================================  ==================

    Conversations **must supply ref_audio** (Voice Direction). With only instruction, the speaker is
    generated together with the text, so synthesizing sentence by sentence gives a different person for every sentence -- we hit this in testing.

    You do not need to record the reference audio: generate a few clips with Voice Design, pick one that sounds good and save it as the permanent
    reference. It should be 5~10 seconds, a single speaker, clean; ``ref_text`` must be its verbatim transcript, otherwise the voice drifts.

    Also, the server's ``--fast-all`` is **incompatible** with the ref_audio path: the text encoder's CUDA graphs
    were only captured for the few shapes seen at warmup, input shapes with reference audio are not among them, and it throws
    "text encoder CUDA graph (4, 32) was not declared in the warmup profile".
    Do not pass that flag when starting the service.
    """

    def __init__(self, base_url=None, instruction=None, cfg_scale=None,
                 ref_audio=None, ref_text=None, seed=None, timeout=60.0, model=None):
        self.base_url = (base_url or os.environ.get("SUPERMEM_BREEZE_URL")
                         or "http://127.0.0.1:7860").rstrip("/")
        self.instruction = instruction or os.environ.get("SUPERMEM_BREEZE_INSTRUCTION") or ""
        if cfg_scale is None:
            env_cfg = os.environ.get("SUPERMEM_BREEZE_CFG_SCALE")
            # The official examples always pair instruction with cfg_scale=4 (instruction strength); without it the model barely follows instructions.
            cfg_scale = float(env_cfg) if env_cfg else (4 if self.instruction else None)
        self.cfg_scale = cfg_scale
        # The reference-audio settings also get env var entry points: the web demo's --config **replaces the whole** built-in
        # CONFIG, and writing an entire json just to set ref_audio makes it easy to drop the persona inside it
        # (reply.llm.config.system) -- drop it and you have neither the persona nor the right language. Using env vars
        # avoids touching that CONFIG.
        self.ref_audio = ref_audio or os.environ.get("SUPERMEM_BREEZE_REF_AUDIO") or None
        self.ref_text = ref_text or os.environ.get("SUPERMEM_BREEZE_REF_TEXT") or None
        # seed only decides where voice design starts from randomly; **it cannot lock the voice**: the speaker is
        # generated autoregressively together with the text, so when the text changes the sampling trajectory changes, and the same seed still grows
        # a different voice. And this side synthesizes sentence by sentence (speak_stream sends a request per sentence), so one
        # reply sends several requests -- in testing, every sentence of the same reply came out in a different voice.
        # The only way to fix the voice is ref_audio; see the three modes in the class docs.
        # seed is kept so "generating candidate voices" is reproducible: same seed + same text gives the same speaker,
        # so once you pick one you can save it as the reference.
        self.seed = BREEZE_SEED if seed is None else seed
        self.timeout = timeout
        self._client = None                 # connection reuse, see _cli()
        self._ref_bytes = None              # reference audio is read from disk only once
        # The server is **single-concurrency**: a second request sent while the first is still streaming gets a 409 Conflict right away
        # (it is rejected, not queued). Callers may be concurrent -- the web side starts the task for the next segment early,
        # to save one client->server round trip per segment -- so this constraint is enforced here:
        # tasks are still created early, they just wait behind the lock and go out as soon as the previous segment finishes.
        # The lock is only in this class; backends that really can run concurrently, like OpenAI, are unaffected.
        self._lock = asyncio.Lock()
        # The server fixes the weights when it starts; we accept this only to keep the signature aligned with other backends --
        # web/run.py's CONFIG passes model in regardless of provider.
        self.model = model

    def _cli(self):
        """Reuse a single client. A reply sends requests **sentence by sentence** (speak_stream sends as soon as a sentence is ready);
        opening a new connection per sentence means redoing the TCP handshake every time -- with the service remote and behind an SSH tunnel
        that trip is tens to hundreds of milliseconds, heard directly as stutter. With keep-alive only the first sentence pays it.
        """
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0))
        return self._client

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _raw(self, text, instruction=None):
        try:
            import httpx  # noqa: F401
        except ImportError as e:                # installed along with openai, normally present
            raise ImportError("The Breeze backend needs httpx: pip install httpx") from e

        ins = instruction or self.instruction
        cfg = self.cfg_scale if self.cfg_scale is not None else (4 if ins else None)
        fields = {"text": text}
        if ins:
            fields["instruction"] = ins
        if cfg is not None:
            fields["cfg_scale"] = str(cfg)
        if self.ref_text:
            fields["ref_text"] = self.ref_text
        if self.seed is not None:
            fields["seed"] = str(self.seed)

        # Put everything into files to send multipart: httpx only sends multipart/form-data when files is non-empty;
        # passing only data= becomes urlencoded, while the server's example uses curl -F.
        files = {k: (None, v) for k, v in fields.items()}
        if self.ref_audio:
            if self._ref_bytes is None:      # no need to read from disk for every sentence
                from pathlib import Path
                ref = Path(self.ref_audio)
                self._ref_bytes = (ref.name, ref.read_bytes())
            files["ref_audio"] = self._ref_bytes

        url = f"{self.base_url}/v1/audio/speech"
        async with self._lock:                           # single concurrency, see __init__
            async with self._cli().stream("POST", url, files=files) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():   # already 24k mono PCM16
                    yield chunk


#: provider name -> built-in implementation. ``local`` is the historical alias of ``piper`` (TTS_BACKEND=local always
#: meant this), so both are kept.
TTS_PROVIDERS = {
    "openai": OpenAITTS,
    "local":  PiperTTS,
    "piper":  PiperTTS,
    "voxcpm": VoxCPMTTS,
    "breeze": BreezeTTS,
}


#: Instances reused by (provider, config). piper / voxcpm load a model (VoxCPM2 is 2B);
#: creating a new one each time means reloading it -- and in the demo each Memory Space has its own SuperMem instance,
#: so without sharing, three spaces means three copies of the model. The backend has no state other than the model, so it can be shared.
_INSTANCES: dict = {}


def make_tts(provider: str | None = None, **cfg):
    """Build a built-in backend by provider name; if ``provider`` is omitted, follow the ``TTS_BACKEND`` env var.

    ``cfg`` differs per provider (openai accepts model/voice/instructions/api_key/base_url,
    piper and voxcpm only accept model); passing the wrong one raises TypeError directly -- easier to find than being silently ignored.

    The same (provider, cfg) returns the same instance. If you need independent ones, construct the class directly.
    """
    name = (provider or TTS_BACKEND).lower()
    cls = TTS_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown tts.provider={provider!r}; "
                         f"options: {' / '.join(sorted(set(TTS_PROVIDERS)))}")
    key = (name, str(sorted(cfg.items())))
    if key not in _INSTANCES:
        _INSTANCES[key] = cls(**cfg)
    return _INSTANCES[key]


# ── Module-level entry points (backward compatible) ────────────────────────────

def _tts_cfg(reply):
    seg = (reply or {}).get("tts") or {}
    return seg.get("provider"), (seg.get("config") or {})


async def tts_stream(text, reply=None, instruction=None):
    """Resolve the backend from the ``reply.tts`` config section and synthesize, yielding a 24kHz PCM16 stream.

    An injected TTS goes through ``vm.utils.get("tts").stream(text)``, not through here; this function
    is for callers that only have a reply config and no SuperMem instance.
    """
    provider, cfg = _tts_cfg(reply)
    async for pcm in make_tts(provider, **cfg).stream(text, instruction):
        yield pcm


async def speak_stream(deltas, reply=None, on_delta=None, tts=None,
                       instruction=None):
    """Text delta stream -> audio stream. Synthesis runs **in parallel** with generation: each complete sentence goes into a queue, another
    coroutine takes it out and synthesizes it, so sound comes out while text is still being generated.

    ``deltas``: an async iterator (``vm.reply_stream(turn)`` is one).
    ``on_delta``: called once per text delta received (pass it if you want to type while speaking).
    ``tts``: the object used for synthesis (``vm.utils.get("tts")`` or your own); if not given, one is resolved on the fly from the
    config in ``reply``.
    """
    queue: asyncio.Queue = asyncio.Queue()
    out: asyncio.Queue = asyncio.Queue()

    async def synth():
        while (seg := await queue.get()) is not None:
            gen = (tts.stream(seg, instruction) if tts is not None
                   else tts_stream(seg, reply, instruction))
            async for pcm in gen:
                await out.put(pcm)
        await out.put(None)

    worker = asyncio.create_task(synth())

    async def feed():
        buf, sent = "", 0
        try:
            async for d in deltas:
                if on_delta:
                    on_delta(d)
                buf += d
                if cut_point(buf, first=sent == 0):
                    await queue.put(buf.strip())
                    buf, sent = "", sent + 1
            if buf.strip():
                await queue.put(buf.strip())
        finally:
            await queue.put(None)               # even if generation fails, let synth() finish

    feeder = asyncio.create_task(feed())
    try:
        while (pcm := await out.get()) is not None:
            yield pcm
    finally:
        for t in (feeder, worker):
            if not t.done():
                t.cancel()
        await asyncio.gather(feeder, worker, return_exceptions=True)
