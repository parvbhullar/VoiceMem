"""supermem core streaming input session: speculative prefetch while listening (EOU 0-300ms).

The third input path alongside "text" and "wav". Two ways to feed it; every chunk returns a ``StreamState``
(this chunk is ``<speak>``/``<silence>``; the chunk where the speaker finishes is ``turn_over`` + speculatively prefetched memory + ``Turn``):

    stream = vm.stream(on_partial=lambda t: print(t))

    # (1) feed audio chunks: supermem's built-in streaming ASR (FunASR paraformer) + silero VAD
    st = await stream.feed(pcm_bytes)          # PCM16 @ src_rate (default 24k)

    # (2) feed partial text from an [external ASR] (FunASR / Whisper / any) -- swapping ASR only changes this line
    st = await stream.feed_partial(text, ended=is_final)

    st.state    # "<speak>" | "<silence>" | "turn_over" (this turn is finished)
    st.memory   # currently speculatively prefetched memory (SearchResult); available while speaking, None if not ready yet
    st.turn     # a Turn only once the turn is finished (otherwise None)

**Stops at the memory result** -- the reply (tts/realtime) is up to the caller once it has the Turn/memory; the core doesn't touch it.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supermem.memory_api import build_memory_context
from supermem.utils.audio.stream_io import resample


@dataclass
class Turn:
    """The memory result already computed by speculative prefetch when a turn ends (or is typed) -- the caller replies with it directly, no second search."""
    text: str
    result: object

    @property
    def memory_context(self) -> str:
        return build_memory_context(self.result)


@dataclass
class StreamState:
    """Returned for every chunk fed (audio or external ASR text): silence/speech for this chunk + current speculative memory + whether the turn is over.

    The perception fields below (``emotion`` / ``speaker_id`` / ``speaker_voiceprint`` /
    ``entity`` / ``schema`` / ``text_embedding``) are all properties **computed only when read**:
    if you don't read them they cost nothing, not a cent or a millisecond, and the 0-300ms speculative prefetch path is completely unaffected.
    """
    state: str                 # "<speak>" | "<silence>" | "turn_over" (turn finished)
    text: str                  # accumulated transcript so far
    memory: object | None      # currently speculatively prefetched memory (SearchResult); None if not ready yet
    turn: Turn | None          # only set once the turn is finished, otherwise None
    _vm: object = None         # capabilities (utils) needed by lazy perception; callers can ignore this
    _pcm: object = None        # this turn's 16k mono audio, for on-demand voiceprint/emotion analysis

    @property
    def memory_context(self) -> str:
        m = self.turn.result if self.turn else self.memory
        return build_memory_context(m) if m is not None else ""

    # ── memory result ───────────────────────────────────────────────────────────

    @property
    def _result(self):
        return self.turn.result if self.turn else self.memory

    @property
    def transcript(self) -> str:
        return self.text

    @property
    def result_leftbrain(self) -> list[str]:
        r = self._result
        return list(r.result_leftbrain) if r is not None else []

    @property
    def result_rightbrain(self) -> list[str]:
        r = self._result
        return list(r.result_rightbrain) if r is not None else []

    @property
    def entity(self) -> list[str]:
        """Named entities in this utterance (already classified during speculative prefetch; taken directly, not recomputed)."""
        r = self._result
        return list(getattr(r.classification, "entities", []) or []) if r is not None else []

    @property
    def schema(self) -> list[str]:
        """Memory slots this utterance was routed to."""
        r = self._result
        return list(getattr(r.classification, "slots", []) or []) if r is not None else []

    # ── acoustic perception (computed only when read) ─────────────────────────────

    @property
    def _perception(self):
        if getattr(self, "_p_cache", None) is None:
            if self._vm is None or self._pcm is None or not len(self._pcm):
                return None
            self._p_cache = _perceive(self._vm, self._pcm, self.text)
        return self._p_cache

    @property
    def emotion(self) -> str:
        p = self._perception
        return getattr(p, "emotion", "") if p else ""

    @property
    def speaker_id(self) -> str:
        p = self._perception
        return (getattr(p, "person_id", None) or "") if p else ""

    @property
    def speaker_voiceprint(self):
        """Voiceprint vector of this turn's speaker (numpy array); None if voiceprint is disabled or can't be computed."""
        if getattr(self, "_vp_cache", None) is None:
            if self._vm is None or self._pcm is None or not len(self._pcm):
                return None
            self._vp_cache = _voiceprint(self._vm, self._pcm)
        return self._vp_cache

    @property
    def text_embedding(self):
        """Text embedding of this transcript; None if there is no text."""
        if getattr(self, "_emb_cache", None) is None:
            if self._vm is None or not self.text.strip():
                return None
            self._emb_cache = _embed(self._vm, self.text)
        return self._emb_cache


# ── implementation of StreamState's perception fields (each runs only when read) ──

def _tmp_wav(pcm) -> str:
    """Write this turn's audio to a temporary wav -- the voiceprint/emotion modules take audio by file path."""
    import tempfile, wave
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def _perceive(vm, pcm, text):
    """Run audio perception once (scene/voiceprint/emotion), reusing the orchestrator's existing preprocess instead of a separate implementation."""
    path = _tmp_wav(pcm)
    try:
        return vm.preprocess(text or "", audio=path)
    except Exception as e:
        print(f"[stream] perception failed: {e}", flush=True)
        return None
    finally:
        import os
        try: os.unlink(path)
        except OSError: pass


def _voiceprint(vm, pcm):
    path = _tmp_wav(pcm)
    try:
        return vm.utils.get("voiceprint").embed(Path(path))
    except Exception as e:
        print(f"[stream] voiceprint extraction failed: {e}", flush=True)
        return None
    finally:
        import os
        try: os.unlink(path)
        except OSError: pass


def _embed(vm, text):
    try:
        emb = vm.utils.get("embedding")
        fn = getattr(emb, "embed_texts", None)
        return fn([text])[0] if fn else emb.embed(text)
    except Exception as e:
        print(f"[stream] text embedding failed: {e}", flush=True)
        return None


#: Keep at most this much audio per turn for on-demand perception (16k mono, 30s ≈ 1.9MB)
_MAX_TURN_SAMPLES = 30 * 16000

#: Keep this much audio before speech onset so the first word isn't clipped (16k mono)
_PREROLL_SAMPLES = int(0.3 * 16000)

#: Sound lasting this long with no transcript still counts as a turn (the play-music-into-the-mic case).
#: Set it long-ish: brief ambient noise (a door closing, a cough) shouldn't become a turn.
MIN_SOUND_ONLY_S = float(os.environ.get("SUPERMEM_SOUND_ONLY_S", "5"))

#: How long a sound-only turn must be silent before it counts as finished.
#:
#: Speech turns use confirm_s (300ms) -- conversation should be that fast. But music isn't conversation: pauses
#: between phrases, weak beats, the gap after an intro easily exceed 300ms, so a song gets chopped into pieces
#: (measured: archived recordings were all 1.5-7.8s), and playback only gives the first few seconds.
#: Playing music doesn't need an instant reply anyway; waiting a bit longer for a complete recording is worth it.
SOUND_ONLY_SILENCE_S = float(os.environ.get("SUPERMEM_SOUND_ONLY_SILENCE_S", "3.0"))

#: Whether this audio chunk has sound (RMS threshold).
#:
#: "End of turn" is normally decided by VAD, but silero VAD detects **human voice** -- music isn't voice,
#: so a whole stretch of music looks like silence to it, the silence counter keeps climbing, and at
#: SOUND_ONLY_SILENCE_S the turn gets cut. Measured: playing a song archived only 3.0s, exactly that threshold.
#: So when not a single word has been transcribed, look at energy instead: if there's still sound it's not
#: silence, and the music is recorded for as long as it plays.
SOUND_LEVEL = float(os.environ.get("SUPERMEM_SOUND_LEVEL", "0.01"))

#: What text to store for a sound-only turn.
#:
#: Not a single word was transcribed in this turn; ingest("") directly extracts no facts, so there's no memory
#: row, and later asking "replay that song from just now" finds nothing. Give it a sentence as a carrier.
#:
#: But it **must be this constant** -- don't let callers each write their own literal: the core uses it to
#: recognise "nobody actually spoke this turn" (see _sound_only in orchestrator); if it isn't recognised, no
#: sound_only tag is set, playback can't tell "the user's speech turn" from "the music turn" when picking
#: candidates, and it plays back the user's own voice.
SOUND_ONLY_TEXT = "The user played a sound for me to listen to."


class VoiceStream:
    """Core streaming input session: speculate while being fed, hand over a Turn when the speaker finishes.

    ``vm.stream(on_partial=None, spec_min_chars=6, gamble_s=0.2, confirm_s=0.3,
    src_rate=24000)``. ``feed`` uses the built-in streaming ASR + silero VAD; ``feed_partial``
    takes text from an external ASR (swapping ASR changes only the line that feeds it). Both share the speculative prefetch logic.

    The two timing parameters go together; don't tune one alone:

        silence 0ms ───────── 200ms ───────── 300ms
                               │               │
                  gamble_s: bet you're done,  confirm_s: VAD confirms you're done,
                  start memory search in bg   memory is ready, speak right away

    The 100ms in between is the window left for search -- if we only started searching after VAD confirmed,
    that time would be added after the user finished, becoming a pause they can hear. A wrong bet (the user
    just paused and kept talking) costs nothing: the next frame with voice cancels the speculation, wasting only one local search.
    """

    def __init__(self, vm, *, on_partial=None, spec_min_chars=6,
                 gamble_s=0.2, confirm_s=0.3, src_rate=24000, vad_threshold=None,
                 emotion=None):
        self.vm = vm
        #: Emotion hint passed to Search during speculative search (the caller may overwrite this attribute at any time).
        #:
        #: **Without it the right brain basically idles.** Almost all right-brain emotional records hang only on the
        #: "emotion" anchor (measured: in one store, 124 of 126 heartnotes had emotion as their anchor), so with no
        #: emotion nothing matches and the right brain is left with the same static profile every turn -- which is
        #: why replies never read like "it remembers this about me". Measured on the same question:
        #:     emotion=None   -> right brain 4 items, all profile
        #:     emotion="sad"  -> 2 emotional records + 1 personality observation + 2 profile
        #:
        #: But **this turn's** emotion can't be read: ``StreamState.emotion`` is a lazy property, and reading it runs
        #: the whole acoustic perception stack synchronously (measured 2.1s), far beyond the 0-300ms prefetch budget.
        #: So the caller should write the ``affect`` returned by the **previous** turn's ingest here -- emotion has
        #: continuity anyway, and it costs 0ms.
        self.emotion = emotion
        self.on_partial = on_partial
        self.spec_min_chars = spec_min_chars
        self.gamble_s = gamble_s
        self.confirm_s = confirm_s
        self.src_rate = src_rate
        self.vad_threshold = vad_threshold
        # ASR/VAD are lazy-loaded: feed_text / feed_partial (external ASR) never touch the audio models.
        self._asr = None
        self._vad = None
        # turn state
        self._text = ""
        self._silence = 0.0
        self._spoke = False
        self._spec = None
        self._spec_text = ""
        self._last_memory = None   # latest computed speculative memory (SearchResult)
        self._pcm = []             # this turn's audio (16k mono), for StreamState's on-demand perception
        self._pcm_len = 0          # samples accumulated; over the cap the oldest are dropped (see _MAX_TURN_S)
        self._preroll = []         # a short stretch before speech onset, prepended to the turn (see _PREROLL_SAMPLES)

    @property
    def asr(self):
        if self._asr is None:
            self._asr = self.vm.utils.get("asr"); self._asr.reset()
        return self._asr

    @property
    def vad(self):
        if self._vad is None:
            if self.vad_threshold is None:
                self._vad = self.vm.utils.get("vad")   # injectable: SuperMem(vad=...) / the vad section of config
            else:
                from supermem.utils.audio.stream_io import make_vad
                self._vad = make_vad(threshold=self.vad_threshold)
        return self._vad

    # ── speculative prefetch (local classifier + local vector Search, 0 LLM/network, in a thread concurrent with mic reads) ──
    async def _speculate(self, text) -> Turn:
        t0 = time.time()

        def work():
            c = self.vm.classify(text)
            return self.vm.search(text, slots=c.slots, entities=c.entities,
                                  emotion=self.emotion or None)

        result = await asyncio.to_thread(work)
        print(f"[speculate] {text[:24]!r} -> {len(result.hits)} hits  "
              f"{(time.time()-t0)*1000:.0f}ms", flush=True)
        return Turn(text, result)

    def _kick(self, text):
        """(Re)start background speculation if the text is long enough and has changed."""
        if text and text != self._spec_text and len(text) >= self.spec_min_chars:
            if self._spec:
                self._spec.cancel()
            self._spec_text = text
            self._spec = asyncio.create_task(self._speculate(text))

    def _ready_memory(self):
        """Get the latest computed speculative memory (SearchResult); if not ready, keep the previous one/None."""
        if self._spec is not None and self._spec.done() and not self._spec.cancelled():
            try:
                self._last_memory = self._spec.result().result
            except Exception:
                pass
        return self._last_memory

    async def _confirm(self) -> Turn:
        try:
            turn = await (self._spec or self._speculate(self._text))
        except asyncio.CancelledError:
            turn = await self._speculate(self._text)
        # flush() may add trailing words not yet decoded during speculation: the final text wins, while memory keeps
        # the prefetched result (only the last few words differ; rerunning Search for them would give back the speculation gain).
        if turn.text != self._text:
            turn = Turn(self._text, turn.result)
        return turn

    def _reset_turn(self):
        if self._asr is not None:
            self._asr.reset()
        self._text, self._silence, self._spoke = "", 0.0, False
        self._spec, self._spec_text, self._last_memory = None, "", None
        self._pcm, self._pcm_len = [], 0
        self._preroll = []

    async def feed_text(self, text) -> Turn:
        """Typed turn: speculate once and return a Turn directly."""
        return await self._speculate(text)

    async def feed_partial(self, text, ended: bool = False) -> StreamState:
        """Take partial text from an [external ASR] (FunASR / Whisper / any streaming ASR).

        Swapping ASR only changes "the line of text fed in"; this method doesn't change at all. New content in text = ``<speak>``
        and (re)starts speculation; ``ended=True`` (external VAD decided the sentence is over) -> hand over a Turn.
        """
        text = (text or "").strip()
        new = bool(text) and text != self._text
        if text:
            self._text = text
        if new and self.on_partial:
            self.on_partial(self._text)
        self._kick(self._text)
        if ended and self._text:
            turn = await self._confirm()
            self._reset_turn()
            return StreamState("turn_over", turn.text, None, turn, self.vm)
        return StreamState("<speak>" if new else "<silence>", self._text,
                           self._ready_memory(), None, self.vm)

    async def feed(self, pcm_bytes) -> StreamState:
        """Feed one PCM16 chunk (``src_rate``, default 24k): built-in streaming ASR + silero VAD + speculation.
        Each chunk returns a ``StreamState`` (``<speak>``/``<silence>`` + current speculative memory + the Turn when finished).
        """
        frame = resample(np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0,
                         src=self.src_rate)
        self._text = self.asr.feed(frame)
        speaking = self.vad.is_speech(frame)

        # Accumulate this turn's audio: StreamState's perception fields use it on demand (voiceprint/emotion), and it's archived for playback.
        #
        # It used to accumulate **every frame**, including the ones where you weren't speaking. So silence between turns
        # kept piling up to the 30s cap, and a "good morning" was stored as 29.9s of audio at RMS 0.005.
        # Given that, the emotion model can only conclude "low energy = sad" -- measured: emotion2vec / SenseVoice /
        # prosody all judged such audio as sad; it looked like the models were inaccurate, but they were fed the wrong thing.
        #
        # Now recording starts only **at speech onset**, with PREROLL seconds kept before it so the start isn't clipped.
        if speaking or self._spoke:
            if not self._spoke and self._preroll:      # just started speaking: prepend the pre-roll
                self._pcm.extend(self._preroll)
                self._pcm_len += sum(len(f) for f in self._preroll)
                self._preroll = []
            self._pcm.append(frame)
            self._pcm_len += len(frame)
            while self._pcm_len > _MAX_TURN_SAMPLES and len(self._pcm) > 1:
                self._pcm_len -= len(self._pcm.pop(0))
        else:
            self._preroll.append(frame)                # not speaking yet: keep only the most recent short stretch
            while sum(len(f) for f in self._preroll) > _PREROLL_SAMPLES and len(self._preroll) > 1:
                self._preroll.pop(0)
        # While not a single word has been transcribed, "has sound" doesn't count as silence -- it's most likely music,
        # and VAD only recognises voice (see SOUND_LEVEL). Once there's a transcript, just follow VAD; don't let ambient
        # noise hold up the end of a sentence.
        audible = speaking or (
            not self._text.strip() and self._spoke
            and float(np.sqrt(np.mean(frame * frame))) >= SOUND_LEVEL)
        if audible:
            if speaking and self._silence > 0 and self._spec:   # barge-in: speaking again -> discard this speculation
                self._spec.cancel(); self._spec, self._spec_text = None, ""
            if speaking:
                self._spoke = True
            self._silence = 0.0
        else:
            self._silence += len(frame) / 16000.0
        if self._text.strip() and self.on_partial:
            self.on_partial(self._text)
        # prefetch while speaking / 200ms bet-you're-done resend
        if self._spoke and self._text.strip() and \
                (self._silence == 0.0 and len(self._text) >= self.spec_min_chars
                 or self._silence >= self.gamble_s):
            self._kick(self._text)
        # A turn needs transcript text -- otherwise every stretch of ambient noise would become a turn.
        # With one exception: **playing music into the mic**. VAD judges music as voice (measured: 357 of 357 chunks),
        # yet ASR transcribes nothing, so it never becomes a turn: no memory, no archived audio, and later asking
        # "replay that song from just now" of course finds nothing. If it plays long enough (>= MIN_SOUND_ONLY_S)
        # hand it over as a turn with empty text, and let the upper layer decide how to remember it.
        sound_only = (not self._text.strip()
                      and self._pcm_len >= MIN_SOUND_ONLY_S * 16000)
        # A turn with not a single transcribed word is most likely music; don't cut it with the conversational 300ms,
        # see SOUND_ONLY_SILENCE_S.
        need_silence = self.confirm_s if self._text.strip() else SOUND_ONLY_SILENCE_S
        if self._spoke and self._silence >= need_silence and (self._text.strip() or sound_only):
            flush = getattr(self._asr, "flush", None)      # chunked ASR (paraformer) pads the partial
            if flush is not None:                          # trailing chunk with zeros and emits it
                self._text = flush() or self._text
            turn = await self._confirm()                   # VAD confirmed finished -> hand over precomputed memory
            pcm = np.concatenate(self._pcm) if self._pcm else None
            self._reset_turn()
            return StreamState("turn_over", turn.text, None, turn, self.vm, pcm)
        return StreamState("<speak>" if speaking else "<silence>", self._text,
                           self._ready_memory(), None, self.vm)
