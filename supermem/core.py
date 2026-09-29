"""supermem top-level facade: one class, SuperMem = left brain + right brain + audio perception + a set of swappable capabilities (utils).

    left brain   factual memory: entities + cognitive graph (slot classification/retrieval), backed by a mem0 vector store
    right brain  emotional memory: per-turn valence-arousal, emotion attribution, personality profile
    utils        pluggable capabilities: embedding / schema (classification) / entity / emotion / voiceprint / asr / memory_engine
                 each has a built-in default; pass a function to swap in your own (local model, another vector store...)

    vm = SuperMem(api_key="sk-...", mode="text_mode")
    vm.ingest("Had ramen with Alex at noon")
    vm.search("What did I eat at noon?")           # searches left and right brain together
    vm.left_brain.search(...) / vm.right_brain.search(...)
    SuperMem(embedding=lambda: MyE(), slots=lambda: MyClassifier())    # swap out one capability

mode decides which capabilities get loaded: left_brain_single / text_mode / multi_modal (with audio).

This file is only the entry point "for others to understand the system": SuperMem is a thin
facade whose lowercase user-facing convenience methods (ingest/search/classify/preprocess/flush/test)
each delegate in one line to an internally held Orchestrator instance (self._o). The real
pipeline (Search/Ingest orchestration, helper methods, forwarding to the left brain / right
brain / audio components, SearchResult / Utils) all lives in orchestrator.py.

You can still call the capitalized orchestration methods the old way (vm.Search / vm.Ingest /
vm.Classify / vm.Flush ...) or reach internal forwarding methods: the facade's __getattr__
forwards transparently to self._o.
"""

from __future__ import annotations

from pathlib import Path

from supermem.orchestrator import Orchestrator, SearchResult, Utils
from supermem.llm_config import MODELS

# SearchResult / Utils are re-exported here so that `from supermem.core import ...` and
# `from supermem import SearchResult, Utils` keep working.
__all__ = ["SuperMem", "SearchResult", "Utils"]


class SuperMem:
    """Top-level facade: left brain + right brain + utils (the system at a glance). Implementation in orchestrator.py.

    Input side ``stream()`` -> ``Turn`` (supermem/stream.py), output side ``reply()`` /
    ``reply_stream()`` (supermem/reply.py); ``SuperMem(reply=fn)`` swaps in your own model.

    The lowercase user-facing convenience methods each delegate in one line to the internal
    ``self._o`` (an ``Orchestrator`` instance); ``left_brain`` / ``right_brain`` / ``utils`` point
    directly at the real components. Constructor arguments are passed through to ``Orchestrator``
    unchanged: ``mode`` goes to mode, capability overrides (``embedding`` / ``schema`` /
    ``memory_engine`` etc.) and ``enable_*`` / ``embedder`` / ``vector_store`` / ``classifier``
    all go through ``**kw``, with exactly the same semantics as ``Orchestrator.__init__``.
    """

    #: Public mode aliases -> internal names. The README uses user-facing terms ("normal" = everything,
    #: "leftbrain_only" = factual memory only); the internal names describe which set of utils gets loaded.
    MODE_ALIASES = {
        "normal":         "multi_modal",
        "leftbrain_only": "left_brain_single",
        "text":           "text_mode",
    }

    def __init__(self, api_key=None, mode="text_mode", memory_root=None,
                 user_id="voice_user", base_url=None, reply=None,
                 openai_key=None, top_k=5, space=None, models=None,
                 memory_language=None, **kw):
        # Language of the text stored in memory: "en" (the only supported value). See supermem/lang.py.
        # Slot names and the 8 canonical emotions are internal enums and are not affected.
        # The language is bound to **the space this instance uses**, not process-global -- see resolve_for_space.
        # openai_key is the old name for api_key; equivalent, kept for compatibility. New code should use api_key.
        # Every model can be chosen separately: {"chat": ..., "reply": ..., "embedding": ...,
        # "tts": ..., "realtime": ...}. Omitted roles fall back to env / defaults as before.
        # Note this is a **process-level** setting, not private to this instance: the left/right brain
        # components load lazily and read the global table at access time. If a second SuperMem in the
        # same process passes different models, the first one changes too (MODELS.update logs a line
        # when that happens). For strict isolation, use separate processes.
        if models:
            MODELS.update(models)
        # space: one directory per memory set, located at ./supermem_memoryspace/<space>/.
        # Defaults to "demo". An explicit memory_root is used as-is (evaluation needs a separate store per conversation).
        self._o = Orchestrator(api_key=api_key or openai_key,
                               mode=self.MODE_ALIASES.get(mode, mode),
                               memory_root=memory_root, space=space,
                               user_id=user_id, base_url=base_url, **kw)
        # The reply layer is a facade-level concern (orchestration stops at memory results), so reply is not passed down.
        # None -> falls back to the built-in openai provider on first use, see _reply_fn.
        #
        # Why reply is not a tenth capability slot (tts is one, and both are on the output side): a capability
        # value can be either a factory or a ready-built object, told apart by "is it a function/class" (Utils.get).
        # But **a reply provider is itself a function** -- whatever `SuperMem(reply=my_async_gen_fn)` passes in
        # would be called once as a factory, which is exactly wrong. This ambiguity is unique to the reply layer,
        # so it goes through normalize() on its own and stays out of the capability table.
        self._reply_src = reply
        self._reply_norm = None
        self._top_k = top_k                  # default number of results for search()
        # The space directory is only known now (Orchestrator resolves space/memory_root), so this comes last.
        from supermem.lang import resolve_for_space
        resolve_for_space(self._o._memory_root, memory_language)
        self.mode = self._o.mode
        self.utils = self._o.utils
        self.left_brain = self._o._left      # the real components
        self.right_brain = self._o._right

    @classmethod
    def from_config(cls, config: dict) -> "SuperMem":
        """Declarative construction: one unified config dict configures every local/api model (mem0-style).

        Each component is written as ``{"provider": ..., "config": {...}}``, so one look at the dict tells
        you whether each model runs locally or via api. This is sugar on top of the existing
        ``SuperMem(embedding=fn, slots=fn, ...)`` injection mechanism, which keeps working. The provider
        mapping lives in ``supermem.config``.::

            vm = SuperMem.from_config({
                "mode": "multi_modal",
                "embedding": {"provider": "local"},   # memory vectors use local E5
                "slots":     {"provider": "local"},   # slot classification uses local E5 (0 LLM)
                "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
            })
        """
        from supermem.config import build_kwargs
        return cls(**build_kwargs(config))

    # ── user-facing convenience methods (each delegates in one line to Orchestrator) ──

    def ingest(self, text=None, audio=None, **kw):
        """Remember one utterance. ``ingest("text")`` stores text; ``ingest(audio="x.wav")`` with audio only
        transcribes locally first, then stores (the same audio still goes through voiceprint/scene/emotion
        perception). If both are given, the given text is used."""
        if audio is not None:
            from supermem import sample_audio   # local import: __init__ imports core, a top-level import would be circular
            audio = sample_audio(audio)
        if text is None:
            if audio is None:
                raise ValueError("ingest() needs either text or audio")
            text = self.transcribe(audio)
        # No audio means no voiceprint, so the speaker can only be the account owner -- use the agreed
        # id "user" (see voice_input_to_messages); otherwise it falls through to the defensive label meant
        # for "unverified voiceprint", and every stored fact becomes "Unidentified speaker Speaker 0 is vegetarian".
        # With audio, keep the orchestrator default and let voiceprint recognition decide the speaker
        # (in multi-speaker settings we can't assume it's the owner).
        if audio is None:
            kw.setdefault("speaker", "user")
        return self._o.Ingest(text, audio_path=audio, **kw)

    def transcribe(self, audio) -> str:
        """Transcribe a whole audio file to text (reuses the streaming ASR in utils, no extra model loaded)."""
        from supermem.utils.audio.stream_io import transcribe_file
        return transcribe_file(self.utils.get("asr"), audio)

    def search(self, query, **kw):
        kw.setdefault("top_k", self._top_k)
        return self._o.Search(query, **kw)

    def classify(self, query):                return self._o.Classify(query)
    def preprocess(self, text, audio=None):   return self._o.preprocess(text, audio_path=audio)
    def flush(self):                          return self._o.Flush()

    def warmup(self, *, audio: bool = True, verbose: bool = True) -> None:
        """Load all local models up front so the first call doesn't have to wait.

        Models are lazy-loaded: without warmup the first ``ingest(audio=...)`` takes twenty-odd seconds
        longer (local E5 ~1.7s, FunASR ~6.5s, the perception stack ~16s), and those seconds land exactly
        when the user first speaks -- the worst place to stall. The web demo has always done this; it now
        lives here so every demo and your own scripts can call it in one line.

        ``audio=False`` warms only the text path (pure leftbrain_only needs no ASR / perception).
        ``verbose=False`` prints nothing at all.
        Repeated calls are safe: once models are loaded each step is just a cheap no-op run.
        """
        import sys
        import time

        total = 4 if audio else 1
        # Only draw a progress bar on a real terminal: when redirected to a file/pipe the \r smears into one line
        bar_ok = verbose and sys.stdout.isatty()
        done = [0]

        def draw(label, finished=False):
            if not verbose:
                return
            if not bar_ok:
                if finished:
                    print(f"[warmup] {label}", flush=True)
                return
            width = 24
            filled = int(width * done[0] / total)
            bar = "█" * filled + "░" * (width - filled)
            pct = int(100 * done[0] / total)
            end = "\n" if done[0] >= total else ""
            # 31 = red
            sys.stdout.write(f"\r\033[31m{bar}\033[0m {pct:3d}%  {label:<28}{end}")
            sys.stdout.flush()

        def step(name, fn):
            draw(f"loading {name} …")
            t0 = time.time()
            try:
                fn()
                label = f"{name} {time.time() - t0:.1f}s"
            except Exception as e:                 # optional dependency missing / model not downloaded
                label = f"{name} skipped ({type(e).__name__})"
            done[0] += 1
            draw(label if done[0] < total else "models ready", finished=True)

        step("embedding / slot classifier", lambda: self.classify("hello"))
        if not audio:
            return

        import numpy as np

        def warm_asr():
            asr = self.utils.get("asr")
            asr.feed(np.zeros(9600, dtype=np.float32))     # load the model and actually run one chunk
            asr.reset()

        step("ASR", warm_asr)
        step("VAD", lambda: self.utils.get("vad").is_speech(np.zeros(512, dtype=np.float32)))

        # The perception stack (scene AST / voiceprint 3D-Speaker / emotion SenseVoice) needs a real file.
        # Measured: the first preprocess takes 2120ms, all model loading; afterwards it's steady at 340-410ms.
        def warm_perceive():
            import tempfile
            import soundfile as sf
            from pathlib import Path
            p = Path(tempfile.gettempdir()) / "supermem_warmup.wav"
            sf.write(p, np.zeros(16000, dtype=np.float32), 16000)
            try:
                self.preprocess("warmup", audio=str(p))
            finally:
                p.unlink(missing_ok=True)

        step("perception (scene / voiceprint / emotion)", warm_perceive)

    def stream(self, **kw):
        """Streaming input path: feed audio chunks / text -> get a Turn (memory result) when the speaker finishes. See supermem/stream.py."""
        from supermem.stream import VoiceStream
        return VoiceStream(self, **kw)

    # ── reply layer (output side): two paths, one entry point, see supermem/reply.py ──

    def _reply_fn(self):
        """Lazy normalization: whatever shape SuperMem(reply=fn) receives -> a uniform async generator function.
        If none was given, use the built-in openai provider (client is built on first call, so import works without a key)."""
        if self._reply_norm is None:
            from supermem.reply import normalize, openai_reply
            self._reply_norm = normalize(self._reply_src or openai_reply())
        return self._reply_norm

    def reply_stream(self, turn_or_text, memory_context=""):
        """Streaming reply: ``async for delta in vm.reply_stream(turn)``.

        The first argument can be a ``Turn``/``StreamState`` directly (text and memory_context are unpacked
        automatically), or a piece of text plus your own pre-rendered memory_context.

        The finished reply is automatically registered with the memory layer (``capture`` -> ``remember_reply``),
        so the next ``ingest()`` includes the agent's half without the caller changing a line.
        """
        from supermem.reply import capture, unpack
        text, ctx = unpack(turn_or_text, memory_context)
        return capture(self._reply_fn()(text, ctx),
                       lambda answer: self._o.remember_reply(text, answer))

    async def reply(self, turn_or_text, memory_context=""):
        """Complete reply: ``answer = await vm.reply(turn)``. Internally just joins reply_stream."""
        return "".join([d async for d in self.reply_stream(turn_or_text, memory_context)])

    def test(self):
        """Startup self-check: tests only the utils this mode needs and prints a 4-tier speed table."""
        from supermem.startup_check import run_util_report
        return run_util_report(self.utils)

    def __getattr__(self, name):
        # Old-style capitalized orchestration methods (Search/Ingest/Classify/Flush...) and internal forwarding
        # methods fall through transparently to the orchestrator; __getattr__ only fires when normal attribute
        # lookup fails, so already-set attributes like self._o / utils / left_brain / right_brain never get here.
        return getattr(self.__dict__["_o"], name)
