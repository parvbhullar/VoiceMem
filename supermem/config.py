"""Unified config entry point: one dict configures every local/API model (modelled on mem0's from_config).

Each component is written as ``{"provider": ..., "config": {...}}``, so opening one dict tells you
whether each model runs locally or via an API. ``build_kwargs(config)`` resolves this declarative dict
into injection arguments that the existing ``SuperMem(**kwargs)`` accepts -- it is sugar layered **on top of**
the existing ``SuperMem(embedding=fn, slots=fn, ...)`` injection mechanism and changes no existing behaviour.

A complete config looks like this (every section's config is optional; omitted means built-in default)::

    CONFIG = {
        "api_key": "sk-...",              # top level, passed through to SuperMem (also written to OPENAI_API_KEY)
        "base_url": None,                 # top level, passed through to SuperMem
        "mode": "multi_modal",            # top level, passed through to SuperMem

        "embedding": {"provider": "local"},                 # memory vectors use local E5
        "slots":     {"provider": "local"},                 # slot classification uses local E5 (0 LLM)
        "vad":       {"provider": "silero"},                # decides "done speaking"; custom to swap in your own
        "memory_engine": {"provider": "mem0"},              # vector store backend (default mem0)
        "tts": {"provider": "openai",                       # speech output (optional layer)
                "config": {"model": "gpt-4o-mini-tts", "voice": "coral"}},
        "llm": {"provider": "openai",                       # internal LLM for left/right brain (labelling/attribution...)
                "config": {"model": "gpt-4o-mini", "api_key": "sk-...", "base_url": None}},

        # reply section: the reply model. Two forms are accepted --
        "reply": {"provider": "openai", "config": {"model": "gpt-4o-mini"}},
        # or the demo's nested form (used by web/run.py): the llm section resolves to reply, the tts section
        # resolves to the ninth replaceable slot, and realtime is still read by the web demo itself:
        # "reply": {
        #     "llm":      {"provider": "openai", "config": {"model": "gpt-4o"}},
        #     "tts":      {"provider": "openai", "config": {"model": "gpt-4o-mini-tts"}},
        #     "realtime": {"provider": "openai", "config": {"model": "gpt-realtime"}},
        # },
    }

provider -> built-in implementation mapping (dead simple, readable at a glance):

    embedding.provider     local  -> LocalE5Embedder (local E5, no network)
                           openai -> OpenAILocalEmbedder (OpenAI Embeddings API)
    slots.provider         local  -> LocalQueryClassifier (local E5, 0 LLM)
                           openai -> QuerySlotClassifier (single LLM call)
    vad.provider           silero -> make_vad (built-in; config may give model / threshold)
                           custom -> the object in config.obj (must have is_speech(frame)->bool)
    memory_engine.provider mem0   -> Mem0BackendStore (default; may also be omitted to use the built-in default)
    tts.provider           openai -> OpenAITTS (OpenAI TTS api, configurable voice/instructions)
                           local  -> PiperTTS (offline piper, alias piper)
                           voxcpm -> VoxCPMTTS (offline VoxCPM2)
                           breeze -> BreezeTTS (Breeze TTS 2 streaming service; tone can be
                                     directed in natural language; weights are non-commercial, so not the default)
    models                 model names for the five roles, see supermem/llm_config.py
    llm.provider           openai -> sets models.chat / OPENAI_API_KEY / OPENAI_BASE_URL
    reply.provider         openai -> supermem.reply.openai_reply (built-in, streaming)
                           custom -> the callable in config.fn (equivalent to SuperMem(reply=fn))

An unknown provider raises a clear error.
"""
from __future__ import annotations

import os
from supermem.llm_config import MODELS


def _split(component: dict | None) -> tuple[str, dict]:
    """Split ``{"provider": ..., "config": {...}}`` into (provider, config); config is optional."""
    component = component or {}
    provider = component.get("provider")
    cfg = component.get("config") or {}
    return provider, cfg


def _bad(component: str, provider, known) -> None:
    raise ValueError(
        f"Unknown {component}.provider={provider!r}; options: {' / '.join(known)}"
    )


def _embedding_factory(provider, cfg):
    """embedding: local -> local E5; openai -> OpenAILocalEmbedder;
    any other name is handed to mem0's EmbedderFactory (ollama / huggingface / gemini / ...)."""
    if provider == "local":
        def make():
            from supermem.leftbrain.local_e5_embedder import LocalE5Embedder
            return LocalE5Embedder()
        return make
    if provider == "openai":
        def make():
            from supermem.leftbrain.local_memory_store import (
                OpenAILocalEmbedder, OpenAILocalEmbedderConfig,
            )
            return OpenAILocalEmbedder(OpenAILocalEmbedderConfig(
                model=cfg.get("model"),
                api_key=cfg.get("api_key"),
                base_url=cfg.get("base_url"),
                dimensions=cfg.get("dimensions"),
            ))
        return make

    # Everything else goes to mem0 -- it ships about ten providers (ollama / huggingface / gemini /
    # bedrock / azure_openai / vertexai / together / lmstudio / fastembed /
    # langchain), and mem0 is already a dependency, so no need to write each one again.
    # The two built-ins don't take this path: local is a local E5 that mem0 lacks, and openai here
    # additionally supports parameters like dimensions.
    from supermem.leftbrain.mem0_embedder import mem0_providers
    known = mem0_providers()
    if provider in known:
        def make():
            from supermem.leftbrain.mem0_embedder import Mem0Embedder
            return Mem0Embedder(provider, cfg)
        return make
    _bad("embedding", provider, ["local", "openai", *sorted(known)])


def _slots_factory(provider, cfg):
    """slots: local -> LocalQueryClassifier; openai -> QuerySlotClassifier."""
    if provider == "local":
        def make():
            from supermem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
            # Share one E5 with the local embedder (saves memory), unless the caller explicitly passed a model.
            kw = dict(cfg)
            if "model" not in kw:
                from supermem.leftbrain.local_e5_embedder import shared_e5
                kw["model"] = shared_e5()
            return LocalQueryClassifier(**kw)
        return make
    if provider == "openai":
        def make():
            from supermem.leftbrain.cognitive_graph.query_slot_classifier import QuerySlotClassifier
            return QuerySlotClassifier()
        return make
    _bad("slots", provider, ["local", "openai"])


def _vad_factory(provider, cfg):
    """vad: silero -> built-in make_vad (configurable model/threshold); custom -> the object in config.obj."""
    if provider in (None, "silero"):
        def make():
            from supermem.utils.audio.stream_io import make_vad
            return make_vad(model=cfg.get("model"), threshold=cfg.get("threshold", 0.5))
        return make
    if provider == "custom":
        obj = cfg.get("obj")
        if obj is None or not hasattr(obj, "is_speech"):
            raise ValueError(
                'vad.provider="custom" needs config.obj to be an object with is_speech(frame)->bool; '
                "injecting directly is simpler: SuperMem(vad=lambda: MyVad())"
            )
        return lambda: obj
    _bad("vad", provider, ["silero", "custom"])


def _memory_engine_factory(provider, cfg):
    """memory_engine: mem0 -> Mem0BackendStore (default; may also be omitted to use the built-in default)."""
    if provider == "mem0":
        # None -> let SuperMem use its built-in default memory_engine (i.e. Mem0BackendStore).
        # No need to construct it explicitly here: the built-in default is already mem0, so not overriding is simplest and equivalent.
        return None
    _bad("memory_engine", provider, ["mem0"])


def _tts_factory(provider, cfg):
    """tts: openai -> OpenAI TTS api; local/piper -> offline piper; voxcpm -> VoxCPM2.

    If provider is omitted it follows the TTS_BACKEND env var. The provider name is validated here --
    if the error waited until the first time we speak, we'd already be mid-conversation.
    """
    from supermem.tts import TTS_PROVIDERS
    if provider is not None and str(provider).lower() not in TTS_PROVIDERS:
        _bad("tts", provider, sorted(set(TTS_PROVIDERS)))

    def make():
        from supermem.tts import make_tts
        return make_tts(provider, **cfg)
    return make


# In the demo's nested form of the reply section (CONFIG["reply"] in web/run.py), these three are sub-section
# names, not provider/config. llm resolves to reply, tts to the tts replaceable slot, realtime is read by web itself.
_REPLY_DEMO_KEYS = ("llm", "tts", "realtime")

#: Top-level keys build_kwargs recognises. A misspelled key used to be **silently ignored** -- the config
#: looked set but had no effect at all, far harder to debug than an error (MODELS.update treats role names the same way).
_KNOWN_TOP = {
    "api_key", "base_url", "mode", "memory_root", "user_id", "space", "models",
    "embedding", "slots", "vad", "memory_engine", "llm", "tts", "reply",
    "top_k", "memory_language",
}


def _check_keys(config: dict) -> None:
    unknown = sorted(set(config) - _KNOWN_TOP)
    if unknown:
        raise ValueError(f"Unknown keys in config: {', '.join(unknown)}. "
                         f"Valid keys are: {', '.join(sorted(_KNOWN_TOP))}")
    # The reply section has two shapes: flat {"provider","config"}, or the demo's nested
    # {"llm","tts","realtime"}. In the nested form the core only consumes llm, tts maps to the top-level capability of the same name,
    # and realtime is read by the web demo itself -- ownership is stated here, not "parsed and then ignored".
    seg = config.get("reply")
    if isinstance(seg, dict) and any(k in seg for k in _REPLY_DEMO_KEYS):
        bad = sorted(set(seg) - set(_REPLY_DEMO_KEYS))
        if bad:
            raise ValueError(f"Unknown keys in reply section (nested form): {', '.join(bad)}. "
                             f"Valid keys are: {', '.join(_REPLY_DEMO_KEYS)}")


def _reply_factory(provider, cfg):
    """reply: openai -> built-in streaming provider; custom -> use the function in config.fn directly."""
    if provider in (None, "openai"):
        from supermem.reply import openai_reply
        return openai_reply(model=cfg.get("model"), api_key=cfg.get("api_key"),
                            base_url=cfg.get("base_url"), system=cfg.get("system"))
    if provider == "custom":
        fn = cfg.get("fn")
        if not callable(fn):
            raise ValueError(
                'reply.provider="custom" needs config.fn to be a callable; '
                "passing the function directly is simpler: SuperMem(reply=fn)"
            )
        return fn
    _bad("reply", provider, ["openai", "custom"])


def build_kwargs(config: dict) -> dict:
    """Resolve the unified config dict into an injection-argument dict that ``SuperMem(**kwargs)`` accepts.

    The returned dict only contains items actually given (omitted components get no key; SuperMem uses its built-in defaults):
    ``api_key`` / ``base_url`` / ``mode`` + injection functions such as ``embedding`` / ``schema`` /
    ``memory_engine`` (zero-arg factories, same semantics as ``SuperMem(embedding=lambda: ...)``).

    The ``reply`` section resolves to ``SuperMem(reply=fn)``; in the demo's nested form ``llm`` resolves to
    reply, ``tts`` to ``SuperMem(tts=...)``, and ``realtime`` is still read by the web demo itself.
    """
    config = config or {}
    kwargs: dict = {}
    _check_keys(config)

    # ── top level: api_key / base_url / mode passed straight through ──
    if config.get("api_key") is not None:
        kwargs["api_key"] = config["api_key"]
    if config.get("base_url") is not None:
        kwargs["base_url"] = config["base_url"]
    if config.get("mode") is not None:
        kwargs["mode"] = config["mode"]
    if config.get("memory_root") is not None:
        kwargs["memory_root"] = config["memory_root"]
    if config.get("user_id") is not None:
        kwargs["user_id"] = config["user_id"]
    if config.get("space") is not None:
        kwargs["space"] = config["space"]
    if config.get("top_k") is not None:
        kwargs["top_k"] = config["top_k"]
    if config.get("memory_language") is not None:
        kwargs["memory_language"] = config["memory_language"]

    # ── models: model names for the five roles, each selectable separately (chat / reply / embedding /
    #    tts / realtime, see supermem/llm_config.py). Outranked by the model in the component sections
    #    below -- the component section is more specific and still wins. ──
    if config.get("models"):
        MODELS.update(config["models"])

    # ── embedding: SuperMem's injection key is embedding ──
    if "embedding" in config:
        provider, cfg = _split(config["embedding"])
        kwargs["embedding"] = _embedding_factory(provider, cfg)

    # ── slots: maps to SuperMem's injection key schema (the classifier used by Classify) ──
    if "slots" in config:
        provider, cfg = _split(config["slots"])
        kwargs["slots"] = _slots_factory(provider, cfg)

    # ── vad: the VAD that decides "done speaking" (used by VoiceStream) ──
    if "vad" in config:
        provider, cfg = _split(config["vad"])
        kwargs["vad"] = _vad_factory(provider, cfg)

    # ── memory_engine: mem0 is the built-in default; returning None means no override ──
    if "memory_engine" in config:
        provider, cfg = _split(config["memory_engine"])
        factory = _memory_engine_factory(provider, cfg)
        if factory is not None:
            kwargs["memory_engine"] = factory

    # ── llm: internal LLM for left/right brain. ``llm.model`` is shorthand for ``models.chat`` (same thing;
    #    if both are given, the llm section, parsed later, wins); this section also handles api_key / base_url.
    #    Existing code reads the OPENAI_MODEL / OPENAI_API_KEY /
    #    OPENAI_BASE_URL env vars, so config is written into those env vars here (api_key/base_url
    #    are also passed through as SuperMem arguments, consistent with the top level). ──
    if "llm" in config:
        provider, cfg = _split(config["llm"])
        if provider not in (None, "openai"):
            _bad("llm", provider, ["openai"])
        if cfg.get("model"):
            # Only set on MODELS, no longer also written to env: checked, no third-party library
            # reads OPENAI_MODEL (neither the openai SDK nor mem0), so writing it just adds another copy
            # of global state. api_key / base_url differ -- the SDK and mem0 read those themselves, so they must be written.
            MODELS.update(chat=cfg["model"])
        if cfg.get("api_key"):
            os.environ["OPENAI_API_KEY"] = cfg["api_key"]
            kwargs.setdefault("api_key", cfg["api_key"])
        if cfg.get("base_url"):
            os.environ["OPENAI_BASE_URL"] = cfg["base_url"]
            kwargs.setdefault("base_url", cfg["base_url"])

    # ── tts: the ninth replaceable slot. Both places are accepted -- top-level "tts", or the demo's reply.tts
    #    (web/run.py always wrote it this way; the core used to ignore it, now it reads it). Top level wins. ──
    tts_seg = config.get("tts")
    if tts_seg is None:
        _r = config.get("reply") or {}
        if any(k in _r for k in _REPLY_DEMO_KEYS):
            tts_seg = _r.get("tts")
    if tts_seg is not None:
        provider, cfg = _split(tts_seg)
        kwargs["tts"] = _tts_factory(provider, cfg)

    # ── reply: the reply model. Two forms -- flat {"provider","config"}, or the demo's nested
    #    {"llm","tts","realtime"} (the core only takes llm; tts/realtime are still read by web itself). ──
    if "reply" in config:
        seg = config["reply"] or {}
        if any(k in seg for k in _REPLY_DEMO_KEYS):
            seg = seg.get("llm") or {}
        provider, cfg = _split(seg)
        kwargs["reply"] = _reply_factory(provider, cfg)

    return kwargs
