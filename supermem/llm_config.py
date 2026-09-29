"""The single resolution point for models and endpoints.

Originally every component did its own thing: some used ``constructor arg -> env -> default``, some only read env, some
hard-coded the model name; env var names were scattered too (``OPENAI_MODEL`` / ``OPENAI_CHAT_MODEL`` /
``OPENAI_EMBEDDING_MODEL`` / ``OPENAI_TTS_MODEL`` / ``OPENAI_REALTIME_MODEL``),
and two of them were read **at import time** -- set env then import and it works; get the order wrong and it silently fails.
The result: "switch a model" meant editing several places, and missing one left the config only half applied.

Here it is collapsed into five **roles**, each optional and separately configurable:

    role        default                   used by
    ─────────────────────────────────────────────────────────────────────────
    chat        gpt-4o-mini               background memory work: extraction / labelling / slot classification /
                                          summaries / inner monologue / right-brain cleanup
    reply       follows chat              the reply the user actually hears
    embedding   text-embedding-3-small
    tts         gpt-4o-mini-tts           (for local backends this holds the voice file path)
    realtime    gpt-realtime

The role name is the **single vocabulary**; all three entry points speak it, and env var names are derived from it (see
``env_name()``): ``chat`` -> ``SUPERMEM_CHAT_MODEL``, ``reply`` ->
``SUPERMEM_REPLY_MODEL``, and so on.

The old ``OPENAI_MODEL`` / ``OPENAI_CHAT_MODEL`` / ``OPENAI_EMBEDDING_MODEL`` /
``OPENAI_TTS_MODEL`` / ``OPENAI_REALTIME_MODEL`` are still honoured (new names win). They were renamed because
these five variables are **read only by this project** -- neither the openai SDK nor mem0 reads any of them -- yet their
OPENAI_ prefix forced users on DeepSeek / Qwen / vLLM to set a variable unrelated to OpenAI.

``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` are **not renamed**: the openai SDK and mem0 read those
directly (``os.environ.get("OPENAI_API_KEY")`` in ``openai/_client.py``);
if we renamed them they couldn't read them and the config would be "half applied" -- this project supports any
OpenAI-compatible endpoint, just point base_url at it; those two variables then mean "the protocol", not "the vendor".

One precedence chain, the closest one wins -- the layer you write it in depends on the situation; it's not "configure it several times":

    explicit component arg  >  models={...}  >  SUPERMEM_*_MODEL  >  legacy OPENAI_*  >  default

    OpenAIAdditiveExtractorConfig(model="o4-mini")     # a single component, the closest
    SuperMem(models={"chat": "gpt-4.1-mini"})          # process-wide (stored on MODELS)
    export SUPERMEM_CHAT_MODEL=gpt-4.1-mini            # set at deploy time, no code change

Values are computed on every lookup, so import order and when env is set don't matter.

``reply`` follows ``chat`` by default: the reply is the audible path and may be given a stronger model separately, but
when it isn't configured we must not end up half applied, e.g. "OPENAI_MODEL is set but replies still use the default model".

Swapping the **implementation** (a local model, a self-hosted service) is a separate matter handled by SuperMem's injectable slots
(embedder / tts / classifier ...); this module only handles **model names and endpoints**.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

#: A role default of None means "follow chat".
_FOLLOW_CHAT = None
#: Distinguishes "no default passed" from "default=None passed".
_UNSET = object()


def env_name(role: str) -> str:
    """The env var name is **mechanically derived** from the role name: ``chat`` -> ``SUPERMEM_CHAT_MODEL``.

    A hand-written role -> variable table will sooner or later have a typo, and a new role will get forgotten --
    and that error is silent (it reads a variable nobody set and quietly falls back to the default). Deriving it
    makes a mismatch impossible.
    """
    return f"SUPERMEM_{role.upper()}_MODEL"


@dataclass(frozen=True)
class Role:
    default: str | None
    #: The variable name before the rename. This can't be derived (the old names never followed a pattern), so they're listed one by one;
    #: still honoured, but the new name wins.
    legacy: str = ""


ROLES: dict[str, Role] = {
    "chat":      Role("gpt-4o-mini",            legacy="OPENAI_MODEL"),
    "reply":     Role(_FOLLOW_CHAT,             legacy="OPENAI_CHAT_MODEL"),
    "embedding": Role("text-embedding-3-small", legacy="OPENAI_EMBEDDING_MODEL"),
    "tts":       Role("gpt-4o-mini-tts",        legacy="OPENAI_TTS_MODEL"),
    "realtime":  Role("gpt-realtime",           legacy="OPENAI_REALTIME_MODEL"),
}


class Models:
    """Process-wide model table. Lazy -- every get reads fresh; nothing is fixed at construction."""

    def __init__(self) -> None:
        self._over: dict[str, str] = {}

    def update(self, mapping: dict[str, str] | None = None, **kw: str) -> "Models":
        """Set overrides. An unknown role raises immediately -- a misspelled name silently doing nothing is far harder to debug.

        This table is **process-wide**, not one per SuperMem instance. A second instance in the same process
        passing different models changes the first one's too -- ``SuperMem(models=...)`` makes this easy to
        miss, so when it actually happens we print a line instead of letting it pass silently.
        """
        for role, name in {**(mapping or {}), **kw}.items():
            if role not in ROLES:
                raise ValueError(f"Unknown model role {role!r}; options: {', '.join(ROLES)}")
            if not name:
                continue
            name = str(name).strip()
            prev = self._over.get(role)
            if prev and prev != name:
                print(f"[models] {role}: {prev} -> {name}. The model table is process-wide; "
                      f"this override also applies to SuperMem instances already created.", flush=True)
            self._over[role] = name
        return self

    def get(self, role: str = "chat", explicit: str | None = None,
            default: str | None = _UNSET) -> str | None:
        """An explicit ``default`` overrides the role's built-in default -- local TTS backends need this:
        their "model" is a voice file path with no universal default, so passing None lets them raise their own error."""
        if role not in ROLES:
            raise ValueError(f"Unknown model role {role!r}; options: {', '.join(ROLES)}")
        spec = ROLES[role]
        name = (explicit or self._over.get(role)
                or os.environ.get(env_name(role), "").strip()
                or (os.environ.get(spec.legacy, "").strip() if spec.legacy else ""))
        if name:
            return name.strip()
        if default is not _UNSET:
            return default
        return spec.default if spec.default is not _FOLLOW_CHAT else self.get("chat")

    def as_dict(self) -> dict[str, str]:
        """The full table currently in effect, for logging/debugging."""
        return {role: self.get(role) for role in ROLES}

    def explain(self) -> str:
        """One line per role: what is in use now and the env var name. For debugging "which model is actually used"."""
        return "\n".join(f"{role:<10} {self.get(role):<24} {env_name(role)}"
                          for role in ROLES)

    # Convenience properties: MODELS.chat / MODELS.reply / ...
    chat = property(lambda self: self.get("chat"))
    reply = property(lambda self: self.get("reply"))
    embedding = property(lambda self: self.get("embedding"))
    tts = property(lambda self: self.get("tts"))
    realtime = property(lambda self: self.get("realtime"))


#: Process-wide singleton. Both SuperMem(models=...) and from_config({"models": ...}) write to it.
MODELS = Models()


def resolve_model(explicit: str | None = None, role: str = "chat",
                  default: str | None = _UNSET) -> str | None:
    """Explicit arg -> override -> env (new name, then legacy name) -> default."""
    return MODELS.get(role, explicit, default)


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Explicit arg -> ``OPENAI_API_KEY`` -> None (let the caller decide what error to raise).

    Not renamed to SUPERMEM_*: the openai SDK and mem0 read it directly, bypassing this project, so a rename would
    leave half the components with a key and half without. For other vendors' models just point ``OPENAI_BASE_URL`` at them;
    these two variables then mean the **protocol**, not the vendor.
    """
    return explicit or os.environ.get("OPENAI_API_KEY")


def resolve_base_url(explicit: str | None = None) -> str | None:
    """Explicit arg -> ``OPENAI_BASE_URL`` -> None (let the SDK use the official endpoint).

    A ``SUPERMEM_BASE_URL`` alias is deliberately not added: mem0 and the openai SDK bypass this project
    and read ``OPENAI_BASE_URL`` directly, so setting only a new name would point half the components at a self-hosted endpoint
    while the other half still hit real OpenAI -- far worse than an ugly name.
    """
    return explicit or os.environ.get("OPENAI_BASE_URL") or None


#: Legacy name, kept so imports don't break.
CHAT_MODEL_DEFAULT = ROLES["chat"].default
