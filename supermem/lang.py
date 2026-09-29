"""Which language the text stored in memory is written in.

Only one value is supported: ``en``. ``zh`` used to be supported; spaces that
recorded ``zh`` are accepted and silently mapped to ``en``.

The left brain has long had ``use_input_language`` (``extract_facts_openai``), so
extracted facts follow the language the user spoke. The right brain had no
equivalent: the trait-label and emotion-word prompts hard-coded a language, so an
English user's memory ended up with English facts but a profile in another
language -- and that profile is spliced into the reply model's system prompt every
turn, leaking the other language into an English conversation.

**Why not "follow the user's language"**: a memory store should have exactly one
language. Following the input means mixing languages in one store; retrieval is
vector-based, and a question in one language sits far from memories in another,
so mixing means half the memories can't be retrieved. Language is a **property of
the store**, not of a single sentence.

**Two things this switch does not affect**, because they are internal enums, not
human-readable text:

  - slot names (emotion / expression_style / thinking_pattern / coping_style /
    likes_dislikes) -- retrieval, quotas and mind-map clustering all key on them;
    renaming them would break everything downstream.
  - the 8 canonical emotions (anxious/sad/wronged/lonely/conflicted/calm/happy/
    tired) -- ``anchor_router`` has a keyword table that normalises onto them, so
    whatever emotion word the model **outputs** still lands on one of these 8
    internal values after normalisation.
"""

from __future__ import annotations

import os

#: Environment variable name. Can also use SuperMem(memory_language="en") or the demo's --lang.
ENV = "SUPERMEM_MEMORY_LANGUAGE"
SUPPORTED = ("en",)
DEFAULT = "en"

# Legacy values that are still accepted (e.g. recorded in existing space JSON)
# and mapped onto a supported language instead of raising.
_LEGACY = {"zh": "en"}

_override: str | None = None


def _check(value: str) -> str:
    v = (value or "").strip().lower()
    v = _LEGACY.get(v, v)
    if v not in SUPPORTED:
        raise ValueError(f"memory_language must be {' / '.join(SUPPORTED)}, got {value!r}")
    return v


def _set(lang: str) -> None:
    global _override
    _override = lang


def set_memory_language(value: str | None) -> None:
    """Process-wide setting. None / "" clears the override, falling back to env / default."""
    global _override
    _override = _check(value) if (value or "").strip() else None


def memory_language() -> str:
    if _override:
        return _override
    env = (os.environ.get(ENV, "") or "").strip()
    return _check(env) if env else DEFAULT


def resolve_for_space(memory_root, explicit: str | None = None) -> str:
    """Settle this instance's language and persist it on its **space**.

    ``SuperMem(memory_language=...)`` used to only write the process-wide
    override, so: create one instance with a language, then another with no
    argument, and the second inherits the first -- the documented default
    became dependent on construction order (issue #9). The root cause was one
    concept stored in two places: the demo read it from the space json, while
    the library only changed the global.

    Here it is unified onto the space:

        explicitly given -> write it into this space's json and apply it
        not given        -> read this space's own record; if the space has no
                            record, fall back to env / default and write the
                            result back (decided once at creation, never changes)

    Each instance reads its own space, so construction order no longer matters.
    """
    import json as _json
    from supermem.utils.common import space as _space
    try:
        path = _space.json_path(memory_root)
    except Exception:
        # No space directory available (rare pure in-memory usage): fall back to the old global behaviour
        set_memory_language(explicit)
        return memory_language()

    stored = ""
    try:
        if path.exists():
            stored = (_json.loads(path.read_text(encoding="utf-8"))
                      .get("space", {}).get("language", "") or "")
    except Exception:
        stored = ""

    if explicit:
        lang = _check(explicit)
    elif stored:
        lang = _check(stored)
    else:
        env = (os.environ.get(ENV, "") or "").strip()
        lang = _check(env) if env else DEFAULT

    if lang != stored:
        try:
            doc = (_json.loads(path.read_text(encoding="utf-8"))
                   if path.exists() else {})
            doc.setdefault("space", {})["language"] = lang
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        except Exception as e:
            print(f"[lang] failed to write space language (usage unaffected): {e}", flush=True)

    _set(lang)
    return lang


def is_zh() -> bool:
    """Kept for compatibility; only English is supported, so always False."""
    return False


def label_rule() -> str:
    """One-sentence language requirement spliced into prompts. Every free-text value stored in memory should carry it."""
    return ("Write every label in English, whatever language the speaker used. "
            "Do not mix in any other language.")


def display_emotion(canonical: str) -> str:
    """Canonical emotion -> how it is written in the store's language.

    The canonical emotions (see anchor_router._CANONICAL_EMOTIONS) are already
    English keys, so the input is returned unchanged.
    """
    return canonical
