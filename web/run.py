"""supermem web demo -- conversation core (EOU 0-300ms speculative prefetch). Pipeline lives in utils.py, rendering in index.html.

Highlights: local ASR+VAD computes while listening; as soon as a partial arrives, a speculative Search starts in the background (local E5 vectors + local slot
classification via the injected LocalQueryClassifier, 0 LLM, 0 network). By the time VAD confirms end-of-speech at 300ms the memory is already computed,
leaving the two ~10-line control flows only "send to LLM / send to Realtime". Pausing mid-sentence and then continuing (barge-in) -> cancel the speculation.

Run (see ``--help`` for options; each one can also take its default from the env var of the same name)::

    export OPENAI_API_KEY=sk-...
    python web/run.py                     # default: realtime
    python web/run.py \\
      --mode llm_tts \\                    # use this when you lack Realtime access
      --port 8787 \\
      --spec_min_chars 6 \\
      --gamble_ms 200 \\
      --confirm_ms 300

The default is ``realtime`` (OpenAI native speech): one round trip straight to audio, unlike llm_tts which needs
"LLM emits text (~1.0s) -> accumulate a sentence -> TTS synthesis (~1.2s)", two serial stages, a noticeably worse experience.
If your key has no Realtime access use ``--mode llm_tts``; that path only needs regular chat + TTS,
and TTS can even be swapped for a local offline model (``TTS_BACKEND=local``).

Note: memory vectors use local 384-dim E5 (the speculation budget cannot afford the network). If you kept a memory store from the old demo
(OpenAI 1536-dim), the dimensions are incompatible -- clear the memory directory before running.
"""
import argparse
import asyncio
import base64
import json
import os
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import uvicorn

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent
sys.path.insert(0, str(HERE))                       # so `import utils` finds the pipeline layer in this directory
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("SUPERMEM_MODELS_DIR", str(_ROOT / "models"))
# The memory space is anchored at the **repo root**, not the current directory. Otherwise `cd web && python run.py` would
# create a separate empty supermem_memoryspace/demo under web/, the user talks to an empty store for ages,
# and concludes memory isn't working (this actually happened).
os.environ.setdefault("SUPERMEM_MEMORYSPACE_ROOT", str(_ROOT / "supermem_memoryspace"))


# ══════════════════ CLI arguments (env vars of the same name give defaults; either works) ══════════════════
# Placed before the two heavy imports below: utils / supermem pull in torch + sentence-transformers,
# so if this came after them `--help` would have to wait for the model libs to load. When imported, don't consume sys.argv (pass []).

def _parse(argv):
    p = argparse.ArgumentParser(description="supermem web demo (brain map + 0-300ms speculative prefetch)")
    p.add_argument("--mode", choices=["llm_tts", "realtime"],
                   default=os.environ.get("DEMO_MODE", "realtime"),
                   help="reply control flow: realtime=OpenAI native speech (default, best experience); "
                        "llm_tts=LLM stream -> TTS stream (no Realtime access needed, TTS can be local)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("SUPERMEM_PORT", 8787)))
    p.add_argument("--spec_min_chars", type=int, default=6,
                   help="start speculative prefetch once the partial transcript reaches this many chars")
    p.add_argument("--gamble_ms", type=int, default=200,
                   help="after this much silence, gamble that you're done and speculate once more")
    p.add_argument("--confirm_ms", type=int, default=300,
                   help="after this much silence VAD confirms the turn ended and hands off the Turn")
    p.add_argument("--config", default=os.environ.get("SUPERMEM_CONFIG"),
                   help="a .json file that overrides the CONFIG below wholesale")
    p.add_argument("--space", default=os.environ.get("SUPERMEM_SPACE", "demo"),
                   help="which memory space to use (supermem_memoryspace/<space>/)")
    p.add_argument("--memory_root", default=os.environ.get("SUPERMEM_MEMORY_ROOT", ""),
                   help="memory store directory to use directly; overrides --space when given")
    p.add_argument("--lang", choices=["en"],
                   default="en",
                   help="language for newly created Memory Spaces: en (the only option). "
                        "Existing spaces keep the language they were created with")
    p.add_argument("--log-file", default=os.environ.get("SUPERMEM_LOG_FILE", ""),
                   help="log file path; if omitted, logs go to results/logs/ automatically")
    p.add_argument("--no-file-log", action="store_true",
                   default=os.environ.get("SUPERMEM_FILE_LOG", "1") == "0",
                   help="terminal output only, don't save a log file")
    return p.parse_args(argv)


ARGS = _parse(None if __name__ == "__main__" else [])

# Must come before the heavy-dependency imports: only then do model loading, library warnings, Uvicorn logs and all later
# prints get fully persisted from the very first moment of the process. Don't create a log file when imported by another module.
# Language: the core defaults to English (the only supported language). Set before SuperMem is built,
# because the extraction prompt picks its example set from it.
from supermem.lang import set_memory_language as _set_lang   # noqa: E402
_set_lang(ARGS.lang)

LOG_FILE = None
if __name__ == "__main__" and not ARGS.no_file_log:
    from logging_utils import setup_file_logging
    LOG_FILE = setup_file_logging(_ROOT, ARGS.log_file)

import compare                                       # noqa: E402  A/B comparison
import utils                                         # noqa: E402  pipeline layer in this directory
from audio_timeline import AudioTimeline, SpeechRateEstimator  # noqa: E402
from session_context import SessionBuffer            # noqa: E402
from supermem import SuperMem                        # noqa: E402
from supermem.audio_timing import TimedAudioChunk    # noqa: E402

BARGE_DEBUG = os.environ.get("BARGE_DEBUG", "1") != "0"
BARGE_THRESHOLD = float(os.environ.get("BARGE_THRESHOLD", "0.45"))  # lower = easier to interrupt
#: The transcript must grow by this many chars since last time before it counts as "he really cut in".
#: This used to be "continuous voice >= 280ms", but pure VAD is too loose -- coughs, doors closing, assistant echo
#: that AEC didn't fully cancel all count as voice; the log was full of "continuous voice -> request barge-in / assistant not speaking, ignore" spinning idle.
#: Switched to waiting for ASR to actually emit characters; costs one more emission (~200-300ms) but it never cuts itself off.
#:
#: This number directly decides "how long before a barge-in is heard": saying N chars takes time; measured, 3 chars take ~2 s.
#: It was once raised to 3 because 2 chars could be fooled by echo (an "an" leaked through -- the assistant's "Annie"
#: coming back into the mic). Echo is now blocked by _is_echo() using **the text the assistant is currently saying**, not brute-forced by char count,
#: so it's back to 2: one char less to say, roughly 300-500ms faster.
BARGE_MIN_CHARS = int(os.environ.get("BARGE_MIN_CHARS", "2"))
BARGE_STABLE_UPDATES = int(os.environ.get("BARGE_STABLE_UPDATES", "2"))
BARGE_REJECT_SILENCE_MS = int(os.environ.get("BARGE_REJECT_SILENCE_MS", "220"))
BARGE_CANDIDATE_TIMEOUT_MS = int(os.environ.get("BARGE_CANDIDATE_TIMEOUT_MS", "1200"))
#: The first moments after the assistant starts speaking can't be interrupted -- the mic hears almost only its own voice then.
BARGE_GRACE_MS = int(os.environ.get("BARGE_GRACE_MS", "500"))
#: Which turn/interruption detection the OpenAI side uses. With semantic_vad the model judges "is this a real interruption",
#: so it's insensitive to backchannels ("mm", "right", "oh"); server_vad only checks for sound, so the assistant's own
#: echo and ambient noise can cut it off. If the model or SDK doesn't support it, it comes back as an error event (not raised);
#: if you see that in the log, switch back to TURN_DETECTION=server_vad.
TURN_DETECTION = os.environ.get("TURN_DETECTION", "semantic_vad")
#: semantic_vad's eagerness to jump in: low waits more for you to finish, high jumps in more.
VAD_EAGERNESS = os.environ.get("VAD_EAGERNESS", "low")
MIC_RATE = 24000                       # upstream sample rate from the frontend (SAMPLE_RATE in index.html)
#: Block strangers by voiceprint. On by default -- warmed up at startup and computed on a background thread; measured zero latency impact
#: (memory_hits still arrive 0.63s before EOU, same as with it off).
SPEAKER_GATE = os.environ.get("SUPERMEM_SPEAKER_GATE", "1") != "0"   # why a barge-in didn't fire: check these log lines
#: How many consecutive turns recognized as someone else before judging "stranger". 1 = flip after one turn (that's what the
#: "switch speakers" demo needs); if the voiceprint store is dirty and keeps taking the owner for a new person, 2 blocks most false positives.
STRANGER_MIN_TURNS = int(os.environ.get("STRANGER_MIN_TURNS", "1"))
#: Log a speaker verdict line every turn (by default only when judged a stranger).
SPEAKER_DEBUG = os.environ.get("SPEAKER_DEBUG", "0") != "0"
MODE = ARGS.mode                                     # llm_tts | realtime
SPEC_MIN_CHARS = ARGS.spec_min_chars                 # partial starts speculation
GAMBLE_S  = ARGS.gamble_ms / 1000                    # gamble that you're done
CONFIRM_S = ARGS.confirm_ms / 1000                   # VAD confirms the end

_RT_PERSONA = (
    # State up front "why you exist". The model's default assistant persona is very strong; without clearly
    # giving it a different footing, it falls back to "Hello, how can I help you?".
    "You are the voice assistant this user has been using for a long time; you've known each other a long while. Your value is that **you remember him** -- "
    "everything you say should be something an assistant without memory couldn't say.\n"
    "\n"
    "[Two kinds of memory, used completely differently]\n"
    "factual memory is facts; you can mention them directly, as if you simply remember "
    "(\"How are you doing with the Annie thing?\", not \"According to the records, Annie is transferring schools\").\n"
    "emotion & characteristics are his personality and emotional attributions; they **only** shape your tone, "
    "what you say first, and what to steer clear of -- never say a word of them out loud.\n"
    "\n"
    # Retrieval is ranked by relevance, but ranking high doesn't mean it relates to this sentence. Without saying so the model forces it in,
    # and it sounds like an off-topic answer or an inexplicable dredging up of old history.
    "[Retrieved != relevant]\n"
    "These memories were retrieved and aren't necessarily all related to what he just said. Use the ones that really are; just be aware of the rest. "
    "When none of them are relevant, simply follow what he said; don't force any memory into it.\n"
    "\n"
    # The most valuable rule. Without it the model makes things up: memory only says "taking the GRE next week", and it opens with
    # "Math has always been your strong suit, right?" -- sounds like real memory, but it's a hallucination, worse than not remembering.
    "[Only say what is really in memory]\n"
    "Details that aren't written -- scores, subjects, what he did, who said what, which day -- don't add a single word of them; "
    "if a memory carries no date, don't mention time. Better to say less than to make things up. If you don't know, just say so.\n"
    "\n"
    # The core of feeling like a product: initiative. This section is the biggest divide between "a product" and "a demo".
    "[Be proactive; don't push the work back to him]\n"
    "x \"Anything you want to talk about?\" \"Anything I can help you with?\" \"How was your day?\" -- "
    "anyone without memory could say these; it's like telling him to his face you remember nothing.\n"
    "v Go straight to something specific: \"How's the prep going for that meeting tomorrow?\".\n"
    "When he's vague (\"so much stress lately\", \"so tired today\"), don't offer generic comfort or just ask \"what's wrong?\". "
    "Pick from memory the specific thing most likely to be the cause, name it, and ask if that's it. If you guess wrong he'll correct you.\n"
    "Ask at most one question per turn, and make it specific. If there's nothing to ask, don't; stop when you're done -- "
    "ending every sentence with a question mark is an interrogation, not a chat.\n"
    "\n"
    # Without this section, "ok ok" gets treated as a brand-new conversation and the model greets again.
    "[Follow the conversation]\n"
    "\"ok\", \"sure\", \"mm-hmm\", \"alright\" and the like are wrap-ups or acknowledgements, **not a new topic**. "
    "Just reply briefly; never greet again, restart the topic, or reintroduce yourself.\n"
    "For where the conversation just was, see the \"recent conversation\" section below.\n"
    "\n"
    "[Saying what kind of person he is]\n"
    "Follow every judgement with the thing that made you think so; don't pile up adjectives -- "
    "a memoryless model could just as well say \"you're really driven\".\n"
    "\n"
    "[How to talk]\n"
    "You are **speaking**, not writing. Short sentences; say one or two sentences at a time, then stop. "
    "Don't repeat back what he just said, don't open with \"I remember you said\", don't recite lists, "
    "and don't introduce yourself with \"as your assistant\" -- you've known each other for ages."
)



# Trimmed from 1147 chars to the current length. What was removed and why -- read this before adding anything back; the original is in git:
#
# · Length/structure rules like "say three or four sentences, two or three points"
#     -> demo taste, not a correctness issue. Also, more sentences means more TTS segments and more audible seams.
#       If you really want to control length, the "say one or two sentences at a time, then stop" line is enough.
# · The whole "speaking style: varied intonation, stress important words, rising tone on questions, no announcer voice" block
#     -> these are **performance directions**; putting them in the text prompt means the text model interprets them and then we hope TTS guesses,
#       two layers removed. The TTS backend has an instruction parameter (Breeze has one, gpt-4o-mini-tts too)
#       that takes this directly and works much better. It has moved there, so don't repeat it here.
# · "Don't start every sentence with the same verbal tic"
#     -> it patched another rule that's already gone (originally "start with mm/hey/oh", which the model treated as mandatory
#       for every sentence). The root cause is gone, so the patch needn't stay.
# · "When asked 'what's your impression of me', focus on him as a person, not recent events"
#     -> tailored to one specific demo question. The "back every judgement with its basis" rule above covers most of it.


_STRANGER = ("The person speaking is not the one you know -- the voiceprint doesn't match. You have no memory of them. "
             "Don't tell them anything about the other person, and don't guess who they are. Treat it as a first meeting, "
             "and say, friendly but honestly, that you don't know them yet.")

#: A line appended when not a single memory was retrieved this turn.
#:
#: The persona's "guess the specific thing from a vague sentence" instruction is the most valuable part of this system when there
#: are memories, but when **nothing at all was retrieved** it becomes a licence to fabricate: create an empty Memory Space,
#: first ask "what can't I eat", and it opens with "for your dietary restrictions, watch out for chili and seafood, you
#: mentioned being allergic to them before" -- both invented. An empty store hits this on the very first sentence, which is exactly
#: the moment someone uses this demo for the first time.
#:
#: Difference from _STRANGER: that one says "the person you know is someone else"; this one says "you don't know this".
#: The language chosen in the UI. The assistant follows it -- replies in this language, and extracted memories are written in it too.
#:
#: The memory language must switch along with it, otherwise the store grows mixed-language: the same thing recorded in one language today
#: and another tomorrow, and retrieval only hits half of each.
UI_LANG = ARGS.lang          # UI language. Switchable anytime at the top right; independent of memory/reply language

#: What language the assistant speaks. It used to follow the **UI language toggle** (the frontend stored the choice in localStorage
#: and POSTed it to the backend on every page load), with the result that the backend default never took effect: after switching once
#: for an English demo recording, every later question in another language still got English answers; changing stores or restarting didn't help.
#: It was then changed to follow **the language of the user's sentence** -- answer in whatever language was asked, regardless of UI.
#: Which language to reply in. Follows the **current space**, not the UI, and not the language of the user's current sentence.
#:
#: It used to be "reply in whatever language the user speaks". Fine for monolingual use, but a space has a language:
#: in an English store the user occasionally drops a sentence in another language, the assistant follows, this turn's memory ends up in that language too,
#: and the store gets mixed. The language is fixed when the space is created; this just enforces it.
_LANG_NOTE = {
    "en": "Always reply in English, even if the user writes in another language.",
    "hi": "Always reply in Hindi (Devanagari script), even if the user writes in "
          "another language. Keep it natural spoken Hindi, not literary Hindi; "
          "English technical words are fine where a Hindi speaker would use them.",
}


#: What language this space uses. **Set once when the space is created, never changed afterwards**.
#:
#: Memory and replies both follow it; the UI language (UI_LANG) is a separate matter and can be switched freely.
#: Why it isn't switchable anytime: language is **a property of the store**. Retrieval is vector-based, and a question in one language
#: sits far from memories in another language in vector space; mixing languages in one store means half the memories can't be retrieved, and
#: nothing reports an error. Switching mid-way either mixes up the store or stores everything twice -- neither is acceptable.
SPACE_LANG = "en"


def space_language(name: str) -> str:
    """The language this space uses. Only en is supported, so every space (old ones included) is en."""
    space_dir(name)                  # still validates the name
    return "en"


def _write_space_language(name: str, lang: str) -> None:
    import json as _json
    d, safe = space_dir(name)
    f = d / f"{safe}.json"
    try:
        doc = _json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        doc.setdefault("space", {})["language"] = lang
        f.write_text(_json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[space] failed to write language (harmless): {e}", flush=True)


def set_lang(lang: str) -> None:
    """The selector at the top right: **switches the UI language only**.

    Memory language and reply language follow the **current space** and aren't affected by this -- see SPACE_LANG.
    A UI in one language while the current space is a store in another is allowed: you can browse
    a memory store through a UI in a different language.
    """
    global UI_LANG
    v = str(lang).lower()
    UI_LANG = v if v in _LANG_NOTE else "en"
    print(f"[lang] UI switched to {UI_LANG} (space \"{ACTIVE_SPACE}\" stays {SPACE_LANG})",
          flush=True)


def _lang_note() -> str:
    """What language the assistant **speaks**.

    Fallback chain: UI language -> space language -> en. UI first is deliberate:
    what it speaks is switchable every turn (one click, top right), while what
    gets **stored** is a property of the library, fixed when it was created (see
    SPACE_LANG). That is how Hindi is supported -- replies come out in Hindi
    while memory is still extracted in the space's language (supermem/lang.py
    only SUPPORTS en/zh), so one library never mixes scripts.
    """
    return _LANG_NOTE.get(UI_LANG) or _LANG_NOTE.get(SPACE_LANG) or _LANG_NOTE["en"]


_NO_MEMORY_NOTE = (
    "You retrieved no relevant memory at all this turn. So: **don't mention anything specific** -- "
    "food, places, names, dates, what he did, what he likes -- not a single one, "
    "and certainly don't say \"you mentioned before\" or \"I remember you said\". "
    "Say honestly that you don't know this yet, then ask him, or just talk about what he said itself. "
    "Better to seem forgetful than to make things up -- he'll see through anything invented at a glance, "
    "and it will make him stop trusting the things you really do remember."
)


#: Only replay when the question is about "a sound". Deliberately dumb -- this is a trigger-word list, not an intent classifier:
#: an extra replay just sounds like "it played that bit back for you", and a miss just falls back to clicking ▶ by hand.
_SOUND_WORDS = ("song", "tune", "melody", "music", "hum", "humming", "that sound", "play it back",
                "play it for me", "play me", "replay", "what sound")


def _has_any(text: str, words) -> bool:
    """Whole-word / whole-phrase match, case-insensitive (so "hum" doesn't hit "human")."""
    t = (text or "").lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in words)


def _musical_memory_ids() -> set[str]:
    """Memories whose audio **really contains music**.

    A turn where music recognition hit during ingest gets tagged ``tune:<tune_id>`` (stored in
    memory_tags just like ``scene:café`` / ``speaker:person_x``). With it we don't need
    the user to have happened to say "I heard a song" -- chat casually in a café and that turn's background music still gets recorded.

    Returns an empty set on read failure: this is only a preference ordering; if unavailable it falls back to the old "first one with audio".
    """
    try:
        tunes = vm._o._audio._music_store().list_tunes()
        ids = [f"tune:{t['tune_id']}" for t in tunes]
        if not ids:
            return set()
        store = vm._o._get_repo()._cognitive_store
        return set(store.memory_ids_for_slots_v2(vm._o._user_id, ids))
    except Exception as e:
        print(f"[web] could not read music tags (replay unaffected): {type(e).__name__}: {e}", flush=True)
        return set()


#: Until when a replay is in progress (monotonic clock). Empty-text turns during this time are all dropped.
_REPLAY_UNTIL = 0.0


def _note_replay(memory_id: str) -> None:
    """Note roughly how long this replay will be audible.

    The played-back recording loops from the speaker into the mic and VAD judges it a new turn -- the transcript is empty, neither the left
    nor right brain hits, the model has nothing in hand, so it replies "sorry, I can't replay that directly", and in the log that
    music is even recognized as "heard for the 2nd time". The page already stops upstream audio during replay (see holdMic in
    supermem.html); this is the second line of defence: in case the frontend didn't take effect (another client, a stale cached page),
    the backend itself can still tell this turn is its own playback.
    """
    global _REPLAY_UNTIL
    path = audio_of(memory_id)
    if not path:
        return
    try:
        import soundfile as sf
        dur = float(sf.info(path).duration)
    except Exception:
        dur = 15.0          # if duration can't be read, hold for the upper bound; better to block one turn too many
    _REPLAY_UNTIL = time.monotonic() + dur + 1.0


def _replaying_now() -> bool:
    return time.monotonic() < _REPLAY_UNTIL


#: The music just heard that hasn't made it into the store yet. ``{"path": wav, "at": monotonic clock}``
#:
#: Why keep a separate copy: ingestion runs on a background thread and a turn takes 15-20 s to be written, while "ask right after listening"
#: is exactly the most natural usage -- measured: play music, then immediately ask "replay that song just now"; that memory's
#: created_at is only seconds from the question time, so it isn't in the store yet when queried. Tune recognition, however, is synchronous
#: (Ingest returns immediately with recognized_tune), so this cache is written by the end of the previous turn,
#: exactly covering that gap of a dozen-odd seconds.
_LAST_TUNE: dict = {"path": "", "at": 0.0}

#: Request the recording above with this fake id. Real memory_ids are uuids, so no collision.
LAST_TUNE_ID = "last:tune"

#: How long the cache counts as "just now". Beyond that it's ignored -- a song heard half an hour ago shouldn't be picked up by "that song just now".
LAST_TUNE_TTL_S = float(os.environ.get("SUPERMEM_LAST_TUNE_TTL", "1800"))


def _remember_tune(audio_path: str) -> None:
    _LAST_TUNE.update(path=str(audio_path or ""), at=time.monotonic())
    print(f"  [replay] remembering this music: {audio_path}", flush=True)


def _last_tune_path() -> str:
    p = _LAST_TUNE.get("path") or ""
    if not p or time.monotonic() - float(_LAST_TUNE.get("at") or 0) > LAST_TUNE_TTL_S:
        return ""
    return p if Path(p).exists() else ""


#: Time qualifiers in the question. Memories already carry created_at; a request like "the one I heard at the café last Wednesday afternoon"
#: is a gamble via semantic retrieval -- "last Wednesday", "afternoon" barely affect the vector, whatever it hits is what you get.
#: So time and place are parsed out separately and used as **hard filters**.
_DAY_WORDS = (
    (("three days ago",), -3), (("day before yesterday",), -2), (("yesterday", "last night"), -1),
    (("today", "this morning", "tonight", "this evening"), 0),
)
_HOUR_WORDS = (
    (("early morning", "midnight", "late at night", "middle of the night"), (0, 6)),
    (("morning", "this morning", "first thing"), (6, 10)),
    (("before noon", "forenoon"), (8, 12)),
    (("noon", "midday", "lunchtime"), (11, 14)),
    (("afternoon",), (12, 18)),
    (("evening", "dusk", "sunset"), (17, 20)),
    (("night", "last night", "tonight"), (18, 24)),
)
#: Weekdays. "Wednesday" alone means this week; together with "last week" it means the previous week.
_WEEKDAYS = (("monday",), ("tuesday",),
             ("wednesday",), ("thursday",),
             ("friday",), ("saturday",),
             ("sunday",))
#: Month-name prefixes, for "<month> <day>" dates.
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")

#: Place phrases -> acoustic scene tags (scene_classifier.SceneTag).
#: Scenes are classified automatically for each recorded turn and stored in memory_tags (scene:café).
_PLACE_WORDS = (
    (("cafe", "café", "coffee shop", "starbucks"), "café"),
    (("office", "at work", "workplace", "my desk"), "office"),
    (("at home", "home", "my room", "the house"), "home"),
    (("outside", "outdoors", "on the street", "park", "on the road"), "outdoor"),
    (("in the car", "subway", "metro", "bus", "commute", "train"), "transit"),
    (("meeting", "conference room"), "meeting"),
)


#: "Which one". Candidates are sorted oldest first, so the ordinal is the index -- when several were heard in the same period,
#: "the first one" / "the previous one" are the most natural phrasing, and memory already has this info (created_at).
_ORDINALS = (
    (("first song", "first one", "first track", "first tune", "earliest one", "earliest song"), 1),
    (("second song", "second one", "2nd song"), 2),
    (("third song", "third one", "3rd song"), 3),
    (("fourth song", "fourth one", "4th song"), 4),
    (("fifth song", "fifth one", "5th song"), 5),
)
#: Counting from the end. -1 = the last one, -2 = second to last (i.e. "the previous one").
_ORDINALS_BACK = (
    (("last song", "last one", "last tune", "latest one", "latest song", "most recent one"), -1),
    (("previous song", "previous one", "song before", "one before", "earlier one", "previous tune"), -2),
)


def _ordinal_of(text: str):
    """"Which one" in the question -> 1-based ordinal (negative counts from the end); None if not said."""
    t = text or ""
    for words, n in _ORDINALS + _ORDINALS_BACK:
        if _has_any(t, words):
            return n
    return None


def _place_of(text: str) -> str:
    """Place mentioned in the question -> scene tag; "" if none."""
    t = text or ""
    return next((tag for words, tag in _PLACE_WORDS if _has_any(t, words)), "")


def _time_window(text: str):
    """Question -> (start, end) as two datetimes; None if no time is mentioned.

    Recognizes: today/yesterday/day before yesterday/three days ago, this week/last week/week before last, Monday-Sunday, this month/last month,
    "<month> <day>", "N days ago", plus the periods early morning/morning/before noon/noon/afternoon/evening/night; they can be combined
    ("last week Wednesday afternoon"). If nothing is recognized, return None and let "take the most recent one" handle it -- don't be clever and guess;
    guessing wrong plays some other recording, which is worse than playing nothing.

    "just now" / "a moment ago" deliberately don't count as a time qualifier: they mean "the most recent one", and the strict branch would
    find nothing to match and play nothing.
    """
    import re
    from datetime import datetime, timedelta
    t = (text or "").lower()
    now = datetime.now()
    day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
    span = None                      # (start day, end day), end day exclusive

    m = re.search(r"\b(\d+)\s*days?\s+ago\b", t)
    if m:
        d = day0 - timedelta(days=int(m.group(1)))
        span = (d, d + timedelta(days=1))

    if span is None:
        wd = next((i for i, words in enumerate(_WEEKDAYS) if _has_any(t, words)), None)
        if wd is not None:
            # Based on this week's Monday; "last week" moves back one more week
            monday = day0 - timedelta(days=day0.weekday())
            if "week before last" in t:
                monday -= timedelta(days=14)
            elif "last week" in t:
                monday -= timedelta(days=7)
            d = monday + timedelta(days=wd)
            span = (d, d + timedelta(days=1))

    if span is None:
        m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
        if m:
            mth, dom = _MONTHS.index(m.group(1)) + 1, int(m.group(2))
            year = now.year - 1 if mth > now.month else now.year
            try:
                d = datetime(year, mth, dom)
                span = (d, d + timedelta(days=1))
            except ValueError:
                span = None
    if span is None:
        m = re.search(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", t)
        if m:
            dom = int(m.group(1))
            try:
                d = day0.replace(day=dom)
                if d > day0:                       # not yet reached this month, so it means last month
                    d = (day0.replace(day=1) - timedelta(days=1)).replace(day=dom)
                span = (d, d + timedelta(days=1))
            except ValueError:
                span = None

    if span is None:
        day = next((d for words, d in _DAY_WORDS if _has_any(t, words)), None)
        if day is not None:
            d = day0 + timedelta(days=day)
            span = (d, d + timedelta(days=1))

    if span is None:
        if "week before last" in t:
            monday = day0 - timedelta(days=day0.weekday() + 14)
            span = (monday, monday + timedelta(days=7))
        elif "last week" in t:
            monday = day0 - timedelta(days=day0.weekday() + 7)
            span = (monday, monday + timedelta(days=7))
        elif "this week" in t:
            monday = day0 - timedelta(days=day0.weekday())
            span = (monday, monday + timedelta(days=7))
        elif "last month" in t:
            first = day0.replace(day=1)
            span = ((first - timedelta(days=1)).replace(day=1), first)
        elif "this month" in t:
            first = day0.replace(day=1)
            span = (first, (first + timedelta(days=32)).replace(day=1))

    hours = next((h for words, h in _HOUR_WORDS if _has_any(t, words)), None)
    if span is None and hours is None:
        return None
    if span is None:                                  # only a period of day was given; default to today
        span = (day0, day0 + timedelta(days=1))
    if hours is None:
        return span
    # Only refine by period when the span is a single day -- "afternoon last week" is meaningless
    if (span[1] - span[0]).days > 1:
        return span
    return span[0] + timedelta(hours=hours[0]), span[0] + timedelta(hours=hours[1])


def _tune_memories() -> list[dict]:
    """All candidates that can be played: tagged tune:, or a sound-only turn, and the original audio still exists.

    Why sound_only counts too: when music is played straight into the mic and acoustics didn't recognize it (loudspeaker, noisy
    surroundings, short clips all cause misses), that turn has neither a tune tag nor speech; all we know is "there's a recording, nobody
    talking". That is precisely the most likely one the user is looking for. Ordinary speech turns don't get mixed in.

    One item = {"id", "at": datetime, "scenes": {...}, "text"}, newest first.
    This is the replay candidate pool -- filtering by time and by place both happen in this pool, independent of whatever this turn's
    retrieval happened to hit.
    """
    from datetime import datetime
    try:
        import sqlite3
        from supermem.utils.common import space as _space
        c = sqlite3.connect(_space.db(vm._o._memory_root))
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """SELECT m.id, m.content, m.created_at,
                      (SELECT group_concat(s.slot) FROM memory_tags s
                        WHERE s.memory_id = m.id AND s.slot LIKE 'scene:%') scenes,
                      (SELECT u.slot FROM memory_tags u
                        WHERE u.memory_id = m.id AND u.slot LIKE 'tune:%' LIMIT 1) tune,
                      EXISTS (SELECT 1 FROM memory_tags o
                               WHERE o.memory_id = m.id AND o.slot = 'sound_only') sound_only
                 FROM memories m
                 WHERE EXISTS (SELECT 1 FROM memory_tags t
                                WHERE t.memory_id = m.id
                                  AND (t.slot LIKE 'tune:%' OR t.slot = 'sound_only'))
                 ORDER BY m.created_at DESC LIMIT 300""").fetchall()
        c.close()
    except Exception as e:
        print(f"[replay] failed to list music memories: {type(e).__name__}: {e}", flush=True)
        return []

    out = []
    for r in rows:
        if not audio_of(r["id"]):
            continue
        try:
            at = datetime.fromisoformat(r["created_at"]).astimezone().replace(tzinfo=None)
        except Exception:
            continue
        out.append({"id": r["id"], "at": at, "text": r["content"] or "",
                    "sound_only": bool(r["sound_only"]),
                    "tune": (r["tune"] or "").split(":", 1)[-1],
                    "scenes": {x.split(":", 1)[1] for x in (r["scenes"] or "").split(",") if ":" in x}})
    return out


def _archived_memory_ids() -> list[str]:
    """All memory ids with archived original audio, newest first.

    Fallback when looking for a recording by time: turns where music recognition missed have no tune: tag, but their audio
    was archived all the same. If we only trusted tags, the user just played a song and it would reply "not saved".
    """
    try:
        import sqlite3
        from supermem.utils.common import space as _space
        c = sqlite3.connect(_space.db(vm._o._memory_root))
        rows = c.execute("SELECT id FROM memories ORDER BY created_at DESC LIMIT 200").fetchall()
        c.close()
        return [r[0] for r in rows]
    except Exception as e:
        print(f"[web] could not list archived memories (replay unaffected): {type(e).__name__}: {e}", flush=True)
        return []


def _created_at(mid: str):
    """When this memory was written. Returns None if not found.

    The vector store only keeps date (day granularity); filtering by "afternoon" needs created_at from sqlite.

    The path must be the _memory_root resolved by vm, **not ARGS.space** -- ``space.db()`` takes
    a directory path; given a bare name ("musictest") it's treated as a relative path, and when not found a new empty
    store is created, so no memory's time can be found and time-based replay silently fails.
    """
    from datetime import datetime
    try:
        import sqlite3
        from supermem.utils.common import space as _space
        c = sqlite3.connect(_space.db(vm._o._memory_root))
        row = c.execute("SELECT created_at FROM memories WHERE id=?", (mid,)).fetchone()
        c.close()
        if not row or not row[0]:
            return None
        return datetime.fromisoformat(row[0]).astimezone().replace(tzinfo=None)
    except Exception:
        return None


#: Max gap between two clips of the same song for them to still count as "one continuous song". VAD cuts at phrase pauses
#: and the next clip starts recording again; that's where the gap comes from.
TUNE_GAP_S = float(os.environ.get("SUPERMEM_TUNE_GAP_S", "90"))


#: A group of clips is requested with a fake id carrying this prefix, followed by comma-separated memory_ids.
#: It still goes through the existing /api/audio/{memory_id}; the frontend needs no change at all.
GROUP_ID_PREFIX = "group:"

#: Stitched full songs live here. Each group is stitched only once, then hit directly.
_STITCH_CACHE: dict = {}


def _stitch(memory_ids: list) -> str:
    """Stitch several recordings into one wav in order and return its path; if stitching fails return the first clip.

    If sample rates differ, resample to the first clip's rate -- archives are all 16k mono, and even if a mismatch occurs
    it shouldn't make the whole replay fail.
    """
    paths = [q for q in (audio_of(m) for m in memory_ids) if q]
    if not paths:
        return ""
    if len(paths) == 1:
        return paths[0]
    key = "|".join(paths)
    if key in _STITCH_CACHE and Path(_STITCH_CACHE[key]).exists():
        return _STITCH_CACHE[key]
    try:
        import numpy as np
        import soundfile as sf
        from supermem.utils.audio.stream_io import resample as _resample
        chunks, sr0 = [], None
        for q in paths:
            x, sr = sf.read(q, dtype="float32", always_2d=False)
            if getattr(x, "ndim", 1) > 1:
                x = x.mean(axis=1)
            if sr0 is None:
                sr0 = sr
            elif sr != sr0:
                x = _resample(x, sr, sr0)
            chunks.append(x)
        out = TURN_AUDIO_DIR / f"stitch_{uuid.uuid4().hex[:12]}.wav"
        sf.write(out, np.concatenate(chunks), sr0)
        _STITCH_CACHE[key] = str(out)
        total = sum(len(c) for c in chunks) / float(sr0 or 16000)
        print(f"  [replay] stitched {len(paths)} clips -> {total:.1f}s", flush=True)
        return str(out)
    except Exception as e:
        print(f"  [replay] stitching failed, playing only the first clip: {type(e).__name__}: {e}", flush=True)
        return paths[0]


def _same_song_group(pool: list, pick: dict) -> list:
    """The picked clip plus the clips before and after it that are **the same song and adjacent in time**, oldest first.

    A recorded turn ends when VAD detects silence, and phrase pauses and weak beats in music are easily long enough --
    measured: a 16-second piece was cut into two turns of 7.8s + 5.0s; replay played only the picked one, and it sounded like
    "it only played the beginning". Music recognition assigns these fragments the same tune_id; stitching them back in time order
    gives the original piece.

    When the tune_id isn't recognized (tune:unidentified), only time adjacency counts -- there's no other evidence then;
    better to stitch fewer clips than to join two different songs together.
    """
    tune = pick.get("tune") or ""
    same = [x for x in pool
            if (x.get("tune") or "") == tune and (tune != "unidentified" or x.get("sound_only"))]
    same.sort(key=lambda x: x["at"])
    if pick not in same:
        return [pick]
    i = same.index(pick)
    lo = i
    while lo > 0 and (same[lo]["at"] - same[lo - 1]["at"]).total_seconds() <= TUNE_GAP_S:
        lo -= 1
    hi = i
    while hi + 1 < len(same) and (same[hi + 1]["at"] - same[hi]["at"]).total_seconds() <= TUNE_GAP_S:
        hi += 1
    return same[lo:hi + 1]


def _group_id(group: list) -> str:
    """A group of clips -> one id. With a single clip just use its own id, no detour."""
    if not group:
        return ""
    if len(group) == 1:
        return group[0]["id"]
    return GROUP_ID_PREFIX + ",".join(x["id"] for x in group)


def _prefer_sound_only(cand: list) -> list:
    """If the candidates include "sound only, no speech" ones, use only those.

    "Let me play you a song" and the music that follows are **two turns** -- a recorded turn runs from detecting voice until VAD
    decides speech is over, so the first turn stores his own sentence (music was already playing in the background, so
    that turn also carries a tune tag), and only the second turn is the music itself. Both are in the pool; pick wrong and what plays is the
    user's own voice, which sounds like "the music got cut off".
    If there is not a single pure-music turn, return as is -- better than playing nothing.
    """
    only = [x for x in cand if x.get("sound_only")]
    return only or cand


def _replay_id(text: str, result) -> str:
    """Whether this turn should play back the original audio from back then; returns the memory_id to play (empty string if not).

    Only plays when the question is about sound (``_SOUND_WORDS``). The candidate pool is **all** memories with a music tag and
    surviving original audio (``_tune_memories``), not whatever this turn's retrieval happened to hit -- "the one I heard at the café
    last Wednesday" is a gamble via semantic retrieval, whereas time and place are already in the memories.

      1. Time or place given -> use as hard filters, take the most recent match.
         An empty result means **there really is none**; return an empty string so the model says so honestly, don't fall back to a random pick:
         falling back, "that song last night" would play this afternoon's recording, and the user would think it got the time wrong,
         when in fact it never searched by time at all.
      2. No conditions given ("that song", "that melody just now") -> the most recent one in the pool.
      3. Pool is empty -> fall back to what this turn's retrieval hit, or the clip just heard that isn't stored yet.
    """
    if not _wants_sound(text):
        return ""

    pool = _tune_memories()
    window, place = _time_window(text), _place_of(text)
    nth = _ordinal_of(text)

    # When "the first one" doesn't say which day, default to today -- "the first one" almost always means the first of
    # today's batch; counting over the whole store would land on one from months ago.
    if nth is not None and window is None:
        from datetime import datetime, timedelta
        day0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        window = (day0, day0 + timedelta(days=1))

    if pool and (window or place):
        cand = pool
        if window:
            cand = [x for x in cand if window[0] <= x["at"] <= window[1]]
        if place:
            cand = [x for x in cand if place in x["scenes"]]
        cand = _prefer_sound_only(cand)
        if cand:
            cand.sort(key=lambda x: x["at"])          # oldest first: the first song comes first
            if nth is None:
                pick = cand[-1]                        # no ordinal given: take the most recent
            elif -len(cand) <= (nth - 1 if nth > 0 else nth) < len(cand):
                pick = cand[nth - 1 if nth > 0 else nth]
            else:
                print(f"  [replay] only {len(cand)} songs in that period, no song #{nth}", flush=True)
                return ""
            print(f"  [replay] picked by conditions {pick['id'][:12]} ({pick['at']:%m-%d %H:%M}"
                  f"{' / ' + place if place else ''}"
                  f"{' / #%d' % nth if nth else ''}, {len(cand)} total)", flush=True)
            return _group_id(_same_song_group(pool, pick))
        print(f"  [replay] {len(pool)} music clips in pool, none match"
              f" ({'time %s~%s ' % (window[0], window[1]) if window else ''}"
              f"{'place ' + place if place else ''})", flush=True)
        return ""

    if pool:
        pick = max(_prefer_sound_only(pool), key=lambda x: x["at"])
        print(f"  [replay] no conditions given, playing the most recent {pick['id'][:12]}"
              f" ({pick['at']:%m-%d %H:%M})", flush=True)
        return _group_id(_same_song_group(pool, pick))

    # Pool is empty: music recognition missed, or it was just heard and isn't stored yet.
    playable = [h.memory_id for h in (getattr(result, "hits", None) or [])
                if audio_of(h.memory_id)]
    if playable:
        print(f"  [replay] no music tag, playing retrieved {playable[0][:12]}", flush=True)
        return playable[0]
    if _last_tune_path():
        print("  [replay] not stored yet, playing the clip just heard", flush=True)
        return LAST_TUNE_ID
    print(f"  [replay] can't play: no music memories at all, and none of the "
          f"{len(getattr(result, 'hits', None) or [])} retrieved hits have audio, "
          f"cache path={_LAST_TUNE.get('path') or '(empty)'}", flush=True)
    return ""


def _turn_detection() -> dict:
    """Turn/interruption detection on the OpenAI side. The two providers' parameters are **not interchangeable** -- semantic_vad
    doesn't accept threshold / prefix_padding_ms / silence_duration_ms; passing them gets the whole thing silently rejected
    (only coming back as an error event). So each has its own version; don't merge them.

    create_response=False   we decide when to reply (after the local end-of-turn decision and memory prefetch)
    interrupt_response=True as soon as the user speaks, the server cuts off the reply being played
    """
    # interrupt_response=False: **don't let the OpenAI side interrupt**.
    # Its VAD is purely acoustic, triggering whenever energy looks like voice, with no resistance to AEC residue -- measured:
    # the assistant got interrupted by its own echo 1.4-1.6s into speaking, showing up as "suddenly stalls mid-sentence"
    # (log: OpenAI VAD heard voice live=True still playing=True 1570ms).
    # The local path is ASR-confirmed: it only counts as a barge-in once a few words are actually transcribed; echo doesn't transcribe into coherent words.
    # Interruption now goes only through the local path: we send response.cancel ourselves in on_speech, same effect.
    base = {"type": TURN_DETECTION, "create_response": False, "interrupt_response": False}
    if TURN_DETECTION == "semantic_vad":
        return {**base, "eagerness": VAD_EAGERNESS}
    # server_vad: the default 0.5 is too dull for barge-in -- a person talking over the speaker, after echo cancellation,
    # already has a weak signal; not reaching the threshold means it can't interrupt.
    return {**base, "threshold": BARGE_THRESHOLD,
            "prefix_padding_ms": 200, "silence_duration_ms": 320}


#: A line appended when replaying. Without it the model "describes" the audio ("you said it was a very lively
#: piano piece...") -- it never heard the audio, the description is all made up; and the user is about to hear it for himself.
_REPLAY_NOTE = ("You have his recording from back then, and it will be played to him right after you say this. "
                "So don't describe what the sound is like -- you haven't heard it, don't make it up. "
                "Just introduce it in one short sentence, like \"I found that clip from back then, have a listen and see if it's the one\", "
                "then stop and let him listen.")

#: A line appended when he's looking for a sound but there really is no archive from that time period.
#: Without it the model casually answers "sure, playing it now" -- and then plays nothing. Saying it'll play and not playing
#: is far worse than just saying it wasn't found.
_NO_REPLAY_NOTE = ("He's looking for a recording, but you **don't have** a recording from the time he mentioned, "
                   "and nothing will be played this turn. So don't say \"playing it now\" or \"here it is\". "
                   "Say plainly that nothing was saved from that time, ask whether he means another time, "
                   "or talk about related things you do remember.")


def _wants_sound(text: str) -> bool:
    return _has_any(text, _SOUND_WORDS)


#: Emotion detected by the perception layer -> one **actionable** performance direction.
#:
#: Just writing "be expressive" in the persona does nothing -- it's an adjective, the model has nothing to aim at. Give it a concrete
#: target ("he's anxious right now, slow down, lower your voice, acknowledge first") and the tone really changes.
#: The emotion itself is computed by acoustic perception (Qwen-Omni attribution + prosody VAD) and differs every turn.
_TONE = {
    "anxious": "He's tense right now. Slow down, keep sentences short, acknowledge him before getting to the point; don't jump straight to solutions.",
    "frustrated": "He's very low right now. Lower and soften your voice, allow pauses; don't rush to comfort him or lecture.",
    "sad": "He's sad right now. Be gentle and slow, keep him company first; don't change the subject.",
    "irritated": "He's a bit irritated right now. Get straight to the point, don't beat around the bush, don't press, and don't use a coaxing tone.",
    "angry": "He's angry. Acknowledge it first, keep your pace steady, don't argue back.",
    "happy": "He's in a good mood. Warm up with him, let your intonation rise, laugh if it fits, don't be stiff.",
    "pleased": "He's in a good mood. Warm up with him, let your intonation rise, laugh if it fits, don't be stiff.",
    "excited": "He's very excited. Get excited too, speak a bit faster and a bit louder, don't pour cold water on it.",
    "proud": "He's proud of himself. Be happy for him, be concrete, don't give hollow praise.",
    "hopeful": "He's looking forward to something. Keep your tone light and think one step ahead with him.",
    "nervous": "He's nervous. Stay steady, keep your voice level and slow, give him a sense of certainty.",
    "wronged": "He feels wronged. Take his side first, soften your tone, don't argue who's right.",
    "calm": "",
}


def _tone_note(emotion: str) -> str:
    return _TONE.get((emotion or "").strip(), "")


#: The baseline speaking style, included every turn. Combined with _TONE it's this turn's full instruction to TTS.
_SPEAK_BASE = os.environ.get("SUPERMEM_SPEAK_BASE", "Speak like an old friend.")


def _speak_instruction(emotion: str) -> str:
    """How to voice this turn, passed straight to TTS.

    The 13 _TONE entries are all **vocal directions** to begin with ("slow down", "lower your voice", "let your intonation rise");
    they used to be concatenated into the text prompt, which meant the text model first interpreted a vocal description and then we hoped TTS
    would guess it from the words -- two layers removed, and measured to have basically no effect. TTS's instruction parameter exists
    for exactly this (Breeze has it, gpt-4o-mini-tts too), so send it there directly.
    """
    tone = _tone_note(emotion)
    return f"{_SPEAK_BASE} {tone}" if tone else _SPEAK_BASE


# ── Short-term conversation history ──────────────────────────────────────────
#: The reply model is **stateless**: reply.py only sends system + the user's current sentence each turn, no previous turns.
#: Long-term memory covers "facts about this person", not "what we were just talking about" -- so after finishing a
#: topic you say "ok ok", it sees a lone "ok ok" and greets you again
#: ("Hi there! Is there anything I can help you with?"). Both kinds of context are indispensable.
#: Short-term context is maintained separately per WebSocket session and Memory Space.
#: The corresponding turn is removed once ingest confirms a persistent memory was created.
#: At most this many chars per sentence go into the prompt. Replies are sometimes long; stuffing them all in pushes memories to the back.
_HISTORY_CHARS = int(os.environ.get("SUPERMEM_HISTORY_CHARS", "200"))
_SESSION_CONTEXT = SessionBuffer(text_limit=_HISTORY_CHARS)


def _history_block(session_id: str, space: str) -> str:
    return _SESSION_CONTEXT.render(session_id, space, "en")


def _push_history(session_id: str, space: str, user_text: str, reply_text: str,
                  interrupted: bool = False) -> str:
    return _SESSION_CONTEXT.add(
        session_id, space, user_text, reply_text, interrupted=interrupted)


def _finish_history_turn(turn_id: str, result: dict) -> None:
    committed = bool((result or {}).get("persistent_memory_created"))
    _SESSION_CONTEXT.mark_complete(turn_id, committed)
    if BARGE_DEBUG:
        state = ("committed to long-term memory, dropped from SessionBuffer" if committed
                 else "not ingested, kept in short-term context only")
        print(f"[context] turn={turn_id[:8] or '-'} {state}", flush=True)


def _realtime_instructions(memory_context: str, stranger: bool = False,
                           replay: bool = False, emotion: str = "",
                           text: str = "", context_session: str = "",
                           context_space: str = "") -> str:
    """Persona + the memories retrieved this turn. It must be clear these are "things you remember", otherwise the model treats them as
    background material to read out, rather than naturally using them as its own memory of this user.

    ``text``: what the user said this turn. Only used to tell "is he looking for a recording that we didn't find" --
    in that case say plainly it wasn't found, otherwise the model casually answers "playing it now" and then plays nothing."""
    if stranger:
        out = f"{_RT_PERSONA}\n\n{_STRANGER}"
        return f"{out}\n\n{_lang_note()}" if _lang_note() else out
    parts = [_RT_PERSONA]
    if memory_context:
        parts.append(memory_context)
    else:
        parts.append(_NO_MEMORY_NOTE)      # nothing retrieved at all: say plainly you don't know, don't make things up
    session_context = _history_block(
        context_session, context_space or ACTIVE_SPACE)
    if session_context:
        parts.append(session_context)
    if _lang_note():
        parts.append(_lang_note())
    tone = _tone_note(emotion)
    if tone:
        parts.append("His current state: " + tone)
    if replay:
        parts.append(_REPLAY_NOTE)
    elif _wants_sound(text):
        parts.append(_NO_REPLAY_NOTE)
    return "\n\n".join(parts)


# ══════════════════ Unified config entry: one dict configures all local/api models ══════════════════
# Open this dict and you know whether each model is local or api. The memory side (embedding/slots) uses local E5
# -> the whole search is 0 LLM, 0 network (Search itself measured ~10ms); the reply part (llm/tts/realtime
# used for replies) is configured here too, rather than scattered across env vars. Missing entries use built-in defaults.
# To plug in a custom config: --config path.json (or SUPERMEM_CONFIG) overrides it wholesale.
CONFIG = {
    "mode": "multi_modal",
    "memory_root": ARGS.memory_root or None,
    "space": ARGS.space,
    "embedding": {"provider": "local"},              # memory vectors via local E5 (0 network)
    "slots":     {"provider": "local"},              # slot classification via local E5 (0 LLM)
    # reply: models used for replies (not the core's concern; read by web). Defaults to OpenAI api for all.
    "reply": {
        "llm":      {"provider": "openai", "config": {"model": utils.CHAT_MODEL,
                                                      "system": _RT_PERSONA}},
        "tts":      {"provider": utils.TTS_BACKEND, "config": {"model": utils.TTS_MODEL}},
        "realtime": {"provider": "openai", "config": {"model": utils.RT_MODEL}},
    },
}

# The json that --config / SUPERMEM_CONFIG points to overrides the CONFIG above wholesale (one file configures everything).
if ARGS.config:
    CONFIG = json.loads(Path(ARGS.config).read_text(encoding="utf-8"))

REPLY = CONFIG.get("reply")                           # passed to utils' reply functions

#: The A/B compare switch and the two arms' config (GET/POST /api/compare).
#: When on, the turn produces no audio and both panels stream text side by
#: side -- same utterance, same retrieval result, differing only in whether
#: memory_context was injected. The viewer judges; there is no automated judge,
#: and that is deliberate.
COMPARE = compare.CompareState()


def _set_compare(state) -> None:
    global COMPARE
    COMPARE = state
    arms = " | ".join(f"{a.label}:{a.model or 'default'}{'+memory' if a.memory else ' no-memory'}"
                      f"{'+cartridge' if a.cartridge and a.memory else ''}"
                      for a in state.arms)
    print(f"[compare] {'on' if state.enabled else 'off'}  {arms}", flush=True)


# ── KV cartridge arm ─────────────────────────────────────────────────────────
#: The active space's memory compiled into one caller cartridge (see
#: supermem/cartridge). A compare panel with "KV cartridge" on puts it at the top
#: of the prompt as a stable prefix, so the engine can reuse its KV from turn to
#: turn instead of re-reading the memory every time.
#:
#: Snapshotted, not recompiled per turn: memory is written after every turn, and
#: recompiling then would change the prefix every turn and nothing would ever be
#: reused. The snapshot is taken when the cartridge is first needed, on
#: POST /api/cartridge, and never mid-turn. What this call adds still reaches the
#: model through the turn's recall hits, which ride after the prefix.
_CARTRIDGES: dict = {}

from supermem.llm_config import resolve_api_key, resolve_base_url, resolve_model  # noqa: E402


def _cartridge(space: str, refresh: bool = False):
    from supermem.cartridge import ContextCompiler, TokenCounter, facts_from_space
    d, safe = space_dir(space)
    if refresh or safe not in _CARTRIDGES:
        model = resolve_model(None, "reply")
        facts, traits = facts_from_space(d)
        comp = ContextCompiler(model, tenant="supermem-web", counter=TokenCounter(model, load=False))
        _CARTRIDGES[safe] = comp.compile_user(safe, facts, traits, display_name=safe)
        c = _CARTRIDGES[safe]
        print(f"[cartridge] compiled space \"{safe}\": {len(facts)} facts, {len(traits)} traits, "
              f"~{c.tokens} tokens, id {c.id}", flush=True)
    return _CARTRIDGES[safe]


def _cartridge_system() -> str:
    """The stable head of a cartridge prompt: persona + language. Session history
    is deliberately NOT here -- it grows every turn and would break the prefix."""
    return "\n\n".join(p for p in (_RT_PERSONA, _lang_note()) if p)


def _cartridge_runtime(space: str):
    from supermem.cartridge import ContextRuntime
    rt = ContextRuntime(persona=_cartridge_system())
    rt.register(space, [_cartridge(space)])
    return rt


def _cartridge_engine(arm):
    from supermem.cartridge import Engine
    base = resolve_base_url(arm.base_url or None) or "https://api.openai.com/v1"
    local = any(h in base for h in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"))
    return Engine(base, resolve_model(arm.model or None, "reply"), arm=arm.label,
                  api_key=resolve_api_key(arm.api_key or None) or "EMPTY",
                  scrape_metrics=local)         # /metrics exists on vLLM, not on hosted APIs


def _cartridge_provider(history: str, space: str):
    """compare.fan_out's ``cartridge`` hook: the provider for a cartridge arm.

    Same inputs as the plain arm (persona, language, history, this turn's
    recall), laid out differently: persona + caller cartridge first and
    byte-stable, history + recall + utterance after it. The engine's own
    accounting (prompt tokens, how many came from KV cache) is left on
    ``fn.last_usage`` for the compare feed."""
    def build(arm):
        rt = _cartridge_runtime(space)
        cart = rt.cartridges(space)[0]
        engine = _cartridge_engine(arm)

        async def fn(text: str, memory_context: str = ""):
            turn = "\n\n".join(x for x in (history, memory_context) if x)
            msgs = rt.messages(space, text, turn_memory=turn)
            try:
                async for kind, val in engine.stream(msgs, max_tokens=512):
                    if kind == "delta":
                        yield val
                        continue
                    if val.error:
                        raise RuntimeError(val.error)
                    fn.last_usage = {
                        "prompt_tokens": val.prompt_tokens, "cached_tokens": val.cached_tokens,
                        "prefill_gpu_ms": val.prefill_gpu_ms,
                        "cartridge_id": cart.id, "cartridge_tokens": cart.tokens,
                    }
            finally:
                await engine.close()

        fn.last_usage = None
        return fn
    return build


async def _prefetch_cartridges(space: str) -> list:
    """Pre-fill the cartridge prefix on every cartridge arm (max_tokens=1), the
    way a telephony integration would on ring. Returns what each engine reported."""
    rt = _cartridge_runtime(space)
    out = []
    for arm in COMPARE.arms:
        if not (arm.cartridge and arm.memory):
            continue
        engine = _cartridge_engine(arm)
        try:
            res = await engine.warm(rt.prefetch_messages(space))
        finally:
            await engine.close()
        out.append({"panel": arm.label, "ms": res.total_ms, "prompt_tokens": res.prompt_tokens,
                    "cached_tokens": res.cached_tokens, "error": res.error})
        print(f"[cartridge] prefetch panel {arm.label}: {res.total_ms:.0f}ms "
              f"prompt={res.prompt_tokens} cached={res.cached_tokens} {res.error or ''}", flush=True)
    return out

# Declarative construction: from_config is sugar on top of the existing injection mechanism (SuperMem(embedding=fn, slots=fn, ...)).
#: One SuperMem instance per Memory Space, built on demand and kept once built.
#:
#: Building a second instance in the same process costs almost nothing: models are lazy-loaded + reused in-process; measured instance build
#: 0.0s, warmup 2.5s (the first one is 6.8s + 4.9s). So switching spaces needs no service restart.
_SPACES: dict = {}


def space_dir(name: str):
    """This space's directory on disk. Names allow only letters, digits and - _, to prevent path traversal."""
    import re as _re
    safe = _re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "", (name or "").strip())[:32]
    if not safe:
        raise ValueError("space name cannot be empty")
    return _ROOT / "supermem_memoryspace" / safe, safe


def get_space(name: str):
    """Get (creating if needed) the SuperMem for this space."""
    _, safe = space_dir(name)
    if safe not in _SPACES:
        cfg = dict(CONFIG)
        cfg["space"] = safe
        t0 = time.monotonic()
        inst = SuperMem.from_config(cfg)
        inst.warmup(verbose=False)
        _SPACES[safe] = inst
        print(f"[space] opened \"{safe}\" in {time.monotonic()-t0:.1f}s", flush=True)
    return _SPACES[safe]


def use_space(name: str) -> str:
    """Switch to this space. Returns the name actually used.

    ``vm`` is a module-level global and everything downstream looks it up by name at runtime, so rebinding it here is enough --
    no need to pass the instance all the way down. Note the callbacks build_app receives must be lambdas, not
    bound methods like ``vm.classify``; a bound method would weld in the instance from before the switch.
    """
    global vm, ACTIVE_SPACE, SPACE_LANG
    vm = get_space(name)
    _, ACTIVE_SPACE = space_dir(name)
    SPACE_LANG = space_language(ACTIVE_SPACE)
    _set_lang(SPACE_LANG)            # memory language follows the space
    return ACTIVE_SPACE


def list_spaces() -> list:
    """Which Memory Spaces exist on disk, and how many memories each has."""
    import sqlite3
    root = _ROOT / "supermem_memoryspace"
    out = []
    for d in sorted(p for p in root.glob("*") if p.is_dir()):
        n = 0
        try:
            from supermem.utils.common import space as _sp
            db = _sp.db(d)
            if Path(db).exists():
                c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                c.close()
        except Exception:
            n = 0
        out.append({"id": d.name, "name": d.name, "count": n,
                    "active": d.name == ACTIVE_SPACE, "open": d.name in _SPACES,
                    "language": space_language(d.name)})
    return out


def create_space(name: str, language: str = "") -> dict:
    """Create a new empty Memory Space: a folder with this name appears on disk, holding a brand-new empty store.

    The instance is built right here (and warmed up), so you can talk immediately after clicking "Create" instead of the first
    sentence getting stuck on model loading.
    """
    d, safe = space_dir(name)
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"\"{safe}\" already exists")
    d.mkdir(parents=True, exist_ok=True)
    get_space(safe)                      # build store + warm up
    lang = "en"
    _write_space_language(safe, lang)    # set once at creation, never changed afterwards
    print(f"[space] created \"{safe}\" (language {lang}) -> {d}", flush=True)
    return {"id": safe, "name": safe, "count": 0, "language": lang}


ACTIVE_SPACE = ""
vm = None
use_space(ARGS.space)


#: Audio from voice turns lands here. The archive table stores paths, so the files themselves must really exist.
TURN_AUDIO_DIR = _ROOT / "results" / "turn_audio"


def save_turn_audio(pcm16k) -> str:
    """Save this turn's PCM as a wav and return its path; return "" if it can't be saved (doesn't affect this turn's conversation).

    Without this step AudioArchive would never have a single record -- it only writes when ingest receives an audio_path.
    The demo used to stream everything over WS and never write to disk, so "play back the original audio from back then" was impossible.
    """
    if pcm16k is None or not len(pcm16k):
        return ""
    try:
        import numpy as np
        import soundfile as sf
        TURN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        path = TURN_AUDIO_DIR / f"turn_{uuid.uuid4().hex[:12]}.wav"
        sf.write(path, np.asarray(pcm16k, dtype="float32"), 16000)
        return str(path)
    except Exception as e:
        print(f"[web] could not save this turn's audio (conversation unaffected): {e}", flush=True)
        return ""


@dataclass
class Pending:
    """The "pre-computed memory" that speculative prefetch has long since computed by the time a turn ends -- control flow replies with it directly, no more searching."""
    text: str
    memory_context: str
    result: object
    spoken: bool = True          # True=voice turn (audio already in the realtime buffer), False=typed turn
    audio_path: str = ""         # this turn's wav on disk; ingest uses it for scene/music/voiceprint perception,
                                 # and binds it to the memory in audio_archive so it can be played back as is later
    stranger: bool = False       # voiceprint says the speaker isn't the owner of this memory store
    replay: str = ""             # which memory's original audio to play back (memory_id); empty = don't play
    emotion: str = ""            # emotion perceived in the previous turn, used to set this turn's tone


# ══════════════════ The two control flows (~10 lines each, only consume the prefetched Pending) ══════════════════

# The sentence-splitting rule (decides how soon the first sound comes out) is cut_point in supermem/tts.py.
_cut_point = utils.cut_point


def _mentioned(name: str, text: str) -> bool:
    """Is the entity name mentioned in this sentence?

    Can't just do `name in text` -- entities in the graph often carry qualifiers ("Jiaqi's boss"), while people say
    "boss". Split into chunks (on the possessive particle and non-word characters) and compare each; any chunk appearing counts as mentioned.
    Single-char chunks don't count: a lone "I" or "a" matches far too easily.
    """
    t = text or ""
    if not name:
        return False
    if name in t:
        return True
    for chunk in re.split(r"[\u7684\s·\u3001,\uff0c]+|[^\u4e00-\u9fffA-Za-z0-9]+", name):
        if len(chunk) >= 2 and chunk in t:
            return True
    return False


#: The acoustic model must be at least this confident to be adopted (emotion2vec+ softmax score).
ACOUSTIC_MIN_SCORE = float(os.environ.get("SUPERMEM_ACOUSTIC_MIN", "0.92"))

#: Only these categories trust the acoustics.
#:
#: Measured: emotion2vec+ **confidently misjudges** real recordings: "oh hey, good morning" judged sad
#: 1.00, "I think it's pretty good" judged sad 1.00 -- raising the threshold can't block it, it gives the wrong answer
#: a perfect score. But when it's wrong it's almost always "sad" (its default bias for this speaker).
#: The high-arousal categories (really laughing, really angry) have obvious acoustic features, exactly what text can't show,
#: so leave those to it; the low-arousal ones go to semantics.
ACOUSTIC_TRUST = set((os.environ.get("SUPERMEM_ACOUSTIC_TRUST")
                      or "happy,wronged,surprised").split(","))

#: Semantic examples for emotions. Local E5 compares this sentence's similarity to these -- broader than a keyword list
#: ("I think it's pretty good" has no emotion word in it), and far more accurate than acoustics. 0 network, 9ms.
_EMO_PROTO = {
    "happy": ["I'm so happy today", "That's great, I'm really glad", "Pretty nice, I'm quite satisfied", "Haha that's so fun"],
    "sad": ["I'm really sad", "My heart feels so heavy", "I feel so let down", "This has got me pretty down"],
    "wronged": ["I'm so mad", "This is infuriating", "Why would they treat me like this", "I think it's really unfair"],
    "anxious": ["I'm under so much pressure", "I'm a bit nervous", "I'm worried I won't finish", "This is keeping me up at night"],
    "tired": ["I'm so tired", "Exhausted, I can't keep going", "I feel completely drained after today"],
    # "calm" needs more examples: everyday statements and questions make up most of a conversation, and with too few examples
    # they can't open enough of a gap and end up judged as "(empty)" all the way -- measured: "I'm allergic to peanuts" and "what can't I eat"
    # were both left unlabelled because of this; adding examples raised the gap from 0.002 to 0.08.
    "calm": ["Nice weather today", "I have a meeting tomorrow", "Good morning", "My name is Alex",
             "Put this on the table", "I'm allergic to peanuts", "I work at a company",
             "There's a meeting next Wednesday at 3pm", "What can't I eat", "How do I use this",
             "Take a look at this for me", "I live downtown"],
}
_PROTO = {}


def _emotion_by_meaning(text: str) -> str:
    """Semantic nearest neighbour. Only give a label if it's similar enough **and** clearly ahead of the runner-up, otherwise empty --
    when ambiguous, no label beats a wrong label."""
    import numpy as np
    if not _PROTO:
        labels, sents = [], []
        for k, vs in _EMO_PROTO.items():
            labels += [k] * len(vs)
            sents += vs
        _PROTO["labels"] = labels
        _PROTO["V"] = np.array(utils.shared_e5().encode(sents, normalize_embeddings=True),
                               dtype=np.float32)
    q = np.array(utils.shared_e5().encode([text], normalize_embeddings=True),
                 dtype=np.float32)[0]
    sims = _PROTO["V"] @ q
    i = int(np.argmax(sims))
    lab, best = _PROTO["labels"][i], float(sims[i])
    other = [float(sims[j]) for j in range(len(sims)) if _PROTO["labels"][j] != lab]
    gap = best - (max(other) if other else 0.0)
    return lab if best >= 0.80 and gap >= 0.02 else ""


#: emotion2vec+ labels -> the emotion labels we use
_E2V_MAP = {"happy": "happy", "sad": "sad", "angry": "wronged", "fearful": "fear",
            "surprised": "surprised", "disgusted": "disgust", "neutral": ""}
_E2V = {}


def _acoustic_emotion(audio_path: str):
    """emotion2vec+ (a model built specifically for speech emotion recognition). Returns (label, score)."""
    if "m" not in _E2V:
        from funasr import AutoModel
        _E2V["m"] = AutoModel(model=os.environ.get("SUPERMEM_E2V_MODEL",
                                                   "emotion2vec/emotion2vec_plus_base"),
                              hub="hf", disable_update=True)
    r = _E2V["m"].generate(audio_path, granularity="utterance", extract_embedding=False)
    if not r:
        return "", 0.0
    lab, score = max(zip(r[0]["labels"], r[0]["scores"]), key=lambda x: x[1])
    en = str(lab).split("/")[-1].strip().lower()
    return _E2V_MAP.get(en, ""), float(score)


_SV = {}


def _sensevoice():
    """SenseVoiceSmall: a single inference yields both a refined transcript and acoustic emotion. Lazy-loaded, reused in-process.

    Note: don't use vm.utils.get("emotion") -- that's the prosody-quadrant heuristic
    (PaperAlignedEmotionDetector); it has no run_with_emotion and it's inaccurate.
    """
    if "t" not in _SV:
        from supermem.utils.audio.asr import Transcriber, pick_device
        _SV["t"] = Transcriber(pick_device())
    return _SV["t"]


def _kick_acoustic(send, audio_path: str) -> None:
    """Push acoustic emotion to the background; once a trustworthy result is computed, send a follow-up tag_update.

    emotion2vec runs over the whole audio clip, measured at 2.3 s. Doing it before sending memory_hits would add those
    2.3 s between "user finishes -> assistant starts speaking", and nine times out of ten its result doesn't reach the trust threshold anyway.
    In the background the hot path owes nothing, and when it does judge correctly the emotion tag in the UI still updates.
    """
    if not audio_path or os.environ.get("SUPERMEM_ACOUSTIC_TAG", "1") == "0":
        return

    async def run():
        try:
            t0 = time.monotonic()
            emo, score = await asyncio.to_thread(_acoustic_emotion, audio_path)
            take = bool(emo) and score >= ACOUSTIC_MIN_SCORE and emo in ACOUSTIC_TRUST
            if BARGE_DEBUG:
                print(f"  [emotion] acoustic (background) {(time.monotonic()-t0)*1000:.0f}ms "
                      f"-> {emo or '-'} {score:.2f} ({'used' if take else 'rejected'})", flush=True)
            if take:
                from supermem.lang import display_emotion
                await send({"type": "tag_update", "emotion": display_emotion(emo),
                            "emotion_from": "acoustic"})
        except Exception as e:
            print(f"[web] background acoustic emotion skipped: {type(e).__name__}: {e}", flush=True)

    asyncio.create_task(run())


def fill_tags(payload: dict, text: str, audio_path: str = "",
              acoustic: bool = True) -> dict:
    """Fill in the emotion / entities the tag bar needs -- both 0 LLM, 0 network.

    Retrieval uses the local slot classifier (no network within the speculation budget), which only outputs slots, not entities;
    emotion can only be computed after this turn's ingest, while memory_hits are sent before the reply.
    The result was that these two cells in the tag bar stayed empty.

    · Emotion: computed once with anchor_router's keyword table (pure lookup).
    · Entities: read directly which entities the memories hit this turn are attached to in the cognitive graph (pure sqlite).
    """
    # (0) Rewrite the right-brain hits into first-person plain language (display only; the raw text stays for brain-map matching).
    #    For this turn, items not yet rewritten show the original and are queued in the background -- see rb_human.
    for h in payload.get("right_brain_hits") or []:
        claim = h.get("claim") or ""
        if not claim:
            continue
        human = rb_human(claim)
        if human and human != claim:
            # Keep the evidence half (" | he said: ..."); it's the support for "why would you say that".
            tail = ""
            for sep in (" | he said: ",):
                if sep in h.get("content", ""):
                    tail = sep + h["content"].split(sep, 1)[1]
                    break
            h["content"] = human + tail

    # (1) If the person states the emotion explicitly, go with what they said (table lookup, 0 network) -- most accurate
    if not payload.get("emotion") and text.strip():
        try:
            from supermem.rightbrain.anchor_router import normalize_emotion_strict
            payload["emotion"] = normalize_emotion_strict(text) or ""
        except Exception:
            pass

    # (2) If not stated, look at the **meaning** of the sentence: local E5 compares semantic similarity with each emotion's example sentences.
    #    9ms, 0 network, and broader than a keyword table ("I think it's pretty good" has no emotion word in it).
    if not payload.get("emotion") and text.strip():
        try:
            emo = _emotion_by_meaning(text)
            if emo:
                payload["emotion"] = emo
                payload["emotion_from"] = "semantic"
        except Exception as e:
            print(f"[web] semantic emotion skipped: {type(e).__name__}: {e}", flush=True)

    # (3) Acoustic (emotion2vec+): overrides the judgement above only when it's **very confident**.
    #
    #    Why it's not the main source: measured on real recordings it judged "oh hey, good morning" as sad
    #    (3.2 s of clean audio, not a too-much-silence problem -- same after trimming); SenseVoice and
    #    the prosody heuristic did the same. With this mic/speaking style, acoustics can't read it accurately.
    #    So keep it, but only count it when score >= ACOUSTIC_MIN_SCORE -- when someone really speaks with emotion
    #    it gives 0.95+, while flat speech gets things like 0.6 or 0.7, which this filters out nicely.
    if acoustic and audio_path and os.environ.get("SUPERMEM_ACOUSTIC_TAG", "1") != "0":
        try:
            t0 = time.monotonic()
            emo, score = _acoustic_emotion(audio_path)
            take = bool(emo) and score >= ACOUSTIC_MIN_SCORE and emo in ACOUSTIC_TRUST
            if take:
                payload["emotion"] = emo
                payload["emotion_from"] = "acoustic"
            if BARGE_DEBUG:
                why = "used" if take else ("confidence too low" if score < ACOUSTIC_MIN_SCORE
                                           else f"{emo} not in the trusted set")
                print(f"[emotion] acoustic {(time.monotonic()-t0)*1000:.0f}ms "
                      f"-> {emo or '-'} {score:.2f} ({why})"
                      f"  final={payload.get('emotion') or '-'}", flush=True)
        except Exception as e:
            print(f"[web] acoustic emotion skipped: {type(e).__name__}: {e}", flush=True)

    # Entities: **no guessing** here.
    #
    # memory_hits are sent before the reply, while entities come out of the same LLM call that extracts facts
    # (at ingest, after the reply). There was once a "zero-cost approximation" here -- picking from the hit old memories
    # the entities this sentence mentioned -- and every sentence ended up with only the speaker:
    #     say "I'm allergic to peanuts" -> tag bar ['Lin'], what was really extracted was ['Lin', 'peanuts']
    #     say "meeting a client at the IFC next Wednesday" -> tag bar empty
    # Showing a wrong one is worse than leaving it empty for now. The frontend fills in the real result after ingest is stored (see
    # loadMemories in supermem.html).

    # anchor_router / emotion2vec both return internal values of the 8 canonical emotions (see
    # supermem/lang.py) -- the UI tag bar needs the spelling for the current memory store language, otherwise
    # an internal key nobody can read would show up in the demo.
    if payload.get("emotion"):
        from supermem.lang import display_emotion
        payload["emotion"] = display_emotion(payload["emotion"])

    rb = payload.get("right_brain_hits") or []
    inner = sum(1 for h in rb if h.get("internal"))
    print(f"[hits] left {len(payload.get('left_brain') or [])}  "
          f"right {len(rb)} ({inner} internal, {len(rb)-inner} shown)  "
          f"emotion={payload.get('emotion') or '-'}  "
          f"entities={', '.join(payload.get('entities') or []) or '-'}", flush=True)
    return payload


async def _announce_turn(pending, send) -> None:
    """Turn opening: transcript + the memories hit this turn + acoustic emotion
    in the background.

    All three control flows (llm_tts / realtime / compare) say the same few
    things, so they live in one place -- there used to be two verbatim copies,
    and adding a third was the moment to merge them.

    Acoustic emotion is **not** computed here. It takes 2.3s (emotion2vec runs
    the whole clip), and these lines sit in the most sensitive stretch there is:
    between the user finishing and the assistant starting. Measured, that step
    alone ate half of a 4.5s gap, and the result was often discarded for low
    confidence anyway -- pure waste. Send the millisecond-level semantic label
    now, run the acoustic one in the background, and issue a tag_update to
    override the emotion in the UI only once it is trustworthy.
    """
    await send({"type": "user_transcript", "text": pending.text})
    note_hits(pending.result)      # make sure the brain-map snapshot has these items on the graph
    await send({"type": "memory_hits",
                **fill_tags(utils.hits_payload(pending.result, has_audio=audio_of,
                                              cluster_of=hit_cluster),
                            pending.text, pending.audio_path or "", acoustic=False)})
    _kick_acoustic(send, pending.audio_path or "")


async def _compare_shared_system(pending, context_session: str, context_space: str) -> str:
    """The system prompt **shared** by both panels: persona + language + session
    history.

    Only memory_context differs per arm -- that is the variable under test. If
    persona, language or history differed between the two, the difference on
    screen would no longer be "with or without memory", and the comparison
    would not count.
    """
    parts = [_RT_PERSONA]
    if pending.stranger:
        parts.append(_STRANGER)
    hist = _history_block(context_session, context_space or ACTIVE_SPACE)
    if hist:
        parts.append(hist)
    if _lang_note():
        parts.append(_lang_note())
    return "\n\n".join(parts)


async def _compare_turn(pending, send, owner, timeline=None, context_session="",
                        context_space="") -> dict:
    """Compare this turn: one utterance to two arms, one with memory injected
    and one without, streaming side by side.

    No audio (the whole TTS path is skipped) and no answer_* messages -- the
    frontend's caption and playback machinery is tied to the playback clock, so
    compare mode uses its own cmp_* messages and that code needs no changes.

    Memory is written **once**, through the existing queue_remember_turn, using
    panel a's reply. Even when both arms fail the turn is still ingested (with
    an empty agent_reply): the facts in what the user just said should not be
    lost to one bad key.
    """
    context_space = context_space or ACTIVE_SPACE
    memory_vm = get_space(context_space)
    system = await _compare_shared_system(pending, context_session, context_space)
    # A stranger's turn gets no memory injected at all (they are not this
    # library's owner), so neither arm has any -- nothing to compare, but the
    # owner's memories must not be read out to someone else either.
    mem_ctx = "" if pending.stranger else (pending.memory_context or "")
    # The cartridge holds the owner's whole memory: a stranger's turn must never
    # get it, so their cartridge arm falls back to the plain (memory-less) provider.
    cartridge = None if pending.stranger else _cartridge_provider(
        _history_block(context_session, context_space), context_space)
    result = await compare.fan_out(pending.text, mem_ctx, COMPARE.arms, send,
                                   system=system, cartridge=cartridge)
    reply = result.get("a", "")
    history_turn_id = _push_history(context_session, context_space,
                                    pending.text, reply)
    queue_remember_turn(pending, reply, owner, history_turn_id,
                        memory_vm=memory_vm)
    if timeline is not None:
        # No audio this turn, but the context really was saved. Without this
        # flag the timeline concludes the turn's context was lost (same line at
        # the end of the llm_tts path).
        timeline.context_saved = True
    return result


async def supermem_llm_tts(pending, send, send_audio, owner, timeline,
                           said=None, context_session="", context_space="",
                           memory_vm=None):
    """Memory was prefetched off the critical path: LLM streaming reply -> TTS streaming speech.

    TTS runs **in parallel** with generation: every full sentence the LLM emits goes into a queue, and another coroutine takes it out, synthesizes and sends the audio.
    If synthesis only started after the whole text was generated, the text would be long done while audio hadn't started (measured: the first TTS frame alone takes
    ~1.2s, plus the few seconds of generation, the user just stares at the text waiting).

    ``said``: an optional dict; the text spoken so far is written into said["text"] as generation proceeds. Barge-in detection uses
    it to block echo (see _is_echo) -- the assistant's speech loops back through the mic into ASR, and the transcribed words are
    indistinguishable from a real barge-in by count; only content can tell them apart.
    """
    memory_vm = memory_vm or vm
    context_space = context_space or ACTIVE_SPACE
    await _announce_turn(pending, send)
    if pending.replay:
        _note_replay(pending.replay)
        await send({"type": "play_memory", "memory_id": pending.replay})
    if COMPARE.enabled:
        # Branch before answer_start: that message resets the frontend's
        # playback and sets aiSpeaking, and a compare turn makes no sound.
        await _compare_turn(pending, send, owner, timeline,
                            context_session=context_session,
                            context_space=context_space)
        return
    await send({"type": "answer_start", "output_id": timeline.output_id,
                "sample_rate": timeline.sample_rate})

    queue: asyncio.Queue = asyncio.Queue()

    # Use the injected TTS (the ninth swappable slot). Swapping the provider in --config, or a library user
    # passing their own implementation via SuperMem(tts=lambda: MyTTS()), takes effect here; if not configured it's the built-in default.
    tts = memory_vm.utils.get("tts")
    # How to voice this turn. Emotion changes per turn, so it's passed per turn, not stored on the instance.
    speak_as = _speak_instruction(pending.emotion)

    def _synth_one(seg):
        """The injected TTS may be user-written and only accept stream(text) -- then fall back;
        it just loses one layer of tone control, and shouldn't make the whole chain error out."""
        try:
            return tts.stream(seg, speak_as)
        except TypeError:
            return tts.stream(seg)

    # A reply is cut into several sentences synthesized one by one, and there's a gap between segments: the next segment's request
    # was only sent after this one finished synthesizing, waiting for "send request -> server prefill -> first byte back"; the audio queue is
    # empty meanwhile, which sounds like a stall at the start of each sentence. With TTS remote (e.g. Breeze on a GPU machine,
    # behind an SSH tunnel too) this gap is especially noticeable.
    # So it's split into two levels: as soon as a segment arrives, start synthesizing **immediately**, each filling its own small queue; playback takes
    # them segment by segment in order. That way the next segment is already being computed while the current one plays.
    # No concurrency limit: whether the server is single-concurrency is its own business (Breeze is); sending a few more requests from here
    # just queues them there, saving a network round trip per segment.
    synths: list[asyncio.Task] = []
    streams: asyncio.Queue = asyncio.Queue()      # each item is one segment's chunk queue

    async def synth():
        while (spec := await queue.get()) is not None:
            seg, text_start, text_end = spec
            chunks: asyncio.Queue = asyncio.Queue()
            state = {"complete": False}

            async def run(seg=seg, chunks=chunks, state=state):
                try:
                    async for chunk in _synth_one(seg):
                        await chunks.put(chunk)
                    state["complete"] = True
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    print(f"[web] TTS failed: {type(e).__name__}: {e}", flush=True)
                finally:
                    await chunks.put(None)        # even on error, let playback wrap up

            synths.append(asyncio.create_task(run()))
            await streams.put((seg, text_start, text_end, chunks, state))
        await streams.put(None)

    async def speak():
        while (item := await streams.get()) is not None:
            seg, text_start, text_end, chunks, state = item
            segment_id = timeline.begin_segment(text_start, text_end)
            # Echo detection must compare against what **the user may have heard**, not what's been generated -- generation is
            # already several segments ahead. This is the moment audio actually starts going out, closest to "said out loud".
            if said is not None:
                said["text"] = (said.get("text") or "") + seg
            try:
                while (chunk := await chunks.get()) is not None:
                    if isinstance(chunk, TimedAudioChunk):
                        if chunk.sample_rate != timeline.sample_rate:
                            raise ValueError(
                                f"TTS output sample rate should be {timeline.sample_rate}Hz, "
                                f"got {chunk.sample_rate}Hz")
                        pcm = chunk.pcm
                        timeline.add_segment_timestamps(
                            segment_id, chunk.timestamps)
                    else:
                        pcm = chunk
                    timeline.append_audio(pcm)
                    await send_audio(pcm)
            except asyncio.CancelledError:
                timeline.finish_segment(segment_id, complete=False)
                raise
            except Exception as e:                # most likely the page was closed mid-listen, not an error
                timeline.finish_segment(segment_id, complete=False)
                print(f"[web] audio send interrupted: {type(e).__name__}", flush=True)
                break
            else:
                timeline.finish_segment(
                    segment_id, complete=state["complete"])

    synther = asyncio.create_task(synth())
    speaker = asyncio.create_task(speak())
    reply, buf, sent = "", "", 0
    interrupted = False
    try:
        # Same instructions as realtime: the two paths must behave the same, otherwise switching --mode would
        # silently drop the persona and the "never read the right brain aloud" constraint.
        # Goes through the core reply layer (persona in CONFIG.reply.llm.config.system; see compose_system in supermem/reply.py:
        # system + memory_context, the same as what the realtime path assembles).
        ctx = _STRANGER if pending.stranger else pending.memory_context
        if not pending.stranger and not (ctx or "").strip():
            ctx = _NO_MEMORY_NOTE          # nothing retrieved at all: say plainly you don't know, don't make things up
        # Emotion is no longer put into the text prompt: it's a **vocal direction** ("lower, soften, leave pauses");
        # having the text model interpret it and then hoping TTS guesses it is two layers removed. The TTS backend's instruction
        # parameter exists for exactly this, so it should move there. Until then pending.emotion has no outlet on this path.
        note = (_REPLAY_NOTE if pending.replay
                else (_NO_REPLAY_NOTE if _wants_sound(pending.text) else ""))
        if note:
            ctx = f"{ctx}\n\n{note}" if ctx else note
        hist = _history_block(context_session, context_space)
        if hist:
            ctx = f"{ctx}\n\n{hist}" if ctx else hist
        if _lang_note():
            ctx = f"{ctx}\n\n{_lang_note()}" if ctx else _lang_note()
        async for d in memory_vm.reply_stream(pending.text, ctx):
            reply += d
            buf += d
            timeline.append_text(d)
            await send({"type": "answer_delta", "text": d})
            if _cut_point(buf, first=sent == 0):
                segment = buf.strip()
                leading = len(buf) - len(buf.lstrip())
                start = len(reply) - len(buf) + leading
                await queue.put((segment, start, start + len(segment)))
                buf, sent = "", sent + 1
        if buf.strip():
            segment = buf.strip()
            leading = len(buf) - len(buf.lstrip())
            start = len(reply) - len(buf) + leading
            await queue.put((segment, start, start + len(segment)))
    except asyncio.CancelledError:
        interrupted = True                      # the user cut in; this turn ends here
    finally:
        await queue.put(None)                   # even if generation errors, let speak() wrap up

    async def _drop_pipeline():
        # Don't leave speak() in the background still sending audio on a connection that has stopped playing.
        # The segments whose synthesis started early must stop too, otherwise they keep occupying the remote TTS queue,
        # and the next turn's first sentence has to wait behind audio nobody wants -- it sounds laggier after an interruption.
        speaker.cancel()
        synther.cancel()
        for t in synths:
            t.cancel()
        await asyncio.gather(
            speaker, synther, *synths, return_exceptions=True)

    if interrupted:
        await _drop_pipeline()
    else:
        # Done generating doesn't mean done speaking: audio is still going out segment by segment, and interruptions mostly land here.
        # If not caught, CancelledError tears down this Task outright and not a single line of the memory-saving below runs,
        # so the interrupted turn would never make it into memory.
        try:
            await synther
            await speaker
        except asyncio.CancelledError:
            interrupted = True
            await _drop_pipeline()
        else:
            timeline.mark_generation_complete()
            try:
                await send({"type": "answer_done", "output_id": timeline.output_id})
                timeout = max(2.0, min(
                    60.0, timeline.sent_samples / timeline.sample_rate + 2.0))
                await asyncio.wait_for(timeline.wait_playback_done(), timeout=timeout)
            except asyncio.TimeoutError:
                timeline.assume_drained()
            except asyncio.CancelledError:
                interrupted = True
                await _drop_pipeline()

    # Record this turn's context; memory writing goes to the background so it doesn't block listening for the next turn.
    context_reply = timeline.heard_text() if interrupted else reply
    if interrupted and BARGE_DEBUG:
        print(f"[context] interrupted at {timeline.rendered_ms()}ms, keeping reply "
              f"{context_reply!r}", flush=True)
    history_turn_id = _push_history(
        context_session, context_space, pending.text, context_reply,
        interrupted=interrupted)
    queue_remember_turn(
        pending, context_reply, owner, history_turn_id, memory_vm=memory_vm)
    timeline.context_saved = True



async def start_realtime_turn(pending, conn, send, timeline,
                              context_session="", context_space=""):
    """Inject the prefetched memory into the Realtime session and trigger this turn's native speech.

    Only "kicks it off"; receiving audio/text and wrapping up all happen in the resident event pump (see realtime_session) --
    OpenAI's event stream can have only one consumer; if each turn read its own they'd cross wires: a leftover response.done
    from a previous interrupted turn would be read by the next turn and taken as its own completion.
    """
    await _announce_turn(pending, send)
    if pending.replay:
        # The frontend holds it and plays it once this turn's reply finishes -- the assistant's reply plays from a queue,
        # and playing early would overlap with the voice.
        _note_replay(pending.replay)
        await send({"type": "play_memory", "memory_id": pending.replay})
    if pending.spoken:
        await conn.input_audio_buffer.commit()
    else:
        await conn.conversation.item.create(item={"type": "message", "role": "user",
                                                  "content": [{"type": "input_text", "text": pending.text}]})
    # Memory goes in response.create's per-response instructions, not
    # session.update. That one is a session-level setting and the model does not
    # read it for the turn you just set it on: asked "what is my cat called?"
    # with "her name is Momo" sitting in the retrieved memory, it still answered
    # "you just mentioned it but I did not catch it".
    print(f"[lat] local VAD confirmed end of speech -> response.create", flush=True)
    await conn.response.create(response={
        "instructions": _realtime_instructions(pending.memory_context, pending.stranger,
                                               replay=bool(pending.replay),
                                               emotion=pending.emotion,
                                               text=pending.text,
                                               context_session=context_session,
                                               context_space=context_space),
    })
    await send({"type": "answer_start", "output_id": timeline.output_id,
                "sample_rate": timeline.sample_rate})


async def truncate_provider_output(conn, provider_item_id: str,
                                   timeline: AudioTimeline) -> None:
    truncate = getattr(conn, "truncate_output", None)
    if callable(truncate):
        await truncate(
            provider_output_id=provider_item_id,
            media_output_id=timeline.output_id,
            audio_end_samples=timeline.rendered_cutoff_samples(),
            sample_rate=timeline.sample_rate,
        )
        return
    conversation = getattr(conn, "conversation", None)
    item_api = getattr(conversation, "item", None)
    truncate = getattr(item_api, "truncate", None)
    if callable(truncate):
        await truncate(
            item_id=provider_item_id, content_index=0,
            audio_end_ms=timeline.rendered_ms())


async def _no_realtime(sock, err):
    """When Realtime can't connect, don't leave people guessing at a traceback.

    Distinguish **network** from **permission**: DNS/connection failures have nothing to do with the key; previously everything was called
    "the key may lack permission", pointing people in the wrong direction.
    """
    name, text = type(err).__name__, str(err)
    network = (isinstance(err, (OSError, TimeoutError, ConnectionError))
               or "gaierror" in name.lower()
               or any(k in text.lower() for k in ("nodename", "temporary failure",
                                                  "name or service", "getaddrinfo",
                                                  "connection refused", "timed out")))
    if network:
        why = ("Can't reach api.openai.com (a DNS/proxy/VPN problem, unrelated to the key). "
               "Make sure you're online and restart; offline, `--mode llm_tts` can't connect either, "
               "since both paths need OpenAI.")
    elif any(k in text for k in ("401", "403", "invalid_api_key", "insufficient", "model_not_found")):
        why = ("This key has no Realtime access or the model is unavailable -- use "
               "`python web/run.py --mode llm_tts` instead; that path only needs regular chat + TTS.")
    else:
        why = ("Check the error itself first; if it's just that Realtime is unavailable, you can use "
               "`python web/run.py --mode llm_tts` instead (regular chat + TTS).")
    msg = f"Can't connect to OpenAI Realtime ({name}: {text}). {why}"
    print(f"[web] {msg}", flush=True)
    try:
        await sock.send_json({"type": "error", "message": msg})
    except Exception:
        pass


# ══════════════════ Driving the supermem core streaming session (vm.stream()) ══════════════════
# ASR + VAD + speculative prefetch (prefetch while speaking / 200ms gamble on done / barge-in / 300ms confirm)
# all live in the core VoiceStream. Here we only do what the demo should: shuttle socket frames, send partials, and wrap each finished
# turn into a Pending for the control flow -- the demo is a usage example of the core, not a parallel rewrite.

def remember_turn(pending, reply: str, owner: dict, history_turn_id: str = "",
                  memory_vm=None) -> None:
    """Store this turn, and note who the speaker is along the way.

    The speaker needn't be computed separately: ingest runs a preprocess internally anyway (scene/voiceprint/emotion),
    and its return value carries speaker_id directly. I used to trigger a whole separate perception pass on the hot path --
    424ms each time (AST taking 361ms), pure duplicated work, and it sat in front of reading the socket.
    """
    # A turn with no transcript = a sound played into the mic (see MIN_SOUND_ONLY_S in stream.py).
    # ingest("") directly extracts no facts, and the audio would be stored for nothing -- later asking "replay that song
    # just now" finds no memory. Give it a sentence so it can be retrieved; what the music is gets tagged tune: by ingest's
    # internal music recognition; here we just give it a memory carrier.
    # Judging "was anything really said" can't just check for empty: music fed into ASR gets forced into a letter or two
    # (measured: cafe_song transcribed as 'i'), non-empty but meaningless. Require at least two CJK characters or three
    # letters to count as speech.
    text = pending.text
    meaningful = re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")
    cjk = len(re.findall(r"[\u4e00-\u9fff]", meaningful))
    if pending.audio_path and cjk < 2 and len(meaningful) < 3:
        from supermem.stream import SOUND_ONLY_TEXT
        text = SOUND_ONLY_TEXT          # must use this constant; the core relies on it to recognize the no-speech turn

    try:
        target_vm = memory_vm or vm
        r = target_vm.ingest(
            text, agent_reply=reply, async_facts=True,
            audio=pending.audio_path or None,
            on_complete=lambda result: _finish_history_turn(history_turn_id, result),
        ) or {}
    except Exception as e:
        print(f"[web] storing this turn failed: {type(e).__name__}: {e}", flush=True)
        return
    # The previous turn's emotion is kept for the next turn's speculative retrieval. Without it the right brain can't fetch emotional records,
    # and every turn would only return the same few static profile items (see the emotion notes in supermem/stream.py).
    affect = r.get("affect")
    if isinstance(affect, dict):
        affect = affect.get("emotion") or affect.get("label") or ""
    owner["emotion"] = str(affect or "").strip()

    # If this turn heard music, remember that recording. Tune recognition is synchronous; by the time we get it here, the background
    # storage has only just started -- whether "ask right after listening" can play anything depends entirely on this line.
    if r.get("recognized_tune") and pending.audio_path:
        _remember_tune(pending.audio_path)
    elif BARGE_DEBUG and _wants_sound(pending.text or ""):
        print(f"  [replay] no music remembered this turn: "
              f"tune={bool(r.get('recognized_tune'))} audio={bool(pending.audio_path)}",
              flush=True)

    sid = r.get("speaker_id") or ""
    if not sid:
        # This turn was too short, the voiceprint wasn't computed at all (see SUPERMEM_SPEAKER_MIN_S in perceiver).
        # Not knowing who it is != a different person, so don't even touch the miss count.
        return
    if not owner["id"]:
        owner["id"] = sid                  # the first to speak counts as this conversation's owner
    owner["last"] = sid
    owner["miss"] = 0 if sid == owner["id"] else owner.get("miss", 0) + 1


# Memory writes run serially in the background, to avoid blocking the realtime event loop and concurrent writes.
# Task references are kept so queued tasks can still complete after the session ends.
_REMEMBER_LOCK = asyncio.Lock()
_REMEMBER_TASKS: set[asyncio.Task] = set()


async def _remember_background(pending, reply: str, owner: dict,
                               history_turn_id: str, memory_vm) -> None:
    queued_at = time.monotonic()
    async with _REMEMBER_LOCK:
        waited = time.monotonic() - queued_at
        if waited > 0.05 and BARGE_DEBUG:
            print(f"[memory] ingest queued {waited:.2f}s", flush=True)
        started = time.monotonic()
        await asyncio.to_thread(
            remember_turn, pending, reply, owner, history_turn_id, memory_vm)
        if BARGE_DEBUG:
            print(f"[memory] ingest main pass {time.monotonic()-started:.2f}s", flush=True)


def queue_remember_turn(pending, reply: str, owner: dict,
                        history_turn_id: str = "", memory_vm=None) -> None:
    memory_vm = memory_vm or vm
    task = asyncio.create_task(
        _remember_background(pending, reply, owner, history_turn_id, memory_vm))
    _REMEMBER_TASKS.add(task)

    def done(t: asyncio.Task) -> None:
        _REMEMBER_TASKS.discard(t)
        try:
            t.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[web] background memory task failed: {type(e).__name__}: {e}", flush=True)

    task.add_done_callback(done)


#: How far back to compare. It used to be 40 chars -- estimated from "speaking wherever generation is at", but generation is
#: much faster than playback (especially now that the next segment is synthesized ahead of time); what the user hears right now was often generated
#: seconds earlier and has long slid out of a 40-char window. Now it compares against **text already spoken**, with a wider window.
ECHO_WINDOW = int(os.environ.get("SUPERMEM_ECHO_WINDOW", "300"))
#: Fuzzy-match threshold: among the new characters, the largest fraction a **contiguous** run matching the assistant's words may occupy.
#:
#: It started with bigram overlap rate -- which was wrong: it ignores contiguity, so as long as a user's barge-in overlaps in vocabulary with what
#: was just discussed ("stress", "GRE", very common), scattered bigrams clear the threshold and a real barge-in gets
#: swallowed as echo, so interruption stops working.
#: Echo is characterized by **a whole contiguous run of the original words**; a real barge-in, even with overlapping words, can't form a long run, so it now uses
#: the longest common substring. ASR being off by a char or two only shortens the run a bit, still far above a real barge-in.
ECHO_RATIO = float(os.environ.get("SUPERMEM_ECHO_RATIO", "0.6"))
#: Below this length only exact matching is done. Two or three chars give too few bigrams, and the overlap rate hits 1.0 all the time --
#: a user repeating a word back ("GRE?") would be swallowed as echo, and that's a real barge-in.
ECHO_FUZZY_MIN = int(os.environ.get("SUPERMEM_ECHO_FUZZY_MIN", "4"))


#: Backchannels: a listener's casual acknowledgement, not an attempt to take the floor.
#:
#: The barge-in criterion is "transcript grew by >= 2 chars", and short backchannels are exactly that long -- in testing
#: "yeah mm" and "right, right" both cut off the reply being played. Saying "mm" or "right" while listening is normal conversation,
#: and cutting off is what's unnatural. The realtime path relies on OpenAI's semantic_vad judging "is this really an
#: interruption"; llm_tts has none, so it can only block by word list.
#:
#: The criterion is **the entire new part is backchannel**: when saying "right, but what I mean is...", the new part
#: has more than "right", so it still interrupts. The only cost is not interrupting on a pure backchannel, which shouldn't interrupt anyway.
BACKCHANNEL_ON = os.environ.get("SUPERMEM_BACKCHANNEL", "1") != "0"
def _bc_norm(s: str) -> str:
    """Normalize: strip spaces and punctuation, keep only letters and digits. The word list and input go through the same function,
    so "got it" (with a space in the list) doesn't forever fail to match "gotit" (input with spaces stripped)."""
    return "".join(ch for ch in (s or "") if ch.isalnum()).casefold()


#: Greedy splitting in descending length order, so longer words like "yeah" are tried before "yes".
_BACKCHANNEL_RAW = {
    # English
    "uh-huh", "mhm", "mm-hmm", "yeah", "yep", "yes", "okay", "ok", "right",
    "sure", "gotcha", "got it", "i see", "cool", "nice", "wow", "hmm", "huh",
}
_BACKCHANNEL = sorted({_bc_norm(w) for w in _BACKCHANNEL_RAW}, key=len, reverse=True)


def _is_backchannel(new_chars: str) -> bool:
    """Are these new characters pure backchannel? Only counts if the whole thing can be consumed by the word list."""
    if not BACKCHANNEL_ON:
        return False
    s = _bc_norm(new_chars)
    if not s:
        return False
    while s:
        for w in _BACKCHANNEL:
            if s.startswith(w):
                s = s[len(w):]
                break
        else:
            return False
    return True


def _is_echo(new_chars: str, said: str) -> bool:
    """Are the characters ASR just emitted the assistant's own voice looping back into the mic?

    When AEC doesn't fully cancel, the assistant's speech enters ASR, and the transcribed characters are indistinguishable from a real barge-in by count.
    But content differs: echo is always a fragment of **the text the assistant just said**.

    ``said`` must be the **already spoken** text (not the generated text); the two can differ by several seconds.
    Case must be flattened -- the very first case that slipped through was the assistant saying "Annie" and ASR emitting lowercase "an".
    """
    s = "".join(ch for ch in (new_chars or "") if ch.strip()).casefold()
    hay = "".join(ch for ch in (said or "")[-ECHO_WINDOW:] if ch.strip()).casefold()
    if not s or not hay:
        return False
    if s in hay:
        return True
    if len(s) < ECHO_FUZZY_MIN:
        return False                       # too short, trust only exact matches
    return _lcs_len(s, hay) / len(s) >= ECHO_RATIO


_INTERRUPT_PREFIXES = tuple(_bc_norm(x) for x in (
    "stop", "wait", "hold on", "hang on", "pause", "quiet",
))
_FILLER_PREFIX = re.compile(r"^(?:u+m+|u+h+|h+m+)+")


def _barge_text(text: str) -> str:
    return _FILLER_PREFIX.sub("", _bc_norm(text))


def _is_explicit_interrupt(text: str) -> bool:
    clean = _barge_text(text)
    return bool(clean) and any(clean.startswith(prefix) for prefix in _INTERRUPT_PREFIXES)


def _has_barge_content(text: str) -> bool:
    """Filter out single syllables and pure filler words; only used to confirm candidates, never triggers cancellation directly."""
    clean = _barge_text(text)
    if not clean or len(set(clean)) == 1 or _is_backchannel(text):
        return False
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in clean)
    latin = sum(ch.isascii() and ch.isalnum() for ch in clean)
    return cjk >= BARGE_MIN_CHARS or latin >= max(3, BARGE_MIN_CHARS)


def _has_strong_final_barge(text: str) -> bool:
    clean = _barge_text(text)
    cjk = sum("\u4e00" <= ch <= "\u9fff" for ch in clean)
    latin = sum(ch.isascii() and ch.isalnum() for ch in clean)
    return cjk >= 3 or latin >= 5


def _lcs_len(a: str, b: str) -> int:
    """Length of the longest common **substring** (contiguous). Rolling one-row DP; strings are short, cost is negligible."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for ch in a:
        cur = [0] * (len(b) + 1)
        for j, cj in enumerate(b, 1):
            if ch == cj:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


async def anticipate(sock, on_frame=None, on_speech=None, owner=None, is_busy=None,
                     said=None, on_candidate=None, on_candidate_reject=None,
                     on_playback_checkpoint=None):
    """Drive the core streaming session, yielding a Pending for each confirmed turn.
    on_frame(raw24k): realtime uses it to feed raw audio to OpenAI in parallel (plan A).
    on_speech(): called once as soon as local VAD hears voice -- realtime uses it for barge-in.
    is_busy(): whether the assistant is speaking right now. The assistant's voice comes back through the mic into ASR, and the transcribed characters
    still go out as partial_transcript -- the user sees what the assistant just said popping up in their own input box.
    So while the assistant is speaking, hold partials back until the transcript really grows by a few characters (confirming a person is cutting in,
    see BARGE_MIN_CHARS), then let them through.
    on_candidate()/on_candidate_reject(): recoverably pause/resume playback on a suspected barge-in; only
    on_speech() is a confirmed interruption."""
    stream = vm.stream(spec_min_chars=SPEC_MIN_CHARS, gamble_s=GAMBLE_S, confirm_s=CONFIRM_S)
    last_partial = ""
    if owner is None:
        owner = {"id": "", "last": "", "miss": 0}   # owner's voiceprint / who spoke last turn / how many consecutive misidentified turns
    barge_base = 0                        # transcript length when a barge-in last fired
    barged = False                        # whether this turn has been confirmed as "a person cutting in"
    candidate = False                     # playback paused, waiting for more evidence
    candidate_updates = 0
    candidate_text = ""
    candidate_silence = 0.0
    candidate_age = 0.0
    discard_candidate_turn = False
    while True:
        msg = await sock.receive()
        if msg.get("type") == "websocket.disconnect":         # page closed/refreshed: wrap up
            return
        if msg.get("text"):                                   # typed turn
            data = json.loads(msg["text"])
            if data.get("type") == "playback_checkpoint":
                if on_playback_checkpoint:
                    await on_playback_checkpoint(data)
                continue
            if data.get("type") == "user_text" and data.get("text", "").strip():
                stream.emotion = owner.get("emotion") or None
                turn = await stream.feed_text(data["text"])
                yield Pending(turn.text, turn.memory_context, turn.result, spoken=False,
                              replay=_replay_id(turn.text, turn.result),
                              emotion=owner.get("emotion", ""))
            continue
        if msg.get("bytes") is None:
            continue
        raw = msg["bytes"]
        if on_frame:
            await on_frame(raw)                               # plan A: audio also goes into the OpenAI buffer
        stream.emotion = owner.get("emotion") or None         # emotion computed in the previous turn
        st = await stream.feed(raw)                           # core: ASR + VAD + speculative prefetch
        cur = st.text.strip()
        busy = bool(is_busy and is_busy())
        frame_s = len(raw) / 2 / MIC_RATE

        # VAD first triggers a recoverable pause; subsequent ASR text is used to confirm whether it's a real interruption.
        if busy and st.state == "<speak>" and not candidate and not barged:
            candidate = True
            discard_candidate_turn = False
            candidate_updates = 0
            candidate_text = ""
            candidate_silence = 0.0
            candidate_age = 0.0
            if BARGE_DEBUG:
                print("[barge] possible interruption -> pausing playback, waiting for ASR", flush=True)
            if on_candidate:
                await on_candidate()

        if candidate and not barged:
            candidate_age += frame_s
            candidate_silence = (candidate_silence + frame_s
                                 if st.state == "<silence>" else 0.0)
            looks_echo = bool(said is not None and cur and _is_echo(cur, said()))
            if cur and not looks_echo and not _is_backchannel(cur) and _has_barge_content(cur):
                normalized = _barge_text(cur)
                if normalized != candidate_text:
                    candidate_text = normalized
                    candidate_updates += 1

            confirmed = (_is_explicit_interrupt(cur) and not looks_echo)
            confirmed = confirmed or candidate_updates >= BARGE_STABLE_UPDATES
            if confirmed:
                candidate = False
                barged = True
                barge_base = len(cur)
                if BARGE_DEBUG:
                    why = "explicit stop command" if _is_explicit_interrupt(cur) else "transcript grew steadily"
                    print(f"[barge] {why} -> interruption confirmed: {cur[-16:]!r}", flush=True)
                if on_speech:
                    await on_speech()
            elif ((candidate_silence * 1000 >= BARGE_REJECT_SILENCE_MS and not cur)
                  or (candidate_age * 1000 >= BARGE_CANDIDATE_TIMEOUT_MS
                      and candidate_updates == 0)):
                candidate = False
                discard_candidate_turn = True
                if BARGE_DEBUG:
                    print("[barge] that sound produced no text -> resuming playback", flush=True)
                if on_candidate_reject:
                    await on_candidate_reject()
        # While the assistant is speaking, ASR is likely mixed with its own echo; those words must not be shown -- the user would see
        # what the assistant just said popping up in their own input box.
        #
        # But the old condition was "busy and barge-in not yet confirmed", i.e. **block everything whenever the assistant is busy**, until
        # enough chars accumulated, the echo check passed, and `barged` flipped true. The user was long done and the screen was still empty.
        # Now _is_echo is accurate enough (contiguous substring comparison), so use it directly to judge whether this text looks like echo:
        # if it does, hide it; if not, show it immediately without waiting for that confirmation.
        echo = (bool(is_busy and is_busy()) and not barged
                and (said is None or _is_echo(cur, said())))
        if st.text.strip() and st.text != last_partial and not echo:
            last_partial = st.text
            await sock.send_json({"type": "partial_transcript", "text": st.text, "replace": True})
        if st.turn:                                           # VAD confirmed end of speech -> memory already prefetched
            if discard_candidate_turn:
                discard_candidate_turn = False
                last_partial = ""
                barge_base = 0
                if BARGE_DEBUG:
                    print(f"[barge] dropping unconfirmed sound-only turn: {st.turn.text!r}", flush=True)
                continue

            if candidate and not barged:
                final_text = st.turn.text.strip()
                looks_echo = bool(said is not None and final_text
                                  and _is_echo(final_text, said()))
                confirmed = (
                    not looks_echo
                    and not _is_backchannel(final_text)
                    and (_is_explicit_interrupt(final_text)
                         or candidate_updates >= BARGE_STABLE_UPDATES
                         or (_has_barge_content(final_text)
                             and _has_strong_final_barge(final_text)))
                )
                candidate = False
                if confirmed:
                    barged = True
                    if BARGE_DEBUG:
                        print(f"[barge] full turn confirms interruption: {final_text!r}", flush=True)
                    if on_speech:
                        await on_speech()
                else:
                    if BARGE_DEBUG:
                        print(f"[barge] full turn judged backchannel/echo/noise -> resuming: {final_text!r}",
                              flush=True)
                    if on_candidate_reject:
                        await on_candidate_reject()
                    last_partial = ""
                    barge_base = 0
                    candidate_updates = 0
                    candidate_text = ""
                    candidate_silence = 0.0
                    candidate_age = 0.0
                    continue

            # We're playing a recording and this turn transcribed nothing: that's our own sound looping back.
            if _replaying_now() and not (st.turn.text or "").strip():
                if BARGE_DEBUG:
                    print("[replay] empty turn during replay, it's our own echo, dropping", flush=True)
                last_partial = ""
                barge_base = 0
                barged = False
                candidate = False
                continue
            last_partial = ""
            barge_base = 0                                    # new turn, transcript grows from scratch
            barged = False
            candidate = False
            candidate_updates = 0
            candidate_text = ""
            candidate_silence = 0.0
            candidate_age = 0.0
            # Who's speaking. The first person to speak counts as this conversation's owner; if another voiceprint shows up later,
            # that's a stranger -- the owner's memories must not be told to them ("Who am I?" -> "You're Jiaqi"
            # was exactly the bug from retrieval never looking at the speaker).
            # Decide using the speaker computed in the previous turn; this turn's is computed in the background.
            # Reading st.speaker_id directly would synchronously run the whole voiceprint+emotion+scene model set -- measured 2.1 s,
            # and inside the event loop, during which the socket isn't even read and mic frames pile up,
            # showing up as "ASR is laggy". The cost is that after a speaker change, the first sentence still counts as the previous person.
            stranger = bool(SPEAKER_GATE and owner["id"]
                            and owner.get("miss", 0) >= STRANGER_MIN_TURNS)
            if stranger or SPEAKER_DEBUG:
                # "Why did it suddenly stop recognizing me?" -- look at this line. It happens when the voiceprint splits one person into two
                # person_* ids: memory gets cleared and the instruction switches to "treat it as a first meeting".
                print(f"[speaker] owner={owner['id'] or '-'} last={owner['last'] or '-'}"
                      f" miss={owner.get('miss', 0)} stranger={stranger}", flush=True)
            yield Pending(st.turn.text,
                          "" if stranger else st.turn.memory_context,
                          st.turn.result, spoken=True,
                          audio_path=await asyncio.to_thread(
                              save_turn_audio, getattr(st, "_pcm", None)),
                          stranger=stranger,
                          replay="" if stranger else _replay_id(st.turn.text, st.turn.result),
                          emotion=owner.get("emotion", ""))



# ══════════════════ Session loop for each mode ══════════════════

async def _session_anticipate(session_id: str, sock, on_close=None, **kwargs):
    try:
        async for pending in anticipate(sock, **kwargs):
            yield pending
    finally:
        if on_close:
            await on_close()
        _SESSION_CONTEXT.clear_session(session_id)


async def llm_tts_session(sock):
    """Barge-in for the llm_tts path.

    It used to be `async for pending in anticipate(sock): await supermem_llm_tts(...)` --
    two problems stacked up, making interruption structurally impossible:
      · on_speech wasn't passed into anticipate, so nobody acted when local VAD heard voice;
      · supermem_llm_tts ends with `await speaker`, returning only after all audio is sent. Meanwhile async for
        doesn't pull the next item, so anticipate stalls and stops reading the socket, and mic frames all
        pile up in the buffer. It felt like "nothing you say matters, it insists on finishing".

    Now the reply runs in a background task and the socket-reading loop never stops; hearing voice cancels that task.
    """
    # until: when the frontend is expected to finish playing the audio already sent (see hearing()).
    turn = {"task": None, "t0": 0.0, "until": 0.0,
            "reply": {"text": ""}, "timeline": None}
    owner = {"id": "", "last": "", "miss": 0}
    speech_rate = SpeechRateEstimator()
    context_session = uuid.uuid4().hex
    candidate_paused = False
    candidate_paused_at = 0.0

    async def pause_candidate():
        nonlocal candidate_paused, candidate_paused_at
        if hearing() and not candidate_paused:
            candidate_paused = True
            candidate_paused_at = time.monotonic()
            await sock.send_json({"type": "answer_pause"})

    async def resume_candidate():
        nonlocal candidate_paused, candidate_paused_at
        if candidate_paused:
            candidate_paused = False
            if turn["until"]:
                turn["until"] += max(0.0, time.monotonic() - candidate_paused_at)
            candidate_paused_at = 0.0
            await sock.send_json({"type": "answer_resume"})

    def hearing() -> bool:
        """Can the user still hear the assistant right now?

        Can't be replaced by `task.done()`: send_audio is **not rate-limited**; it pushes into the socket as fast as TTS produces,
        so a reply of a dozen-odd seconds is pushed in two or three. The task is long done while the frontend
        is still playing the remaining dozen seconds -- a barge-in during that time made the old stop_reply return immediately,
        the frontend never got answer_interrupt, and that's exactly "interrupting does nothing, it insists on finishing".
        So compute from **the duration of audio already sent**, consistent with the realtime path: 24k PCM16,
        2 bytes per sample.
        """
        t = turn["task"]
        timeline = turn["timeline"]
        task_active = (t is not None and not t.done()
                       and not (timeline and timeline.playback_done))
        return (candidate_paused or task_active
                or time.monotonic() < turn["until"])

    async def playback_checkpoint(data):
        timeline = turn["timeline"]
        if timeline is None or data.get("output_id") != timeline.output_id:
            return
        timeline.update_checkpoint(
            data.get("rendered_samples", 0), data.get("sample_rate", MIC_RATE),
            data.get("state", "playing"))
        if timeline.playback_done:
            turn["until"] = 0.0

    async def send_audio(pcm: bytes):
        """Send audio and keep the books. The frontend plays from a queue (pcm-player-worklet); here we track
        the same timeline: this chunk follows once the previous one finishes."""
        turn["until"] = max(turn["until"], time.monotonic()) + len(pcm) / 2 / MIC_RATE
        if not turn["t0"]:
            # The grace period starts from the moment **the assistant actually makes sound**, not from task creation. The first TTS frame takes
            # ~1.2s; counted from task creation, the grace period would be over before it even spoke, i.e. useless.
            turn["t0"] = time.monotonic()
        await sock.send_bytes(pcm)

    async def stop_reply(force: bool = False):
        nonlocal candidate_paused, candidate_paused_at
        if not hearing():
            turn["task"] = None
            return
        since = (time.monotonic() - turn["t0"]) * 1000 if turn["t0"] else 0.0
        if not force and turn["t0"] and since < BARGE_GRACE_MS:
            if BARGE_DEBUG:
                print(f"[barge] only {since:.0f}ms spoken, still within grace period, not interrupting", flush=True)
            return
        if BARGE_DEBUG:
            left = max(0.0, turn["until"] - time.monotonic()) * 1000
            print(f"[barge] ★ interrupt: triggered by transcript (frontend still has {left:.0f}ms unplayed)", flush=True)
        task = turn["task"]
        timeline = turn["timeline"]
        heard_text = timeline.heard_text() if timeline else ""
        output_id = timeline.output_id if timeline else ""
        if timeline:
            timeline.mark_interrupted()
        if task is not None and not task.done():
            task.cancel()
        turn["task"], turn["until"] = None, 0.0
        candidate_paused = False
        candidate_paused_at = 0.0
        try:
            await sock.send_json({"type": "answer_interrupt",
                                  "output_id": output_id,
                                  "heard_text": heard_text})
        except Exception:
            pass
        if task is not None and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def close_session():
        task = turn["task"]
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[web] reply wrap-up failed: {type(e).__name__}: {e}", flush=True)

    async for pending in _session_anticipate(
            context_session, sock, on_speech=stop_reply, owner=owner,
            is_busy=hearing, said=lambda: turn["reply"]["text"],
            on_candidate=pause_candidate, on_candidate_reject=resume_candidate,
            on_playback_checkpoint=playback_checkpoint, on_close=close_session):
        # The whole turn is backchannel while the assistant is still speaking: treat it as unheard.
        #
        # _is_backchannel used to guard only the "mid-sentence" path (in anticipate), but when VAD finishes a whole turn it takes
        # **another** path -- the stop_reply(force=True) below. So just saying "mm", VAD considers a turn finished,
        # and the new turn cut off the playing reply in the name of "replacing the old turn"; the backchannel list was never
        # consulted. That's exactly how it cut off in the measured logs.
        # An "mm" while the assistant isn't speaking still goes through as a normal turn.
        if _is_backchannel(pending.text) and hearing():
            if BARGE_DEBUG:
                print(f"[barge] whole turn is backchannel {pending.text!r}, not a turn, keep talking", flush=True)
            continue
        # The previous turn is replaced by a new one before it finished playing. force: this must not be blocked by the grace period; if blocked,
        # the old task keeps pouring audio into the same socket and the two turns play interleaved.
        await stop_reply(force=True)
        turn["t0"] = turn["until"] = 0.0
        turn["reply"] = {"text": ""}            # new turn, echo comparison starts empty
        reply_state = turn["reply"]
        context_space = ACTIVE_SPACE
        memory_vm = vm
        timeline = AudioTimeline(
            prebuffer_seconds=0.16, rate_estimator=speech_rate)
        turn["timeline"] = timeline

        async def run_reply(pending=pending, timeline=timeline,
                            context_space=context_space, memory_vm=memory_vm,
                            reply_state=reply_state):
            try:
                await supermem_llm_tts(
                    pending, sock.send_json, send_audio, owner, timeline,
                    said=reply_state, context_session=context_session,
                    context_space=context_space, memory_vm=memory_vm)
            finally:
                if not timeline.context_saved:
                    reply = timeline.heard_text()
                    history_turn_id = _push_history(
                        context_session, context_space, pending.text, reply,
                        interrupted=True)
                    queue_remember_turn(
                        pending, reply, owner, history_turn_id,
                        memory_vm=memory_vm)
                    timeline.context_saved = True

        task = asyncio.create_task(run_reply())
        turn["task"] = task

        def reply_done(done_task):
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[web] reply task failed: {type(e).__name__}: {e}", flush=True)

        task.add_done_callback(reply_done)


async def realtime_session(sock):
    """Plan A: the whole mic audio is fed to OpenAI Realtime in parallel; local ASR+VAD only handles speculative memory +
    judging turns at 500ms (OpenAI's built-in server_vad is turned off)."""
    connected = False
    context_session = uuid.uuid4().hex
    try:
        async with utils.realtime_connect(REPLY) as conn:
            # A successful handshake doesn't mean it works: without permission/with a wrong model name, OpenAI sends
            # close 4000 (invalid_model / permission error) **after connecting**. So it only counts once the first interaction succeeds.
            # turn_detection only borrows OpenAI's VAD for **interruption**, not to take over turns:
            #   create_response=False    -> we still decide when to reply (only after local VAD finishes
            #                              a turn and memory is prefetched do we response.create)
            #   interrupt_response=True  -> as soon as the user speaks, the server cuts off the reply being played
            # Interruption detection sits on the OpenAI side because it works directly on the audio stream; local VAD has to wait
            # for mic frames to pass through the whole chain (browser AEC -> ws -> resample -> silero), and a real person
            # cutting in over the speaker already has a weak signal, easily missed.
            # turn_detection lives under session.audio.input, **not at the top level** -- written at the top level
            # it's silently rejected ("Unknown parameter: session.turn_detection", only coming back as an error
            # event; if nobody looks, you think it's set). The old {"turn_detection": None} never
            # took effect, so server_vad stayed on and eagerly auto-replied: the response it generated didn't carry
            # our injected memory, and our own response.create then failed colliding with "a response is already running"
            # -- that's how voice turns "didn't use memory".
            #   create_response=False    -> we decide when to reply (only after local VAD finishes a turn
            #                              and memory is prefetched do we response.create)
            #   interrupt_response=True  -> as soon as the user speaks, the server cuts off the reply being played;
            #                              this detection runs on the OpenAI side directly on the audio stream, more
            #                              reliable than local VAD (behind AEC + network + resampling)
            await conn.session.update(session={
                "type": "realtime",
                "audio": {
                    "input": {"turn_detection": _turn_detection()},
                    "output": {"voice": utils.RT_VOICE},
                },
            })
            connected = True

            # This turn's state: who's speaking, what was said, what the user's input was this turn.
            # until: when the frontend is expected to finish playing the audio already sent (see hearing()).
            turn = {"live": False, "reply": "", "pending": None,
                    "t0": 0.0, "until": 0.0, "first": False,
                    "timeline": None, "response_done": False,
                    "provider_item_id": "", "space": "", "memory_vm": None}
            owner = {"id": "", "last": "", "miss": 0}
            speech_rate = SpeechRateEstimator()
            timelines: dict[str, AudioTimeline] = {}
            playback_tasks: set[asyncio.Task] = set()
            candidate_paused = False
            candidate_paused_at = 0.0
            # Realtime allows only one response at a time; the next turn can be created only after cancellation completes.
            response_idle = asyncio.Event()
            response_idle.set()

            def hearing() -> bool:
                """Can the user still hear the assistant right now?

                Can't be replaced by turn["live"]: realtime pushes audio much faster than real-time playback; a reply of
                a dozen-odd seconds is pushed in two or three, and live turns False as soon as response.done arrives,
                while the frontend is still playing the remaining dozen seconds. If the user cuts in then, on_speech sees
                live=False and "ignores" it, and the frontend never gets answer_interrupt -- exactly
                "interrupting does nothing, it insists on finishing".
                So compute from **the duration of audio already sent**: 24k PCM16, 2 bytes per sample.
                """
                timeline = turn["timeline"]
                buffered = bool(timeline and not timeline.playback_done
                                and time.monotonic() < turn["until"])
                return (candidate_paused or turn["live"] or buffered
                        or time.monotonic() < turn["until"])

            async def pause_candidate():
                nonlocal candidate_paused, candidate_paused_at
                if hearing() and not candidate_paused:
                    candidate_paused = True
                    candidate_paused_at = time.monotonic()
                    await sock.send_json({"type": "answer_pause"})

            async def resume_candidate():
                nonlocal candidate_paused, candidate_paused_at
                if candidate_paused:
                    candidate_paused = False
                    if turn["until"]:
                        turn["until"] += max(0.0, time.monotonic() - candidate_paused_at)
                    candidate_paused_at = 0.0
                    await sock.send_json({"type": "answer_resume"})

            def close_turn(interrupted=False):
                p, reply = turn["pending"], turn["reply"]
                timeline = turn["timeline"]
                space = turn["space"] or ACTIVE_SPACE
                memory_vm = turn["memory_vm"] or vm
                turn.update(live=False, reply="", pending=None, timeline=None,
                            response_done=False, provider_item_id="",
                            space="", memory_vm=None)
                if p is None:
                    return
                if interrupted:
                    reply = timeline.heard_text() if timeline else ""
                    if BARGE_DEBUG and timeline:
                        print(f"[context] interrupted at {timeline.rendered_ms()}ms, keeping reply "
                              f"{reply!r}", flush=True)
                history_turn_id = _push_history(
                    context_session, space, p.text, reply,
                    interrupted=interrupted)
                queue_remember_turn(
                    p, reply, owner, history_turn_id, memory_vm=memory_vm)
                if timeline:
                    timelines.pop(timeline.output_id, None)

            async def playback_checkpoint(data):
                timeline = timelines.get(str(data.get("output_id") or ""))
                if timeline is None:
                    return
                timeline.update_checkpoint(
                    data.get("rendered_samples", 0),
                    data.get("sample_rate", MIC_RATE),
                    data.get("state", "playing"))
                if turn["timeline"] is timeline and timeline.playback_done:
                    turn["until"] = 0.0
                    if turn["response_done"]:
                        close_turn(interrupted=False)

            async def playback_fallback(timeline):
                delay = max(0.0, turn["until"] - time.monotonic()) + 0.5
                await asyncio.sleep(delay)
                if turn["timeline"] is timeline and turn["response_done"]:
                    timeline.assume_drained()
                    turn["until"] = 0.0
                    close_turn(interrupted=False)

            def schedule_playback_fallback(timeline):
                task = asyncio.create_task(playback_fallback(timeline))
                playback_tasks.add(task)
                task.add_done_callback(playback_tasks.discard)

            async def pump():
                """Resident event pump: OpenAI's event stream has only this one consumer.

                Nothing is forwarded to the frontend while turn["live"] is false -- after an interruption OpenAI keeps emitting
                leftover audio for a while; forwarding it would queue new audio right after the frontend's stopPlayback, making the interruption unclean.
                """
                async for ev in conn:
                    t = getattr(ev, "type", "")
                    if t.endswith("output_audio.delta"):
                        if turn["live"]:
                            if not turn["first"]:
                                # First-frame audio latency: time from sending response.create to OpenAI emitting the first
                                # chunk of sound. This is the half of "it reacts slowly" that we can't control.
                                turn["first"] = True
                                print(f"[lat] realtime first frame "
                                      f"{(time.monotonic()-turn['t0'])*1000:.0f}ms", flush=True)
                            pcm = base64.b64decode(ev.delta)
                            timeline = turn["timeline"]
                            if timeline:
                                timeline.append_audio(pcm)
                                timestamps = getattr(ev, "timestamps", ()) or ()
                                if timestamps:
                                    timeline.add_timestamps(tuple(timestamps))
                            turn["provider_item_id"] = (
                                getattr(ev, "item_id", "") or turn["provider_item_id"])
                            # The frontend plays from a queue (nextPlay in index.html); here we track
                            # the same timeline: this chunk follows once the previous one finishes.
                            turn["until"] = (max(turn["until"], time.monotonic())
                                             + len(pcm) / 2 / 24000)
                            await sock.send_bytes(pcm)
                    elif t.endswith("output_audio_transcript.delta"):
                        if turn["live"]:
                            turn["reply"] += ev.delta
                            timeline = turn["timeline"]
                            if timeline:
                                timeline.append_text(ev.delta)
                                timestamps = getattr(ev, "timestamps", ()) or ()
                                if timestamps:
                                    timeline.add_timestamps(tuple(timestamps))
                            turn["provider_item_id"] = (
                                getattr(ev, "item_id", "") or turn["provider_item_id"])
                            await sock.send_json({"type": "answer_delta", "text": ev.delta})
                    elif t == "error" or t.endswith(".error"):
                        err = getattr(ev, "error", None)
                        code = getattr(err, "code", "")
                        # server_vad commits the audio buffer itself after finishing a sentence, so our subsequent
                        # commit hits an empty buffer. Both cases must keep the manual commit (when speech is too short
                        # server_vad doesn't auto-commit), so this one is expected; ignore it.
                        # response_cancel_not_active: interruption has two paths (local VAD's
                        # on_speech + server_vad's interrupt_response) backing each other up;
                        # whichever arrives first wins, and the slower one missing is normal.
                        if code not in (
                                "input_audio_buffer_commit_empty",
                                "response_cancel_not_active"):
                            print(f"[web] realtime event error: {err or ev}", flush=True)
                    elif t.endswith("input_audio_buffer.speech_stopped"):
                        # The moment OpenAI decides "you're done". Compared with local silero finishing (the moment we send
                        # response.create), whichever is earlier -- the earlier one is the
                        # real EOU lower bound, and the later part is wasted waiting.
                        turn["stopped"] = time.monotonic()
                        if BARGE_DEBUG:
                            print("[lat] OpenAI decided the speaker finished", flush=True)
                    elif t.endswith("input_audio_buffer.speech_started"):
                        # OpenAI's VAD heard voice: its side has already cut the reply, we wrap up in sync
                        since = (time.monotonic() - turn["t0"]) * 1000
                        if BARGE_DEBUG:
                            print(f"[barge] OpenAI VAD heard speech (live={turn['live']}, "
                                  f"still_playing={hearing()}, {since:.0f}ms)", flush=True)
                        # Grace period, same as the local path. Local VAD sends response.create as soon as a turn ends (500ms silence),
                        # while OpenAI's server_vad considers the next utterance started after just 320ms of silence --
                        # the tail of the user's speech, breathing, ambient noise are all enough to trigger it.
                        # Without this gate, speech_started would arrive before the assistant makes a sound,
                        # and the reply gets cut before the first audio chunk: not a word is heard, and no error is reported.
                        # Only logged, no longer used to interrupt -- see the notes in _turn_detection.
                        # This log line is kept because it's the most direct evidence of "was the echo cancelled cleanly":
                        # frequent occurrences while the assistant speaks mean AEC has residue.
                    elif t.endswith("response.done") or t.endswith("response.cancelled"):
                        # done/cancelled events release the barrier for creating the next turn's response.
                        response_idle.set()
                        if BARGE_DEBUG and not turn["live"]:
                            print("[barge] old Realtime response has exited", flush=True)
                        if turn["live"]:
                            timeline = turn["timeline"]
                            if timeline:
                                timeline.mark_generation_complete()
                            if t.endswith("response.cancelled"):
                                response_idle.clear()
                                heard = timeline.heard_text() if timeline else ""
                                provider_item_id = turn["provider_item_id"]
                                if timeline:
                                    timeline.mark_interrupted()
                                await sock.send_json({
                                    "type": "answer_interrupt",
                                    "output_id": timeline.output_id if timeline else "",
                                    "heard_text": heard,
                                })
                                close_turn(interrupted=True)
                                if provider_item_id and timeline:
                                    try:
                                        await truncate_provider_output(
                                            conn, provider_item_id, timeline)
                                    except Exception as e:
                                        if BARGE_DEBUG:
                                            print(f"[barge] provider context truncation failed: {e}",
                                                  flush=True)
                                response_idle.set()
                            else:
                                turn["live"] = False
                                turn["response_done"] = True
                                await sock.send_json({
                                    "type": "answer_done",
                                    "output_id": timeline.output_id if timeline else "",
                                })
                                if timeline and timeline.playback_done:
                                    close_turn(interrupted=False)
                                elif timeline:
                                    schedule_playback_fallback(timeline)

            async def on_frame(raw):
                await conn.input_audio_buffer.append(audio=base64.b64encode(raw).decode())

            async def on_speech():
                """User speaks while the assistant is talking -> interrupt. Idempotent: stops firing once hearing() turns false."""
                nonlocal candidate_paused, candidate_paused_at
                if not hearing():
                    if BARGE_DEBUG:
                        print("[barge] voice detected but the assistant isn't speaking, ignoring", flush=True)
                    return
                # The first moments after it starts speaking can't be interrupted: the mic hears almost only the assistant's own voice then,
                # echo cancellation hasn't caught up, and it easily cuts itself off as soon as it speaks.
                since = (time.monotonic() - turn["t0"]) * 1000
                if since < BARGE_GRACE_MS:
                    if BARGE_DEBUG:
                        print(f"[barge] only {since:.0f}ms spoken, still within grace period, not interrupting", flush=True)
                    return
                if BARGE_DEBUG:
                    left = max(0.0, turn["until"] - time.monotonic()) * 1000
                    print(f"[barge] ★ interrupt: triggered by transcript (frontend still has {left:.0f}ms unplayed)",
                          flush=True)
                active_response = not response_idle.is_set()
                timeline = turn["timeline"]
                heard_text = timeline.heard_text() if timeline else ""
                output_id = timeline.output_id if timeline else ""
                provider_item_id = turn["provider_item_id"]
                if timeline:
                    timeline.mark_interrupted()
                turn["live"], turn["until"] = False, 0.0
                candidate_paused = False
                candidate_paused_at = 0.0
                # Tell the frontend to stop playback first -- that's a local operation and takes effect immediately; response.cancel() has to
                # wait for an OpenAI round trip. The order used to be reversed, so after cutting in you still heard that round trip's
                # worth of audio, which felt like "interrupting is useless, it insists on finishing".
                await sock.send_json({"type": "answer_interrupt",
                                      "output_id": output_id,
                                      "heard_text": heard_text})
                close_turn(interrupted=True)
                if active_response:
                    await conn.response.cancel()
                if provider_item_id and timeline:
                    try:
                        await truncate_provider_output(
                            conn, provider_item_id, timeline)
                    except Exception as e:
                        if BARGE_DEBUG:
                            print(f"[barge] provider context truncation failed: {e}", flush=True)
                if active_response:
                    try:
                        await asyncio.wait_for(response_idle.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        # Local playback has already stopped; the next turn
                        # keeps waiting for the done/cancelled event.
                        print("[barge] timed out waiting for Realtime to confirm "
                              "the cancel; holding off on the next turn", flush=True)

            pump_task = asyncio.create_task(pump())
            try:
                async for pending in anticipate(sock, on_frame=on_frame,
                                                on_speech=on_speech, owner=owner,
                                                said=lambda: turn["reply"],
                                                on_candidate=pause_candidate,
                                                on_candidate_reject=resume_candidate,
                                                on_playback_checkpoint=playback_checkpoint):
                    # Same as llm_tts: if the whole turn is backchannel while the assistant is still speaking, treat it as unheard.
                    if _is_backchannel(pending.text) and hearing():
                        if BARGE_DEBUG:
                            print(f"[barge] whole turn is backchannel {pending.text!r}, not a turn, keep talking",
                                  flush=True)
                        continue
                    if hearing():                        # previous turn replaced by a new one before it finished playing
                        await on_speech()
                    if not response_idle.is_set():
                        if BARGE_DEBUG:
                            print("[barge] waiting for the old response to exit before creating the next turn", flush=True)
                        await response_idle.wait()
                    if COMPARE.enabled:
                        # realtime is **one** audio stream and cannot carry two
                        # panels, so this turn goes through the
                        # chat-completions fan-out instead (silent).
                        # The mic audio in the buffer must be cleared: the
                        # realtime connection is long-lived, and audio that is
                        # neither committed nor cleared keeps piling up -- the
                        # next commit after compare is switched off would speak
                        # several earlier turns at once.
                        await conn.input_audio_buffer.clear()
                        await _announce_turn(pending, sock.send_json)
                        await _compare_turn(pending, sock.send_json, owner,
                                            context_session=context_session,
                                            context_space=ACTIVE_SPACE)
                        continue
                    context_space = ACTIVE_SPACE
                    memory_vm = vm
                    timeline = AudioTimeline(
                        prebuffer_seconds=0.08, rate_estimator=speech_rate)
                    timelines[timeline.output_id] = timeline
                    turn.update(
                        live=True, reply="", pending=pending,
                        t0=time.monotonic(), until=0.0, first=False,
                        timeline=timeline, response_done=False,
                        provider_item_id="", space=context_space,
                        memory_vm=memory_vm)
                    response_idle.clear()
                    try:
                        await start_realtime_turn(
                            pending, conn, sock.send_json, timeline,
                            context_session=context_session,
                            context_space=context_space)
                    except Exception:
                        response_idle.set()
                        raise
            finally:
                pump_task.cancel()
                try:
                    await pump_task
                except asyncio.CancelledError:
                    pass
                if turn["pending"] is not None:
                    timeline = turn["timeline"]
                    fully_played = bool(
                        turn["response_done"] and timeline and timeline.playback_done)
                    close_turn(interrupted=not fully_played)
                for task in playback_tasks:
                    task.cancel()
                if playback_tasks:
                    await asyncio.gather(
                        *list(playback_tasks), return_exceptions=True)
    except Exception as e:
        if connected:
            raise
        await _no_realtime(sock, e)
    finally:
        _SESSION_CONTEXT.clear_session(context_session)


#: Right-brain slot -> the three brain-map clusters.
#:
#: The right brain's real classification unit is its 5 slots (emotion / likes_dislikes / coping_style /
#: expression_style / thinking_pattern), not memory_class -- that only has heartnote /
#: response_experience, which can't tell anything apart. The brain map has only three clusters, so here the slots
#: are collapsed into 3.
#:
#: In retrieved hit content, trait hits carry the slot at the end ("<claim> (likes_dislikes) | he said: ...");
#: heartnotes are individual emotional moments ("Emotional note: ... (inner note: ...)") and go to emotion.
#: Keys are matched against the lowercased text with spaces turned into underscores.
SLOT_TO_CLUSTER = {
    "emotion":          "emotion",
    "emotional_note":   "emotion",
    "inner_note":       "emotion",
    "likes_dislikes":   "preference",
    "thinking_pattern": "preference",
    "coping_style":     "experiences",
    "expression_style": "experiences",
    "avoid_repeating":  "experiences",
}
_CALM = ("", "calm", "neutral")


def rb_cluster(content: str, memory_class: str = "", emotion: str = "") -> str:
    """Which brain-map cluster a right-brain memory belongs to. 0 LLM, only looks at the slot name."""
    # First strip a date prefix like "[2026-06-20] ", otherwise it fills the small window we compare against,
    # and a heartnote's "Emotional note" falls outside the window.
    text = re.sub(r"^\s*\[[0-9-]{6,12}\]\s*", "", content or "")
    low = text.lower().replace(" ", "_")
    # Trait hits: "<claim> (<slot>)", possibly followed by " | he said: ...".
    for slot, cluster in SLOT_TO_CLUSTER.items():
        if f"({slot})" in low:
            return cluster
    head = low[:40]
    found = [(head.find(slot), cluster) for slot, cluster in SLOT_TO_CLUSTER.items() if slot in head]
    if found:
        return min(found)[1]
    if str(memory_class) == "response_experience":
        return "experiences"
    if emotion not in _CALM:
        return "emotion"
    return "experiences"
    if emotion not in _CALM:
        return "emotion"
    return "experiences"


def audio_of(memory_id: str) -> str:
    """Where this memory's original audio from back then is; returns "" if never archived, or cleared after the retention period.

    Goes through the core's GetOriginalAudio -- it already checks "does the file still exist", no need to rewrite that here.
    ``LAST_TUNE_ID`` is the exception: it refers to the clip just heard that hasn't been stored yet (see _LAST_TUNE).
    """
    if memory_id == LAST_TUNE_ID:
        return _last_tune_path()
    if memory_id.startswith(GROUP_ID_PREFIX):     # a group of clips, stitch a complete one on the fly
        return _stitch([m for m in memory_id[len(GROUP_ID_PREFIX):].split(",") if m])
    try:
        r = vm._o._audio.GetOriginalAudio(memory_id)
        return r.get("audio_path") or "" if r.get("found") else ""
    except Exception as e:
        print(f"[web] archived audio lookup failed: {e}", flush=True)
        return ""


def hit_cluster(content: str, source: str) -> str:
    """For retrieval hits: only content and source, no metadata."""
    return rb_cluster(content, source, "")


def _rb_cluster(m) -> str:
    """For snapshots: take fields from a RightBrainMemory object."""
    meta = getattr(m, "metadata", None) or {}
    return rb_cluster(getattr(m, "content", ""),
                      str(getattr(m, "memory_class", "")),
                      meta.get("emotion", ""))


def fact_index(uid: str) -> dict:
    """Left-brain memory id -> original fact text.

    The original text lives in the vector store; the cognitive graph's memories table only has id/slot/heat etc., no text --
    at first get_memory_record was used, and every heartnote's cause came out empty.
    """
    try:
        entries = vm._o._get_repo()._vector_store.list_entries(user_id=uid)
        return {e["id"]: e["text"] for e in entries}
    except Exception as e:
        print(f"[web] reading left-brain facts failed: {e}", flush=True)
        return {}


#: At most this many entities drawn per right-brain slot on the brain map.
RB_ENTITIES_PER_SLOT = int(os.environ.get("SUPERMEM_RB_GRAPH_PER_SLOT", "6"))
#: At most this many memories drawn per left-brain slot on the brain map.
LB_ENTRIES_PER_SLOT = int(os.environ.get("SUPERMEM_LB_GRAPH_PER_SLOT", "7"))


# ── The "plain language" version of right-brain judgements ───────────────────
# The claims stored in the library are for the **model**: compact, third person, label-like ("Facing a lot of
# pressure recently"). Those can't change -- that density is exactly what the prompt needs.
# But the page is for **people**, and the same sentence shown there reads like a case file. Here we make a display-only
# first-person rewrite.
#
# Async + cache: memory_snapshot is synchronous and can't wait for an LLM round trip inside. So the first time
# shows the original, the background rewrite lands in the cache, and the frontend's next poll (watchMemories polls anyway) swaps in
# the plain-language version. The rewrite only touches wording, adds no facts.
#: What this space's owner is called. The right-brain lines are **about him**; always writing "he" loses the feeling of
#: knowing him -- "Jiaqi goes quiet when he's nervous" vs "he goes quiet when he's nervous", only the former feels like you know him.
#: The name comes from the voiceprint registry (bound when he said "my name is X"); if unavailable it falls back to a generic reference.
_OWNER_NAME_CACHE: dict = {}
#: Question words. "What's my name?" was once bound as a self-introduction, and the registry really held an entry
#: name="what name" (see the top of voiceprint/speaker_identity.py). Block it before display.
_BAD_NAME_CHARS = "?"


def owner_name(space: str = "") -> str:
    """This space owner's name; returns "" if it can't be recognized (the caller falls back to a generic reference)."""
    space = space or ACTIVE_SPACE
    if space in _OWNER_NAME_CACHE:
        return _OWNER_NAME_CACHE[space]
    name = ""
    try:
        import json
        from supermem.utils.common import space as _sp
        d, _ = space_dir(space)
        p = _sp.mm(d, "voiceprint_registry.json")
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            cands = []
            for key, v in (data or {}).items():
                if not isinstance(v, dict) or v.get("role") != "user":
                    continue
                n = (v.get("name") or "").strip()
                # Block fake names bound from questions, and placeholder keys like "user"
                if not n or n.lower() == "user" or any(c in n for c in _BAD_NAME_CHARS):
                    continue
                cands.append((bool(v.get("entity_id")), n))
            if cands:
                # Those with an entity_id are more trustworthy (they really landed in the graph)
                cands.sort(key=lambda t: not t[0])
                name = cands[0][1]
    except Exception as e:
        print(f"[rb] reading owner name failed: {type(e).__name__}: {e}", flush=True)
    _OWNER_NAME_CACHE[space] = name
    return name


_RB_HUMAN: dict = {}          # original claim -> plain-language version
_RB_HUMAN_PENDING: set = set()
#: Turn off to always show the raw claim (when you don't want to spend money on display).
RB_HUMANIZE = os.environ.get("SUPERMEM_RB_HUMANIZE", "1") != "0"

_RB_HUMANIZE_PROMPT = (
    # Lessons from four versions, don't regress:
    # (1) Just saying "first person" isn't enough -- without saying **who** is speaking, the model only does synonym swaps.
    # (2) "Don't add facts" gets read as "don't change the wording", degrading into sticking "I noticed" in front of the original.
    #    The two must be separated: no facts added, but the wording must be restated.
    # (3) "I found / I noticed / it seems to me" are all **reporting verbs** -- grammatically first person, but the tone is still
    #    an observer reporting. What we want is a little companion who gets him and feels for him, so these openers must be banned.
    # (4) The first version of the name rule said "alternate with 'he'", too soft; all ten used "he" -- the examples also all used
    #    "he", and the model follows the examples. The rule must be firm, and the examples must use the name too.
    #    Later changed to **always use the name**: each item on the page is its own line, not a continuous paragraph,
    #    so the repeated name reads as "this is about whom", not as wordiness.
    # (5) Display language follows the **UI**, not the original. The store may be mixed-language (half translated); following the original
    #    would mix languages on the page. Display is for people; one screen shouldn't be half-baked.
    "Each line below is a judgement about {who}, from a little assistant who has always been by {who}'s side -- "
    "like a small animal that really gets him, quietly staying nearby, taking everything in.\n"
    "Rewrite each one as something this little assistant would say.\n"
    "\n"
    "Tone:\n"
    "· Short. The observation half is best at around ten words; it should read as a sentence, not a record.\n"
    "· Warm, with a slight protective feeling toward him -- caring, not analysing.\n"
    "· **Don't start with \"I found\", \"I noticed\", \"it seems to me\"** -- that's a reporting tone. "
    "Just say the thing itself, or how you feel for him.\n"
    "· No gushing, no cutesiness, no piles of exclamation marks, no lecturing and no consoling.\n"
    "{name_rule}"
    "\n"
    "Content:\n"
    "· **You must rephrase it**. The original uses summary terms (\"brief responses\", \"seeks validation\"); "
    "turn them back into how a person would describe it (\"he stops saying much\", \"he wants someone to catch him\").\n"
    "· But **never add any facts**: no extra details, no guessing causes, no added judgements. Change the wording, not the content.\n"
    # Key distinction: adding facts about the user = hallucination; saying what **I myself** plan to do = the assistant's stance, safe.
    # And this half is only shown on the page, it never enters the model's prompt, so a mistake can't affect replies.
    "· After the observation, you may add **what you yourself plan to do**, separated by \" —— \". "
    "Only say your approach (\"I don't press\", \"I let him finish\"); "
    "**don't say one more thing about him**.\n"
    # Without this rule it writes "I'll find him a quiet place to study" -- a voice assistant can't do that, and it reads as fake.
    "· The only thing you can do is **talk**: how to open, what to bring up first, what to avoid, "
    "when to stay quiet. Don't promise real-world actions (finding places, setting alarms, doing things for him); "
    "you can't do them.\n"
    "· This second half is optional: if no natural approach comes to mind, keep only the observation, don't force it. "
    "Three or four out of ten is enough; having it on every line sounds like reciting rules.\n"
    "· Write everything in {lang}, whatever language the original is in.\n"
    "\n"
    "Output line by line, with exactly the same number and order of lines as the input; no numbering, quotes or extra words.\n"
    "Example:\n"
    "{examples}"
)

#: The examples. They carry two things: **tone** and **output language**.
#: The model follows examples far more strongly than rules -- both the name rule and the language rule tripped on this:
#: the rule said "use the name every time" but the examples used "he", so the output was all "he"; the rule said "output in English"
#: but the examples were in another language, so the output was too. So the examples must match the target language, and all use the name.
_RB_HUMANIZE_EXAMPLES = {
    "en": (
        "  input   Tends to give brief responses when anxious\n"
        "  output  {who} goes quiet the moment he tenses up —— I don't push, I wait\n"
        "  input   Easily distracted during exams\n"
        "  output  {who}'s mind wanders in exams —— I won't bring it up, it'd only add pressure\n"
        "  input   Doesn't like being ignored when feeling low\n"
        "  output  When {who} is down, being ignored is the worst of it\n"
        "  input   Feels accomplished when recognized\n"
        "  output  A little praise and {who} lights right up\n"
        "  input   Hates being interrupted\n"
        "  output  Cut {who} off mid-sentence and you'll lose him —— I let him finish\n"
    ),
}


def _rb_humanize_now(claims: list, name: str = "", lang: str = "en") -> None:
    """Runs on a background thread: one LLM round trip rewrites a batch, results land in _RB_HUMAN."""
    lang_name = "English"
    try:
        from openai import OpenAI
        who = name or "this person"
        sysmsg = _RB_HUMANIZE_PROMPT.format(
            who=who,
            lang=lang_name,
            # The rule must be firm. The version that said "alternate" used "he" in all ten.
            name_rule=(f"· **Call him \"{name}\" directly every time**, don't substitute \"he\" -- "
                       "each item on the page is its own line, and using the name says \"this is about whom\".\n"
                       if name else ""),
            examples=_RB_HUMANIZE_EXAMPLES["en"]
                     .replace("{who}", who))
        r = OpenAI().chat.completions.create(
            model=utils.CHAT_MODEL, temperature=0.7,
            messages=[{"role": "system", "content": sysmsg},
                      {"role": "user", "content": "\n".join(claims)}],
        )
        lines = [x.strip() for x in (r.choices[0].message.content or "").splitlines() if x.strip()]
        if len(lines) != len(claims):      # if line counts don't match, drop the whole batch; don't pair them misaligned
            print(f"[rb] rewrite line count mismatch ({len(lines)}!={len(claims)}), skipping this batch", flush=True)
            return
        for c, h in zip(claims, lines):
            _RB_HUMAN[(lang, name, c)] = h
    except Exception as e:
        print(f"[rb] judgement rewrite failed: {type(e).__name__}: {e}", flush=True)
    finally:
        _RB_HUMAN_PENDING.difference_update((lang, name, c) for c in claims)


def rb_human(claim: str) -> str:
    """The sentence used for display. Returns the original until the rewrite is ready, and queues it meanwhile."""
    if not RB_HUMANIZE or not claim:
        return claim
    name, lang = owner_name(), SPACE_LANG
    hit = _RB_HUMAN.get((lang, name, claim))
    if hit:
        return hit
    if (lang, name, claim) not in _RB_HUMAN_PENDING:
        _RB_HUMAN_PENDING.add((lang, name, claim))
        import threading
        threading.Thread(target=_rb_humanize_now,
                         args=([claim], name, lang), daemon=True).start()
    return claim


def rb_human_batch(claims: list) -> None:
    """Queue everything missing in one go -- the brain map shows dozens of items per screen, one thread each is too wasteful."""
    if not RB_HUMANIZE:
        return
    name, lang = owner_name(), SPACE_LANG
    todo = [c for c in dict.fromkeys(claims)
            if c and (lang, name, c) not in _RB_HUMAN
            and (lang, name, c) not in _RB_HUMAN_PENDING]
    if not todo:
        return
    _RB_HUMAN_PENDING.update((lang, name, c) for c in todo)
    import threading
    threading.Thread(target=_rb_humanize_now, args=(todo, name, lang), daemon=True).start()


def right_brain_tree(uid, facts):
    """Right hemisphere of the brain map: slot -> judgement -> evidence.

    Reads the right brain's judgement table (supermem/rightbrain/traits_store.py). The old
    slot->entity->heartnote structure is no longer written; _right_brain_tree_v1 is only for viewing old data.
    """
    try:
        store = vm._o._right._traits()
    except Exception as e:
        print(f"[web] reading judgement table failed: {type(e).__name__}: {e}", flush=True)
        return []

    traits = list(store.all(uid, per_slot=RB_ENTITIES_PER_SLOT))
    rb_human_batch([t.claim for t in traits])     # queue the missing ones in one go, see rb_human
    out = []
    for t in traits:
        out.append({
            "cluster": t.cluster,
            "slot": t.slot,
            # Node titles use the plain-language version; until the rewrite is ready it's the original, swapped in on the next poll.
            # raw is kept -- the frontend matches it against each turn's hits; the plain-language version wouldn't match.
            "raw": t.claim,
            "text": rb_human(t.claim),
            "desc": "",               # the judgement is itself a summary; no extra line of raw facts
            "notes": [{"text": e.quote, "emotion": e.emotion, "cause": e.cause}
                      for e in t.evidence],
        })
    return out


def _right_brain_tree_v1(uid: str, facts: dict) -> list:
    """The right brain's real three-layer structure: slot -> entity -> the heartnotes hanging under it.

    Nodes on the brain map are **entities** ("wronged", "dislikes nuts and allergies", "chooses silent endurance"), not
    individual heartnotes -- the entity is the "point" the right brain has generalized, and heartnotes are the evidence
    supporting it. Each heartnote also carries the left-brain fact that triggered it, so "why wronged" is two clicks away.

    Read-only, entirely through graph_store's public methods.
    """
    graph = vm._o._right._rb_graph_store()
    repo = vm._o._right._rb_repo()
    notes = {}
    for m in repo.list_all(uid):
        # response_experience records "how the assistant answered last time", an internal note for the reply layer,
        # not an understanding of the user as a person. It doesn't belong on the brain map -- the nodes it grows contain the assistant's own
        # words, looking like "the system took its own replies as knowledge about you".
        if getattr(m, "memory_class", "") == "response_experience":
            continue
        meta = getattr(m, "metadata", None) or {}
        notes[m.id] = {
            "text": m.content,
            "emotion": meta.get("emotion", ""),
            "cause": facts.get(meta.get("left_memory_id", ""), ""),
        }

    out = []
    for slot in graph.list_slots(uid):
        # Skip slots without a description: the right brain hasn't generalized them yet, and what hangs underneath is a heap of raw entities
        # dumped in (measured: under the old "people/places/attitudes" slot were Jiaqi / CS undergrad /
        # NUS / September 2026 -- names, education, institutions, dates, not attitudes).
        # Drawn on the brain map it's just noise, and it takes up space in the right hemisphere.
        # This used to be "skip the whole slot if it has no description".
        # description is only generated by **long-term attribution** (_summarize_slot at session boundaries);
        # a newly created space has never run it -> all six slots have empty descriptions -> every one is skipped -> the right
        # hemisphere has no nodes at all. But the nodes draw **entities**, unrelated to whether this slot has a one-line
        # profile summary: if the entity exists and the evidence exists, it should be drawn.
        cluster = SLOT_TO_CLUSTER.get(slot.name, "experiences")
        # Only draw the first few entities per slot -- after running a while the store has 89; drawing them all turns the right hemisphere into
        # a blur (the left brain had only 26 in the same period).
        #
        # Ordering can't look only at evidence count. It used to be purely by count descending, and among 24 entities under
        # "likes_dislikes" the new one had only 1 piece of evidence, always ranked last, **never making it onto the map** -- the user says
        # something brand new, and the right brain grows no new node, looking like it didn't remember.
        # Now half the seats are reserved for the most recently added: half by evidence count (stable profile), half by recency
        # (what was just said shows up immediately).
        def rows(newest):
            out = []
            for ent in graph.get_entities_for_slot(uid, slot.id, newest_first=newest):
                mids = [i for i in graph.get_memories_for_entity(ent.id) if i in notes]
                out.append((len(mids), ent, mids))
            return out

        fresh_n = max(1, RB_ENTITIES_PER_SLOT // 2)
        by_recent = rows(True)[:fresh_n]                    # truly most recently added (rowid descending)
        taken = {t[1].id for t in by_recent}
        by_evidence = [t for t in sorted(rows(False), key=lambda t: -t[0])
                       if t[1].id not in taken]
        picked = by_recent + by_evidence[:RB_ENTITIES_PER_SLOT - len(by_recent)]

        for _, ent, mids in picked:
            # Nodes with no memory hanging under them aren't drawn. The emotion slot pre-seeds 8 emotion-word
            # entities when a space is created (sad/calm/lonely...), so a brand-new empty store would open with six lonely dots, looking
            # like it already remembered something, with not a single piece of evidence behind them.
            if not mids:
                continue
            ns = [notes[i] for i in mids]
            # The entity description uses only the sentence generalized by the consolidation step (_summarize_entity).
            #
            # There was once a fallback here: with no description, use the first evidence's cause. That was wrong --
            # cause is the **raw left-brain fact**, so the card became
            #     title: Jiaqi
            #     body: The user's name is Jiaqi, 20 years old, studying at NUS...   <- raw fact
            #     note: [wronged] uh I'm Jiaqi, I'm twenty now...                     <- verbatim words
            # And several entities extracted from the same sentence (Jiaqi/CS major/NUS) share the same cause, so three cards
            # had identical bodies. Better left empty -- the title itself should be the summary, with no extra line in between.
            desc = (getattr(ent, "description", "") or "").strip()
            out.append({
                "cluster": cluster,
                "slot": slot.name,
                "text": ent.name,                      # this is what's shown on the brain map
                "desc": desc,
                "notes": ns,
            })
    return out


#: Memory ids hit by the most recent retrieval. The snapshot must guarantee they're on the map -- what lights up on the brain map must be
#: what the backend really retrieved; hit but not drawn, and "it remembers this" isn't shown.
_LAST_HIT_IDS: set = set()


def note_hits(result) -> None:
    """Record which left-brain memories this turn's retrieval hit."""
    _LAST_HIT_IDS.clear()
    for h in (getattr(result, "hits", None) or []):
        mid = getattr(h, "memory_id", "")
        if mid:
            _LAST_HIT_IDS.add(str(mid))


def memory_snapshot(limit: int = 48) -> dict:
    """Memories already in the store, so the frontend can fill the brain map when the page opens.

    Read-only, no models: left brain via list_entries + the cognitive graph's slot annotations, right brain via list_all.
    If the store is empty (new user) return empty lists, and the frontend grows from an empty map as before.
    """
    from supermem.leftbrain.cognitive_graph.types import SlotV2

    uid = vm._o._user_id
    left, right = [], []
    try:
        repo = vm._o._get_repo()
        entries = repo._vector_store.list_entries(user_id=uid)
        # Slots are annotated in the cognitive graph, not on memory entries -- build an id -> slot reverse lookup first
        cog = repo._cognitive_store
        slot_of = {}
        for slot in SlotV2:
            for mid in cog.memory_ids_for_slots(uid, [slot]):
                slot_of.setdefault(mid, slot.value)
        # The assistant's own words are also stored as is (see 3ed67f7), but the brain map draws "memories about the
        # user" -- growing the assistant's replies into nodes would treat its own words as knowledge about the user.
        entries = [e for e in entries if e.get("role") != "assistant"]
        # Each slot is capped too. A slot on the brain map is a fixed-size sector -- when daily_life
        # accumulates twenty-odd items that sector blurs, while other slots have three or four dots. With caps all clusters have similar density.
        #
        # But **the items hit by this turn's retrieval must stay**, even if they rank beyond the cap: what lights up on the map
        # must be what the backend really retrieved; hit but not drawn looks like "retrieved 5, only 3 lit up".
        # Hits go in first, and the remaining slots are filled in the original order.
        per_slot, kept = {}, []
        hit_first = ([e for e in entries if str(e["id"]) in _LAST_HIT_IDS] +
                     [e for e in entries if str(e["id"]) not in _LAST_HIT_IDS])
        for e in hit_first:
            sl = slot_of.get(e["id"], "daily_life")
            hit = str(e["id"]) in _LAST_HIT_IDS
            per_slot[sl] = per_slot.get(sl, 0) + 1
            if hit or per_slot[sl] <= LB_ENTRIES_PER_SLOT:
                kept.append(e)
        entries = kept
        # Likewise, the hit items must not be cut off by limit
        head = [e for e in entries if str(e["id"]) in _LAST_HIT_IDS]
        rest = [e for e in entries if str(e["id"]) not in _LAST_HIT_IDS]
        for e in (head + rest)[:max(limit, len(head))]:
            # list_entries' date just takes the first 10 chars of time_start; a pure time string gets cut into
            # something like "09:20:37". If it doesn't look like a date, blank it; don't send garbage to the frontend.
            d = str(e.get("date", ""))
            # Include which entities this memory is attached to: the frontend uses this to connect two memories about the same person/thing
            # -- lines on the brain map then reflect real relations, not arbitrary ones.
            # What's stored are entity ids (person_jiaqi_5ea413); the frontend shows them as labels and also
            # connects same-named entities, so swap in names here.
            try:
                ents = []
                for eid in cog.entity_ids_for_memory(e["id"]) or []:
                    ent = cog.get_entity(eid)
                    nm = (getattr(ent, "name", "") if ent else "").strip()
                    ents.append(nm or eid)
            except Exception:
                ents = []
            # hit: this item was hit by this turn's retrieval and hence kept on the map as a guarantee.
            # The frontend uses "which item is new on the map" to judge "what entities did the sentence just said extract", and the guaranteed
            # ones are **old memories** -- without marking them, saying "la la la" would make last time's entities
            # (Jiaqi, roommate, gaming...) pop up in the tag bar.
            left.append({"text": e["text"], "date": d if d[:4].isdigit() else "",
                         "slot": slot_of.get(e["id"], "daily_life"),
                         "hit": str(e["id"]) in _LAST_HIT_IDS,
                         "entities": list(ents)[:6]})
    except Exception as e:
        print(f"[web] left-brain snapshot read failed: {e}", flush=True)
    try:
        right = right_brain_tree(uid, fact_index(uid))
    except Exception as e:
        print(f"[web] right-brain snapshot read failed: {e}", flush=True)
    return {"left": left, "right": right}


# classify must be wrapped: passing vm.classify directly would weld in **the current** instance,
# so after switching spaces the brain map would still classify for the old space.
app = utils.build_app(MODE, realtime_session if MODE == "realtime" else llm_tts_session,
                      lambda *a, **k: vm.classify(*a, **k), memory_snapshot, audio_of,
                      spaces=(list_spaces, create_space, use_space, lambda: ACTIVE_SPACE),
                      set_lang=set_lang,
                      compare=(lambda: COMPARE, _set_compare))


@app.get("/api/cartridge")
def api_cartridge() -> dict:
    """The active space's cartridge (manifest only, never the memory text)."""
    try:
        return {"space": ACTIVE_SPACE, **_cartridge(ACTIVE_SPACE).manifest()}
    except Exception as e:  # noqa: BLE001 -- e.g. a brand-new space with no database yet
        return {"space": ACTIVE_SPACE, "error": f"{type(e).__name__}: {e}"}


@app.post("/api/cartridge")
async def api_cartridge_refresh() -> dict:
    """Re-snapshot the active space's memory into its cartridge and pre-fill it
    on every cartridge panel (what a telephony integration does on ring)."""
    try:
        cart = _cartridge(ACTIVE_SPACE, refresh=True)
    except Exception as e:  # noqa: BLE001
        return {"space": ACTIVE_SPACE, "error": f"{type(e).__name__}: {e}"}
    return {"space": ACTIVE_SPACE, **cart.manifest(),
            "prefetch": await _prefetch_cartridges(ACTIVE_SPACE)}


def _warm_network() -> None:
    """Move the first call on both network paths to startup.

    The local models are warmed above, but the two **network** paths are still
    cold: the reply model and the transcription endpoint. Measured, the first
    call is ~2.0s against ~0.65s after -- that second and a half lands on the
    user's first sentence, the one that decides whether this demo feels fast.
    What it pays for is DNS + TLS + client construction.

    Both fire in parallel and each only prints on failure rather than raising:
    warmup is an optimisation, and no network at boot should still boot. The
    cost is negligible (one token from the reply model, 0.3s of silence for
    transcription).
    """
    import asyncio as _aio

    async def _reply() -> None:
        from supermem.reply import openai_reply
        fn = openai_reply(model=utils.CHAT_MODEL, system="warmup")
        agen = fn("hi", "")
        async for _ in agen:            # the first token is enough, the handshake is already paid for
            await agen.aclose()
            break

    def _asr() -> None:
        warm = getattr(vm.utils.get("asr"), "warmup", None)
        if warm:                        # local ASR has no such method, and doesn't need it
            warm()

    t0 = time.monotonic()
    thread = threading.Thread(target=_asr, daemon=True)
    thread.start()
    try:
        _aio.new_event_loop().run_until_complete(_reply())
    except Exception as e:              # noqa: BLE001
        print(f"[warmup] reply-model warmup failed (ignored): {type(e).__name__}: {e}", flush=True)
    thread.join(timeout=20)
    print(f"[warmup] network (reply model + transcription) {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    print(f"[web] mode={MODE} spec>={SPEC_MIN_CHARS} chars gamble={ARGS.gamble_ms}ms "
          f"confirm={ARGS.confirm_ms}ms -> http://localhost:{ARGS.port}/", flush=True)
    # Warm everything here so the first sentence never waits on a model load.
    # The ASR is lazy: pulling it up when the user starts speaking costs several
    # seconds, and that audio piles up in the socket buffer meanwhile. Catching
    # up feeds it to the VAD frame by frame, the silence instantly exceeds
    # confirm_ms, and the first sentence gets cut off -- which sounds like "the
    # first sentence is both slow and wrong".
    print("[web] warming local models (embedding / ASR / VAD / perception)…", flush=True)
    vm.warmup(verbose=True)
    _warm_network()
    print("[web] ready", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
