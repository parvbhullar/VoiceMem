"""Merge three LLM calls -- "extract facts / annotate / right-brain traits" -- into one.

Measured: one ingest makes 12 chat calls, and these three consume the **same
utterance** with outputs that don't depend on each other:

    extract_facts_openai   extracts the fact text
    CognitiveAnnotator     tags each fact with slot / entity / relation
    _extract_rb_traits     reads "what this person is like" from the utterance

Merging them into one call has an ordering problem to solve: the annotator's input
is the extractor's output, so it looks like they must run serially. The approach is
to have the extractor emit the annotations too, in one go, and stash them here; the
annotator and _extract_rb_traits check the stash first, use it on a hit, and only
fall back to their own LLM call on a miss.

This needs no function signature changes, and **it falls back to the old path
automatically when broken** -- if the merged output is missing fields, fails to
parse, or the model ignores the format, each downstream step makes its own call as
before, behaving exactly as it did pre-merge.

``SUPERMEM_MERGED_EXTRACTION=0`` disables merging.
"""
from __future__ import annotations

import os
import threading

_lock = threading.Lock()
_annotations: dict[str, dict] = {}      # fact text -> {slot, entities, relations}
_traits: dict[str, list] = {}           # utterance -> [(slot, label), ...]
_MAX = 512


def enabled() -> bool:
    return os.environ.get("SUPERMEM_MERGED_EXTRACTION", "1") != "0"


def _put(store: dict, key: str, value) -> None:
    if not key:
        return
    with _lock:
        if len(store) >= _MAX:
            store.clear()
        store[key] = value


def _take(store: dict, key: str):
    """Remove on read: a fact is only annotated once, keeping it just wastes memory."""
    with _lock:
        return store.pop(key, None)


def put_annotation(fact_text: str, ann: dict) -> None:
    _put(_annotations, (fact_text or "").strip(), ann)


def take_annotation(fact_text: str) -> dict | None:
    return _take(_annotations, (fact_text or "").strip())


# Emotion and traits belong to the **whole utterance**, not to any one fact, and
# extraction and the right-brain write happen back to back in the same turn, on the
# same thread. Keying by the utterance used to miss -- the utterance extraction saw
# carried a "Speaker 0: " prefix that didn't match the text the right brain received.
# Now it's a thread-local "last value": matched back to back within one thread, and
# threads (concurrent eval runs) don't interfere with each other.
_local = threading.local()


def put_emotion(utterance: str, emo: str) -> None:
    _local.emotion = emo


def take_emotion(utterance: str = "") -> str | None:
    v = getattr(_local, "emotion", None)
    _local.emotion = None          # clear on read so the last turn's value doesn't leak into the next
    return v


def put_traits(utterance: str, traits: list) -> None:
    _local.traits = traits


def take_traits(utterance: str = "") -> list | None:
    v = getattr(_local, "traits", None)
    _local.traits = None
    return v


# The block appended after the extraction prompt. Deliberately short -- it is sent on
# every ingest, and the original extraction prompt is already long. Field names match
# the annotator's output format, so not one line of downstream parsing has to change.
PROMPT_ADDENDUM = """

Additionally, for EACH item in "memory", include these three fields:
- "slot": one of [{slots}]
- "entities": [{{"name": "...", "entity_type": "user|person|project|task|knowledge|preference|place|routine|asset|organization|event", "role": "subject|object|context|owner"}}]
- "relations": [{{"from": "...", "to": "...", "relation_type": "...", "confidence": 0.9}}]
  Relation direction must match reality: a boss manages the user, not the reverse.

And add ONE top-level field "traits": subjective things this utterance reveals about
the speaker. Each item {{"slot": "...", "label": "<5-15 chars>"}}, slot is one of:

{traits_body}

Outside that case, only include a category the utterance clearly shows —
for a one-off event or a plain question, "traits": [] is the right answer.

Each label becomes the TITLE of a node on a graph, so write it as a short
pattern — roughly 5-15 characters (or 3-8 words), no subject, no full stop.

{label_rule}

Copy the SHAPE of these, never the wording:
{label_examples}

Also add ONE top-level field "emotion": how the speaker feels, as a SINGLE
English word (pleased/happy/calm/anxious/sad/wronged/angry/surprised/tired/
disappointed/…).
**Judge from what they actually say.** "I love strawberries" is pleased, not sad;
"I'm so angry" is angry, not anxious. If the utterance carries no clear feeling
(a plain fact, a question), return "" — an empty string is the right answer
far more often than a guess. A wrong label is worse than none: it gets shown
to the user as [label] next to their own words.

Never invent entities, traits or feelings that are not in the text.

Keep one-off requests OUT of "memory": asking for a recommendation, asking what
you remember about them, asking you to do something right now. When the same
sentence ALSO states a lasting fact, write only the lasting half — never both in
one item. "I'm taking the GRE next week, any books you'd recommend?" gives exactly
one memory, "User is taking the GRE next week", and nothing about the book request.

OUTPUT SHAPE — your JSON object must have EXACTLY these three top-level keys:
{{"memory": [...], "emotion": "...", "traits": [...]}}
The prompt above describes only the "memory" key. "emotion" and "traits" are
REQUIRED as well; omitting them is an error. Use "" and [] when there is nothing.

════════════════════════════════════════════════════════════════════════
LANGUAGE — this overrides every example above.
{label_rule}
It applies to EVERY string you output: the "memory" texts, every trait
"label", and "emotion". The only exception is "slot", which is an internal
key and must stay exactly as listed (emotion / coping_style / expression_style /
thinking_pattern / likes_dislikes). If the speaker wrote in English, every one of those strings must
be English — copying the Chinese wording from the examples is a mistake.
════════════════════════════════════════════════════════════════════════"""


#: Examples are given in **one** language only, the same language as this turn.
#:
#: Putting examples in two languages side by side once turned out worse than a single
#: language: the model started writing even the left-brain facts in the other
#: language. Examples pull on the output far harder than rules do -- so keep the rules,
#: but the examples must be chosen right first.
#: The traits description block is swapped as a whole, not just the good/bad examples
#: at the end -- the examples built into the slot descriptions are what the model
#: copies; changing only the final set didn't help in testing.
_TRAITS_BODY = {"en": """  emotion           WHEN they feel WHAT — the situation plus the feeling it triggers.
                    "tense before design reviews", "annoyed when interrupted",
                    "anxious when a project slips"
  coping_style      what they DO about a feeling, or how they want to be treated.
                    "wants comfort under stress", "needs to be alone when upset"
  expression_style  habits of speaking and communicating
  thinking_pattern  how they think, weigh things, decide
  likes_dislikes    what they like or dislike

emotion vs coping_style is the one people get wrong: "annoyed when interrupted" is
emotion (a feeling appearing), "walks away when interrupted" is coping_style (an
action taken). If the label has no verb of doing or wanting in it, it is emotion.

For "emotion" the label must read as **a pattern, not a bare feeling word**:
"tense before design reviews", "annoyed when interrupted", "calm when alone"
— NOT "anxious" / "happy". It becomes the title of a node on a graph; a bare
word tells the user nothing.

When the utterance states a RECURRING tendency about the speaker — "every time
…", "always", "never", "I'm the kind of person who…", or any habit/reaction
that clearly holds beyond this one moment — a trait is REQUIRED. "I zone out in
long meetings" is likes_dislikes "dislikes long meetings"; "I can't sleep before a
demo" is emotion "can't sleep before a demo".
Do not skip it just because the same content also went into "memory": "memory"
records WHAT HAPPENED, "traits" records WHAT THIS PERSON IS LIKE, and one
sentence very often carries both.

"""}

_EXAMPLES = {
    "en": ("  good: hates being interrupted / wants comfort under stress / "
           "conclusion first\n"
           "  bad: The user tends to plan in detail. (a full sentence with a subject)\n"
           "  bad: I major in computer science (copying the utterance / a plain fact)"),
}


def prompt_addendum() -> str:
    """The block appended to the end of the **user message**.

    Note: the user message, not system. At the end of system, the model honours only
    the per-item fields (slot/entities come out) and drops the top-level
    emotion/traits -- the output format in the user message hard-codes "memory" as the
    only top-level key, and the model follows it strictly. The right brain then gets
    emotion="" + traits=[] every turn, write() returns early, and the mind map never
    grows a single node. The final OUTPUT SHAPE section restates the top-level
    structure explicitly for exactly this reason; don't remove it.
    """
    from supermem.leftbrain.cognitive_graph.slot_v2 import ALL_SLOT_V2_VALUES
    from supermem.lang import label_rule, memory_language
    key = memory_language()
    return PROMPT_ADDENDUM.format(slots=", ".join(ALL_SLOT_V2_VALUES),
                                  label_rule=label_rule(),
                                  label_examples=_EXAMPLES[key],
                                  traits_body=_TRAITS_BODY[key])
