"""Aligned with Mem0 OSS / Platform V3: additive (ADD-only) memory extraction.

- System: ``data/additive_extraction_prompt.txt`` (identical to upstream ``ADDITIVE_EXTRACTION_PROMPT``)
- User: ``mem0_additive_prompt_build.generate_additive_extraction_prompt``
- OpenAI Chat JSON: top-level ``memory`` array; items have ``id`` / ``text`` / ``attributed_to`` / optional ``linked_memory_ids``

Default chat model: ``gpt-4o-mini`` (the ``chat`` role; override with ``SUPERMEM_CHAT_MODEL`` /
``SuperMem(models={"chat": ...})`` / ``OpenAIAdditiveExtractorConfig(model=...)``)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from supermem.leftbrain.mem0_additive_prompt_build import (
    generate_additive_extraction_prompt,
    load_additive_system_prompt,
)

from supermem.utils.common.cost_log import log_usage as _log_usage
from supermem.llm_config import resolve_api_key, resolve_model


def remove_code_blocks(content: str) -> str:
    text = content.strip()
    m = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.I)
    return m.group(1).strip() if m else text


def extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return text[start_idx : end_idx + 1]
    return text


@dataclass(frozen=True)
class ExtractedAdditiveMemory:
    """A single additive extraction result (aligned with the Mem0 output schema)."""

    local_id: str
    text: str
    attributed_to: str
    linked_memory_ids: tuple[str, ...] = ()
    # "whose which attribute", without the value, e.g. "User's favorite restaurant" / "User's job".
    # Used only for the second candidate retrieval in the conflict-resolution stage (see
    # voice_input.py); empty string = an event/experience with no attribute to overwrite.
    # Emitted alongside by the extraction prompt (_ATTRIBUTE_ADDENDUM), zero extra calls.
    attribute: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExtractedAdditiveMemory:
        lids = raw.get("linked_memory_ids") or []
        if not isinstance(lids, list):
            lids = []
        attr = raw.get("attribute")
        return cls(
            local_id=str(raw.get("id", "")),
            text=str(raw.get("text", "")).strip(),
            attributed_to=str(raw.get("attributed_to", "user")),
            linked_memory_ids=tuple(str(x) for x in lids),
            attribute=str(attr).strip() if isinstance(attr, str) else "",
        )


#: Two kinds of junk the extraction model keeps forcing in despite the prompt; blocked
#: here by literal match.
#:
#: Why it has to be blocked in code: the upstream prompt opens with "every memorable piece
#: of information must be captured; missing one loses context", and the "don't extract
#: requests / don't extract the assistant's own words" we wrote in the addendum can't
#: override it -- two prompt revisions in testing, and it still extracted them.
#:
#: Why these two kinds are especially harmful: they are worded almost exactly like the
#: current conversation, so the very next question scores them high and they push the
#: truly relevant memories out of top-k. In testing, after "recommend me some dishes"
#: was stored, the next time the user asked what to eat, the allergy memory fell out of
#: the top five and the model recommended a dish with meat.
_JUNK_PATTERNS = (
    # The assistant's own words taken as facts about the user
    "assistant recommended", "assistant suggested", "assistant provided",
    "assistant replied", "assistant explained", "assistant told",
    # This turn's temporary request, not a lasting fact
    # Asking the assistant "do you still remember me / what do you think of me" is also a
    # one-off request. These are especially toxic: worded almost like the next similar
    # question, they rank first and push the real profile out of top-k.
    "asked for recommendations", "requested recommendations",
    "asked for suggestions", "wants suggestions", "is asking for",
)


def _is_junk(text: str) -> bool:
    low = text.lower()
    return any(pat in text or pat in low for pat in _JUNK_PATTERNS)


def _strip_request_clause(text: str) -> str:
    """When one memory holds both a lasting fact and a one-off request, cut off the request half.

    "I'm taking the GRE next week, any books you'd recommend?" gets extracted as one item,
    "User is taking the GRE next week, and asked for recommendations on GRE books." -- the
    whole item hits "asked for recommendations" in _JUNK_PATTERNS, so **the real fact
    "taking the GRE next week" is thrown away with it**. That is exactly how it got lost in
    testing: the right brain stored the utterance, the left brain had nothing, and when the
    user asked about it later there was no recollection at all.

    Conservative approach: cut only at the last comma, and what remains must still be a
    complete fact (>= 8 chars, no longer matching junk); otherwise keep the old behaviour
    and drop it -- better to miss a memory than stuff a half-sentence into the store.
    """
    for sep in (", and ", ", "):
        if sep not in text:
            continue
        head = text.rsplit(sep, 1)[0].strip().rstrip(",")
        if len(head) >= 8 and not _is_junk(head):
            return head + ("." if not head.endswith(".") else "")
    return ""


def parse_additive_memory_response(raw_json: str) -> list[ExtractedAdditiveMemory]:
    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("additive response must be a JSON object")
    mem = data.get("memory")
    if mem is None:
        raise ValueError('additive JSON missing "memory" array')
    if not isinstance(mem, list):
        raise ValueError('"memory" must be a list')
    out: list[ExtractedAdditiveMemory] = []
    for item in mem:
        if not isinstance(item, dict):
            continue
        t = (item.get("text") or "").strip()
        if not t:
            continue
        if _is_junk(t):
            kept = _strip_request_clause(t)
            if kept:
                print(f"[extract] cut off the request half: {t[:36]} → {kept[:36]}", flush=True)
                t = kept
            else:
                print(f"[extract] dropped (assistant's own words / one-off request): {t[:40]}", flush=True)
                continue
        # Extra annotations returned by the merged call: stash them for the downstream
        # annotator, saving that LLM call. If the fields are missing nothing is stored,
        # and the annotator, finding nothing, makes its own call as before.
        if any(k in item for k in ("slot", "entities", "relations")):
            from supermem.leftbrain import merged_extraction
            merged_extraction.put_annotation(t, {
                "slot": item.get("slot", ""),
                "entities": item.get("entities") or [],
                "relations": item.get("relations") or [],
                "confidence": item.get("confidence", 0.9),
            })
        out.append(ExtractedAdditiveMemory.from_dict(item))

    # Right-brain labels belong to the **whole utterance**, not to any single fact
    traits = data.get("traits")
    if isinstance(traits, list):
        from supermem.leftbrain import merged_extraction
        pairs = [(str(x.get("slot", "")).strip(), str(x.get("label", "")).strip())
                 for x in traits if isinstance(x, dict)]
        merged_extraction.put_traits(_MERGED_UTTERANCE.get("text", ""),
                                     [p for p in pairs if p[0] and p[1]])
    emo = data.get("emotion")
    if isinstance(emo, str):
        from supermem.leftbrain import merged_extraction
        merged_extraction.put_emotion(_MERGED_UTTERANCE.get("text", ""), emo.strip())
    return out


# The parse function can't see the utterance (it only receives the JSON string), yet
# traits must be keyed by the utterance for _extract_rb_traits to find them. extract()
# puts the utterance here before the call.
_MERGED_UTTERANCE: dict[str, str] = {}


@dataclass
class OpenAIAdditiveExtractorConfig:
    """OpenAI Chat client configuration."""

    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    use_input_language: bool = True

    def resolved_model(self) -> str:
        return resolve_model(self.model)


# Appended after the upstream extraction prompt (additive_extraction_prompt.txt itself is
# untouched). Purpose: have every fact also carry a value-free "whose which attribute"
# phrase. The conflict-resolution stage uses it for a second candidate search -- the
# vectors of "favorite restaurant is A" and "favorite restaurant is B" aren't similar
# (different values), but both are close to "User's favorite restaurant", so the old
# value is guaranteed to enter the comparison window (the main cause of missed UPDATEs
# on the benchmark was the old value not making it into top-k). Extraction already costs
# one LLM call, so one more field is free.
_LANGUAGE_RULE = """

# LANGUAGE

Write each memory in the SAME language the user spoke it in. A Chinese utterance
becomes a Chinese memory, an English one an English memory. Never translate.

Retrieval is embedding-based: a Chinese question and an English memory about the
same thing are far apart in vector space, so a translated memory quietly becomes
unfindable. Keep proper nouns and product names as the user said them.
"""


_VOICE_ADDENDUM = """

# VOICE CONTEXT: what is NOT worth remembering

These transcripts come from live speech, so much of what is said is about the
conversation itself rather than about the user. Do NOT extract:
- Audibility and connection checks from either side ("can you hear me?",
  "are you there?"), and the replies confirming audibility.
- Identity probes that carry no new information ("do you know who I am?",
  "do you remember my name?"). If the user actually states their name, extract the name —
  but never the question itself.
- Remarks about the exchange rather than the user's life ("why aren't you replying",
  "don't interrupt me", "you replied so fast", "is it running?").
- Fragments the ASR clearly garbled: isolated syllables, filler, or text with no
  recoverable meaning ("lolo", "um that", "uh").
- Anything whose only content is that something was said or asked. Never write a
  memory of the form "User asked X" or "Assistant said X" unless X itself is a fact
  about the user's world.
- The user's request for this turn. "User wants dinner suggestions", "User asked for
  restaurant recommendations" — wanting something right now is not a lasting fact.
  A standing preference is ("User only drinks pour-over coffee"); a one-off ask is not.
- Never write "Speaker 0" (or any "Speaker N") into a memory as if it were a name —
  it means the voice has not been matched to a known person yet. In that case say
  "the user" instead, and do not assume they are anyone named elsewhere
  in the conversation.
  When the message DOES carry a real name ("Jiaqi: I moved to Hangzhou"), keep that name as
  the subject. Rewriting a named speaker to "the user" makes the memory unfindable by
  questions that use the name ("Which city did Jiaqi move to?").
- The assistant's own answer: suggestions it made, options it listed, information it
  looked up. Those are the assistant's output, not facts about the user. Extract from
  an assistant turn only when the user CONFIRMED something about themselves in it.

These two are the most damaging kind of junk: they are worded like the current
conversation, so they score high on the very next query and push the real memories
out of top-k — the allergy stops being retrieved right when it matters.

The test: would this still be useful to know a week from now, in a different
conversation? If not, do not extract it. Extracting nothing from a turn is a valid
and common outcome — an empty result is better than a worthless memory.

"""

_ATTRIBUTE_ADDENDUM = """

# ADDITIONAL FIELD: attribute

For each memory object, ALSO output an "attribute" field (string):
- If the memory states a fact of the form "<subject>'s <attribute> is <value>" (a preference, status, or personal detail that could later change), set "attribute" to "<subject>'s <attribute>" WITHOUT the value. Examples:
  - "User's favorite restaurant is Sushi Zen" -> "attribute": "User's favorite restaurant"
  - "User works as a data analyst at Grab" -> "attribute": "User's job"
  - "User lives in Clementi" -> "attribute": "User's residence"
  - "User's sister Jiahui is in her final year of high school" -> "attribute": "Jiahui's school year"
- If the memory is an event, experience, or one-off narrative ("User went hiking last Sunday", "User had a fight with a coworker"), set "attribute" to "".
- Keep the subject name explicit ("User" for the account owner, the person's name otherwise). Keep it short (2-6 words). Write it in English regardless of the memory language.
"""


class OpenAIMem0V3AdditiveExtractor:
    """Extraction with OpenAI Chat + the Mem0 additive system/user prompts."""

    def __init__(self, config: OpenAIAdditiveExtractorConfig | None = None) -> None:
        self._cfg = config or OpenAIAdditiveExtractorConfig()
        self._system = load_additive_system_prompt() + _LANGUAGE_RULE + _ATTRIBUTE_ADDENDUM + _VOICE_ADDENDUM

    def extract(
        self,
        *,
        new_messages: list[dict[str, str]],
        summary: str | dict[str, str] | None = None,
        recently_extracted_memories: list[Any] | None = None,
        existing_memories: list[dict[str, Any]] | None = None,
        last_k_messages: list[dict[str, str]] | None = None,
        observation_date: str | None = None,
        current_date: str | None = None,
        custom_instructions: str | None = None,
    ) -> list[ExtractedAdditiveMemory]:
        """Run one additive ADD extraction over ``new_messages``."""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Please install: pip install openai>=1.0") from e

        api_key = resolve_api_key(self._cfg.api_key)
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY (or pass it via OpenAIAdditiveExtractorConfig(api_key=...))")

        # timeout must be set explicitly: the openai client's default timeout is 10 minutes,
        # and one occasional hung upstream request stalls the whole ingest; if nothing comes
        # back within 60s, let the retry mechanism take over
        kw_client: dict[str, Any] = {"api_key": api_key, "timeout": 60.0, "max_retries": 2}
        if self._cfg.base_url:
            kw_client["base_url"] = self._cfg.base_url
        client = OpenAI(**kw_client)

        user_content = generate_additive_extraction_prompt(
            summary=summary,
            recently_extracted_memories=recently_extracted_memories,
            existing_memories=existing_memories,
            new_messages=new_messages,
            last_k_messages=last_k_messages,
            current_date=current_date,
            timestamp=observation_date,
            custom_instructions=custom_instructions,
            use_input_language=self._cfg.use_input_language,
        )

        # Merged mode: have this one call also emit slot/entities/relations/right-brain
        # labels, saving the separate LLM round trips of the downstream annotator and
        # _extract_rb_traits.
        #
        # Appended to the end of the **user message**. When attached after system, the
        # model drops the top-level emotion/traits (the output format in the user message
        # hard-codes "memory" as the only top-level key, and it complies), so the right
        # brain gets nothing every turn and never grows a node. See merged_extraction.
        from supermem.leftbrain import merged_extraction
        system = self._system
        if merged_extraction.enabled():
            user_content = user_content + merged_extraction.prompt_addendum()
            _MERGED_UTTERANCE["text"] = " ".join(
                (m.get("content") or "") for m in new_messages
                if (m.get("role") or "user") != "assistant").strip()
        else:
            _MERGED_UTTERANCE.pop("text", None)

        resp = client.chat.completions.create(
            model=self._cfg.resolved_model(),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        _log_usage("extract", self._cfg.resolved_model(), getattr(resp, "usage", None))
        raw_text = (resp.choices[0].message.content or "").strip()
        json_str = extract_json(remove_code_blocks(raw_text))
        return parse_additive_memory_response(json_str)

    def extract_from_asr(
        self,
        user_text: str,
        *,
        assistant_reply: str | None = None,
        summary: str | dict[str, str] | None = None,
        recently_extracted_memories: list[Any] | None = None,
        existing_memories: list[dict[str, Any]] | None = None,
        last_k_messages: list[dict[str, str]] | None = None,
        observation_date: str | None = None,
        current_date: str | None = None,
        custom_instructions: str | None = None,
    ) -> list[ExtractedAdditiveMemory]:
        """Convenience: this turn's user ASR + optional assistant reply; ``existing_memories`` is for dedup/linking (UUID ``id`` + ``text``)."""
        msgs: list[dict[str, str]] = [{"role": "user", "content": user_text.strip()}]
        if assistant_reply is not None and assistant_reply.strip():
            msgs.append({"role": "assistant", "content": assistant_reply.strip()})
        return self.extract(
            new_messages=msgs,
            summary=summary,
            recently_extracted_memories=recently_extracted_memories,
            existing_memories=existing_memories,
            last_k_messages=last_k_messages,
            observation_date=observation_date,
            current_date=current_date,
            custom_instructions=custom_instructions,
        )


# ── Mem0 V1-style Conflict Resolver ────────────────────────────────────────────

_CONFLICT_SYSTEM_PROMPT = """You are a smart memory manager which controls the memory of a system.
You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to:
- ADD: Add it to the memory as a new element
- UPDATE: Update an existing memory element (use the existing ID, keep the most informative version)
- DELETE: Delete an existing memory element that directly contradicts the new fact
- NONE: Make no change (fact already present or not worth storing)

Guidelines:
1. ADD if the fact is genuinely new information not covered by existing memories.
2. UPDATE if the fact updates/refines an existing memory about the same topic (same ID, new text).
3. DELETE if the fact directly contradicts an existing memory (e.g. user switched habits, changed preference).
4. NONE if the fact duplicates an existing memory with no meaningful new context.

Return JSON only:
{
  "memory": [
    {"id": "<existing-id or new sequential id>", "text": "<memory text>", "event": "ADD|UPDATE|DELETE|NONE", "old_memory": "<old text, only for UPDATE>"},
    ...
  ]
}

Rules:
- For ADD: use a new sequential string id ("new_0", "new_1", ...).
- For UPDATE/DELETE/NONE: use the exact existing UUID id from the provided list.
- For UPDATE: include "old_memory" with the original text.
- Only emit entries where something actually changes (ADD or UPDATE or DELETE). You may omit NONE entries.
"""

# The version above is a condensed rewrite (1.5k chars) that phrased the UPDATE trigger
# as "merge if it's the same topic". In testing that was too loose: July's "took the kids
# to the museum" was judged the same topic as May's "likes painting to relax" and merged
# in, flattening the specifics (sunrise, and "known the friend for 4 years" vanished
# entirely), and the date stayed that of the memory it was merged into.
# Replaced here with mem0's official original (DEFAULT_UPDATE_MEMORY_PROMPT from
# github.com/mem0ai/mem0 mem0/configs/prompts.py, 5310 chars, copied byte for byte): it
# restricts UPDATE to "talking about the same thing", gives 4 pairs of positive/negative
# examples, and defaults new facts to ADD. The extraction prompt is already the official
# copy, so with this step the whole write path is aligned with mem0.
_CONFLICT_SYSTEM_PROMPT = (
    Path(__file__).resolve().parent / "data" / "mem0_update_memory_prompt.txt"
).read_text(encoding="utf-8")

# mem0's official prompt is designed for single-user memory; its "same topic = update"
# rule is dangerous in a multi-person household: several people may be asked the same
# kind of question ("which is your favorite charity"), and if the decision looks only at
# the topic and not the person, a later speaker overwrites a completely unrelated
# person's memory (in testing: Nancy stated her charity preference first, and when
# Kenneth/Eric/Jennifer each stated theirs later, one of them actually overwrote Nancy's).
# A mandatory rule is appended here on top of the official text, without changing
# mem0_update_memory_prompt.txt itself.
_CONFLICT_SYSTEM_PROMPT += """

IMPORTANT — multi-person household rule (in addition to the rules above):
Memories in this system can be about DIFFERENT named people living in the same household (e.g. "Nancy's preferred charity is X" vs "Jennifer's preferred charity is Y" are about two different people, not the same fact).
Before choosing UPDATE or DELETE for any existing memory, check whether the named subject/person in the new fact is the SAME as the named subject/person in that existing memory:
- If the subjects are different named people, you MUST NOT use UPDATE or DELETE on that memory, even if the topic/category is identical — treat the new fact as ADD instead.
- Only use UPDATE/DELETE when the existing memory is unambiguously about the SAME person as the new fact (same name, or both clearly refer to the generic account owner "User").
- If a fact's subject is ambiguous or not named, prefer ADD over UPDATE — a duplicate is far cheaper to have than silently overwriting a different person's information.
"""

_SINGLE_VALUED_RULE = """
IMPORTANT — single-valued attribute rule (in addition to the rules above):
Some facts describe a SINGLE-VALUED attribute of a person: something that has exactly one current value and gets replaced when it changes — favorite X (favorite food/restaurant/color/team/song...), current job/employer/role, residence/address, age, phone number, relationship status, current school/grade, the ONE pet's name, the date of a specific planned event.
- If a new fact and an existing memory are about the SAME person and the SAME single-valued attribute but state DIFFERENT values, the new fact SUPERSEDES the old one: choose UPDATE with the new value. This applies even when the new fact contains NO negation or explicit reference to the old value ("User's favorite restaurant is B" simply replaces "User's favorite restaurant is A"). The most recent statement wins.
- MULTI-VALUED attributes (hobbies, friends, foods the person likes in general, places visited, skills, allergies) accumulate: a new value is ADD, not UPDATE, unless it directly contradicts an existing one.
- Events, experiences, and one-off narratives ("went hiking on Sunday", "had dinner with Mom") are NEVER merged into unrelated preference memories and never replace each other: always ADD. Do not fold specific details of one event into another memory.
"""


def _build_update_memory_message(
    retrieved_old_memory: list, new_facts: list, speaker_name: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Build the full message sent to the model, same as mem0's official ``get_update_memory_messages``.

    The official code joins "decision prompt + old memories + new facts + output structure
    description" into a single user message, without splitting system/user. Building it the
    same way has two practical benefits: the output structure description matches our
    parser field for field; and that description contains the word "JSON" -- the official
    decision prompt body contains no "json" at all, so sent separately it would be rejected
    outright by OpenAI's ``response_format=json_object`` (400).

    speaker_name:
        The real name of the speaker of this turn's new facts (passed when known). The fact
        text usually already contains the name, but that is a "soft signal" -- stating the
        current speaker explicitly gives the model a harder anchor for judging "is the
        existing memory about the same person as the new fact", instead of guessing from
        the two texts alone.
    """
    if retrieved_old_memory:
        current_memory_part = f"""
    Below is the current content of my memory which I have collected till now. You have to update it in the following format only:

    ```
    {retrieved_old_memory}
    ```

    """
    else:
        current_memory_part = """
    Current memory is empty.

    """

    return f"""{system_prompt if system_prompt is not None else _CONFLICT_SYSTEM_PROMPT}

    {current_memory_part}

    The new retrieved facts are mentioned in the triple backticks. You have to analyze the new retrieved facts and determine whether these facts should be added, updated, or deleted in the memory.
    {f"These new facts were all spoken by: {speaker_name}. Use this as the authoritative subject when checking whether an existing memory is about the same person (per the multi-person household rule above)." if speaker_name else ""}

    ```
    {new_facts}
    ```

    You must return your response in the following JSON structure only:

    {{
        "memory" : [
            {{
                "id" : "<ID of the memory>",                # Use existing ID for updates/deletes, or new ID for additions
                "text" : "<Content of the memory>",         # Content of the memory
                "event" : "<Operation to be performed>",    # Must be "ADD", "UPDATE", "DELETE", or "NONE"
                "old_memory" : "<Old memory content>"       # Required only if the event is "UPDATE"
            }},
            ...
        ]
    }}

    Follow the instruction mentioned below:
    - Do not return anything from the custom few shot prompts provided above.
    - If the current memory is empty, then you have to add the new retrieved facts to the memory.
    - You should return the updated memory in only JSON format as shown below. The memory key should be the same if no changes are made.
    - If there is an addition, generate a new key and add the new memory corresponding to it.
    - If there is a deletion, the memory key-value pair should be removed from the memory.
    - If there is an update, the ID key should remain the same and only the value needs to be updated.

    Do not return anything except the JSON format.
    """


@dataclass
class ConflictResolution:
    event: str          # ADD | UPDATE | DELETE | NONE
    memory_id: str      # existing UUID for UPDATE/DELETE, "new_N" for ADD
    text: str           # new text (ADD/UPDATE) or empty (DELETE)
    old_memory: str = ""


class ConflictResolver:
    """Mem0 V1 style: make ADD/UPDATE/DELETE/NONE decisions for newly extracted facts against existing memories."""

    def __init__(self, config: OpenAIAdditiveExtractorConfig | None = None,
                 single_valued_rule: bool = True) -> None:
        self._cfg = config or OpenAIAdditiveExtractorConfig()
        # Single-valued attribute "new value overrides" rule (the SUPERMEM_CONFLICT_WIDE ablation switch is passed in via voice_input)
        self._system = _CONFLICT_SYSTEM_PROMPT + (_SINGLE_VALUED_RULE if single_valued_rule else "")

    def resolve(
        self,
        new_facts: list[str],
        existing_memories: list[dict[str, str]],  # [{"id": uuid, "text": ...}]
        speaker_name: str | None = None,
    ) -> list[ConflictResolution]:
        """Return the operation decision for each fact. All ADD when existing_memories is empty.

        speaker_name: the real name of the speaker of this turn's new facts, passed when
        known, to help the model judge whether an UPDATE/DELETE candidate is really the
        same person as the new fact (see the multi-person household rule at the top of
        the module).
        """
        if not new_facts:
            return []

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Please install: pip install openai>=1.0") from e

        api_key = resolve_api_key(self._cfg.api_key)
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")

        kw: dict[str, Any] = {"api_key": api_key}
        if self._cfg.base_url:
            kw["base_url"] = self._cfg.base_url
        client = OpenAI(**kw)

        # UUID -> index mapping (anti-hallucination): the official prompt's example ids are
        # all small integers like 0/1/2; hand it 36-char UUIDs directly and the model easily
        # gets a character or two wrong, so UPDATE/DELETE points at a nonexistent memory and
        # is silently dropped. mem0 does the same step officially (main.py "Map UUIDs to
        # integers (anti-hallucination)"), mapping the results back to real UUIDs.
        uuid_mapping: dict[str, str] = {}
        indexed: list[dict[str, str]] = []
        for idx, mem in enumerate(existing_memories or []):
            uuid_mapping[str(idx)] = str(mem.get("id", ""))
            indexed.append({"id": str(idx), "text": str(mem.get("text", ""))})

        user_content = _build_update_memory_message(indexed, new_facts, speaker_name=speaker_name,
                                                    system_prompt=self._system)
        # The upstream prompt requires **a decision for every candidate**, including
        # "event":"NONE". Candidates are picked by similarity (top-15 per new fact + top-10
        # per attribute), 25 per turn in testing, so the model has to emit 25 JSON objects,
        # twenty-odd of them NONE -- this one call takes 10.2s, half the whole ingest turn,
        # and it gets slower as the store grows.
        # Changed to emit only the ones that actually change: anything not mentioned is
        # NONE. The consumer already iterates by event and doesn't assume a one-to-one
        # mapping (voice_input.py "if resolve succeeds, execute the decisions"), so a few
        # missing NONEs don't affect any behaviour.
        if os.environ.get("SUPERMEM_RESOLVE_OMIT_NONE", "1") != "0":
          user_content += (
            "\n\nIMPORTANT — output size:\n"
            "Return ONLY the memories whose event is ADD, UPDATE or DELETE.\n"
            "OMIT every memory you would mark NONE — anything absent from your\n"
            "output is treated as NONE. Most turns change nothing, so "
            '{"memory": []} is a normal and correct answer.'
          )

        resp = client.chat.completions.create(
            model=self._cfg.resolved_model(),
            messages=[{"role": "user", "content": user_content}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()

        try:
            data = json.loads(remove_code_blocks(raw))
        except json.JSONDecodeError:
            data = json.loads(extract_json(raw))

        results: list[ConflictResolution] = []
        for item in data.get("memory", []):
            event = str(item.get("event", "NONE")).upper()
            if event not in ("ADD", "UPDATE", "DELETE", "NONE"):
                event = "NONE"
            raw_id = str(item.get("id", ""))
            # UPDATE/DELETE must land on a real UUID; if the index doesn't map, the model
            # made up a nonexistent id, so downgrade to ADD rather than modifying the store
            # with a fake id (a failed modification = a lost fact)
            if event in ("UPDATE", "DELETE"):
                if raw_id in uuid_mapping:
                    raw_id = uuid_mapping[raw_id]
                elif raw_id not in set(uuid_mapping.values()):
                    event = "ADD" if event == "UPDATE" else "NONE"
            results.append(ConflictResolution(
                event=event,
                memory_id=raw_id,
                text=str(item.get("text", "")).strip(),
                old_memory=str(item.get("old_memory", "")),
            ))
        return results
