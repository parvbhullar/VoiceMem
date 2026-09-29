"""voice_input.py — input adapter for the voice module

Converts the voice module's structured output into the SuperMem left-brain ingest format.

Voice module output format:
{
  "id": "str",
  "time_stamp": {"begin": "str", "end": "str"},
  "slots": ["str", ...],
  "contents": [
    {
      "sub_id":        "str",
      "time_start":    "str",
      "time_end":      "str",
      "sentence":      "str",
      "voiceprint_id": "str",
      "emotion":       "str"   # emotion2vec output, may be empty
    }
  ]
}

The voiceprint_id → person name / entity_id mapping is managed by VoiceprintRegistry.
"""
from __future__ import annotations

import os as _os

import os

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class VoiceContent:
    sub_id: str
    time_start: str
    time_end: str
    sentence: str
    voiceprint_id: str
    emotion: str = ""          # emotion2vec result, e.g. "neutral"/"anxious"

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceContent":
        return cls(
            sub_id=str(d.get("sub_id", "")),
            time_start=str(d.get("time_start", "")),
            time_end=str(d.get("time_end", "")),
            sentence=str(d.get("sentence", "")).strip(),
            voiceprint_id=str(d.get("voiceprint_id", "")),
            emotion=str(d.get("emotion", "") or ""),
        )


@dataclass
class VoiceInput:
    id: str
    time_stamp: dict          # {"begin": str, "end": str}
    slots: list[str]
    contents: list[VoiceContent]
    environment: str = ""     # AST/CLAP background sound description, e.g. "background sounds: Washing machine(0.82)"
    #: the agent's half (contents is the user's half); the extractor uses it to disambiguate
    agent_reply: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceInput":
        ts = d.get("time_stamp") or {}
        if isinstance(ts, str):
            ts = {"begin": ts, "end": ts}
        return cls(
            id=str(d.get("id", "")),
            time_stamp=ts,
            slots=[str(s) for s in (d.get("slots") or [])],
            contents=[VoiceContent.from_dict(c) for c in (d.get("contents") or [])],
            agent_reply=str(d.get("agent_reply", "") or ""),
        )

    @property
    def begin_time(self) -> str:
        return self.time_stamp.get("begin", "")

    @property
    def end_time(self) -> str:
        return self.time_stamp.get("end", "")

    def full_transcript(self) -> str:
        return " ".join(c.sentence for c in self.contents if c.sentence)

    def dominant_emotion(self) -> str:
        """Return the most frequent non-empty/non-neutral emotion; empty string if all are neutral."""
        from collections import Counter
        neutral = {"-", "neutral", "unknown", ""}
        emos = [c.emotion for c in self.contents if c.emotion and c.emotion not in neutral]
        if not emos:
            return ""
        return Counter(emos).most_common(1)[0][0]


# ── Voiceprint registry ────────────────────────────────────────────────────────

@dataclass
class VoiceprintEntry:
    """Registration info for a single voiceprint."""
    role: str         # "user" | "assistant"
    name: str         # display name, e.g. "Mr. Zhou"; same as voiceprint_id when unset
    entity_id: str    # id in the left brain's cognitive_graph entities table; may be empty


class VoiceprintRegistry:
    """Persistent registry of voiceprint_id → person name + entity_id.

    Core features:
      - every unknown voiceprint defaults to role="user" (multi-person conversations)
      - bind() binds a voiceprint_id to a known person name and/or entity_id
        → voice_input_to_messages() replaces "Speaker N" with the real name
        → the LLM sees "Mr. Zhou: ..." during extraction, and CognitiveAnnotator naturally links the memory to the Mr. Zhou entity
      - entity_id is stored in memory metadata for later direct lookup
    """

    ROLE_USER      = "user"
    ROLE_ASSISTANT = "assistant"

    def __init__(self, registry_path: Path, entity_resolver: Any = None) -> None:
        self._path = registry_path
        self._entries: dict[str, VoiceprintEntry] = {}
        #: person name -> cognitive graph entity_id. Injected by the orchestrator; without it we fall back to the original behaviour.
        self._resolve_entity = entity_resolver
        self._load()

    # ── Read / write ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for vpid, v in data.items():
                if isinstance(v, dict):
                    self._entries[vpid] = VoiceprintEntry(
                        role=v.get("role", self.ROLE_USER),
                        name=v.get("name", vpid),
                        entity_id=v.get("entity_id", ""),
                    )
                else:
                    # compatible with the old format (plain string role)
                    self._entries[vpid] = VoiceprintEntry(
                        role=str(v), name=vpid, entity_id=""
                    )
        except Exception:
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            vpid: {"role": e.role, "name": e.name, "entity_id": e.entity_id}
            for vpid, e in self._entries.items()
        }
        self._path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ── Public API ──────────────────────────────────────────────────────────────

    def bind(
        self,
        voiceprint_id: str,
        *,
        name: str | None = None,
        entity_id: str | None = None,
        role: str = ROLE_USER,
    ) -> VoiceprintEntry:
        """Bind a voiceprint to a person name and/or entity_id (incremental: only the given fields are overwritten)."""
        if role not in (self.ROLE_USER, self.ROLE_ASSISTANT):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        existing = self._entries.get(voiceprint_id)
        entry = VoiceprintEntry(
            role=role,
            name=name or (existing.name if existing else voiceprint_id),
            entity_id=entity_id or (existing.entity_id if existing else ""),
        )
        self._entries[voiceprint_id] = entry
        self._save()
        return entry

    # backward compatible with the old interface
    def register(self, voiceprint_id: str, role: str, name: str | None = None) -> None:
        self.bind(voiceprint_id, role=role, name=name)

    def get(self, voiceprint_id: str) -> VoiceprintEntry:
        """Return the registration info; unknown voiceprints get a default entry (role=user, name=voiceprint_id)."""
        return self._entries.get(
            voiceprint_id,
            VoiceprintEntry(role=self.ROLE_USER, name=voiceprint_id, entity_id=""),
        )

    def resolve(self, voiceprint_id: str) -> str:
        return self.get(voiceprint_id).role

    def display_name(self, voiceprint_id: str) -> str:
        return self.get(voiceprint_id).name

    def entity_id(self, voiceprint_id: str) -> str:
        """Return it directly if bound; otherwise look the person name up in the cognitive graph once, and backfill if found.

        This is resolved lazily rather than in bind() because a self-introduction ("I'm Xiao Li") happens **before**
        extraction creates the entity -- at bind time the person is usually not in the graph yet, so looking up then would always miss.
        """
        entry = self.get(voiceprint_id)
        if entry.entity_id or not self._resolve_entity:
            return entry.entity_id
        # Only look up registered ones: for a voiceprint never bound, get() returns a temporary entry whose name is the id itself,
        # and querying the graph with that as a name is meaningless. If not found (person not in the graph yet) return "" and retry next time.
        if voiceprint_id not in self._entries or not entry.name:
            return ""
        try:
            eid = self._resolve_entity(entry.name) or ""
        except Exception:
            return ""
        if eid:
            self.bind(voiceprint_id, name=entry.name, entity_id=eid, role=entry.role)
        return eid

    def all_display_names(self) -> list[str]:
        """List of all bound real names (excluding default entries that were never bound, whose name still equals the voiceprint_id
        itself). Used in multi-person conversations to decide "does this candidate memory name someone else"."""
        return [e.name for vpid, e in self._entries.items() if e.name and e.name != vpid]

    def to_dict(self) -> dict:
        return {
            vpid: {"role": e.role, "name": e.name, "entity_id": e.entity_id}
            for vpid, e in self._entries.items()
        }


# ── Emotion mapping (emotion label → SuperMem affect vocabulary) ────────────
# This table was originally built against emotion2vec's fixed nine-class vocabulary and only covered
# how those nine words are written. After paper_emotion_detector.py was wired in, labels came from
# Qwen2.5-Omni's free-form emotion words instead -- no longer a closed set of nine (synonymous variants like
# "happiness"/"frustration"/"anxiety" can all appear), so exact string matching would miss a lot and the affect field
# would silently fall back to None. Fallback: when the exact match misses, first run anchor_router.normalize_emotion()
# keyword fuzzy matching (the same 8-class normalization the right-brain anchors already use), then
# map the normalized result onto the affect vocabulary, so free-form labels in any wording work.
_EMO_TO_AFFECT: dict[str, str] = {
    "happy": "excited",
    "sad": "sad",
    "angry": "angry",
    "anxious": "anxious",  "fear": "anxious",   "fearful": "anxious",
    "disgust": "disgusted", "disgusted": "disgusted",
    "surprised": "curious",
    "satisfied": "satisfied",
}

# Fallback mapping from anchor_router._CANONICAL_EMOTIONS (8 classes) → affect vocabulary.
_CANONICAL_TO_AFFECT: dict[str, str] = {
    "happy": "excited",
    "sad": "sad",
    "wronged": "angry",
    "lonely": "sad",
    "conflicted": "anxious",
    "calm": "satisfied",
    "anxious": "anxious",
    "tired": "sad",
}


def emotion_to_affect(emotion: str) -> str | None:
    """Emotion label → SuperMem affect field (None for neutral/unknown)."""
    if not emotion:
        return None
    e = emotion.strip()
    direct = _EMO_TO_AFFECT.get(e)
    if direct:
        return direct
    from supermem.rightbrain.anchor_router import normalize_emotion
    canonical = normalize_emotion(e)
    if canonical == "calm" and e not in ("calm", "neutral"):
        # normalize_emotion falls back to "calm" for unrecognized input -- distinguish "the model really
        # said calm" from "we have never seen this word and fell back passively", so that an unseen word
        # in free-form output is not silently treated as "calm", hiding a real mapping gap.
        return None
    return _CANONICAL_TO_AFFECT.get(canonical)


# ── Core adapter ───────────────────────────────────────────────────────────────

def _looks_like_voiceprint_id(vpid: str) -> bool:
    """Is this id a system-generated voiceprint number, or a person name passed in by the caller?

    Voiceprint numbers look like person_86fed148 / utt_9f2a / spk_3; person names have no such prefix.
    If we cannot tell them apart, a person name is treated as "unidentified", the memory subject is rewritten to "user",
    and questions by name no longer retrieve anything ("Which city did Jiaqi move to?" vs "user moved to Hangzhou").
    """
    v = (vpid or "").lower()
    return v.startswith(("person_", "utt_", "spk_", "speaker_", "voiceprint_"))


def voice_input_to_messages(
    vi: VoiceInput,
    registry: VoiceprintRegistry,
) -> list[dict[str, str]]:
    """Convert VoiceInput.contents into OpenAI messages format.

    - consecutive sentences from the same voiceprint are merged into one message
    - "Speaker N" is replaced with the registered real name (if bound)
      → the LLM sees "Mr. Zhou: ...", and CognitiveAnnotator naturally links it to the Mr. Zhou entity
    - a voiceprint with no bound name must not expose its internal person_id as a "name" to the extraction model:
      tested (see the extract_facts_openai tests), a prefix like "person_f7b840f9: ..."
      is basically ignored by the model as noise, and it falls back to the official mem0 prompt's default assumption
      -- that the sentence was said by the account owner (User); once the account owner's real name
      already appears in existing memories, the text gets attributed straight to the owner (even if the speaker is
      actually a completely different person). Use an explicit "unidentified speaker" label instead, while keeping
      the person_id suffix so that several different unidentified speakers are not merged into one person by the model.
    """
    if not vi.contents:
        return []

    messages: list[dict[str, str]] = []
    current_vpid: str | None = None
    current_sentences: list[str] = []

    def _flush() -> None:
        if not current_sentences or current_vpid is None:
            return
        entry = registry.get(current_vpid)
        label = entry.name
        if label == current_vpid:
            if current_vpid.lower() in ("user", "voice_demo_user"):
                # Caller-supplied account-owner channel (text-mode demos pass
                # the literal id "user" for every utterance): this IS the
                # account owner by definition, so the defensive unidentified
                # label below would be actively wrong here -- it made every
                # stored fact read "Unidentified speaker user stated..."
                # (real user complaint). That label exists for unverified
                # VOICEPRINT ids, where assuming account-owner identity
                # mis-attributes speech across real people; a fixed text
                # channel has no such ambiguity. Once the user self-
                # identifies ("my name is Jiaqi"), the registry binding (see
                # core.py's text-mode binding) replaces this with their
                # actual name via the normal entry.name path above.
                label = "User"
            elif _looks_like_voiceprint_id(current_vpid):
                # Really a voiceprint with no bound name. The label must satisfy two things: not be written into memory as a person name
                # (that is how "Unidentified speaker is a vegetarian" happened),
                # and not be a long descriptive phrase -- that would steer the language of the whole extraction and break retrieval.
                # Use a neutral code name like "Speaker N"; the explanation goes in a separate system hint.
                label = "Speaker 0"
            # Remaining case: the caller passed a person name directly (the eval adapter passes speaker="Jiaqi",
            # multi-person conversations pass the other party's name). Not being in the registry just means it was never registered,
            # not that the identity is unknown -- use it as a name as is. Rewriting it to "user" would make every question by name
            # fail to retrieve ("Which city did Jiaqi move to?" → memory says "user moved to Hangzhou").
        content = f"{label}: " + " ".join(current_sentences)
        messages.append({"role": entry.role, "content": content})

    for c in vi.contents:
        if not c.sentence:
            continue
        if c.voiceprint_id != current_vpid:
            _flush()
            current_vpid = c.voiceprint_id
            current_sentences = [c.sentence]
        else:
            current_sentences.append(c.sentence)

    _flush()

    # the agent's half goes last: the user says "that one then", and the reference can only be resolved from the reply
    if vi.agent_reply and vi.agent_reply.strip():
        messages.append({"role": "assistant", "content": vi.agent_reply.strip()})

    return messages


# ── Slot mapping table ───────────────────────────────────────────────────────
# The targets are real SlotV2 enum values (work/finance/relationships/health/goals/
# daily_life/knowledge, see cognitive_graph/slot_v2.py) -- the values this used to map to,
# "task_todo"/"plan_future"/"event"/"fact"/"routine_habit"/"relationship" etc.,
# did not belong to any real taxonomy, and downstream _write_slotv2_hints also used the wrong store
# attribute/method names; the two problems together meant this path wrote nothing and failed silently.

_VOICE_SLOT_TO_SLOTV2: dict[str, str] = {
    "work": "work", "task": "work", "todo": "work", "deadline": "work",
    "project": "work", "meeting": "work",
    "finance": "finance", "money": "finance", "salary": "finance",
    "goal": "goals", "plan": "goals",
    "fact": "knowledge", "knowledge": "knowledge", "info": "knowledge",
    "event": "daily_life", "experience": "daily_life", "memory": "daily_life",
    "preference": "daily_life", "like": "daily_life", "dislike": "daily_life",
    "routine": "daily_life", "habit": "daily_life", "daily": "daily_life",
    "relationship": "relationships", "people": "relationships",
    "family": "relationships", "friend": "relationships",
    "health": "health", "medical": "health",
    "place": "daily_life", "location": "daily_life",
}


def map_voice_slots_to_slotv2(voice_slots: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for s in voice_slots:
        mapped = _VOICE_SLOT_TO_SLOTV2.get(s.lower().strip())
        if mapped and mapped not in seen:
            seen.add(mapped)
            result.append(mapped)
    return result


# ── Ingest ───────────────────────────────────────────────────────────────────

@dataclass
class VoiceIngestResult:
    voice_id: str
    memory_ids: list[str]
    facts_count: int
    begin_time: str
    end_time: str
    slots: list[str]
    messages_count: int
    affect: str | None = None   # emotion signal from emotion2vec
    error: str | None = None


def ingest_voice_input(
    vi: VoiceInput,
    user_id: str,
    *,
    registry: VoiceprintRegistry,
    repo: Any,
    extractor: Any,
    extra_metadata: dict | None = None,
    session_id: int | str | None = None,
) -> VoiceIngestResult:
    """Ingest one VoiceInput into the left-brain memory store.

    Flow:
      1. contents → messages (replace Speaker N with real names)
      2. retrieve candidate old memories (semantic search over the whole turn) as dedup reference for extraction
      3. extractor.extract(messages, existing_memories=...) → atomic fact sentences (ADD-only extraction)
      4. retrieve candidate old memories per new fact (rather than one search for the whole turn)
      5. ConflictResolver.resolve(facts, existing) → ADD/UPDATE/DELETE/NONE decisions
      6. apply decisions: ADD→append_extracted, UPDATE→update_memory, DELETE→delete_memory
      7. write slot pre-classification into memory_tags
    """
    messages = voice_input_to_messages(vi, registry)
    if not messages:
        return VoiceIngestResult(
            voice_id=vi.id, memory_ids=[], facts_count=0,
            begin_time=vi.begin_time, end_time=vi.end_time,
            slots=vi.slots, messages_count=0, error="empty_contents",
        )

    # This turn's speaker real name (only when bound; if unbound, display_name() falls back to the voiceprint id itself,
    # which is not a name, and passing it to ConflictResolver as the "current speaker" would mislead it) -- used to
    # give the model a hard anchor for "who said this batch of new facts" during conflict resolution, see the
    # multi-person household rule in extract_facts_openai.
    speaker_name: str | None = None
    if vi.contents:
        vpid = vi.contents[0].voiceprint_id
        resolved = registry.display_name(vpid)
        if resolved and resolved != vpid:
            speaker_name = resolved

    # List of known "other people" names -- the existing_memories candidate pool is searched by user_id (the whole household),
    # not filtered by speaker, so when two different people say similarly templated things (e.g. both
    # talking about "which charity I pick for this year's donation drive"), the other person's old memories get pulled into the candidates.
    # ConflictResolver's system prompt already says "different person names must not
    # UPDATE", but that is only a soft hint the model is not guaranteed to follow (actually reproduced: in Jennifer's turn,
    # a memory under Nancy was UPDATEd to Jennifer's answer, "Nancy...is the downtown
    # botanical society"). So add a hard filter in code: candidates whose text explicitly names
    # someone else and does not mention the current speaker's own name are never shown to the model, cutting off
    # accidental UPDATE/DELETE damage at the root.
    other_person_names = (
        [n for n in registry.all_display_names() if n != speaker_name]
        if speaker_name else []
    )

    def _drop_other_named_people(cands: list[dict[str, str]]) -> list[dict[str, str]]:
        if not speaker_name or not other_person_names:
            return cands
        out = []
        for c in cands:
            text = str(c.get("text", ""))
            if speaker_name in text:
                out.append(c)
                continue
            if any(name in text for name in other_person_names):
                continue
            out.append(c)
        return out

    # ── Step 1.5: candidate old memories (dedup reference for extraction) ─────────
    # Previously extraction passed no existing_memories at all, so the model knew nothing about what was already stored,
    # and the "Existing Memories are only for dedup/linked_memory_ids" defence designed into the extraction prompt
    # was useless -- the same thing was re-extracted in full again and again, relying entirely on the downstream ConflictResolver
    # to catch it. Here we first do a rough recall with the whole turn text and feed it to the extractor as dedup reference.
    # Recall candidates using only the user's half: the agent's reply is much longer and would push the old memories we need to compare out of the top 10
    query_text = "\n".join(str(m.get("content", "")) for m in messages
                           if m.get("role") != "assistant").strip()
    existing_for_extraction: list[dict[str, str]] = []
    if query_text and hasattr(repo, "search"):
        try:
            existing_for_extraction = [{"id": h.memory_id, "text": h.text}
                                        for h in repo.search(query_text, user_id=user_id, top_k=10)]
        except Exception:
            pass
    if not existing_for_extraction and hasattr(repo, "existing_for_extractor"):
        try:
            existing_for_extraction = repo.existing_for_extractor(user_id=user_id)
        except Exception:
            pass
    existing_for_extraction = _drop_other_named_people(existing_for_extraction)

    # ── Step 2: extract atomic facts ─────────────────────────────────────────
    # Raw-text fallback: when extraction (via OpenAI) fails or yields no facts, store the whole raw sentence as one memory,
    # so the local E5 vector store can still grow the brain map and retrieve (0 OpenAI). Without a key the downstream ConflictResolver
    # automatically degrades to ADD-only.
    def _raw_fallback() -> list:
        # Off by default: only store high-quality facts extracted by the LLM. Only with SUPERMEM_INGEST_RAW_FALLBACK=1 is the raw text
        # stored as a fallback when extraction fails / yields no facts (for no-key / offline demos).
        if os.environ.get("SUPERMEM_INGEST_RAW_FALLBACK", "0") != "1":
            return []
        raw = " ".join(c.sentence for c in vi.contents if c.sentence).strip()
        if not raw:
            return []
        from supermem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
        return [ExtractedAdditiveMemory(local_id="0", text=raw, attributed_to="user")]

    try:
        extracted = extractor.extract(
            new_messages=messages,
            existing_memories=existing_for_extraction,
            observation_date=vi.begin_time,
            current_date=vi.begin_time,
        )
    except Exception as e:
        extracted = _raw_fallback()
        print(f"[ingest] extraction failed ({e}) -> {'storing raw text as fallback' if extracted else 'no raw text, skipping'}", flush=True)
        if not extracted:
            return VoiceIngestResult(
                voice_id=vi.id, memory_ids=[], facts_count=0,
                begin_time=vi.begin_time, end_time=vi.end_time,
                slots=vi.slots, messages_count=len(messages),
                error=f"extraction_failed: {e}",
            )

    if not extracted:
        extracted = _raw_fallback()
        if extracted:
            print("[ingest] extraction found no facts -> storing raw text as fallback", flush=True)
        else:
            return VoiceIngestResult(
                voice_id=vi.id, memory_ids=[], facts_count=0,
                begin_time=vi.begin_time, end_time=vi.end_time,
                slots=vi.slots, messages_count=len(messages),
            )

    # ── Step 3-4: Conflict resolution (Mem0 V1 style) ───────────────────────
    from supermem.leftbrain.extract_facts_openai import ConflictResolver

    # Candidate old memories: search separately for each new fact (consistent with official mem0: main.py embeds each
    # new_retrieved_fact separately and searches, rather than joining the whole turn into one query).
    # Previously we searched top-10 once with the whole raw turn text -- when one turn covers several things at once
    # (e.g. timeline + budget + partner), the semantic signal of the old memory that actually needs updating is diluted by other topics
    # and easily drops out of the top 10; the model never sees the candidate, can only decide ADD, nobody deletes the old version,
    # and the store keeps several versions of the same thing (the direct cause of "old numbers still in the store after an update").
    # Window size and a second retrieval path (the direct cause of missed UPDATE/DELETE on the benchmark):
    #   1) top-5 is too narrow -- with a few hundred memories, the old value of the same attribute often does not make the top 5, the model cannot see
    #      the candidate and can only decide ADD; widening to 15 adds only a few hundred tokens to the resolver.
    #   2) the vectors of "favorite restaurant is A" and "favorite restaurant is B" are not that similar (the values differ,
    #      and the value is exactly the most informative word in the sentence), so also search with the attribute phrase the extractor provides,
    #      "User's favorite restaurant" (without the value): the old-value memory is highly
    #      similar to that phrase and always enters the window. Event-type facts have an empty attribute and do not trigger the second path.
    # SUPERMEM_CONFLICT_WIDE=0 reverts to the old behaviour (vector top-5 per fact only, no attribute search,
    # resolver without the single-valued attribute rule), for ablation comparisons.
    wide = os.environ.get("SUPERMEM_CONFLICT_WIDE", "1") != "0"
    new_fact_texts = [m.text for m in extracted if m.text]
    existing_map: dict[str, dict[str, str]] = {}
    if hasattr(repo, "search"):
        queries: list[tuple[str, int]] = [(t, 15 if wide else 5) for t in new_fact_texts]
        if wide:
            queries += [(m.attribute, 10) for m in extracted
                        if m.text and getattr(m, "attribute", "")]
        for q, k in queries:
            try:
                for h in repo.search(q, user_id=user_id, top_k=k):
                    existing_map[h.memory_id] = {"id": h.memory_id, "text": h.text}
            except Exception:
                continue
    existing = list(existing_map.values())
    if not existing and hasattr(repo, "existing_for_extractor"):
        try:
            existing = repo.existing_for_extractor(user_id=user_id)
        except Exception:
            pass
    existing = _drop_other_named_people(existing)

    # Conflict detection (new facts vs what is stored → ADD/UPDATE/DELETE) is an LLM call, and
    # the prompt must include existing memories, so **the bigger the store, the slower**: measured 10.2s for this one call on a 95-memory store,
    # half of a whole ingest turn.
    # SUPERMEM_ALWAYS_ADD=1 skips it and always adds -- every memory has a timestamp anyway, and retrieval prefers
    # the latest. The cost is that old and new values of the same attribute both stay in the store ("favorite restaurant is A" and "...is B"),
    # and the answering model picks by date.
    always_add = _os.environ.get("SUPERMEM_ALWAYS_ADD", "0") == "1"

    resolutions = []
    if existing and new_fact_texts and not always_add:
        try:
            resolver = ConflictResolver(single_valued_rule=wide)
            resolutions = resolver.resolve(new_fact_texts, existing, speaker_name=speaker_name)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("ConflictResolver failed, falling back to ADD-only: %s", e)

    # ── Step 5: apply decisions ───────────────────────────────────────────────
    # if resolve succeeded, apply its decisions; otherwise fall back to the original ADD-only path
    if resolutions:
        # build a fact_text → ExtractedAdditiveMemory map so ADD can reuse metadata
        fact_map = {m.text: m for m in extracted if m.text}
        to_add: list = []
        for r in resolutions:
            if r.event == "ADD" and r.text:
                orig = fact_map.get(r.text) or next(iter(fact_map.values()), None)
                if orig:
                    from supermem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
                    to_add.append(ExtractedAdditiveMemory(
                        local_id=r.memory_id,
                        text=r.text,
                        attributed_to=orig.attributed_to,
                    ))
            elif r.event == "UPDATE" and r.memory_id and r.text:
                # UPDATE is **treated as ADD by default**; old memories are no longer rewritten.
                #
                # "Which old memory is this sentence correcting" is itself an unreliable judgement; the cost of getting it wrong
                # (overwriting a good memory) far outweighs the benefit of getting it right (saving one duplicate). Measured over 158
                # UPDATEs: 34% were identical old vs new (pure no-ops that still recomputed the embedding and reset the LLM-assigned
                # slot tags to the vector version), 11% got shorter and lost information, some wrote ASR noise
                # into the old memory ("mentioned considering something" → "mentioned being a bit slow slowly..."),
                # and some even oscillated ("has plans next Wednesday" → more specific → back to vague).
                # Newer versions of upstream mem0 also removed update.
                #
                # With appending instead, there will be more duplicate memories -- that is left to dedup/archiving, a problem that **can be
                # fixed afterwards**; information lost to overwriting cannot be recovered afterwards.
                # For the old behaviour: SUPERMEM_APPLY_UPDATE=1.
                if os.environ.get("SUPERMEM_APPLY_UPDATE", "0") == "1" and hasattr(repo, "update_memory"):
                    # Pass this session's date: the updated memory now describes what was said this time,
                    # so the timestamp must move with it, otherwise the new fact hangs on the merged memory's old date.
                    repo.update_memory(r.memory_id, r.text, session_id=session_id,
                                       observed_at=vi.begin_time, user_id=user_id)
                else:
                    # take metadata from fact_map just like the ADD branch, so this rewritten one does not
                    # lose attributed_to (without it, it would go down the "unverified voiceprint" defensive path).
                    orig = fact_map.get(r.text) or next(iter(fact_map.values()), None)
                    if orig:
                        from supermem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
                        to_add.append(ExtractedAdditiveMemory(
                            local_id=r.memory_id, text=r.text,
                            attributed_to=orig.attributed_to,
                        ))
            elif r.event == "DELETE" and r.memory_id:
                if hasattr(repo, "delete_memory"):
                    repo.delete_memory(r.memory_id)
        extracted = to_add  # only append the ADD part

    # entity_id binding: collect every bound voiceprint in this batch
    speaker_entity_map = {
        vpid: registry.entity_id(vpid)
        for vpid in {c.voiceprint_id for c in vi.contents}
        if registry.entity_id(vpid)
    }

    # emotion2vec → affect
    affect = emotion_to_affect(vi.dominant_emotion())

    meta = {
        "turn_id":            vi.id,
        "voice_id":           vi.id,
        "time_start":         vi.begin_time,
        "time_end":           vi.end_time,
        "voice_slots":        vi.slots,
        "source":             "voice",
        "speaker_entity_map": speaker_entity_map,
        "affect":             affect,
        **({"session_id": session_id} if session_id is not None else {}),
        **({"background_sounds": vi.environment} if vi.environment else {}),
        **(extra_metadata or {}),
    }
    memory_ids = repo.append_extracted(extracted, user_id=user_id, extra_metadata=meta)
    print(f"[ingest] stored {len(memory_ids or [])}: {[m.text[:20] for m in extracted][:3]}", flush=True)

    # write pre-classified slots into memory_tags
    slotv2_hints = map_voice_slots_to_slotv2(vi.slots)
    if slotv2_hints and memory_ids:
        _write_slotv2_hints(repo, user_id, memory_ids, slotv2_hints)

    return VoiceIngestResult(
        voice_id=vi.id,
        memory_ids=memory_ids or [],
        facts_count=len(extracted),
        begin_time=vi.begin_time,
        end_time=vi.end_time,
        slots=vi.slots,
        messages_count=len(messages),
        affect=affect,
    )


def _write_slotv2_hints(repo: Any, user_id: str, memory_ids: list[str], slotv2_tags: list[str]) -> None:
    """The voice module's own coarse slot hints are a medium-confidence signal -- they share the same
    memory_tags table with the LLM tags from core.py (confidence=0.95) and the tags assigned automatically by embedding at write time (real cosine
    scores, see memory_repository_v2.py._tag_memory_slots_v2), so they get a fixed confidence lower than both, and never override a more
    trustworthy judgement (upsert_memory_tags is an upsert; a later write for the same slot overwrites the confidence)."""
    try:
        store = repo._cognitive_store  # type: ignore[attr-defined]
        if store is None or not hasattr(store, "upsert_memory_tags"):
            return
        tags = [(slot, 0.5) for slot in slotv2_tags]
        for mid in memory_ids:
            store.upsert_memory_tags(mid, user_id, tags)
    except Exception:
        pass
