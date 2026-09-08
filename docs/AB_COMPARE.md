# A/B compare demo — how it works

One reference for everything built on top of VoiceMem's web demo: what the
compare view is, how a turn flows end to end, what actually gets written into
memory and what gets sent to the LLM, and where every piece lives in the code.

Line references are to the current tree.

---

## 1. What this is

The demo's claim is *"inject retrieved memory into an ordinary LLM and the
replies get better."* With a single reply on screen that claim is unprovable —
you see a good answer, but not what the same model would have said without the
memory.

So every turn now runs **twice**, side by side:

| arm | gets |
|---|---|
| **With VoiceMem** | persona + language + session history + **retrieved memory** |
| **Without VoiceMem** | persona + language + session history |

Same utterance, same retrieval result, same model, same persona. The only
variable is whether the memory block is in the system prompt. Whatever differs
on screen came from the memory.

There is no automated judge. You read the two replies and decide.

---

## 2. A turn, end to end

```
 mic ──► VAD + streaming ASR ──► partial text ──► speculative retrieval
  │                                  │                    │
  │                                  │            Classify → Search
  │                                  │                    │
  │                              (turn ends)      Pending{ text,
  │                                  │              memory_context, result }
  │                                  ▼
  │                        _compare_turn()  web/run.py:1413
  │                                  │
  │            ┌─────────────────────┴─────────────────────┐
  │            ▼                                           ▼
  │   arm A: openai_reply(model)                  arm B: openai_reply(model)
  │      system = shared + memory                    system = shared
  │            │                                           │
  │            └──────────────► cmp_* over ws ◄────────────┘
  │                                  │
  ▼                                  ▼
 audio archived            queue_remember_turn(pending, reply_A)
                                     │
                            vm.ingest(text, agent_reply=…)
```

Key property: **retrieval happens while you are still speaking.** The partial
transcript drives `Classify` + `Search` so that by the time VAD confirms you
stopped, `Pending.memory_context` is already computed. Neither arm waits for it.

### Where the pieces live

| step | code |
|---|---|
| streaming ASR + VAD + speculation | `voicemem/stream.py` |
| turn announce (transcript, hits, emotion) | `web/run.py:1375` `_announce_turn` |
| shared system prompt | `web/run.py:1396` `_compare_shared_system` |
| the compare turn | `web/run.py:1413` `_compare_turn` |
| two-arm fan-out | `web/compare.py:226` `fan_out` |
| one arm's stream | `web/compare.py:189` `_run_arm` |
| memory on/off for an arm | `web/compare.py:160` `context_for` |
| ingest, once | `web/run.py:1819` `queue_remember_turn` → `:1740` `remember_turn` |

---

## 3. What goes into memory

This is the part most people get wrong, so it is worth being precise: **the
user's words and the agent's words are treated differently.**

`VoiceMem.Ingest()` — `voicemem/orchestrator.py:1092` — is called once per turn
with both halves:

```python
vm.ingest(text, agent_reply=reply, async_facts=True, …)
```

### The user's utterance → extracted into facts

`_finish_ingest` (`voicemem/orchestrator.py:1259`) runs LLM fact extraction and writes to
both hemispheres:

- **Left brain** — hard facts (`ingest_facts`, `voicemem/orchestrator.py:1291`)
  `"User has a cat named Mocha who is three years old."`
- **Right brain** — persona and emotion notes (`_write_right_brain`, plus
  `learn_from_reaction` for emotion attribution)

### The agent's reply → stored verbatim, never extracted

`voicemem/orchestrator.py:1308` writes one row with `attributed_to="assistant"` and **no
fact extraction**. The reason is in the code comment, and it matters:

- Extracting it yields things like *"the assistant recommended asparagus"* —
  that records what the assistant did, **not who the user is**. And its wording
  overlaps heavily with the current conversation, so it scores high on the next
  query and pushes genuinely relevant memories out of top-k.
- It is still stored, because otherwise *"what did you tell me earlier?"* has no
  answer.

### Retrieval excludes the assistant by default

`voicemem/leftbrain/mem0_backend_store.py:402`:

```python
if not include_assistant:
    flt["role"] = {"ne": "assistant"}
```

Those rows come back only when the question is *about the assistant* —
`asks_about_assistant(query)`, `voicemem/leftbrain/brain.py:147`.

### The reply is also used twice, for two different jobs

`Ingest` keeps two separate replies (`voicemem/orchestrator.py` ~1145):

| value | which reply | used by |
|---|---|---|
| `agent_reply` | **this** turn's | left brain, to disambiguate the user's sentence ("yes, that one") |
| `prior_reply` | the **previous** turn's | right brain, for emotion attribution — what the user is reacting to |

### Ingestion is asynchronous

`async_facts=True` runs extraction on a background thread, so the next turn's
recording is never blocked. Typical: 2.5–5.5s. A turn with no new fact returns
in ~0.01s — that is dedup working, not a failure.

---

## 4. What goes to the LLM

### Building the memory block

`Search()` returns hits; `build_memory_context()`
(`voicemem/memory_api.py:40`) renders them:

```
factual memory CONTEXT you know about the user (top5 in left brain):
- [2026-09-08] Sanyam works on voice AI at Unpod.
- [2026-09-08] Sanyam Sharma has knee pain and is inquiring about when doctors are available.

user's emotion & characteristics (top5 in right brain):
- …
Let these shape your tone, what you bring up, and what you leave alone.
Never state them back to the user.
```

Right-brain content gets its own heading and that explicit instruction on
purpose: those are internal notes. Without the separation the model reads them
aloud, and replies turn into *"your coping mechanism is going to the gym"* —
which sounds like someone reading a file about you.

### Splitting shared from variable

```
_compare_shared_system()        ← identical for both arms
    persona (_RT_PERSONA)
  + language note (_lang_note)
  + session history (_history_block)

context_for(arm, memory_context) ← the only difference
    arm.memory ? memory_context : ""
```

Two deliberate decisions here:

- **Session history is shared.** A plain LLM with a conversation window really
  does see recent turns. VoiceMem's value is recall *beyond* that window, so
  giving only one arm the history would be measuring the wrong thing.
- **A `memory=False` arm gets `""`, not `_NO_MEMORY_NOTE`.** That note ("say you
  don't know, don't invent") is for a memory-enabled turn that retrieved
  nothing — a third condition. Telling the baseline it has no memories would be
  testing a differently-prompted model, not a model without memory.

### Consequence worth knowing before you demo

Both arms share the persona, which tells the model it is this person's
companion. A companion with no memory **confabulates rather than admitting
ignorance**. Observed, same question, separate runs:

> *"You work at that tech company…"* · *"that marketing agency…"* · *"Kartik
> Supply Co."*

That is the honest baseline — same prompt, no memory — and it is the contrast
the demo exists to show. It is not a bug.

### Asking the right question

| ask | works? |
|---|---|
| "What did I tell you about my knee?" | ✅ answer exists only in memory |
| "Where do I work?" | ✅ fact from an earlier turn |
| "Hi, I'm X from Y working as Z" | ❌ all three are **in this message** |

Both arms always receive the current user message. An arm repeating something
you just said is not recall. Memory only shows up on things said *earlier*.

---

## 5. Ingest happens exactly once

Two traps, both handled:

**1. `vm.reply_stream()` remembers on your behalf.** `voicemem/core.py:241`
wraps the provider in `capture(…, remember_reply)`. Calling it once per arm
would store **two agent replies for one user turn**. So the arms call the plain
provider `voicemem.reply.openai_reply()` directly, and the turn is ingested by
the existing `queue_remember_turn` path — with **panel A's** reply. Panel B's
reply is a baseline; it never enters memory.

**2. Both arms share the persona, not the model.** Each arm may point at a
different model, endpoint or key. Anything else being different would make the
comparison dishonest, so only `memory_context` and the model config vary.

---

## 6. Speech to text

Local streaming ASR was replaced with the transcription API. Two local models
ship with the repo and **both carry Chinese in their vocabulary**, so they fall
back to it on unclear audio:

```
spoken: "hello how are you"
paraformer-zh (Chinese-only)  → 'lohow are youmycat is nameda and sheis three yearsold'
sherpa zh-en (bilingual)      → 'HELLO HOW ARE YOU MY CAD IS NAMED MOKA…'   then 狗日按摩
transcription API             → 'Hello, how are you? My cat is named Mocha…'
```

That garbage transcript is what got extracted and stored, so one bad turn
poisoned the memory. `models/asr/` is now empty; nothing is downloaded.

`OpenAIStreamingASR` — `voicemem/utils/audio/asr.py:222`:

- **Partials also go over the network,** one request at a time (single-flight).
  `feed()` returns the last text it has and starts a request only when none is
  in flight, so the audio loop never blocks. This is not optional: the
  end-of-turn check in `stream.py` is gated on `self._text.strip()`, so an ASR
  that stays silent until the end would never let a turn finish — and the
  speculative prefetch eats the partial text.
- `flush()` is the one synchronous call, and its result is the turn's final text.
- A failed partial is swallowed (flush re-transcribes everything anyway); a
  failed flush keeps the last good partial rather than dropping the turn.

### Language must be pinned

`VOICEMEM_ASR_LANGUAGE=en`, not `auto`. Spoken Hindi and Urdu are the same
language phonetically, so `auto` flip-flopped between scripts:

```
"hello, how are you?"  →  ہیلو، ہاو آر یو؟
(a later turn)         →  تویی نمیره بنپالا
```

Retrieval is vector-based. The same fact written in Latin, Devanagari and Arabic
script lands in three different places and most of the memory stops being
findable. **One script per memory space.** Set `hi` for a Hindi demo instead —
but then speak only Hindi.

Note the split: what the assistant **says** follows `UI_LANG` (switchable per
turn, `_lang_note()`), while what gets **stored** is a property of the space,
fixed at creation — `voicemem/lang.py` supports only `en`/`zh`. So Hindi replies
come out of an English memory space without mixing scripts in one vector store.

---

## 7. The UI

```
┌──────── middle ────────┐┌──────────── right pane ─────────────┐
│ LIVE INPUT             ││          BRAIN  (42%)               │
│  transcript, type box  ││   you ●──▶ work ●──▶ ● memory       │
│  emotion/entity/schema ││                                     │
├────────────────────────┤├─────────────────────────────────────┤
│ TOP-K RECALL           ││ YOU  where do i work?               │
│  left facts + scores   ││ ┌── With VoiceMem ─┐┌─ Without ───┐ │
│  right profile         ││ │ 1072ms · 1756ms  ││ 1053ms      │ │
├────────────────────────┤│ │ You work on…     ││ You work at…│ │
│ AI REPLY  ⚖ A/B        ││ └──────────────────┘└─────────────┘ │
│  ⚙ model/memory/key    ││ ▸ Injected memory context 212 chars │
│  Idle · Pause · Talk   ││ YOU  what's my name?          ↓scroll│
└────────────────────────┘└─────────────────────────────────────┘
```

- **Turn feed** — every turn appends a block, so you scroll back through the
  conversation. It follows new output only when already at the bottom; pulling
  the view down while someone reads an earlier turn is the worst thing it could
  do.
- **Middle column is a control panel** — which arm has memory, model, endpoint,
  key. Its reply bodies are hidden; the same text in two places just reads
  twice.
- **Labels follow the server, not the checkbox.** `cmp_start` carries the
  server's own per-arm `memory` flag; the heading, the "injected into" badge and
  the checkbox all read that. Earlier they read the local checkbox, so when the
  two drifted the screen described the opposite of what happened — *"212 chars →
  A"* above a panel titled *With VoiceMem*, while the server had handed the
  memory to B. A demo whose purpose is to be believed cannot lie about which
  side got the memory.
- **Compare mode is silent.** One audio stream cannot speak two replies. Switch
  ⚖ A/B off for a single reply with voice.

### Latency is time to first token

Total is misleading here: it tracks how long the answer is, so a chatty arm
looks slower than a terse one even when it started speaking first. What a
listener feels is the wait before the first word. `_run_arm` stamps TTFB on the
first delta so the UI can show it mid-stream, and repeats it on `cmp_done`
alongside the total. An arm that produced nothing reports `ttfb: null` — zero
would read as instantaneous when the truth is it never spoke.

Measured from this machine, `gpt-4o-mini`:

| condition | median TTFB |
|---|---|
| small prompt (507 chars) | 927ms |
| big prompt (2350 chars) | 684ms |
| `gpt-4o`, big prompt | 660ms |

**Prompt size and model choice barely matter.** ~650ms is the network floor from
here to OpenAI. Two arms running concurrently push the observed number to
0.8–1.6s, and it varies several hundred ms run to run.

The first call of a process used to cost ~2.0s (DNS, TLS, client construction).
`_warm_network()` now pays that at boot — both the reply model and the
transcription endpoint, in parallel, before uvicorn starts. Startup reports
~2.3s of warmup and the first turn is no longer an outlier. This removes a
reliable 2s penalty; it does not make replies faster.

### The brain draws the actual signal path

Each beam carries its own clock, so stages fire in order rather than all at
once:

```
you ──in(blue)──▶ the slot clusters this turn classified into
                    ──recall(gold)──▶ the memory nodes that matched
                    ──recall(gold)──▶ the right hemisphere, later still
      ◀──out(mint)── when the memory arm's first token arrives
```

Every argument is something the backend computed for that turn — `slots` from
the local E5 classifier (the same values in the `schema` chip), the node lists
from the retrieval hits. It is the pipeline drawn, not decoration over it.

Practical limits: partials spark the origin at 400ms intervals (they arrive
~100ms apart and lighting each one washes the canvas out), and fan-out is capped
at 4 targets per source for the same reason.

---

## 8. Running it

```bash
bash run_demo.sh          # http://localhost:8787
```

Everything comes from `.env` (chmod 600):

```
OPENAI_API_KEY=…
VOICEMEM_ASR=openai
VOICEMEM_ASR_MODEL=gpt-4o-mini-transcribe
VOICEMEM_ASR_LANGUAGE=en
```

Overrides:

```bash
VOICEMEM_PORT=8788           bash run_demo.sh   # 8787 taken
VOICEMEM_REPLY_MODEL=gpt-4o-mini bash run_demo.sh
VOICEMEM_ASR_LANGUAGE=hi     bash run_demo.sh   # Hindi speech
```

### Compare state over HTTP

```bash
curl localhost:8787/api/compare
curl -X POST localhost:8787/api/compare -H 'Content-Type: application/json' \
  -d '{"enabled":true,"arms":[{"label":"a","memory":true},{"label":"b","memory":false}]}'
```

Compare is **on by default** (`CompareState.enabled = True`) — both replies are
the point, and having to find a toggle first was the confusion this replaced.

A per-panel `api_key` lives only in process memory: masked in `Arm.__repr__`,
reported as `has_key` by `GET`, never echoed back. Fine for a localhost demo,
and the reason it must not be bound to a public interface.

**Known gap:** `POST /api/compare` from outside the page (curl, a second tab)
changes the server state but does not update an already-open page. The page
syncs on load and on its own actions. There is no cross-client sync requirement.

---

## 9. WS protocol additions

Existing `answer_*`, `memory_hits`, `user_transcript`, `partial_transcript` and
`tag_update` are unchanged. **In compare mode the server emits no `answer_*` at
all** — which is why the caption and PCM playback machinery, tied to the
playback clock, cannot regress.

| type | payload | when |
|---|---|---|
| `cmp_ctx` | `{context, chars}` | once per turn, before the arms start |
| `cmp_start` | `{panel, model, memory}` | per arm, on first call |
| `cmp_delta` | `{panel, text, ttfb?}` | per chunk; `ttfb` on the first only |
| `cmp_done` | `{panel, text, ms, ttfb, error?}` | per arm, on completion or failure |

### Failure handling

- One arm failing (bad model, 401, endpoint down) produces `cmp_done{error}` for
  that panel; the other completes normally.
- Both failing additionally sends `{"type":"error"}`. The turn is **still
  ingested** with an empty agent reply — the facts in what the user just said
  should not be lost to one bad key.
- A disconnect is **not** an arm failure. `client_gone()`
  (`web/compare.py:172`) recognises both shapes Starlette uses
  (`WebSocketDisconnect`, and `RuntimeError` mentioning a sent close message);
  `_run_arm` re-raises those, and `fan_out` gathers with
  `return_exceptions=True` so one arm's disconnect cannot cancel the other
  before the exception reaches the ws handler. Before this, closing the tab
  mid-turn logged two bogus panel errors and then raised
  `RuntimeError('Cannot call "send" once a close message has been sent.')` with a
  full traceback.

---

## 10. Tests

```bash
python tests/test_compare.py        # 28  fan-out, arms, secrets, ttfb, disconnect
python tests/test_openai_asr.py     # 14  single-flight partials, flush, warmup
node   tests/test_compare_ui.mjs    #  8  panel labels follow the server
node   tests/test_brain_signal.mjs  #  7  signal cascade staging
```

No network, no models — providers and transcribe calls are injected, and the two
`.mjs` suites load the real functions out of `voicemem.html` against a stub DOM
(there is no browser on this machine).

`tests/` is in `.gitignore`; the existing test files are force-added, so new ones
need `git add -f`.

---

## 11. Limits worth knowing

- **No Hindi speech input while pinned to `en`.** Hindi replies work; Hindi
  speech needs `VOICEMEM_ASR_LANGUAGE=hi`, and then don't mix in English.
- **Every turn is a network round trip now.** Local ASR was ~150ms; the API is
  ~650ms plus concurrency. That is the cost of not shipping a model.
- **Dedup is similarity-based, not exact.** The same fact in different wording
  can land as separate rows:
  ```
  • Sanyam works on voice AI at Unpod.
  • Sanyam works on voice AI at Unpod.
  • Sanyam Sharma is a voice AI engineer at Unpod.ai.
  ```
  Duplicates eat top-5 slots and push real matches out.
- **One memory space, one person.** Facts about two different people in one
  library make the comparison meaningless — a question about one can retrieve
  the other's history.
- **Proper nouns are the weak spot** of any speech-to-text. Names and companies
  get mangled (`MOKA` for Mocha, `Unport.ai` for Unpod), and the wrong spelling
  is what gets stored.
- **Memory Space is gone from the UI** — one library, no picker. `activeSpace`
  still exists internally; the memories really do live in a directory, the user
  just no longer has to know that.
