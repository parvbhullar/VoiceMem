# A/B Compare UI — same turn, memory injected vs not

Date: 2026-09-07
Status: approved, ready for implementation plan

## Problem

VoiceMem's value claim is "inject retrieved memory into an existing LLM and the
replies get better". Today the web demo shows only one reply, so the claim is
unproven on screen: a viewer sees a good answer but cannot see what the same
model would have said without the memory block.

We want the demo to answer that in one glance — two panels, same user turn, one
arm with the memory context injected and one without, streaming side by side.

## Goal

In the existing web demo, a `⚖ Compare` toggle splits the reply area into two
panels. Each panel has its own model (user-supplied) and its own memory on/off
switch. The user speaks once; both arms answer from the same transcript and the
same retrieval result. The user judges the difference by eye.

Non-goals (explicitly out of the first cut):

- No automated LLM judge, verdict line, or win/loss scoreboard. The user judges.
- No audio for the compare panels. Compare mode is text-only; mic input still
  works through the existing VAD/ASR path. A per-panel "listen" toggle is a
  possible phase 2, reusing `utils.tts_stream`.
- No export, no persisted compare history.

## Constraints discovered in the codebase

1. **`vm.reply_stream()` writes the reply into memory.** `voicemem/core.py:241`
   wraps the provider in `capture(..., lambda answer: remember_reply(text, answer))`.
   Calling it once per arm would store two agent replies for one user turn and
   corrupt the memory space. Arms must therefore call the plain provider
   `voicemem.reply.openai_reply(...)` directly, and the turn must be ingested
   exactly once by the existing path.

2. **Memory context is already computed off the critical path.** The
   speculative prefetch in `anticipate()` fills `Pending.memory_context`
   (`web/run.py:1107`) before the user finishes speaking. Both arms consume that
   same string — no second retrieval, so the comparison is a true ablation of
   injection, not of retrieval.

3. **`realtime` mode is one audio stream.** It cannot carry two arms. When
   compare is on, the turn is served by the chat-completions fan-out regardless
   of `--mode`. The realtime connection stays open but its input buffer must be
   cleared per diverted turn (`conn.input_audio_buffer.clear()`, present in the
   installed openai 3.8.0 SDK) — otherwise the uncommitted mic audio piles up
   and the next non-compare commit replays earlier turns. This is why compare
   does not require restarting the server in `--mode llm_tts`, which matters
   because `run_demo.sh` defaults to realtime.

4. **The frontend reply renderer is coupled to the playback clock.**
   `answer_start/_delta/_done` in `web/voicemem.html:2085-2109` drive spoken
   captions, PCM output and live chat drafts. Compare must not reuse those
   messages; it gets its own `cmp_*` message types and its own render state.

5. **`web/run.py` is already 3151 lines.** Compare logic goes into a new module
   rather than growing it further.

## Architecture

```
mic → existing VAD/ASR (untouched) → Pending{text, memory_context, result}
                                          │
                    compare OFF ──────────┴──────── compare ON
                          │                              │
              vm.reply_stream(text, ctx)      compare.fan_out(text, ctx, arms)
              (auto-remembers reply)            ├─ arm A: openai_reply(model_a)(text, ctx_a)
                          │                     └─ arm B: openai_reply(model_b)(text, ctx_b)
                    TTS → audio                        ctx_x = ctx if arm.memory else ""
                          │                              │
                    remember_turn(reply)          remember_turn(reply_A)  ← exactly once
```

### Components

**`web/compare.py`** (new, ~120 lines, no FastAPI or voicemem-core imports at
module load beyond `voicemem.reply`)

```python
@dataclass
class Arm:
    label: str           # "a" | "b"
    model: str = ""      # "" -> server default via llm_config resolution
    memory: bool = True
    base_url: str = ""
    api_key: str = ""    # masked in __repr__

@dataclass
class CompareState:
    enabled: bool = False
    arms: tuple[Arm, Arm] = (Arm("a"), Arm("b", memory=False))

async def fan_out(text, memory_context, arms, send, system) -> dict
```

`fan_out` builds one `openai_reply(model, api_key, base_url, system)` provider
per arm, runs both generators concurrently, and emits `cmp_start` / `cmp_delta`
/ `cmp_done` per panel as tokens arrive. It returns
`{"a": text, "b": text, "latency_ms": {"a": int, "b": int}, "errors": {...}}`.

Both arms get the **same** `system`: persona (`_RT_PERSONA`) + language note +
session-history block — everything that is identical for the two arms. Only
`memory_context` differs, so the on-screen difference can only come from the
memory. Sharing the persona has a consequence worth knowing before demoing:
the persona tells the model it is this user's companion, so the **no-memory arm
confabulates rather than admitting ignorance** (observed: "your cat is about
three years old", then "about six" on a re-run). That is the honest baseline —
same prompt, no memory — and it is the difference the demo is meant to show.

`ctx_for(arm)` is `memory_context if arm.memory else ""`. Note the existing
`_NO_MEMORY_NOTE` fallback (`web/run.py:298`) is **not** applied to a
`memory=False` arm — that arm must represent a plain LLM with no memory wiring
at all, not an LLM told it has no memories.

**`web/run.py`** (~40 lines changed)

- Module-level `COMPARE = compare.CompareState()`.
- In `voicemem_llm_tts()`: if `COMPARE.enabled`, send `cmp_ctx`, call
  `compare.fan_out(...)`, skip the TTS queue, the `synth`/`speak` tasks and all
  `answer_*` sends, then fall through to the existing `_push_history` +
  `queue_remember_turn(pending, replies["a"], owner, history_turn_id,
  memory_vm=memory_vm)`. The `timeline` object is left empty (no
  `append_audio` / `append_text`) but still gets `context_saved = True` at the
  end, mirroring `web/run.py:1555`, so the timeline bookkeeping cannot conclude
  the turn's context was lost.
- In `realtime_session()`, at the single turn dispatch site
  (`web/run.py:2611`): when `COMPARE.enabled`, `await
  conn.input_audio_buffer.clear()`, skip the `AudioTimeline` creation and the
  `turn.update(live=True, ...)` registration (the event pump's completion
  handler keys off `turn["pending"]` and must not see a compare turn), run
  `compare.fan_out(...)`, then call `_push_history` and `queue_remember_turn`
  inline. `response_idle` is left set — no realtime response is created.
- `build_app(..., compare=(get_fn, set_fn))` gains:
  - `GET /api/compare` → `{"enabled": bool, "arms": [{label, model, memory, base_url, has_key}]}`
    (never returns the key itself)
  - `POST /api/compare` → accepts the same shape plus `api_key`, validates, returns the sanitized state.

**`web/voicemem.html`**

- `⚖ Compare` toggle in the header; calls `POST /api/compare`.
- When on, a compare section renders below the transcript: two columns, each with
  a model text input, a memory switch, a latency badge, and the streaming reply.
- A collapsible **Injected memory context** block showing the exact string sent
  to the memory-on arm (from `cmp_ctx`), so "what VoiceMem contributed" is
  visible verbatim.
- Four new `case 'cmp_start' | 'cmp_delta' | 'cmp_done' | 'cmp_ctx'` handlers
  writing into a new `s.ui.cmp` state object, plus one new render function called
  from the existing `paint(s)`. No changes to the `answer_*` cases.

### WS protocol additions

| type | payload | when |
|---|---|---|
| `cmp_ctx` | `{context: str, facts: int, rb: int}` | once per turn, before the arms start |
| `cmp_start` | `{panel: "a"\|"b", model: str, memory: bool}` | per arm, on first call |
| `cmp_delta` | `{panel, text}` | per token chunk |
| `cmp_done` | `{panel, text, ms: int, error?: str}` | per arm, on completion or failure |

Existing `answer_*`, `memory_hits`, `user_transcript`, `partial_transcript` and
`tag_update` messages are unchanged. In compare mode the server emits no
`answer_*` messages at all.

## Data flow per turn (compare on)

1. VAD confirms end of utterance → `Pending{text, memory_context, result}` (unchanged).
2. Server sends `user_transcript` and `memory_hits` (unchanged — brain graph still lights up).
3. Server sends `cmp_ctx` with `pending.memory_context`.
4. `fan_out` streams both arms concurrently into `cmp_*` messages.
5. `_push_history(...)` records panel A's reply as the session context.
6. `queue_remember_turn(pending, replies["a"], ...)` ingests the turn once, in
   the background, exactly as today.

## Error handling

- `asyncio.gather(..., return_exceptions=True)` per arm. One arm failing (bad
  model name, 401, endpoint down) produces `cmp_done{panel, error}` for that
  panel; the other arm continues and completes normally.
- If both arms fail, additionally send the existing `{"type": "error"}` so the
  page's current error toast fires. The turn is **still ingested**, with an
  empty agent reply: the facts in what the user just said must not be lost
  because of one bad key, and an empty `agent_reply` is exactly what
  `Memory.remember(text)` does. (This reverses an earlier draft of this spec
  that skipped ingestion.)
- `POST /api/compare` rejects a payload with a non-string model or a missing
  panel label with HTTP 400 and leaves the previous state intact.
- A `memory=True` arm with an empty `memory_context` (nothing retrieved) is not
  an error: it sends `cmp_start{memory: true}` and the frontend shows an
  "0 facts injected" badge, which is itself a useful demo state.

## Secrets

Per-panel `api_key` lives only in the in-process `CompareState`. It is never
written to disk, never returned by `GET /api/compare` (which exposes only
`has_key: bool`), and `Arm.__repr__` masks it so an exception traceback or a
debug print cannot leak it. Blank `api_key`/`base_url` fall through to the
server's own `.env` key and the normal `llm_config` resolution chain.

This endpoint accepts a credential over plain HTTP on localhost. That is
acceptable for a local demo and must not be exposed on a public interface; the
`POST /api/compare` handler docstring states this.

## Testing

`tests/test_compare.py`, `unittest` (matching `tests/test_offline_engine.py` and
`tests/test_session_context.py`; no pytest dependency). Arms are injected fake
async generator providers, so no network and no model downloads.

1. `fan_out` passes `memory_context` to a `memory=True` arm and `""` to a
   `memory=False` arm.
2. Both arms receive the identical `text` and the identical `system`.
3. `cmp_delta` messages carry the right `panel` label and interleave without
   cross-contaminating the two accumulated strings.
4. One arm raising mid-stream still yields a complete `cmp_done` for the other,
   plus `cmp_done{error}` for the failed one.
5. Both arms raising produces an `error` message and returns empty replies.
6. `repr(Arm(api_key="sk-secret"))` does not contain `sk-secret`.
7. `CompareState` round-trips through the sanitize function used by
   `GET /api/compare` without exposing `api_key`.
8. A `memory=True` arm with `memory_context == ""` still calls its provider with
   `""` and does not substitute `_NO_MEMORY_NOTE`.

End-to-end verification actually run (server on 8787, `gpt-4o-mini` both arms,
typed turns over `/ws`):

1. `POST /api/compare` enables compare; `GET` reports it; an unknown panel is a
   400 and leaves the previous state in place.
2. A statement turn ("My cat is called Mocha and she is three years old") with
   both arms memory-off streams two replies, emits **zero** `answer_*` messages,
   and is ingested once — `/api/memories` then holds "User has a cat named
   Mocha who is three years old."
3. Asking "How old is my cat again?" with panel A memory-on sends a non-empty
   `cmp_ctx` ("factual memory CONTEXT … - [2026-09-07] User has a cat named
   Mocha who is three years old.") and panel A answers "Mocha is three years
   old."
4. The same question with both arms memory-off produces two different invented
   ages across runs, confirming no memory leaks into a memory-off arm.

Browser verification (headless Chromium 149 via patchright, 1280x720, real ws
turns against the running server):

5. Toggle round-trip: `#cmp` shows, `#reply` goes `display:none`, panel A gains
   the `mem` class, and `GET /api/compare` reports `enabled: true`; clicking
   again reverses all of it. Zero console errors, zero failed requests.
6. A full turn with panel A memory-on renders "Mocha is three years old." beside
   panel B's "Your cat is nine years old.", with the context badge reading
   `134 chars → A`.
7. With both arms memory-off the badge reads "neither arm has memory on — this
   went into no prompt", and the two arms invent different ages (5 and 3; across
   six raw-ws runs: 6, 4, 7, 3, 7, "college"). That variance is what rules out a
   memory leak into a memory-off arm — the baseline confabulates.

Two notes for anyone driving this headlessly: Chromium 149 blocks `ws://localhost`
from a page under Local Network Access checks, so a browser-driven turn needs
`--disable-features=LocalNetworkAccessChecks`; and the page has no microphone, so
turns were typed into `#textIn` after clicking `#talkBtn`.

Layout was measured rather than eyeballed, because the first version collapsed:
`.center` is a fixed-height flex column, so the two arms were squeezed to 64px at
1440x900 and 5px at 1280x720, with the text inside `overflow:hidden`. Fixed with
a `min-height` floor on `.cmp-cols` plus a `cmp-on` class that makes the middle
column scrollable only while compare is open. Re-measured: arms are 190-240px tall
at 1280x720, 1440x900 and 1600x1000, with no page overflow and the footer still
on screen.

A second layout bug turned up only once the page was looked at on a real screen:
`.cmp` was `flex:1`, i.e. `flex:1 1 0%`, so it shrank *below its own content*.
Because `.cmp-cols` has a 190px floor, the context block then overflowed `.cmp`
and landed on top of `.out-foot` — "Injected memory context" rendered on the same
line as the Start-talking button. Reproduced at 1920x1080 and 1440x900, fixed
with `flex:1 0 auto` (grow, never shrink under content) and re-measured at
1280x720 / 1440x900 / 1920x1080 / 1920x1130: no overflow and no overlap anywhere,
arms 190-319px.

The page now defaults to English (`localStorage 'vm-lang' || 'en'`); the top-bar
selector still switches to 中文 and the choice persists. `renderCmp()` was added
to the language-switch handler — the compare placeholders are not part of
`renderAll()`, so they previously stayed in the old language until the next turn.

Known gap: `POST /api/compare` from outside the page (curl, a second tab) changes
the server state but does not update an already-open page — the page syncs on
boot and on its own actions only. There is no cross-client sync requirement.

## File-by-file change list

| File | Change |
|---|---|
| `web/compare.py` | new — `Arm`, `CompareState`, `fan_out`, `sanitize` |
| `web/run.py` | `COMPARE` state + `_set_compare`; `_compare_shared_system`; `_compare_turn`; branch in `voicemem_llm_tts`; realtime divert; `_announce_turn` extracted; pass `compare=` into `build_app` |
| `web/utils.py` | `build_app` gains the optional `compare` param and delegates to `compare.register_routes` |
| `web/voicemem.html` | toggle, compare section (per-panel model / memory / base_url / api_key), 4 `cmp_*` handlers, `renderCmp`, i18n keys, CSS |
| `tests/test_compare.py` | new — 20 unit tests (`git add -f`: `tests/` is gitignored, matching the two already-tracked test files) |

Two changes to the plan above, made while implementing:

* The routes live in `web/compare.py` (`register_routes`) rather than being
  written inline in `build_app`. `web/utils.py` imports torch and the TTS stack,
  so routes defined there cannot be unit-tested without loading models; in
  `compare.py` they run against a bare `FastAPI()` app in milliseconds.
* `_announce_turn` was extracted. The transcript + `memory_hits` +
  acoustic-emotion block was already duplicated verbatim between
  `voicemem_llm_tts` and `start_realtime_turn`; the compare path would have
  made a third copy.


## Round two: the UI was unreadable

Feedback after the first build was "I can't understand any of it". Five changes,
all of them about the demo explaining itself:

1. **Compare is on by default** (`CompareState.enabled = True`). Both replies are
   the point; needing to find a toggle first was the confusion. Switching it off
   returns the page to one reply with voice.
2. **The panels say what they are** — "With VoiceMem" / "Without VoiceMem" — instead
   of "A" / "B" plus a `memory` checkbox. The model, endpoint, key and the memory
   switch moved behind a ⚙ per panel, so the default view is two labels and two
   replies. The configurability is unchanged, just no longer the first thing shown.
3. **The Chat / Memory Space tab is gone.** The right pane is always the brain
   (`.track` pinned to its second screen). The compare panels already show both
   replies in the middle column, so a chat log on the right was a third copy of the
   same turn. The conversation list in the left sidebar stays.
4. **The brain reacts to every turn.** `VMBrain.beam(left, right)` only draws when
   *both* hemispheres have hits, so an empty space or a facts-only turn looked
   inert. Added `VMBrain.think(ids)`, which pulses the `you` node and beams from it
   to whatever was hit; it fires on `user_transcript` (before retrieval returns) and
   on `memory_hits` when only one side matched.
5. **English + Hindi** instead of English + Chinese. A 92-key `hi` block was added
   (verified at parity with `en`, including `{0}` placeholder counts) and the
   selector now offers EN / हिंदी. The `zh` block stays in the file so a browser
   with `vm-lang=zh` in localStorage still works, and the language handler also
   calls `renderCmp()` and `syncButtons()` — neither the compare placeholders nor
   the Start/End button are `data-i18n` driven, so both used to keep the old
   language until the next turn.

Hindi replies work through `_lang_note()`, which now prefers `UI_LANG` over
`SPACE_LANG`. That split is deliberate: what the assistant *says* is switchable per
turn, while what gets *stored* is a property of the memory space, fixed at creation
(`voicemem/lang.py` supports only `en`/`zh`). So Hindi replies come out of an
English memory space and nothing mixes languages inside one vector store. Verified:
`तेरी बिल्ली मोचा तीन साल की है।` from the memory arm against
`तुम्हारी बिल्ली लगभग चार साल पुरानी है` from the baseline.

### A crash found by closing the tab

Disconnecting mid-turn left this in the log:

```
[compare] panel b failed: WebSocketDisconnect:
[compare] panel a failed: RuntimeError: Cannot call "send" once a close message has been sent.
Traceback (most recent call last): ...
```

`_run_arm` caught every `Exception`, so a disconnect became a panel error, and then
the both-arms-failed branch tried to report the turn-level error on the closed
socket. `compare.client_gone(exc)` now recognises both shapes Starlette uses
(`WebSocketDisconnect`, and `RuntimeError` mentioning a sent close message);
`_run_arm` re-raises those, and `fan_out` gathers with `return_exceptions=True` so
one arm's disconnect cannot cancel the other mid-await before the exception is
re-raised for the ws handler to treat as a normal close. Four tests cover it,
including one asserting an unrelated `RuntimeError` is still reported per panel
rather than reclassified as a disconnect. Re-verified: a deliberate mid-turn
disconnect now leaves zero tracebacks and zero panel errors in the log.

Note for future browser work: patchright evaluates in an isolated world, so
`page.evaluate` cannot read the page's own globals (`window.VMBrain`, `CMP`) — they
read as `undefined` even though the page is fine. Verify through the DOM instead;
canvas animation was checked by diffing `canvas.toDataURL()` across a turn.


## Round three: the Chinese that survived the language switch

Picking EN still left ten-odd Chinese strings on screen: the sidebar's "对话"
label, the speaker chip ("你", `speaker 7 · recognized as "你"`), the type-here
placeholder, the "Top-K 召回" and "AI 回复" panel headers, both recall column
headings, the brain legend, and the Idle / Pause / Start-talking row.

Two separate causes:

* **Markup with no `data-i18n`.** Those nodes were never in the translation pass,
  so `applyLang()` walked straight past them. Fixed by adding `data-i18n` /
  `-ph` / `-title` (and a new `-aria`, which `applyLang` now handles) to every
  one of them, with nine new keys: `dialogue`, `searchChats`, `typeHint`,
  `topk`, `aiReply`, `langTitle`, `dlScopeTitle`, `langChinese`, `langEnglish`.
  Elements that own child nodes — the sidebar label with its count span, the AI
  reply header with the compare toggle, the two legend swatches — needed the
  text wrapped in its own span first: `data-i18n` assigns `innerHTML`, so
  tagging the parent would have deleted the child.
* **Chinese string literals in JS.** `'你'` was the hardcoded fallback speaker
  name in five places (`blankUI`, `addTurn`, `renderIO`, the export, the
  `user_transcript` handler); all now call `i18n('you')`. `SPK_COLOR` is keyed by
  the speaker's display name, so it needed the Hindi form as a third key.

Two latent bugs turned up in the same sweep:

* `SLOT_DESC` fell back to **Chinese** for any unknown language
  (`SLOT_DESC_I18N[LANG] || SLOT_DESC_I18N.zh`), so picking हिंदी would have put
  Chinese on the ten most prominent labels in the brain graph. Fallback is now
  `en`, and a `hi` block was added.
* `KIND_CN` called `i18n()` once at load, so node-type labels kept the language
  the page started in. It is a Proxy now, resolved per read.

The Chinese fallback text in the HTML source was also replaced with English, so
the pre-JS paint is English rather than a flash of Chinese.

Verified by walking the rendered DOM in both languages — every visible text node
plus every `placeholder` / `title` / `aria-label` on a visible element — looking
for CJK: zero hits in EN, zero in हिंदी, no console errors in either. All three
tables are at 101 keys with matching `{0}` placeholder counts, and neither the
`en` nor the `hi` table contains any CJK.
