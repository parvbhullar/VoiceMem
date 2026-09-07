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

Both arms get the **same** `system` (the demo persona `_RT_PERSONA`). Differing
personas would make the comparison dishonest.

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
  page's current error toast fires, and skip ingestion for that turn (no reply
  to remember).
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

Manual verification: run `bash run_demo.sh`, toggle Compare, set panel A memory
on / panel B memory off with the same model, speak a turn that depends on a
stored fact, and confirm panel A uses the fact while panel B asks for it. Then
confirm the memory space gained exactly one turn (one user utterance, one agent
reply) for that turn.

## File-by-file change list

| File | Change |
|---|---|
| `web/compare.py` | new — `Arm`, `CompareState`, `fan_out`, `sanitize` |
| `web/run.py` | `COMPARE` state; compare branch in `voicemem_llm_tts`; realtime divert; pass `compare=` into `build_app` |
| `web/utils.py` | `build_app` gains the optional `compare` param and the two `/api/compare` routes |
| `web/voicemem.html` | toggle, compare section, 4 `cmp_*` handlers, one render fn |
| `tests/test_compare.py` | new — 8 unit tests above |
