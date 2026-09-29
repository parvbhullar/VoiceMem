# examples

Six runnable examples, from "use it as a memory store" to "hook up a Realtime voice model".

```bash
pip install supermem
export OPENAI_API_KEY=sk-...
```

| | What it does | Extra requirements |
|---|---|---|
| [`01_memory.py`](01_memory.py) | Store and search -- the minimal usage | The audio half needs `bash scripts/download_models.sh` |
| [`02_streaming.py`](02_streaming.py) | Streaming API: feed audio chunks and see what each turn computes | Same as above |
| [`03_simple_agent_with_supermem_memory.py`](03_simple_agent_with_supermem_memory.py) | Full voice agent: fetches memory while listening, can be interrupted while speaking | A microphone |
| [`04_all_local_l40s.py`](04_all_local_l40s.py) | All open-source components, streaming on one L40S, nothing leaves the machine | A local vLLM + `pip install voxcpm` |
| [`05_realtime_gpt_qwen.py`](05_realtime_gpt_qwen.py) | Hooks up gpt-realtime / qwen-omni-realtime, memory injected with each response | Nothing to install (all in the base dependencies) |
| [`06_mic_memory.py`](06_mic_memory.py) | **Listen only, no answers**: mic -> transcription -> memory retrieval. No LLM, no TTS | A microphone |

## 01 · Store and search

```bash
python examples/01_memory.py
```

Two kinds of input; the only difference is `mode`:

```python
SuperMem(mode="normal")           # audio -> both brains (ASR / speaker ID / scene / emotion all run)
SuperMem(mode="leftbrain_only")   # text -> left brain only (facts), no emotion attribution
```

Results come in two halves: `result.result_leftbrain` is facts, `result.result_rightbrain`
is profile and emotion.

## 02 · Streaming API

```bash
python examples/02_streaming.py speech.wav
```

Feed audio in chunks and get a state back for each; when VAD decides the speaker has finished, `state` becomes
`turn_over`, and by then `memory_context` has long been computed -- retrieval ran in the background while you
were still talking, so it takes no time before the reply.

All the fields available at `turn_over`:

```
result_leftbrain / result_rightbrain    memories retrieved this turn
speaker_id / speaker_voiceprint         who is speaking
emotion / transcript                    emotion / transcription
entity / schema / text_embedding        extracted entities, slots, vectors
```

## 03 · Full voice agent

```bash
python examples/03_simple_agent_with_supermem_memory.py
```

Mic -> supermem prefetches memory while listening -> OpenAI answers with the memory -> OpenAI TTS speaks,
and **it goes quiet the moment you start talking**. For the web version (with brain map and memory panel), use `python web/run.py`.

## 04 · All open source, one L40S

```bash
vllm serve Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.5   # in another terminal
python examples/04_all_local_l40s.py
```

No API anywhere in the pipeline: FunASR transcription, silero end-of-speech, local E5 for memory vectors and
slot classification, fact extraction and replies sent to the local vLLM, VoxCPM2 for speech. To switch models
just edit the `VLLM` dict -- vLLM is OpenAI-compatible, so filling in `base_url` for the `llm` and `reply`
sections is all it takes.

A 48G card splits roughly like this: 0.5 for vLLM (Qwen3-8B bf16 ~16G + KV), VoxCPM2 ~5G,
E5 / FunASR / silero ~2G together. If VRAM is tight, switch TTS to piper (`{"tts": {"provider": "local"}}`,
tens of MB).

First-frame latency depends on TTS: VoxCPM is the quality option (0.5s+, and it competes with vLLM for the card),
piper is the low-latency option (tens of ms). All models load lazily; before `[ready]` the example warms up
E5 / FunASR / silero / VoxCPM once each -- otherwise the first sentence loads them on the spot and waits several seconds.

Echo cancellation and barge-in live in [`_audio.py`](_audio.py) (shared by 04 / 05), see the section below.

## 05 · Realtime: gpt / qwen

```bash
OPENAI_API_KEY=sk-...    python examples/05_realtime_gpt_qwen.py gpt
DASHSCOPE_API_KEY=sk-... python examples/05_realtime_gpt_qwen.py qwen
```

Every mic chunk takes two paths: one goes via `input_audio_buffer.append` into realtime for native speech out,
the other via `stream.feed()` into supermem to prefetch memory. When local VAD decides the turn is over the memory
is already ready and is sent along with `response.create`.

**Memory must go through per-response `instructions`, not `session.update`** -- the latter is a session-level
setting and in testing the model did not read it this turn: asked "what's my cat's name", the store clearly
retrieved "named Momo", yet it answered "you just mentioned it but I didn't catch it".

Server VAD is only borrowed for barge-in (`create_response=False` + `interrupt_response=True`); when to reply is
still decided locally -- otherwise the response it rushes to generate would not contain the memory we inject.

Both providers use the same event names; only the `session` structure differs (gpt's `turn_detection` lives under
`session.audio.input`, and at the top level it is silently rejected; qwen's is flat). Input sample rates differ
too, gpt 24k and qwen 16k; the example records at one rate and resamples.

Three things are written this way for latency reasons, not by accident:

* **Playback goes through a buffer.** realtime pushes audio much faster than real-time playback; calling
  `out.write()` directly in the event pump would block -- and the event pump is the websocket's only consumer, so
  once it stalls nothing is read, TCP backpressure builds, and `speech_started` / `response.done` are all delayed
  until the buffered audio finishes playing.
* **Barge-in must clear the local buffer.** `interrupt_response=True` only stops generation on the server; the
  seconds already received locally play out anyway, and it sounds like "interrupting doesn't work".
* **Uplink audio and local ASR each get their own coroutine.** The ASR inside `stream.feed()` runs on the event
  loop; if they were chained, any hiccup there would also make the realtime uplink audio stutter.

Storing memory (`vm.ingest`) is a synchronous call that takes seconds, so both examples push it into a thread --
on the event pump or the main loop it would freeze the whole session.

Barge-in has two paths backing each other up: server VAD's `interrupt_response`, and local AEC's
`speech_probability`. Whichever arrives first wins; the slower one hits `response_cancel_not_active`,
which is expected and ignored in the example.

## 06 · Listen only

```bash
python examples/06_mic_memory.py
```

02 feeds audio from a file; this one uses the mic. There is no generative model anywhere in the pipeline --
speaking, transcription, retrieval and storing are all visible:

```
[hear] I'm allergic to peanuts
[recall] Memories already retrieved the moment you finished:
       left brain   User is vegetarian
[store] Written, retrievable next time
```

The point of the `[recall]` step is that it **takes no time after you finish** -- retrieval ran in the background
while you were still talking (0-500ms speculative prefetch). To verify it really remembers: say "I'm allergic
to peanuts", then a few sentences later ask "What can't I eat?".

## `_audio.py` · Echo cancellation

With speakers on, the mic records "your voice + the assistant's voice from the speaker". **AEC** (acoustic echo
cancellation) takes the signal the speaker is playing as reference and subtracts it from the mic signal; what
remains is the real human voice. Skipping this breaks two things: transcription stores the assistant's words in
memory as if you said them, and VAD keeps thinking someone is talking, so the assistant cuts itself off as soon as
it starts speaking.

So playback must go through the **same** `sd.Stream` as recording -- only within the same callback can you get a
reference signal exactly aligned with this mic frame. The whole device runs at 16k: WebRTC's APM only takes
8/16/32/48k, while TTS and realtime output 24k, so `play()` downsamples once.

`AudioIO` also handles barge-in: while the assistant is speaking, if `speech_probability` stays above the
threshold for 0.5s it clears the playback buffer and fires the callback. The first 0.6s after the assistant starts
is a grace period -- then the mic holds almost only the assistant's own voice and AEC has not converged yet;
without the grace period it would cut itself off at the first sound.

Browsers don't need any of this: `getUserMedia` has built-in AEC (`web/run.py` relies on it). 03 inlines a copy of
the same thing; that example deliberately stays a self-contained single file.

## Why all examples use `from_config` instead of `SuperMem(openai_key=...)`

These examples run embedding and slots on local E5:

```python
vm = SuperMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},   # memory vectors: local, 0 network
    "slots":     {"provider": "local"},   # slot classification: local, 0 LLM
    "api_key":   os.environ["OPENAI_API_KEY"],   # only used for write-side fact extraction
})
```

This is not an optional optimisation -- **the 0-500ms speculative prefetch budget cannot include network calls**.
Retrieval runs in the background while you talk and memory must be ready when you finish; if every embedding sent
an HTTP request to OpenAI, the round trip alone would eat the whole budget. The 134ms in the README refers to this
setup; in testing, search itself takes ~10ms.

The default `SuperMem(openai_key=...)` constructor uses OpenAI embeddings, which also works, but every retrieval
goes over the network.

## One more pitfall: stores cannot be mixed

Without `memory_root`, all examples and the web demo share one store (`results/voice_memory` under the package
directory). **Stores with different vector dimensions cannot be mixed** -- local E5 is 384-dim, OpenAI is
1536-dim, and pointing both at one directory fails with:

```
ValueError: shapes (25,384) and (1536,) not aligned
```

The examples now use the same local config as the demo, so they don't collide. To keep them separate, set
`memory_root` explicitly.
