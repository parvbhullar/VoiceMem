# SuperMem

**Long-term memory for real-time voice agents, served as reusable KV context.**

SuperMem listens to a conversation, remembers the facts, the person and how they
feel, and gives a voice agent exactly the context it needs for the next turn,
retrieved while the caller is still speaking. On top of that, SuperMem compiles a
caller's long-lived memory into **KV context cartridges** that the inference engine
reuses instead of re-reading every turn: *more context, less compute*.

## Architecture

SuperMem keeps two complementary memories:

* **Left brain**: factual memory, organised by schema (slots) and entities for accurate retrieval.
* **Right brain**: persona, emotion and relationships, as independent and cross-entity memory nodes.

The pipeline is **streaming**: while the user is still speaking, SuperMem segments
audio, transcribes, classifies the query and prefetches the relevant memories, so
retrieval is off the critical path. At query time it **routes first, ranks second, and
injects only the top-K memories** into the model context.

On the serving side, the **Context Compiler** turns memory into versioned, tenant-scoped
cartridges (organisation, caller, caller x organisation) in a canonical order, and the
**Context Runtime** attaches them as a stable prompt prefix and pre-fills them before the
caller speaks, so the engine (vLLM + LMCache, NVIDIA Dynamo) reuses their KV cache.

### Key features

* **Dual-brain memory**: what the user said *and* who the user is.
* **Multimodal**: speech, speakers (voiceprint), sound events, emotion from real-world audio.
* **Streaming retrieval**: memory is ready when the turn ends, not after.
* **Small context**: only the top-K memories are injected per turn.
* **KV context cartridges**: long-lived memory as reusable KV, pre-filled on ring.
* **Pluggable**: every component (ASR, embedding, vector engine, reply LLM, TTS) is replaceable; any OpenAI-compatible endpoint works.


## Quick Start

### Installation

**Prerequisite:** Python 3.10+

```bash
cd SuperMem

# Install the memory system (bundles ASR / speaker ID / scene / emotion / local embedding)
pip install -e .
```

### Required Model Download

```bash
bash scripts/download_models.sh
```

### Basic Usage

#### Run as an Offline Memory Engine

```python
from supermem import SuperMem

vm = SuperMem(
    mode="normal",
    openai_key="api_xxx",
    top_k=5,
)

# Local models load lazily -- warm them up so the first call doesn't pay for it.
vm.warmup()

# Store an audio file.
# SuperMem internally runs ASR / speaker ID / scene / emotion / embedding extraction.
print("ingest start")
vm.ingest(audio="assets/input.wav")  # I am vegetarian and allergic to nuts.
print("ingest done")

# Writing is slow because it extracts facts, tags them and builds the graph.
# Reading is a pure vector lookup -- independent of write cost.
print("search start")
result = vm.search("What are my dietary restrictions?")
print("search done")

print(result.result_leftbrain, result.result_rightbrain)


# Store Left Brain factual text directly (no emotional information).
vm = SuperMem(
    mode="leftbrain_only",
    openai_key="api_xxx",
    top_k=5,
)

vm.ingest("I am vegetarian and allergic to nuts.")

result = vm.search("What are my dietary restrictions?")
```

#### Run SuperMem in Streaming Mode

Think of SuperMem's streaming interface as a VAD interface that continuously processes audio.

The example below stores one fact explicitly, then feeds a **question** as audio to show how the memory is already retrieved before the speaker finishes. It ends, as always, with the ingest decision.

```python
import asyncio
import os
from pprint import pprint

import numpy as np
import soundfile as sf

from supermem import SuperMem

# Reuses the vm above; building one here so the block runs standalone
vm = SuperMem(mode="normal", openai_key=os.environ["OPENAI_API_KEY"], top_k=5)

# Local models load lazily -- warm them up so the first audio chunk doesn't wait
vm.warmup()

# Store one fact first, so the question below has something to find
vm.ingest("I am vegetarian and allergic to nuts.")

SPEC_MIN_CHARS = 6
searching = False


def on_partial(text):
    """Partial transcripts as they arrive. Long enough = the search already started."""
    global searching
    print(f"\r[partial] {text}", end="", flush=True)
    if not searching and len(text) >= SPEC_MIN_CHARS:
        searching = True
        print("\n[search start] speaker isn't done yet, retrieval already running", flush=True)


async def main():
    # This audio is a question: "What are my dietary restrictions?"
    audio, sr = sf.read("assets/question.wav", dtype="float32")
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    stream = vm.stream(src_rate=sr, vad_threshold=0.5, on_partial=on_partial)
    step = int(sr * .032)

    for i in range(0, len(pcm), step):
        st = await stream.feed(pcm[i:i + step].tobytes())
        if st.state != "turn_over":
            continue

        # VAD confirmed end of turn. Memory was fetched while the user spoke -- just read it
        print("[search end]")
        print("transcript  ", st.transcript)
        print("left brain  ", st.result_leftbrain)
        print("right brain ", st.result_rightbrain)
        pprint({k: getattr(st, k) for k in
                ["speaker_id", "speaker_voiceprint", "emotion",
                 "entity", "schema", "text_embedding"]})

        # Every turn runs the ingest decision
        print("[ingest] LLM deciding whether this is worth storing...", flush=True)
        res = vm.ingest(st.transcript)
        print(f"[ingest] extracted {res['facts_count']} facts -> {res['memory_ids']}")


asyncio.run(main())
```

### Interactive Demo with SuperMem

Run from the repo root (or use `bash ../run_demo.sh`, which also loads `.env`).

```bash
python web/run.py
```

Then open:

```text
http://localhost:8787
```

By default, the demo mirrors terminal output — including Python logging and
Uvicorn's own logs — to `results/logs/supermem-TIME-PID.log`, one timestamped
line per record, tagged stdout or stderr. The resolved path is printed at
startup. To choose a path or disable file logging:

```bash
python web/run.py --log-file results/logs/debug.log
python web/run.py --no-file-log
```

Reply context combines the current input, turns from this session that are not
yet represented by persistent memory, and retrieved memory. Each turn enters an
in-memory SessionBuffer first. The asynchronous ingest completion callback
removes it only after persistent memory is created. Buffers are isolated by
Memory Space and WebSocket session.

Barge-in uses two stages during playback. VAD first pauses playback while
preserving the audio queue. An explicit stop command or stable ASR updates
confirm cancellation; backchannels, echo, non-text sounds, and isolated
syllables resume playback. `BARGE_REJECT_SILENCE_MS` and
`BARGE_CANDIDATE_TIMEOUT_MS` configure rejection timing.

Both reply modes share a PCM-sample media timeline. The browser AudioWorklet
reports actual rendered progress, so interrupted context contains only the
heard prefix. TTS providers may return `TimedAudioChunk` alignment metadata;
plain PCM providers use segment duration and an adaptive speech-rate fallback.

#### Managing memories

Open `http://localhost:8787/memories` (or **Memories →** in the demo's top bar)
to work on any brain without switching the demo's live one:

- browse, search (substring or semantic), edit and delete facts, filtered by
  slot, entity and role; read the right-brain profile;
- add memories from pasted text or a `.txt`, `.md`, `.json`, `.pdf` or `.docx`
  file. Transcripts are `Name: text` lines or a JSON list of turns
  (`speaker`/`role` + `text`/`content`); name the owner speaker, whose turns
  become the user's memories. Loading runs in the background and can be
  cancelled;
- create, clear and delete brains. The brain live in the demo cannot be
  deleted, and can be cleared only while no voice session is connected.

Limits: 10 MB per file, 500 chunks per load, and each chunk costs several LLM
calls (about 5-20 s), so a large file takes a while. Split bigger files.

The server has no authentication and these routes delete data. Keep it on
localhost or a trusted network.


## Customize Your Voice Agent with SuperMem

You can integrate SuperMem with your own voice model to build a real-time voice agent with long-term memory.

The basic flow is:

**microphone → SuperMem listens and prefetches relevant memories → your model reads those memories and generates a response**

```bash
export OPENAI_API_KEY=sk-...
# Only used for fact extraction when writing memories.
# Memory retrieval runs entirely locally.

python examples/03_simple_agent_with_supermem_memory.py
```

To use your own model, replace the generation step — the memory half stays exactly as is:

```python
def my_reply(text, memory_context):        # a sync function is fine, it runs off-thread
    return my_model.generate(system=memory_context, user=text)

vm = SuperMem(reply=my_reply)
```


## KV context cartridges

See **[docs/CARTRIDGES.md](docs/CARTRIDGES.md)**: the cartridge contract, the web demo's
"KV cartridge" panel, and the benchmark (full prefill vs KV cartridge, p50/p95, recomputed
tokens, prefill GPU time, answer accuracy) with the GPU serving script.


## Evaluation


### Run Evaluation

A benchmark can be started with a single command:

```bash
export OPENAI_API_KEY=sk-...

# Start with the small example included in the repository
# to make sure everything is set up correctly.
# 2 conversations, 5 questions.
python evaluation/run.py \
    --dataset locomo \
    --data evaluation/examples/locomo_sample.json

# Then run the full dataset.
python evaluation/run.py \
    --dataset locomo \
    --data data/locomo.json
```

Before running a full evaluation, add `--inspect` to check how the dataset is parsed.

This mode does not call the model and does not incur API costs:

```bash
python evaluation/run.py \
    --dataset locomo \
    --data data/locomo.json \
    --inspect
```

During evaluation, the answering model receives **only the retrieved memories**, not the original conversation history.

If the model receives the full conversation, the benchmark becomes a reading-comprehension test rather than an evaluation of the memory system itself.

See **[evaluation/README.md](evaluation/README.md)** for the complete evaluation protocol and instructions for adding a new benchmark. Adding a benchmark only requires one file and two functions.


## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
