# SuperMem × NVIDIA Dynamo: what we built, how it works, where we are

*Prepared 30 Sep 2026 for the NVIDIA Voice AI Ecosystem Meetup (showcase deadline: 1 Oct, 2 PM IST).*

---

## 1. TL;DR

- **SuperMem** is our long-term memory layer for voice agents. It remembers each caller (facts + personality/emotion) and hands the agent the right context for every turn, retrieved *while the caller is still speaking*.
- **Problem:** rich memory makes the agent right, but carrying it costs a full **prefill** on every turn. That means higher time-to-first-token (TTFT) and more GPU-seconds per call.
- **What we added: KV Context Cartridges.** The caller's long-lived memory is compiled into a stable, versioned prompt block whose **KV cache the inference engine reuses** instead of recomputing it. It is pre-filled when the call starts, before the caller says anything.
- **Where Dynamo fits:** Dynamo is the NVIDIA system that stores, moves and routes KV cache (KV-aware router, KVBM memory tiers, LMCache/NIXL). Cartridges are designed to make SuperMem's memory exactly the kind of KV that Dynamo manages well.
- **Status:**
  - **Built and working:** cartridges, the A/B demo in the SuperMem UI, and the benchmark harness.
  - **Measured locally (hosted OpenAI):** 85–95% of each prompt was served from cache on the cartridge path.
  - **Not yet measured:** the GPU latency/cost numbers on vLLM + LMCache / Dynamo. That run on our GCP GPU is the next step.

---

## 2. What we did

| Area | Work done |
|---|---|
| **KV Context Cartridges** (`supermem/cartridge/`) | **Contract:** kinds (organisation / caller / caller×organisation), canonical order, and an id hashed from content + version + model + tokenizer + tenant. **Compiler:** SuperMem memory → canonical cartridges (deduped, stable sort). **Runtime:** prompt layout + pre-fill. **Engine client:** measures TTFT, cached tokens and prefill GPU time. **Report:** p50/p95 table. |
| **Demo, inside the existing SuperMem UI** | The A/B compare panel gets a **"KV cartridge"** switch. Each answer card shows the engine-reported **"KV reused NN% · cached/prompt tokens"**. Any panel can point at any OpenAI-compatible endpoint (our GCP vLLM, an in-house LLM, NIM, Dynamo). |
| **Benchmark** (`evaluation/cartridges/`) | Deterministic 16K-context workload: hospital policy + 10 callers × 10 questions = 100 turns. Arms: *no memory / full prefill / KV cartridge / cartridge + pre-fill / CacheBlend (experimental)*. Outputs the showcase table: TTFT p50/p95, % context recomputed, GPU prefill ms/turn, answer accuracy, end-of-speech → first audio. |
| **GPU serving** (`scripts/gcp_serve_cartridges.sh`) | One script starts two engines on one GCP GPU: the baseline (vLLM, prefix caching off) and the cartridge engine (vLLM + LMCache CPU tier). Same model, same GPU; only KV reuse differs. |
| **Product clean-up** | Renamed to **SuperMem** everywhere (package, UI, env vars, data); the whole codebase, prompts, logs and UI are now **English-only** (UI in English + Hindi); stored memory keys migrated to English; new README. |
| **Quality** | 60+ automated tests pass (cartridge, compare, UI, engine). App verified end to end in the browser. A backup of the pre-change state is kept. |

---

## 3. How SuperMem works (one voice turn)

```
Caller speaks
   │
   ├─► streaming ASR ─► partial text ─► SPECULATIVE RECALL (starts before the caller finishes)
   │                                      ├─ Left brain : facts (vector search + entity/slot graph)
   │                                      └─ Right brain: personality / emotion / how to talk to them
   │                                      → top-K memories, ~300 tokens, ready at end of turn
   ▼
End of speech
   │
   ▼
Prompt = [persona] [CALLER CARTRIDGE: stable, KV reused] [history + this turn's memories + utterance]
   │
   ▼
LLM (vLLM / Dynamo) ─► reply ─► TTS ─► caller hears the answer
   │
   ▼ (after the reply, in the background)
Memory write: extract new facts and emotions, update the graph
```

---

## 4. The techniques that make it fast *and* accurate

| # | Technique | Why it's fast | Why it's accurate |
|---|---|---|---|
| 1 | **Speculative retrieval on partial ASR** | Memory search runs while the caller is still talking, so it is off the critical path. | Retrieval is re-checked against the final transcript. |
| 2 | **Route first, rank second, top-K only** | Only ~300 tokens of memory per turn instead of the whole history. | Slot/entity routing picks *relevant* memories, not just similar-sounding ones. |
| 3 | **Local embeddings** | Recall takes milliseconds, with no network call. | Same embedding space for writes and reads. |
| 4 | **Dual-brain memory** | n/a | The agent knows *what* the caller said and *how* to talk to them (tone, concerns), without reading those notes back to them. |
| 5 | **KV Context Cartridge (stable prefix)** | The caller's memory is identical token-for-token across turns, so the engine reuses its KV and only computes the new part. | **Exact prefix reuse is lossless**: same tokens give the same KV, so the answer quality is identical to full prefill. |
| 6 | **Snapshot per call, volatile data last** | New facts learned mid-call ride in the per-turn part, so the prefix never breaks. | Nothing is lost: the new facts still reach the model every turn. |
| 7 | **Pre-fill on call start (pre-ring)** | The cartridge's KV is computed during ring/greeting time, so even turn 1 is warm. | n/a |
| 8 | **Versioned, tenant-scoped cartridge ids** | A warm cartridge is recognised across calls. | A memory change produces a new id, so stale context is never served. Mixed-tenant or mixed-model attach is refused. |
| 9 | **Honest measurement** | n/a | Numbers come from the engine itself (cached tokens, prefill time). Every benchmark arm starts cold, requests run one at a time, and results are reported as p50/p95. Missing data prints "not measured". |

---

## 4a. How much SuperMem remembers, and how it merges what callers say across calls

**No limit on calls.** SuperMem doesn't keep "the last N calls". After each call it extracts **facts** and stores them on disk (SQLite + vector index), one memory space per caller. Raw transcripts are not replayed into the prompt. (Our demo space currently holds 149 facts.)

**Small prompt, no matter how much history.** Each turn sends only the **top-5 facts + top-5 personality notes** relevant to *this* question (~300–400 tokens). Five calls or five hundred, the prompt stays the same size.

**Forgetting.** Facts that keep getting used gain "heat". Heat halves every 14 days. An archive step for cold facts (older than 30 days, low heat) exists but is **not automatic**, so by default nothing is deleted.

**Merging new information (left brain, facts).** After each call, an LLM **conflict resolver** compares every new fact with similar stored ones and decides:

| Decision | When | Example |
|---|---|---|
| **ADD** | New information | "Allergic to penicillin" |
| **UPDATE** | A single-valued attribute changed (address, phone, job, favourite X): **newest wins** | "I moved to Whitefield" replaces the old address |
| **NONE** | Already known | "I take metformin" (again) |
| **DELETE** | The caller corrects an earlier statement | "I'm not diabetic, I misspoke" |

Multi-valued attributes (allergies, hobbies) **accumulate**. Events ("went to hospital on Sunday") are **never merged**; each is kept with its date.

**Personality notes (right brain).** Every 50 new notes a clean-up runs. Duplicates are removed. When two notes contradict each other, the old one is **kept but marked outdated** and down-weighted, so the agent knows both "how they were" and "how they are now".

### Example: one caller, five calls

| Call | What Rahul says | What SuperMem does |
|---|---|---|
| **1 · 1 Jun** | "I'm Rahul, I live in Indiranagar, I'm diabetic, on metformin, and I have knee pain. Is it serious?" | 5 facts **ADD**. Right brain: *worries about severity* |
| **2 · 8 Jun** | "My number changed, it's 98xxx now. I'm allergic to penicillin." | Phone **UPDATE** (old one replaced); allergy **ADD** |
| **3 · 20 Jun** | "I moved to Whitefield. The knee is fine now." | Address **UPDATE** → Whitefield; "20 Jun: knee recovered" **ADD** (dated) |
| **4 · 5 Jul** | "I'm allergic to ibuprofen too. Still on metformin." | Allergy **ADD** (now two); metformin **NONE** (duplicate) |
| **5 · 20 Jul** | "The pain is back." | Retrieves knee pain (1 Jun) + recovered (20 Jun) + diabetic + metformin + both allergies + Whitefield |

The agent's reply on call 5:
> "Rahul ji, the knee pain is back? It had settled in June. There's a Tuesday slot at the Whitefield branch, and I've noted your penicillin and ibuprofen allergies, so the doctor will choose the painkiller."

After five calls, memory holds:
- **Address:** Whitefield (current)
- **Phone:** 98xxx (current)
- **Allergies:** penicillin + ibuprofen
- **Knee:** a dated timeline (pain → recovered → back)
- **Personality:** a "worries about severity" note, so the agent leads with the concrete next step, then reassurance

**Link to KV cartridges:** at the start of each call, this *current* memory is compiled into the caller's cartridge and pre-filled. Anything learned during the call rides in the per-turn part until the next call's cartridge.

**Caveats:** ADD/UPDATE decisions are made by an LLM and can occasionally be wrong. Every fact keeps its date, and `SUPERMEM_ALWAYS_ADD=1` switches merging off (append-only). This example illustrates the documented rules; it is not a recorded test run.

---

## 5. SuperMem + Dynamo: how they fit together

```
                 SuperMem (decides WHAT the agent should know)
   memory ─► Context Compiler ─► cartridges (org · caller · caller×org), versioned
                                   │
            call rings ─► pre-fill │  caller speaks ─► prompt with cartridge prefix
                                   ▼
                 NVIDIA Dynamo (decides WHERE and HOW that context lives on GPUs)
   KV-aware router ─► sends the caller to the worker that already holds their KV
   KVBM / LMCache ─► keeps cartridge KV in GPU → CPU RAM → SSD → remote instead of deleting it
   NIXL           ─► moves KV between workers and tiers fast
   agent hints    ─► live replies high priority; memory-writing LLM calls background priority
                                   │
                                   ▼
                          vLLM / TensorRT-LLM workers
```

**In one line:** *Dynamo decides where and how inference runs; SuperMem decides what long-lived context should be compiled, reused and attached to every inference.*

---

## 6. What the user gets

| For | Benefit |
|---|---|
| **The caller** | No repeating themselves ("my patient ID", "my knee issue"). The agent answers with their details, in their language and tone, without a longer pause. |
| **The business** | Shorter calls (fewer turns to resolution), fewer wrong or generic answers, and memory that doesn't raise GPU cost per call. |
| **The platform team** | Any OpenAI-compatible model works (our own, in-house, NIM). Memory is a drop-in layer, and the per-turn KV reuse % is visible in the UI. |

---

## 7. What Dynamo adds that we didn't have before

| | **Before (memory as plain prompt text)** | **Cartridges on one vLLM (prefix cache)** | **Cartridges on Dynamo (+ LMCache/KVBM)** |
|---|---|---|---|
| Per-turn cost of memory | Full prefill every turn | Reused while it stays in that GPU's cache | Reused across turns *and* calls |
| Many callers at once | n/a | GPU cache fills up and cartridges get evicted and recomputed | Evicted KV moves to CPU/SSD tiers and comes back without recompute |
| Returning caller on a multi-GPU fleet | Lands on any GPU, recomputes | Lands on any GPU, likely recomputes | **KV-aware router** sends them to the GPU that holds their memory |
| Live reply vs background memory writing | Compete for the same GPU queue | Same | **Priority hints**: live reply first |
| Keep a caller's context safe mid-call under load | No | No | **Cache pinning** (experimental in Dynamo 1.0) |

The core technique works on any engine, and Dynamo makes it hold **at fleet scale** (many callers, many GPUs, long-lived memory). That scale is exactly where memory normally becomes too expensive.

---

## 8. Results so far

| What | Status | Result |
|---|---|---|
| Cartridge path in the SuperMem UI (hosted OpenAI model) | **Measured**, a handful of live turns | **85–95%** of each ~5.7K-token prompt served from cache; answers match the non-cartridge memory path. (Hosted API, so latency is dominated by network. No latency claim from this.) |
| Full pipeline on laptop simulator | Rehearsal only | Used to test the harness and UI. **Not a measurement; not for slides.** |
| GPU: full prefill vs KV cartridge (vLLM + LMCache), 100 turns, p50/p95 | **Pending**: GCP run | Targets from the NVIDIA brief: TTFT p50 ≥2× faster, p95 ≥1.5×, ≤15% context recomputed, ≥40% lower prefill GPU time, accuracy within 2 pp. |
| Same through Dynamo (KV router + KVBM) | **Pending**: after the vLLM run | We only label a result "Dynamo" if it was measured through Dynamo. |

---

## 9. Next steps (to the 1 Oct submission and 7 Oct meetup)

1. **GCP GPU run:** `bash scripts/gcp_serve_cartridges.sh`, then `python evaluation/cartridges/run_bench.py … --turns 100` → the real table for the slide.
2. **Demo video (≤90 s):** SuperMem UI with panel A = memory + KV cartridge on the GCP engine, panel B = no memory / full prefill.
3. **One slide** in NVIDIA's template (problem, approach, stack, measured results, next steps, demo), plus the reply mail. Due 1 Oct, 2 PM IST.
4. **If selected:** put the cartridge engine behind a Dynamo frontend (KV-aware router, KVBM) and re-measure; try CacheBlend (non-prefix reuse) and pinning.

---

## 10. How to use SuperMem

### a) Install

```bash
cd SuperMem
pip install -e .                 # Python 3.10+
bash scripts/download_models.sh  # local models: ASR, VAD, embeddings, speaker ID
```

Put the key in `.env` next to `run_demo.sh` (or export it):
```
OPENAI_API_KEY=sk-...
```

### b) Web demo (what we show)

```bash
bash run_demo.sh                 # → http://localhost:8787
```

1. **Start talking**, or type in the left box and press Enter.
2. **Top-K recall** (left side) shows the facts (left brain) and personality notes (right brain) found for this turn. The brain map shows where they live.
3. **A/B compare** (AI REPLY → **A/B**): the same question is answered by two panels side by side.
   - Panel ⚙ → **memory**: include SuperMem's memory or not (with vs without).
   - Panel ⚙ → **KV cartridge**: carry the memory as a reusable KV cartridge. The card then shows **"KV reused NN% · cached/prompt tokens"**, as reported by the engine.
   - Panel ⚙ → **model / own endpoint / own key**: point the panel at any OpenAI-compatible LLM (our GCP vLLM, an in-house model, NVIDIA NIM, a Dynamo frontend). Example: model `Qwen/Qwen2.5-3B-Instruct`, endpoint `http://<gpu-vm>:8002/v1`, key `EMPTY`.
4. **Memory space** (top right): one space = one person's memory. Create or switch spaces there.

### c) In your own code

One line into an existing chat loop (OpenAI-style messages):
```python
from supermem import inject
reply = llm.chat(inject(messages, user_id="caller-4411"))   # adds that caller's memory, stores the new turn
```

Or the two primitives:
```python
from supermem.memory_api import Memory
m = Memory(user_id="caller-4411")
context = m.recall("the pain is back")      # prompt-ready memory text for this turn
m.remember("I moved to Whitefield")         # stored in the background
```

Full engine (audio, streaming, voice agent):
```python
from supermem import SuperMem
vm = SuperMem(mode="normal", top_k=5)
vm.warmup()
vm.ingest("I am vegetarian and allergic to nuts.")
print(vm.search("What are my dietary restrictions?").result_leftbrain)
```

More examples are in `examples/` (01 memory, 02 streaming, 03 voice agent, 06 microphone).

### d) Choosing models

| What | How |
|---|---|
| Reply LLM for a demo panel | Panel ⚙: model + own endpoint + own key |
| Reply model for the whole app | `SUPERMEM_REPLY_MODEL=...` |
| Background memory model (fact extraction, tagging) | `SUPERMEM_CHAT_MODEL=...` |
| Endpoint for all OpenAI-compatible calls | `OPENAI_BASE_URL=...` (affects the transcription API too, so set it with care) |
| Speech recognition | `SUPERMEM_ASR=openai` (default in `run_demo.sh`) or a local model |

### e) KV cartridge benchmark on GPU (for the NVIDIA numbers)

```bash
# on the GCP GPU VM
bash scripts/gcp_serve_cartridges.sh        # :8001 full prefill, :8002 vLLM + LMCache
python evaluation/cartridges/run_bench.py --model Qwen/Qwen2.5-3B-Instruct \
  --full-url http://localhost:8001/v1 --cartridge-url http://localhost:8002/v1 \
  --turns 100 --gpu-label "GCP L4 24GB"
# → results/cartridges/<run>/report.md  (TTFT p50/p95, % recomputed, prefill GPU ms, accuracy)
```

Details: `docs/CARTRIDGES.md`.

---

## 11. Open items

- **Model weights** are still downloaded from the original model repo on HuggingFace. They should be mirrored to our own HF org (`SUPERMEM_MODELS_REPO`).
- **Sample audio** in `assets/` is Mandarin speech. It should be replaced with English samples.
- The repo keeps its **Apache-2.0 `LICENSE`** and a short **`NOTICE`**, as the license requires.
- **CacheBlend** (non-prefix reuse) is experimental upstream. It is kept as a separate arm and not claimed.
- One older UI test (`test_brain_signal.mjs`) was already failing before this work.
