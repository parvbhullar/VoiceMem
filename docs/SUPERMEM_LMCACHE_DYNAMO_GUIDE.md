# SuperMem + LMCache + NVIDIA Dynamo: roles, how it works, how to set it up and use it

*30 Sep 2026*

---

## 1. One-paragraph summary

SuperMem decides **what** a voice agent should know about a caller and packages it as a **KV context cartridge**, a stable block of prompt text. The LLM engine turns that text into **KV cache** (the model's internal "already read" state). Today our in-house server (plain vLLM) keeps that KV **only in GPU memory, only on one server**. **LMCache** extends it to CPU RAM, SSD and shared storage, so it survives when the GPU fills up. **NVIDIA Dynamo** runs many GPU workers behind one endpoint and **routes each caller to the worker that already holds their KV**. Together, memory stops costing a full re-read on every turn, at fleet scale.

---

## 2. Where we are today (measured on our in-house server)

- **Server:** plain vLLM 0.30.0, Gemma-4-26B-A4B-FP8, one engine.
- **Metrics show:** no LMCache connector (`external_prefix_cache_queries_total = 0`) and no Dynamo. Only vLLM's built-in GPU prefix cache is active.

Test with SuperMem's real demo memory (~7,100-token cartridge):

| Request | Prompt | GPU prefill |
|---|---|---|
| No memory | 44 tokens | 17 ms |
| Memory, **no reuse** (full prefill) | 7,130 | 163 ms |
| Memory, **cartridge reused** | 7,130 | **18 ms** |

**What this shows:** with the cartridge, 7K tokens of memory cost the GPU about the same as no memory at all (~9× less prefill than without reuse). A handful of requests only; the 100-turn p50/p95 benchmark is the next step.

**Why the demo screen still shows "with memory" slower than "without memory":**
- The client (laptop) uploads ~28 KB of prompt to GCP every turn.
- Both A/B panels hit the same GPU at the same time.
- Memory can never be *faster* than no memory; the goal is *as fast as*.

Running SuperMem next to the GPU removes the upload cost.

---

## 3. The roles

| Component | Role | Analogy |
|---|---|---|
| **SuperMem** | Decides *what* to remember about each caller. Compiles it into cartridges (org / caller / caller×org), versioned and tenant-scoped. Pre-fills on call start. | The librarian who prepares the caller's file |
| **vLLM** | Runs the model. Its **prefix cache** reuses KV for identical prompt prefixes, but only in that GPU's memory. | The reader who remembers files they read recently, while their desk has room |
| **LMCache** | A KV cache layer plugged into vLLM. When GPU memory is full, KV goes to **CPU RAM → local SSD → remote storage** instead of being deleted, and is loaded back instead of recomputed. Also supports **CacheBlend** (reuse a cartridge even when it isn't at the start of the prompt). | A filing cabinet next to the desk |
| **Dynamo frontend + KV-aware router** | One OpenAI-compatible endpoint in front of many workers. Knows which worker holds which KV blocks and **sends a returning caller to that worker**. | The receptionist who sends Rahul to the counter that already has his file |
| **Dynamo KVBM / NIXL** | KV Block Manager: manages KV across tiers cluster-wide. NIXL moves KV blocks fast between GPUs, CPU and storage. | The building's internal mail system |
| **Agent hints / priority** | Live replies run before background work (SuperMem's memory-writing LLM calls). | Phone calls before paperwork |

---

## 4. How a call flows with everything in place

```
Phone rings (caller ID = Rahul)
  │
  ▼
SuperMem: load Rahul's memory → cartridge (id = hash of content+version+model+tenant)
          → send a 1-token "pre-fill" request
  │
  ▼
Dynamo frontend (KV-aware router)
  ├─ Rahul's cartridge KV already on worker 2?  → route to worker 2 (no recompute)
  ├─ On no worker's GPU, but in LMCache (CPU/SSD)? → load it back (fast, no recompute)
  └─ Nowhere? → compute once on the least-loaded worker, then keep it
  │
Rahul speaks → SuperMem recalls this turn's top memories (while he is still speaking)
  │
  ▼
Prompt = [persona][Rahul's cartridge: KV reused][history + this turn's memories + utterance]
  │                 └── only this last part is computed fresh
  ▼
Worker replies → TTS → Rahul hears the answer
  │
After the call: SuperMem writes new facts (background priority).
Next call: new cartridge version, pre-filled on ring.
```

---

## 5. What each layer adds

| Situation | vLLM only (today) | + LMCache | + Dynamo |
|---|---|---|---|
| Same caller, next turn | ✅ reused (GPU) | ✅ | ✅ |
| Many callers; GPU KV fills up | ❌ evicted → full recompute | ✅ KV moves to CPU/SSD, loaded back | ✅ |
| Caller calls again tomorrow | ❌ usually evicted | ✅ from CPU/SSD/remote | ✅ |
| Server restart | ❌ all lost | ✅ if disk/remote tier is used | ✅ |
| Several GPUs/servers | ❌ caller lands anywhere, recompute | ⚠️ shared only with remote storage | ✅ router sends caller to the worker holding their KV |
| Cartridge not at the start of the prompt | ❌ | ✅ CacheBlend (experimental) | ✅ |
| Live reply vs memory-writing work | Same queue | Same | ✅ priority hints |

**For a single caller on one GPU**, today's vLLM prefix cache is enough; that is why the demo already shows 163 → 18 ms. **LMCache and Dynamo matter at scale**: many callers, repeat calls over days, several GPUs.

---

## 6. Setup

All commands run on the **GPU VM**. Dynamo and LMCache flags change between releases, so check with `--help` on the installed version before running.

### Phase 1: LMCache on the existing vLLM (~1 hour; needs a short restart)

```bash
pip install lmcache

cat > lmcache.yaml <<'EOF'
chunk_size: 256
local_cpu: true
max_local_cpu_size: 40      # GB of host RAM for KV
EOF

LMCACHE_CONFIG_FILE=lmcache.yaml \
vllm serve /home/gcp-vm/models/gemma-4-26B-A4B-it-FP8-dynamic \
  --served-model-name gemma-4-26b-a4b-it-fp8 gemma-4-31b-it-fp8 \
  --port 8000 \
  --enable-prompt-tokens-details \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

**Verify:** after some traffic, `curl -s localhost:8000/metrics | grep external_prefix_cache` shows values **> 0**. With `--enable-prompt-tokens-details`, SuperMem's UI shows **"KV reused NN%"** on the cartridge panel.

### Phase 2: Dynamo in front (~0.5–1 day)

```bash
python -m venv ~/dynamo && source ~/dynamo/bin/activate
pip install "ai-dynamo[vllm]" lmcache

git clone https://github.com/ai-dynamo/dynamo && cd dynamo
docker compose -f deploy/docker-compose.yml up -d          # etcd + NATS

python -m dynamo.frontend --http-port 8000 --router-mode kv &

# one worker per GPU
CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=lmcache.yaml \
python -m dynamo.vllm --model /home/gcp-vm/models/gemma-4-26B-A4B-it-FP8-dynamic \
  --served-model-name gemma-4-26b-a4b-it-fp8 --connector lmcache &
CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=lmcache.yaml \
python -m dynamo.vllm --model /home/gcp-vm/models/gemma-4-26B-A4B-it-FP8-dynamic \
  --served-model-name gemma-4-26b-a4b-it-fp8 --connector lmcache &
```

**Verify:** `curl localhost:8000/v1/models` lists the model, and a chat request answers. Routing shows up in the frontend logs and metrics as the same caller hitting the same worker.

---

## 7. How to use it from SuperMem

Nothing changes in SuperMem: Dynamo exposes the same OpenAI-compatible API as vLLM.

- **Web demo:** AI REPLY → A/B → panel ⚙
  - model: `gemma-4-26b-a4b-it-fp8`
  - own endpoint: `http://<gpu-vm>:8000/v1` (the Dynamo frontend)
  - own key: `EMPTY`
  - tick **memory** + **KV cartridge**
- **Whole app:** `SUPERMEM_REPLY_MODEL=gemma-4-26b-a4b-it-fp8` plus the reply endpoint in the app config.
- **In code:**
  ```python
  from supermem.cartridge import Engine
  Engine("http://<gpu-vm>:8000/v1", "gemma-4-26b-a4b-it-fp8", arm="prod")
  ```
  Or point any OpenAI client there.

For a fair live demo, run SuperMem on (or next to) the GPU VM, so network upload doesn't hide the GPU savings.

---

## 8. How we will measure it (for NVIDIA)

Run the benchmark **on the GPU VM** against each setup and compare the `report.md` tables:

```bash
python evaluation/cartridges/run_bench.py --model gemma-4-26b-a4b-it-fp8 \
  --single-url http://localhost:8000/v1 --arms nomem,full,cartridge,prefetch \
  --turns 100 --gpu-label "<GPU>"                      # baseline: full prefill via cache_salt
python evaluation/cartridges/run_bench.py ... --callers 50 --turns 500   # pressure: GPU KV overflows
```

| Run | Shows |
|---|---|
| vLLM only, 10 callers | Cartridge benefit (prefix reuse) |
| vLLM only, 50 callers | Where reuse breaks down (evictions) |
| vLLM + LMCache, 50 callers | LMCache recovers it (CPU tier) |
| Dynamo + LMCache, 2 workers | Router keeps callers on their KV |

Report: TTFT p50/p95, % context recomputed, GPU prefill ms/turn, answer accuracy. A result is labelled "LMCache" or "Dynamo" **only if measured through it**.

---

## 9. Plan and owners

| When | What | Who |
|---|---|---|
| Today | Phase 1 (LMCache) + 100-turn benchmark → slide numbers | Infra (restart) + us |
| 1 Oct, 2 PM | Slide + demo video submitted (measured: cartridge + vLLM prefix cache, plus LMCache if ready) | Us |
| 2–6 Oct | Phase 2 (Dynamo, 2 workers) + 50-caller benchmark | Infra + us |
| 7 Oct | Meetup: live demo on Dynamo endpoint | Us |

**Risks:**
- The shared inference server needs a restart for Phase 1. Use a maintenance window or a separate GPU VM.
- Dynamo/LMCache flags differ by version; verify with `--help`.
- LMCache is x86 only.
- CacheBlend is experimental; keep it out of claims.

References: [LMCache in Dynamo](https://docs.nvidia.com/dynamo/v1.3.0/integrations/kv-cache-integrations/lm-cache) · [Dynamo KV cache offloading (vLLM)](https://docs.nvidia.com/dynamo/v1.3.0/backends/v-llm/kv-cache-offloading) · [LMCache + Dynamo 1.0](https://blog.lmcache.ai/en/2026/03/16/lmcache-nvidia-dynamo-1-0-a-match-made-in-inference-heaven/)
