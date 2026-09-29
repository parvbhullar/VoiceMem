# KV Context Cartridges: runbook

SuperMem decides **what** an agent should know about a caller. Cartridges make
the model able to see it **without re-reading it every turn**: long-lived
context is compiled into canonical, versioned blocks whose KV cache the engine
keeps (vLLM prefix cache → LMCache CPU tier → NVIDIA Dynamo KVBM) and reuses.

```
SuperMem memory ──► Context Compiler ──► cartridges (org · caller · caller×org)
                                             │   canonical text, id = hash(content, version,
                                             │   model, tokenizer, codec, tenant)
                                             ▼
caller rings ──► Context Runtime ──► pre-ring prefill (max_tokens=1)  ─┐
caller speaks ─► Context Runtime ──► [system][org][caller][account]    │ same prefix
                                     [history][turn hits + utterance] ─┘ → KV reused
                                             ▼
                      vLLM + LMCache  /  NVIDIA Dynamo (KV router + KVBM)
```

## What's in the repo

| Path | What |
|---|---|
| `supermem/cartridge/contract.py` | The cartridge contract: kinds, canonical order, id, tenant/model guards |
| `supermem/cartridge/compiler.py` | SuperMem memory (sqlite space or records) → cartridges; dedupe + stable sort |
| `supermem/cartridge/runtime.py` | Prompt layout, pre-ring prefetch, CacheBlend layout, SuperMem reply provider |
| `supermem/cartridge/engine.py` | Measuring client: TTFT, time-to-first-sentence, cached tokens, prefill GPU time |
| `supermem/cartridge/report.py` | p50/p95 and the showcase table; missing data prints "not measured" |
| `supermem/cartridge/simulator.py` | Fake engine for laptop rehearsal. **Not a measurement**; labelled everywhere |
| `evaluation/cartridges/dataset.py` | Deterministic hospital workload: ~11K-token org + ~3K-token callers + accounts, 10 questions each |
| `evaluation/cartridges/run_bench.py` | The benchmark (arms `nomem`, `full`, `cartridge`, `prefetch`, `blend`) |
| `web/compare.py`, `web/run.py`, `web/supermem.html` | The existing A/B compare, extended: a panel can carry its memory as a KV cartridge (⚙ → "KV cartridge"), and its turn card shows how much of the prompt the engine served from KV cache |
| `scripts/gcp_serve_cartridges.sh` | Starts baseline (:8001) and cartridge (:8002) vLLM engines on one GPU |
| `tests/test_cartridge.py`, `tests/test_compare.py`, `tests/test_compare_ui.mjs` | no GPU; `python tests/test_cartridge.py`, `python tests/test_compare.py`, `node tests/test_compare_ui.mjs` |

## 1. In the SuperMem web demo (local, your OpenAI key)

```bash
bash ../run_demo.sh            # -> http://localhost:8787
```

1. Turn on **A/B** in the AI REPLY box.
2. Panel A: ⚙ → tick **memory** and **KV cartridge**. Ticking it snapshots the active
   space's memory into a cartridge and pre-fills it (toast shows tokens + time).
3. Panel B: ⚙ → tick **memory** only (same memory, old layout) or untick it (no memory).
4. Speak or type. Panel A's turn card shows `KV reused NN% · cached/prompt tok`: the
   engine's own `usage.prompt_tokens_details.cached_tokens`, nothing estimated.

What it shows locally: the whole caller memory (~4K tokens for `demo`) rides in panel
A's prompt and ~90–95% of the prompt is served from cache from the second request on
(OpenAI prompt caching). TTFT on a hosted API is dominated by network, so the latency
claim is measured on the GPU (section 2), not here. Point panel A's endpoint at the GCP
vLLM/LMCache engine (⚙ → own endpoint `http://<vm>:8002/v1`, model name) to see it on
NVIDIA hardware in the same UI.

`POST /api/cartridge` re-snapshots the memory (do it after a call; memory written during
a call reaches the model through the turn's recall hits, so the prefix stays stable).

### Benchmark rehearsal without a GPU (simulated)

```bash
python -m supermem.cartridge.simulator --port 8001 --no-cache &
python -m supermem.cartridge.simulator --port 8002 &
python evaluation/cartridges/run_bench.py --model Qwen/Qwen2.5-3B-Instruct \
    --full-url http://127.0.0.1:8001/v1 --cartridge-url http://127.0.0.1:8002/v1
```

Every report and JSON from the simulator says **SIMULATED**. None of it goes on the slide.

## 2. Measure on GCP (real numbers)

```bash
# on a GCP VM with one L4 (or A100); see the header of the script for gcloud + pip
bash scripts/gcp_serve_cartridges.sh
python evaluation/cartridges/run_bench.py --model Qwen/Qwen2.5-3B-Instruct \
    --full-url http://localhost:8001/v1 --cartridge-url http://localhost:8002/v1 \
    --turns 100 --gpu-label "GCP g2-standard-8, 1x L4 24GB"
# -> results/cartridges/<run>/report.md, summary.json, turns_<arm>.jsonl
```

`report.md` is the slide table (Full Prefill vs KV Cartridge, targets, PASS/MISS).

Optional:
- `--tts openai` (or any SuperMem TTS provider) adds **end-speech → first audio**:
  time-to-first-sentence + TTS first-audio, per turn. ASR endpointing is identical
  across arms and excluded; the report says so.
- `BLEND=1 bash scripts/gcp_serve_cartridges.sh` + `--arms full,blend --blend-url http://localhost:8003/v1`
  runs LMCache CacheBlend (non-prefix reuse). Experimental: check the LMCache version's docs first.
- `--callers 50` stresses the cache (more distinct callers than fit in GPU KV), which is
  where the LMCache CPU tier / Dynamo KVBM earn their keep.

### Dynamo instead of plain vLLM for the cartridge arm
Point `--cartridge-url` (or panel A's endpoint in the web demo) at a Dynamo frontend
(KV-aware router + vLLM worker with the LMCache/KVBM connector). Use the Dynamo docs
for your installed version to start the frontend and worker. Only label a result
"Dynamo" if it was measured through Dynamo.

## How each number is measured

| Metric | Source |
|---|---|
| TTFT p50/p95 | wall clock, request sent → first streamed token; nearest-rank over all turns |
| Context recomputed | `1 − cached_tokens / prompt_tokens` from the engine's `usage` (needs `--enable-prompt-tokens-details`) |
| GPU prefill time/turn | delta of vLLM `vllm:request_prefill_time_seconds_sum` around each request (requests run one at a time) |
| Answer accuracy | every expected string (ID, amount, time, name) present in the answer; temperature 0 |
| Context leverage | prompt tokens presented ÷ tokens actually recomputed |
| Pre-ring prefetch | time of the `max_tokens=1` warm-up; hidden behind the ring, reported separately |

Fairness: arms run one after another, round-robin across callers, each arm
starting from a cold cache (a run-scoped epoch line at the top of the prompt
changes every block hash). The first turn of every call in the `cartridge` arm
is a real cold prefill; the `prefetch` arm pays it during the ring instead.
