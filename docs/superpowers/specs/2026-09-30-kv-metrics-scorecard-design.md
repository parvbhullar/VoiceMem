# KV cartridge metrics scorecard (web demo)

## Goal
Show the NVIDIA benchmark structure (TTFT p50/p95, % context recomputed,
GPU-ms/turn, quality parity, end-of-speech → first audio) in the live demo,
filled only from numbers the current code already measures.

## Scope
Frontend only: `web/supermem.html`. No backend / websocket changes.

## Data
- Source: existing `cmp_start` (`memory`, `cartridge`, `model`) and `cmp_done`
  (`ttfb`, `ms`, `usage`, `error`) messages.
- Each `cmp_done` is recorded on the current session (`s.metrics`), so switching
  sessions / "New chat" shows that chat's numbers.
- Arm role: memory + cartridge → Cartridge; memory only → Full prefill;
  no memory → No memory (baseline flagged as not a full-prefill comparison).
- p50 / p95: nearest rank, same as `supermem/cartridge/report.py`. Error turns excluded.

## Rows
| Row | Filled from | Otherwise |
|---|---|---|
| TTFT p50 / p95 | `ttfb` per arm; improvement = baseline ÷ cartridge | — |
| % context recomputed | `usage.prompt_tokens`, `usage.cached_tokens` | "not measured" (engine not reporting `cached_tokens`) |
| GPU-ms / turn | mean `usage.prefill_gpu_ms` | "not measured" |
| Quality parity | — | "not measured" (offline benchmark) |
| EoS → first audio | — | "not measured" |

Header line: N turns per arm, context tokens (median prompt tokens), model, GPU "not reported".
Targets column: TTFT ≥2×, recompute ≤15%, quality ≤1–2 pp drop, EoS→audio −25–50%.

## UI
Collapsible `<details>` strip between brain map and chat feed; open state kept in
localStorage (wrapped in try/catch). Existing dark tokens.

## Rule
Never print an example or placeholder number: a cell without measured data says "not measured".
