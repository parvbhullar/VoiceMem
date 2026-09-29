"""Turn per-request results into the showcase table.

Rules the table follows (NVIDIA asked for honest measurements):
* a cell is only filled from measured requests; missing data prints
  "not measured", never a placeholder number;
* p50 / p95 are nearest-rank over every turn in the arm, errors excluded and
  counted separately;
* every table carries its conditions (N, context, model, GPU, engine).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

NOT_MEASURED = "not measured"


def pct(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, math.ceil(p / 100 * len(vals)) - 1)
    return vals[k]


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


@dataclass
class ArmSummary:
    arm: str
    n: int
    errors: int
    ttft_p50: float | None
    ttft_p95: float | None
    ttfs_p50: float | None
    prompt_tokens: float | None
    recomputed_frac: float | None       # recomputed / prompt tokens, averaged over turns
    prefill_gpu_ms: float | None        # mean per turn
    accuracy: float | None              # fraction of turns answered correctly
    e2e_p50: float | None               # end of speech -> first audio (only if TTS was measured)
    e2e_p95: float | None
    context_leverage: float | None      # prompt tokens presented / tokens actually recomputed
    prefetch_p50: float | None = None   # pre-ring warm-up time (hidden behind the ring)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def summarize(arm: str, turns: list[dict]) -> ArmSummary:
    ok = [t for t in turns if not t.get("error")]
    prompt = [t["prompt_tokens"] for t in ok if t.get("prompt_tokens") is not None]
    recomputed = [t["recomputed_tokens"] for t in ok
                  if t.get("recomputed_tokens") is not None and t.get("cached_tokens") is not None]
    fracs = [t["recomputed_frac"] for t in ok
             if t.get("recomputed_frac") is not None and t.get("cached_tokens") is not None]
    correct = [t["extra"]["correct"] for t in ok if "correct" in t.get("extra", {})]
    e2e = [t["extra"]["e2e_ms"] for t in ok if t.get("extra", {}).get("e2e_ms") is not None]
    prefetch = [t["extra"]["prefetch_ms"] for t in turns
                if t.get("extra", {}).get("prefetch_ms") is not None]
    leverage = None
    if recomputed and prompt and len(recomputed) == len(prompt):
        leverage = sum(prompt) / max(1, sum(recomputed))
    return ArmSummary(
        arm=arm, n=len(ok), errors=len(turns) - len(ok),
        ttft_p50=pct([t["ttft_ms"] for t in ok], 50),
        ttft_p95=pct([t["ttft_ms"] for t in ok], 95),
        ttfs_p50=pct([t["ttfs_ms"] for t in ok], 50),
        prompt_tokens=mean(prompt),
        recomputed_frac=mean(fracs),
        prefill_gpu_ms=mean([t["prefill_gpu_ms"] for t in ok]),
        accuracy=(sum(correct) / len(correct)) if correct else None,
        e2e_p50=pct(e2e, 50), e2e_p95=pct(e2e, 95),
        context_leverage=leverage,
        prefetch_p50=pct(prefetch, 50),
    )


# ── the showcase table ───────────────────────────────────────────────────────

def _ms(v):
    return NOT_MEASURED if v is None else f"{v:,.0f} ms"


def _pctf(v):
    return NOT_MEASURED if v is None else f"{v * 100:.1f}%"


def _speedup(base, new, need):
    if base is None or new is None or new <= 0:
        return NOT_MEASURED, None
    x = base / new
    return f"{x:.2f}x faster", x >= need


def _lower(base, new, need):
    if base is None or new is None or base <= 0:
        return NOT_MEASURED, None
    drop = 1 - new / base
    return f"{drop * 100:.0f}% lower", drop >= need


def showcase_rows(full: ArmSummary, cart: ArmSummary) -> list[tuple[str, str, str, str, str]]:
    """(metric, full prefill, kv cartridge, target, result) rows, in the order
    of the NVIDIA showcase slide."""
    rows = []
    r, ok = _speedup(full.ttft_p50, cart.ttft_p50, 2.0)
    rows.append(("TTFT p50", _ms(full.ttft_p50), _ms(cart.ttft_p50), ">=2x faster", _verdict(r, ok)))
    r, ok = _speedup(full.ttft_p95, cart.ttft_p95, 1.5)
    rows.append(("TTFT p95", _ms(full.ttft_p95), _ms(cart.ttft_p95), ">=1.5x faster", _verdict(r, ok)))
    ok = None if cart.recomputed_frac is None else cart.recomputed_frac <= 0.15
    rows.append(("Context recomputed", _pctf(full.recomputed_frac), _pctf(cart.recomputed_frac),
                 "<=15%", _verdict(_pctf(cart.recomputed_frac), ok)))
    r, ok = _lower(full.prefill_gpu_ms, cart.prefill_gpu_ms, 0.40)
    rows.append(("GPU prefill time/turn", _ms(full.prefill_gpu_ms), _ms(cart.prefill_gpu_ms),
                 ">=40% lower", _verdict(r, ok)))
    if full.accuracy is None or cart.accuracy is None:
        r, ok = NOT_MEASURED, None
    else:
        delta = (cart.accuracy - full.accuracy) * 100
        r, ok = f"{delta:+.1f} pp", abs(delta) <= 2.0
    rows.append(("Answer accuracy", _pctf(full.accuracy), _pctf(cart.accuracy), "<=2pp delta",
                 _verdict(r, ok)))
    r, ok = _lower(full.e2e_p50, cart.e2e_p50, 0.25)
    rows.append(("End-speech -> first audio (p50)", _ms(full.e2e_p50), _ms(cart.e2e_p50),
                 ">=25% lower", _verdict(r, ok)))
    return rows


def _verdict(text, ok):
    if ok is None:
        return text
    return f"{text} {'PASS' if ok else 'MISS'}"


def showcase_markdown(full: ArmSummary, cart: ArmSummary, conditions: dict) -> str:
    cond = " · ".join(f"{k} {v}" for k, v in conditions.items() if v)
    lines = [
        "## Baseline -> Unpod KV Cartridges",
        "",
        f"**{cond}**",
        "",
        f"| Metric | Full Prefill (`{full.arm}`) | KV Cartridge (`{cart.arm}`) | Target | Result |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {m} | {a} | {b} | {t} | {r} |" for m, a, b, t, r in showcase_rows(full, cart)]
    if cart.context_leverage is not None:
        lines += ["", f"**Context leverage:** {cart.context_leverage:.1f}x "
                      f"(prompt tokens presented / tokens actually recomputed)"]
    lines += ["", f"Turns: full {full.n} ok / {full.errors} errors · "
                  f"cartridge {cart.n} ok / {cart.errors} errors. All values measured on this run."]
    return "\n".join(lines)


def arms_markdown(summaries: list[ArmSummary]) -> str:
    lines = ["| Arm | N | TTFT p50 | TTFT p95 | Prompt tok | Recomputed | Prefill GPU/turn | "
             "Accuracy | Leverage | Pre-ring prefetch p50 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(
            f"| `{s.arm}` | {s.n} | {_ms(s.ttft_p50)} | {_ms(s.ttft_p95)} | "
            f"{NOT_MEASURED if s.prompt_tokens is None else f'{s.prompt_tokens:,.0f}'} | "
            f"{_pctf(s.recomputed_frac)} | {_ms(s.prefill_gpu_ms)} | {_pctf(s.accuracy)} | "
            f"{NOT_MEASURED if s.context_leverage is None else f'{s.context_leverage:.1f}x'} | "
            f"{_ms(s.prefetch_p50) if s.prefetch_p50 is not None else '-'} |")
    return "\n".join(lines)
