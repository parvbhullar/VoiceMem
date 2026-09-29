"""Full prefill vs KV cartridges: the showcase benchmark.

Same model, same GPU, same prompts, same questions. Arms differ only in how
the engine treats the KV of the cartridge prefix:

    nomem     no memory at all (shows what memory buys: accuracy)
    full      memory in the prompt, full prefill every turn (the baseline)
    cartridge memory as cartridges, KV reused by the engine (vLLM APC / LMCache / Dynamo)
    prefetch  cartridge + pre-ring prefill: the caller's cartridges are computed
              while the phone rings, so even turn 1 is warm
    blend     cartridges under LMCache CacheBlend (non-prefix reuse), needs a
              blending-enabled engine at --blend-url

Two ways to get a "full prefill" baseline on the same GPU:
  --full-url    a second vLLM with prefix caching off (recommended)
  --single-url  one engine for everything; the full arm sends a unique
                cache_salt per request so nothing is ever reused

Run on the GPU box (see scripts/gcp_serve_cartridges.sh):

    python evaluation/cartridges/run_bench.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --full-url http://localhost:8001/v1 --cartridge-url http://localhost:8002/v1 \\
        --gpu-label "GCP g2 L4 24GB" --turns 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.cartridges.dataset import ORG_ID, TENANT, build, score, turn_order  # noqa: E402
from supermem.cartridge.compiler import ContextCompiler, TokenCounter  # noqa: E402
from supermem.cartridge.engine import Engine  # noqa: E402
from supermem.cartridge.report import arms_markdown, showcase_markdown, summarize  # noqa: E402
from supermem.cartridge.runtime import ContextRuntime  # noqa: E402

ARMS = ("nomem", "full", "cartridge", "prefetch", "blend")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--full-url", help="engine with prefix caching OFF (baseline)")
    p.add_argument("--cartridge-url", help="engine with KV reuse ON (vLLM APC / LMCache / Dynamo frontend)")
    p.add_argument("--blend-url", help="engine with LMCache CacheBlend enabled (blend arm only)")
    p.add_argument("--single-url", help="one engine for all arms; full arm uses a unique cache_salt per request")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--arms", default="nomem,full,cartridge,prefetch")
    p.add_argument("--showcase-arm", default="",
                   help="arm compared against full in the showcase table (default: prefetch if run, else cartridge)")
    p.add_argument("--turns", type=int, default=100)
    p.add_argument("--callers", type=int, default=10)
    p.add_argument("--org-tokens", type=int, default=11000)
    p.add_argument("--user-tokens", type=int, default=3000)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--gpu-label", default="", help="e.g. 'GCP g2-standard-8, 1x L4 24GB'")
    p.add_argument("--tts", default="", help="measure end-of-speech -> first audio with this supermem TTS provider")
    p.add_argument("--out", default=str(ROOT / "results" / "cartridges"))
    return p.parse_args(argv)


def _gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip().replace("\n", " + ")
    except (OSError, subprocess.SubprocessError):
        return ""


async def _tts_first_audio_ms(tts, sentence: str) -> float | None:
    t0 = time.perf_counter()
    try:
        async for _pcm in tts.stream(sentence, None):
            return (time.perf_counter() - t0) * 1000
    except Exception as e:  # noqa: BLE001
        print(f"  tts failed: {e}", flush=True)
    return None


def _first_sentence(text: str) -> str:
    for sep in (". ", "? ", "! ", "\n"):
        if sep in text:
            return text.split(sep, 1)[0] + sep.strip()
    return text


async def run_arm(arm: str, args, engines: dict, workload, compiler, run_id: str, tts) -> list[dict]:
    runtime = ContextRuntime(org=compiler.compile_org(ORG_ID, workload.org_sections, version=1),
                             epoch=f"{run_id}-{arm}")
    for c in workload.callers:
        runtime.register(c.user_id, [
            compiler.compile_user(c.user_id, c.facts, c.traits, display_name=c.name, version=1),
            compiler.compile_rel(ORG_ID, c.user_id, c.account, version=1),
        ])

    engine = engines[arm]
    mode = {"nomem": "nomem", "blend": "blend"}.get(arm, "cartridge")
    salted = arm == "full" and args.single_url and not args.full_url
    histories: dict[str, list[dict]] = {}
    turns = []
    order = turn_order(workload, args.turns)
    print(f"\n== arm {arm}: {len(order)} turns on {engine.base_url} (mode {mode}) ==", flush=True)

    for i, (caller, q, t_idx) in enumerate(order):
        extra = {"caller": caller.user_id, "turn_in_call": t_idx, "question": q.text,
                 "source": q.source, "expect": q.expect}
        if arm == "prefetch" and t_idx == 0:
            # The phone is ringing: caller ID is known, nobody has spoken yet.
            warm = await engine.warm(runtime.prefetch_messages(caller.user_id, mode))
            if warm.error:
                extra["prefetch_error"] = warm.error
            else:
                extra["prefetch_ms"] = warm.total_ms
        hist = histories.setdefault(caller.user_id, [])
        msgs = runtime.messages(caller.user_id, q.text, mode=mode, history=hist)
        res = await engine.complete(msgs, max_tokens=args.max_tokens,
                                    cache_salt=uuid.uuid4().hex if salted else None)
        extra["correct"] = score(res.text, q.expect) if not res.error else False
        if tts is not None and res.text and res.ttfs_ms is not None:
            first_audio = await _tts_first_audio_ms(tts, _first_sentence(res.text))
            if first_audio is not None:
                extra["tts_first_audio_ms"] = first_audio
                # Measured from the moment the final transcript is handed to the
                # LLM; ASR endpointing is identical across arms and excluded.
                extra["e2e_ms"] = res.ttfs_ms + first_audio
        res.extra = {**res.extra, **extra}
        hist += [{"role": "user", "content": q.text}, {"role": "assistant", "content": res.text}]
        turns.append(res.to_dict())
        mark = "ERR" if res.error else ("ok " if extra["correct"] else "x  ")
        ttft = f"{res.ttft_ms:7.0f}" if res.ttft_ms is not None else "      -"
        print(f"  [{i + 1:3d}] {mark} ttft {ttft} ms  prompt {res.prompt_tokens}  "
              f"cached {res.cached_tokens}  {caller.user_id} t{t_idx}  {res.error or ''}", flush=True)
    return turns


async def main(argv=None) -> int:
    args = parse(argv)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; choose from {ARMS}")

    def url_for(arm):
        if arm == "full":
            return args.full_url or args.single_url
        if arm == "blend":
            return args.blend_url
        if arm == "nomem":
            return args.full_url or args.cartridge_url or args.single_url
        return args.cartridge_url or args.single_url

    engines = {}
    for arm in arms:
        url = url_for(arm)
        if not url:
            raise SystemExit(f"arm {arm!r} has no engine URL (see --help)")
        engines[arm] = Engine(url, args.model, arm=arm, api_key=args.api_key)

    workload = build(args.callers, args.org_tokens, args.user_tokens, args.seed)
    counter = TokenCounter(args.model)
    if not counter.exact:
        print("! tokenizer not loadable here; cartridge token counts are estimates "
              "(the engine's own prompt_tokens are still exact)", flush=True)
    compiler = ContextCompiler(args.model, TENANT, counter)

    tts = None
    if args.tts:
        from supermem.tts import make_tts
        tts = make_tts(args.tts)

    # Engine warm-up (CUDA graphs, allocator), not counted and not memory-related:
    # a throwaway prompt that shares no prefix with the workload.
    for e in {id(e): e for e in engines.values()}.values():
        await e.complete([{"role": "user", "content": "warm-up " + uuid.uuid4().hex}], max_tokens=4)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)

    all_turns, summaries = {}, []
    for arm in arms:
        turns = await run_arm(arm, args, engines, workload, compiler, run_id, tts)
        all_turns[arm] = turns
        (out / f"turns_{arm}.jsonl").write_text("\n".join(json.dumps(t) for t in turns) + "\n")
        summaries.append(summarize(arm, turns))

    versions = {arm: await e.version() for arm, e in engines.items()}
    for e in engines.values():
        await e.close()

    by_arm = {s.arm: s for s in summaries}
    show = args.showcase_arm or ("prefetch" if "prefetch" in by_arm else "cartridge")
    conditions = {
        "context": f"~{by_arm['full'].prompt_tokens:,.0f} prompt tokens" if "full" in by_arm
        and by_arm["full"].prompt_tokens else "",
        "model": args.model,
        "GPU": args.gpu_label or _gpu_name() or "unknown GPU",
        "N": f"{args.turns} turns / {args.callers} callers",
        "engine": ", ".join(sorted({f"vLLM {v}" for v in versions.values() if v})),
    }
    simulated = "simulator" in versions.values() or any(
        t.get("extra", {}).get("simulated") for turns in all_turns.values() for t in turns)
    md = [f"# Cartridge benchmark {run_id}", ""]
    if simulated:
        md += ["> **SIMULATED ENGINE -- NOT A MEASUREMENT.** These numbers come from "
               "`supermem.cartridge.simulator` and must not go on a slide.", ""]
    if "full" in by_arm and show in by_arm:
        md += [showcase_markdown(by_arm["full"], by_arm[show], conditions), ""]
    md += ["## All arms", "", arms_markdown(summaries), "",
           "Method: turns run one at a time, round-robin across callers; each arm starts from a cold "
           "cache (run-scoped epoch line at the top of the prompt); temperature 0; accuracy = every "
           "expected string present in the answer. End-speech -> first audio excludes ASR endpointing "
           "(identical across arms) and is only reported when --tts was given."]
    (out / "summary.json").write_text(json.dumps({
        "run_id": run_id, "simulated": simulated, "conditions": conditions, "showcase_arm": show,
        "args": vars(args), "arms": [s.to_dict() for s in summaries],
    }, indent=2))
    (out / "report.md").write_text("\n".join(md) + "\n")
    latest = Path(args.out) / "latest.json"
    latest.write_text(json.dumps({"run_dir": str(out)}))
    print("\n" + "\n".join(md))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
