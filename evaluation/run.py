#!/usr/bin/env python3
"""SuperMem evaluation entry point: run one benchmark with one command.

    python evaluation/run.py --dataset locomo --data data/locomo.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# mem0 telemetry opens a global qdrant in ~/.mem0 with an exclusive file lock; multi-process/threaded evaluation
# hits "already accessed by another instance". Evaluation does not need it. Must be set before the import.
os.environ.setdefault("MEM0_TELEMETRY", "False")

from evaluation import datasets  # noqa: E402


# ── (4) Answer model / judge: both use the OpenAI-compatible API, configured in one place ──

def make_llm(model: str):
    """Return ``fn(system, user) -> str``. Set OPENAI_BASE_URL to use your own endpoint."""
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)

    def call(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()
    return call


def provenance() -> dict:
    """The environment of this evaluation run. Written into the results file -- so a number seen six months later can be traced to the code that produced it."""
    import platform
    import subprocess
    from datetime import datetime, timezone

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""

    def version(pkg):
        from importlib.metadata import PackageNotFoundError, version as v
        try:
            return v(pkg)
        except PackageNotFoundError:
            return ""

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),   # run with uncommitted changes: the number maps to no code version
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("supermem", "mem0ai", "openai", "qdrant-client")},
    }


ANSWER_SYSTEM = """Answer the question based on the "memories".

"User" in the memories and the person named in the question are the same person -- the memories were extracted
from that person's own words, and "User" as the subject is just the wording used during extraction. Do not conclude
the memories lack something just because the name differs.

Answer only from the memories below; do not invent information that is not in them. Only answer "I don't know"
when the memories really do not contain it.
Keep the answer short -- give the answer directly, do not restate the question or explain your reasoning.

Memories:
{memory}"""


# ── (2)(3) Full evaluation of one conversation ─────────────────────────────────

def run_conversation(conv, args, answer_llm, judge_llm) -> dict:
    """Build a store -> ingest the conversation -> retrieve and answer each question -> score. Returns this conversation's results."""
    from supermem import SuperMem
    from supermem.memory_api import build_memory_context

    # One independent memory store per conversation: mixing them would feed in answers from other conversations
    root = Path(args.memory_root) / conv.id
    vm = SuperMem(memory_root=str(root), user_id=conv.id, mode=args.mode)

    t0 = time.time()
    for turn in conv.turns:
        try:
            vm.ingest(turn.text, speaker=turn.speaker or "user",
                      observed_at=turn.observed_at or None, async_facts=False)
        except Exception as e:
            print(f"  [{conv.id}] ingest failed (skipping this turn): {e}", flush=True)
    ingest_s = time.time() - t0

    items, got, total = [], 0.0, 0.0
    for q in conv.questions:
        t1 = time.time()
        result = vm.search(q.text, top_k=args.top_k)
        search_ms = (time.time() - t1) * 1000
        memory = build_memory_context(result)

        answer = answer_llm(ANSWER_SYSTEM.format(memory=memory or "(no relevant memories)"), q.text)
        s = (datasets.Score(0.0, 1.0, "not scored") if args.no_score
             else ds_score(args, q, answer, judge_llm))

        got += s.correct
        total += s.total
        items.append({
            "question_id": q.id, "question": q.text, "gold": q.answer,
            "predicted": answer, "correct": s.correct, "total": s.total,
            "note": s.note, "category": q.category,
            "search_ms": round(search_ms, 1),
            "memory_tokens": len(memory) // 2,      # rough estimate: ~2 characters per token (calibrated for Chinese)
            "memory": memory if args.save_memory else "",
        })

    return {"conversation_id": conv.id, "ingest_seconds": round(ingest_s, 1),
            "score": got, "total": total, "items": items}


def ds_score(args, q, answer, judge_llm):
    return datasets.get(args.dataset).score(q, answer, judge_llm)


# ── Summary ──────────────────────────────────────────────────────────────────

def summarize(results: list[dict], dataset: str) -> dict:
    got = sum(r["score"] for r in results)
    total = sum(r["total"] for r in results)
    by_cat: dict[str, list[float]] = {}
    lat, toks = [], []
    for r in results:
        for it in r["items"]:
            by_cat.setdefault(it["category"] or "all", []).append(
                it["correct"] / it["total"] if it["total"] else 0.0)
            lat.append(it["search_ms"])
            toks.append(it["memory_tokens"])
    return {
        "dataset": dataset,
        "conversations": len(results),
        "questions": sum(len(r["items"]) for r in results),
        "score": round(got, 2), "total": round(total, 2),
        "accuracy": round(got / total, 4) if total else 0.0,
        "by_category": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())},
        "median_search_ms": round(sorted(lat)[len(lat) // 2], 1) if lat else 0.0,
        "median_memory_tokens": sorted(toks)[len(toks) // 2] if toks else 0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="SuperMem evaluation: run one benchmark with one command")
    p.add_argument("--dataset", required=True, choices=datasets.names(),
                   help="which benchmark to run (see evaluation/datasets/)")
    p.add_argument("--data", required=True, help="dataset file path")
    p.add_argument("--out", default="", help="results json, default results/<dataset>.json")
    p.add_argument("--answer-model", default=os.environ.get("EVAL_ANSWER_MODEL", "gpt-4o-mini"),
                   help="model that answers using the memories")
    p.add_argument("--judge", default=os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini"),
                   help="judge model for scoring")
    p.add_argument("--mode", default="left_brain_single",
                   help="SuperMem mode; left_brain_single is enough for text-only evaluation, "
                        "use text_mode to include the right brain")
    p.add_argument("--top-k", type=int, default=5, help="memories retrieved per question")
    p.add_argument("--limit", type=int, default=0, help="only run the first N conversations (for debugging)")
    p.add_argument("--workers", type=int, default=4, help="conversations to run concurrently")
    p.add_argument("--memory-root", default="", help="directory for memory stores, default results/<dataset>_memory")
    p.add_argument("--resume", action="store_true", help="resume the previous run, skipping finished conversations")
    p.add_argument("--save-memory", action="store_true", help="also save each question's retrieved memories in the results (large, useful for review)")
    p.add_argument("--inspect", action="store_true", help="only parse the dataset and print the first few entries, no evaluation")
    p.add_argument("--no-score", action="store_true",
                   help="generate answers without scoring; score later with evaluation/score.py (switching judges needs no rerun)")
    args = p.parse_args()

    out = Path(args.out or f"results/{args.dataset}.json")
    if not args.memory_root:
        args.memory_root = str(out.parent / f"{args.dataset}_memory")

    # (1) Load the dataset
    convs = datasets.get(args.dataset).load(args.data)
    if args.limit:
        convs = convs[:args.limit]

    if args.inspect:                        # confirm parsing is right before spending money
        print(f"Parsed {len(convs)} conversations, "
              f"{sum(len(c.questions) for c in convs)} questions\n")
        for c in convs[:2]:
            print(f"[{c.id}] {len(c.turns)} turns / {len(c.questions)} questions")
            for t in c.turns[:3]:
                print(f"   {t.observed_at or '(no date)'} {t.speaker}: {t.text[:60]}")
            for q in c.questions[:2]:
                print(f"   Q: {q.text[:60]}")
                print(f"   A: {q.answer[:60]}" if q.answer else f"   rubric: {q.rubric[:2]}")
            print()
        return

    done: dict[str, dict] = {}
    if args.resume and out.exists():
        done = {r["conversation_id"]: r
                for r in json.loads(out.read_text(encoding="utf-8")).get("results", [])}
        print(f"Resuming: {len(done)} conversations already done, skipping", flush=True)

    prov = provenance()
    if prov["git_dirty"]:
        print("Warning: working tree is dirty; these numbers map to no commit", flush=True)
    answer_llm, judge_llm = make_llm(args.answer_model), make_llm(args.judge)
    todo = [c for c in convs if c.id not in done]
    results = list(done.values())

    print(f"{datasets.display_name(args.dataset)}: {len(todo)} conversations to run "
          f"(answer={args.answer_model}, judge={args.judge}, top_k={args.top_k})", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_conversation, c, args, answer_llm, judge_llm): c
                   for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            conv = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"  [{conv.id}] failed: {e}", flush=True)
                continue
            results.append(r)
            acc = r["score"] / r["total"] if r["total"] else 0
            print(f"  [{i}/{len(todo)}] {r['conversation_id']}  "
                  f"{r['score']:.0f}/{r['total']:.0f} ({acc:.0%})", flush=True)
            # write after every conversation: a multi-hour run that dies midway need not restart from scratch
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"summary": summarize(results, args.dataset),
                                       "config": vars(args), "provenance": prov,
                                       "results": results},
                                      ensure_ascii=False, indent=2), encoding="utf-8")

    s = summarize(results, args.dataset)
    name = datasets.display_name(args.dataset)
    print(f"\n{name}: {s['conversations']} conversations \u00b7 {s['questions']} questions\n")
    print(f"Score: {s['score']:.0f}/{s['total']:.0f} = {s['accuracy']:.1%}\n")
    if len(s["by_category"]) > 1:
        width = max(len(k) for k in s["by_category"])
        for k, v in s["by_category"].items():
            print(f"  {k:<{width + 2}}{v:.1%}")
        print()
    print(f"Median retrieval latency: {s['median_search_ms']:.0f} ms")
    print(f"Median retrieved memory: {s['median_memory_tokens']} tokens")
    print(f"\nSaved to {out}")
    if args.no_score:
        print(f"Not scored. Score with: python evaluation/score.py --file {out}")


if __name__ == "__main__":
    main()
