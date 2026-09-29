#!/usr/bin/env python3
"""Re-score a finished evaluation run -- without re-running retrieval and answering.

    python evaluation/score.py --file results/locomo.json
    python evaluation/score.py --file results/locomo.json --judge gpt-4o --out results/locomo-gpt4o.json

Retrieval and answering are the expensive half (one search + one generation per question); scoring is the cheap half.
With them split: switching judge models, fixing a scoring bug, or checking whether scores are stable across judges
all re-run only the cheap half. run.py --no-score does only the expensive half.

The original questions are re-read from the dataset (matched by question_id), not reconstructed from the results file --
fields needed for scoring such as rubric and meta are not stored in the results file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation import datasets                              # noqa: E402
from evaluation.run import make_llm, provenance, summarize   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Re-score existing evaluation results")
    p.add_argument("--file", required=True, help="results json produced by run.py")
    p.add_argument("--out", default="", help="where to write, default overwrites --file")
    p.add_argument("--judge", default="", help="judge model, defaults to the one from the original run")
    p.add_argument("--dataset", default="", choices=[""] + datasets.names(),
                   help="defaults to the one recorded in the results file")
    p.add_argument("--data", default="", help="dataset file path, defaults to the one recorded in the results file")
    args = p.parse_args()

    src = Path(args.file)
    blob = json.loads(src.read_text(encoding="utf-8"))
    cfg = blob.get("config", {})

    dataset = args.dataset or cfg.get("dataset", "")
    data = args.data or cfg.get("data", "")
    judge_model = args.judge or cfg.get("judge", "gpt-4o-mini")
    if not dataset or not data:
        raise SystemExit("The results file does not record dataset/data; specify --dataset and --data")
    if not Path(data).exists():
        raise SystemExit(f"Dataset not found: {data}\nUse --data to point to its current location")

    ds = datasets.get(dataset)
    # Index by (conversation id, question id): question_id is unique only within one conversation and collides across them
    # (every LoCoMo conversation has q0/q1/...), so using q.id alone would pick up another conversation's gold answer.
    questions = {(c.id, q.id): q for c in ds.load(data) for q in c.questions}
    judge = make_llm(judge_model)

    n = sum(len(r["items"]) for r in blob["results"])
    print(f"Re-scoring: {n} questions, judge {judge_model} (originally {cfg.get('judge', '?')})", flush=True)

    changed, missing = 0, 0
    for r in blob["results"]:
        got = total = 0.0
        for it in r["items"]:
            q = questions.get((r["conversation_id"], it["question_id"]))
            if q is None:               # the dataset changed and this question no longer matches -- keep the old score and count it
                missing += 1
                got += it["correct"]
                total += it["total"]
                continue
            s = ds.score(q, it["predicted"], judge)
            if s.correct != it["correct"]:
                changed += 1
            it["correct"], it["total"], it["note"] = s.correct, s.total, s.note
            got += s.correct
            total += s.total
        r["score"], r["total"] = got, total

    blob["summary"] = summarize(blob["results"], dataset)
    blob["config"] = {**cfg, "judge": judge_model, "rescored": True}
    blob["provenance"] = provenance()

    out = Path(args.out or src)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")

    s = blob["summary"]
    if missing:
        print(f"Warning: {missing} questions not found in the dataset, kept their original scores")
    print(f"Changed verdict on {changed}/{n} questions")
    print(f"Score {s['score']:.0f}/{s['total']:.0f}  =  {s['accuracy']:.1%}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
