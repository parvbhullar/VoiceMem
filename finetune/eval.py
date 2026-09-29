import argparse
import collections
import json

from swift import InferRequest, RequestConfig, TransformersEngine

from dataset import answer, load, prompt, question
from utils import BASE, DATA

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--data", default=str(DATA))
p.add_argument("--base", default=BASE)
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--out", default="")
p.add_argument("--quiet", action="store_true")
args = p.parse_args()

rows = load(args.data)
engine = TransformersEngine(args.base, adapters=[args.adapter])
config = RequestConfig(max_tokens=args.max_tokens, temperature=0.0)
resps = engine.infer([InferRequest(messages=prompt(r)) for r in rows], config)

out = []
for row, resp in zip(rows, resps):
    pred = resp.choices[0].message.content
    ref = answer(row)
    out.append({
        "question": question(row),
        "ref": ref,
        "pred": pred,
        "meta": row["meta"],
        "pred_chars": len(pred or ""),
        "ref_chars": len(ref or ""),
    })
    if not args.quiet:
        print(json.dumps(out[-1], ensure_ascii=False))

if args.out:
    from utils import write_jsonl
    write_jsonl(out, args.out)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def table(key):
    groups = collections.defaultdict(list)
    for o in out:
        groups[o["meta"][key]].append(o)
    for name, rows_ in sorted(groups.items()):
        empty = sum(1 for o in rows_ if not (o["pred"] or "").strip())
        print(f"  {name:<12} n={len(rows_):<4} "
              f"median answer length {median([o['pred_chars'] for o in rows_]):<5} "
              f"(ref {median([o['ref_chars'] for o in rows_])})"
              f"{f'  empty answers {empty}' if empty else ''}")


print(f"\n{'=' * 56}")
print(f"{len(out)} rows · base {args.base} · adapter {args.adapter}")
print("By category:")
table("category")
print("By lang:")
table("lang")
print("\nNote: this only reports coverage and length, not accuracy -- reply quality needs human review or a separate judge model.")
