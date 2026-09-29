# evaluation — one command, one number

```bash
export OPENAI_API_KEY=sk-...

python evaluation/run.py --dataset locomo --data data/locomo.json
```
The result is printed when the run finishes and also written to `results/locomo.json`:

```text
LoCoMo: 10 conversations · 152 questions

Score: 139/152 = 91.4%

  multi_hop     88.2%
  single_hop    95.1%
  temporal      85.7%

Median retrieval latency: 12 ms
Median retrieved memory: 298 tokens

Saved to results/locomo.json
```

## Common options

| Option | What it does | Default |
|---|---|---|
| `--dataset` / `--data` | Which adapter / data file to use | required |
| `--answer-model` | Model that answers using the memories | `gpt-4o-mini` |
| `--judge` | Judge model for scoring | `gpt-4o-mini` |
| `--top-k` | Memories retrieved per question | `5` |
| `--mode` | `left_brain_single` = fact memory only; `text_mode` = include the right brain | `left_brain_single` |
| `--workers` | Conversations run concurrently | `4` |
| `--limit` | Only run the first N conversations (for debugging) | all |
| `--resume` | Continue the previous run, skipping finished conversations | off |
| `--save-memory` | Also store each question's retrieved memories in the results, for manual review | off |
| `--inspect` | Only parse the dataset and print it, no evaluation | off |
| `--no-score` | Generate answers without scoring; score later with `score.py` | off |

Results are written to disk after every conversation, so if a multi-hour run dies midway, add `--resume` to continue.

## Re-scoring

Retrieval + answering is the expensive half (one search plus one generation per question); scoring is the cheap
half. To switch judge models, or after fixing a scoring bug, you don't need to re-run the expensive half:

```bash
python evaluation/score.py --file results/locomo.json --judge gpt-4o
```

The original questions are re-read from the dataset (matched by conversation id + question id), not reconstructed
from the results file -- fields needed for scoring such as rubric and meta are not stored there. It prints how many
verdicts changed.

To split the run into two stages entirely, pass `--no-score` when generating.

## What's in the results file

```json
{
  "summary":    { "accuracy": ..., "by_category": {...}, "median_search_ms": ... },
  "config":     { every parameter used for this run },
  "provenance": { "git_commit": ..., "git_dirty": ..., "python": ..., "packages": {...} },
  "results":    [ gold / predicted / scoring rationale for every question of every conversation ]
}
```

`provenance` exists so that a number seen six months later can still be traced to the code and environment that
produced it. `git_dirty` true means the working tree had uncommitted changes during the run and the number maps
to no commit -- commit before producing official results.

## Evaluating a new benchmark

One file, two functions, and not a single line of the main flow changes.

**1. Copy `datasets/locomo.py` to `datasets/your_dataset.py`** and implement two functions:

```python
def load(path: str) -> list[Conversation]:
    """Read your data file and convert it to the common structure.
    Conversation(id, turns=[Turn(speaker, text, observed_at)], questions=[Question(...)])
    """

def score(q: Question, answer: str, judge) -> Score:
    """Decide whether this question is answered correctly. judge(system, user) -> str is the injected judge model.
    Score(correct=1.0, total=1.0, note="scoring rationale")
    For rubric scoring: correct = points satisfied, total = total points
    """
```

**2. Register it in `DATASETS` in `datasets/__init__.py`**:

```python
DATASETS = {"locomo": "evaluation.datasets.locomo", "your_dataset": "evaluation.datasets.your_dataset"}
```

**3. Run**:

```bash
python evaluation/run.py --dataset your_dataset --data data/xxx.json --inspect   # verify parsing first
python evaluation/run.py --dataset your_dataset --data data/xxx.json
```
