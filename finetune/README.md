# finetune

Train your own SuperMem reply adapter.

```bash
pip install ms-swift==4.5.2 bitsandbytes    # install torch for your own platform
python finetune/train.py            # first run the pipeline end to end on the bundled 5 samples
python finetune/train.py --data data/train.jsonl
```

Default hyperparameters live in `finetune/utils.py` (`BASE` / `ADAPTER` / `TRAIN`) and match the
training run of the released adapter, so **running with defaults reproduces that run** (base
`Qwen/Qwen3.6-35B-A3B`, LoRA rank 32 / alpha 64, 4-bit).

## Files

| | |
|---|---|
| `train.py` | Training entry point, ms-swift's `sft_main` |
| `eval.py` | Runs a trained adapter over the data and prints `ref` / `pred` for each row |
| `dataset.py` | Reads JSONL and **validates every row**; on a violation it reports the row and message number |
| `utils.py` | Default hyperparameters, system prompts, warmup conversion |
| `data/sample.jsonl` | 5 sample rows |

## Data format

```json
{
  "messages": [
    {"role": "system",    "content": "<chosen by category/lang, see below>"},
    {"role": "user",      "content": "what was my cat's name again\n\nMEMORY CONTEXT (things you remember about the user):\n- [2023-05-08] User has a British Shorthair cat named Momo, three years old."},
    {"role": "assistant", "content": "It's Momo, your three-year-old British Shorthair."}
  ],
  "meta": {"lang": "en", "category": "knowledge", "session_id": "s_0001", "turn": 1}
}
```
The memory block is appended to **the last user turn**; history turns carry no memory. Only the
last assistant message contributes to the loss (`loss_scale="last_round"`); history turns do not.

## Common options

```bash
python finetune/train.py --data data/train.jsonl \
    --out out/my-adapter --epochs 3 --lr 1e-4 --no-4bit
```

| | | Default |
|---|---|---|
| `--data` | Training data | `finetune/data/sample.jsonl` |
| `--out` | Output directory | `out/supermem-qlora` |
| `--base` | Base model | `Qwen/Qwen3.6-35B-A3B` |
| `--rank` / `--alpha` | LoRA rank / alpha | 32 / 64 |
| `--epochs` / `--lr` | Epochs / learning rate | 2 / 2e-4 |
| `--max-len` | Max sequence length | 2048 |
| `--no-4bit` | Skip 4-bit quantization (only with enough VRAM) | 4-bit on by default |

**Switching base models requires changing `target_regex`** -- it is `ADAPTER["target_modules"]` in
`utils.py`, hard-coded to Qwen3.6-35B-A3B module names. If unsure, change it to `all-linear`.

## Evaluation

```bash
python finetune/eval.py --adapter out/supermem-qlora --out preds.jsonl
```

Prints `question` / `ref` / `pred` / `meta` per row; `--out` saves them as JSONL.
Temperature is fixed at 0 for reproducible comparisons.

For metrics on memory retrieval itself, see [`evaluation/`](../evaluation/).

## Notes

- **Training data is not in this repository.** Before publishing it, document its source, license, consent status and preprocessing.
- The adapter cannot be distributed as a standalone model; check the base model's license and access terms yourself.
- For multi-GPU training, `utils.warmup_steps()` is computed per process; divide by the number of GPUs.
