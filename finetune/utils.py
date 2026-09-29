import json
from pathlib import Path

DATA = Path(__file__).parent / "data" / "sample.jsonl"

# Hyperparameters used to train the released adapter (Qwen3.6-35B-A3B QLoRA v2); running with defaults reproduces that run.
# They used to be read from a manifest under models/; that manifest was removed with the model directory, so the values live here.
BASE = "Qwen/Qwen3.6-35B-A3B"
ADAPTER = {
    "format": "PEFT LoRA",
    "rank": 32,
    "alpha": 64,
    "dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    # Hard-coded to Qwen3.6-35B-A3B module names; must change when switching base models (use "all-linear" if unsure).
    "target_modules": (
        r"^(model\.language_model(?=\.).*\.(shared_expert_gate|down_proj|out_proj|"
        r"in_proj_a|in_proj_b|q_proj|in_proj_z|gate_proj|up_proj|in_proj_qkv|"
        r"k_proj|v_proj|o_proj))$"
    ),
}
TRAIN = {
    "epochs": 2,
    "seed": 42,
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.03,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "precision": "bf16",
    "gradient_checkpointing": True,
    "max_sequence_length": 2048,
    "optimizer": "adamw_torch_fused",
    "weight_decay": 0.1,
    "adam_beta": [0.9, 0.95],
}

MEMORY_CATEGORIES = ("knowledge", "emotion", "persona")

SYSTEM = {
    ("memory", "en"): "You are SuperMem, a personal AI companion with memory. Reply naturally "
                      "using the retrieved memories and user profile below; do not recite them "
                      "verbatim to the user.",
    ("casual", "en"): "You are SuperMem, a warm and thoughtful AI companion. Reply naturally "
                      "based on the current conversation.",
}


def system_prompt(category, lang):
    kind = "memory" if category in MEMORY_CATEGORIES else "casual"
    return SYSTEM[(kind, lang)]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def warmup_steps(n_rows, epochs):
    # transformers 5.x removed warmup_ratio and only keeps warmup_steps, so convert it here.
    # Computed for a single process; divide by the number of GPUs for multi-GPU.
    per_step = TRAIN["per_device_train_batch_size"] * TRAIN["gradient_accumulation_steps"]
    return max(1, round(n_rows * epochs / per_step * TRAIN["warmup_ratio"]))
