"""Lightweight token-usage logging, so eval scripts can account for real spend.

Configured via environment variables:
  COST_LOG_PATH - which jsonl file to write to (unset = no logging, zero overhead in production)
  COST_TAG      - label for this batch run (e.g. "locomo-final"), for grouping when aggregating
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_COST_LOG_PATH = os.environ.get("COST_LOG_PATH")
_COST_TAG = os.environ.get("COST_TAG", "?")


def log_usage(step: str, model: str, usage) -> None:
    if not _COST_LOG_PATH or usage is None:
        return
    line = json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "tag": _COST_TAG,
        "step": step,
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
    }, ensure_ascii=False)
    with open(_COST_LOG_PATH, "a") as f:
        f.write(line + "\n")
