"""Qwen Omni text-generation helpers: consolidate generate kwargs and silence known harmless warnings."""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Any


def build_omni_text_gen_kwargs(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tokenizer: Any,
) -> dict[str, Any]:
    """For text JSON output; with temperature=0 no sampling params are passed, avoiding a transformers UserWarning."""
    do_sample = float(temperature) > 0.0
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    return kwargs


class _OmniInferenceLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "System prompt modified" in msg:
            return False
        if "audio output may not work" in msg:
            return False
        return True


@contextmanager
def suppress_omni_text_inference_noise():
    """Silence known Omni+transformers noise from custom system prompts / greedy decoding."""
    log_filter = _OmniInferenceLogFilter()
    root = logging.getLogger()
    root.addFilter(log_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*do_sample.*", category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message=".*System prompt modified.*",
                category=UserWarning,
            )
            yield
    finally:
        root.removeFilter(log_filter)
