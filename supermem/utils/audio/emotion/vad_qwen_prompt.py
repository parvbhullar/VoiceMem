from __future__ import annotations

import json
import re
from typing import Any

from supermem.utils.audio.emotion.omni_generate import (
    build_omni_text_gen_kwargs,
    suppress_omni_text_inference_noise,
)
from supermem.utils.audio.emotion.types import VAD
from supermem.utils.audio.emotion.vad_audio import VADEstimator

_DEFAULT_SYSTEM_EN = """You are an acoustic affect annotator. Listen ONLY to the user's audio (speech prosody, pace, sighs—not any separate text channels).
Rate how the speaker sounds on two dimensions (do not transcribe semantic content unless it informs tone):
- valence: float in [-1, 1]. −1 unpleasant/negative sounding, +1 pleasant/positive sounding, 0 neutral.
- arousal: float in [0, 1]. 0 calm/low activation, 1 highly activated/agitated/excited.

Reply with ONE JSON object only, valid JSON on a single line, no markdown fences, no extra keys:
{"valence": <float>, "arousal": <float>}
"""


def _get_sequences(generate_out: Any) -> Any:
    if hasattr(generate_out, "sequences"):
        return generate_out.sequences
    if isinstance(generate_out, (tuple, list)) and len(generate_out) > 0:
        return generate_out[0]
    return generate_out


def parse_vad_from_model_text(raw: str) -> VAD:
    """Parse JSON from model output (tolerates leading/trailing noise and occasional markdown fences)."""
    s = raw.strip()
    if not s:
        raise ValueError("empty model output for VAD")

    # strip markdown ```json ... ```
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, re.I)
    if fence:
        s = fence.group(1).strip()

    start = s.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in model output: {s[:200]!r}")
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s[start:])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in VAD response: {e}; snippet={s[start : start + 280]!r}") from e

    if not isinstance(obj, dict):
        raise ValueError(f"VAD JSON must be object, got {type(obj)}")

    v = float(obj["valence"])
    a = float(obj["arousal"])
    return VAD(valence=v, arousal=a)


class QwenOmniPromptVADEstimator:
    """Estimate V/A via prompt with Qwen2.5-Omni (or another multimodal model supporting chat_template + audio path).

    By design, the upstream loads ``processor`` / ``model`` / ``tokenizer`` once and injects them into this class,
    to avoid the dialogue main loop and this class each holding a VRAM copy (separate processes loading their own is fine).
    """

    def __init__(
        self,
        processor: Any,
        model: Any,
        tokenizer: Any,
        *,
        system_prompt: str = _DEFAULT_SYSTEM_EN,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> None:
        self.processor = processor
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    # for runtime checks: satisfies the VADEstimator protocol
    def estimate(self, audio_path: str) -> VAD:
        import torch

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "audio", "path": audio_path}],
            },
        ]

        gen_kwargs = build_omni_text_gen_kwargs(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            tokenizer=self.tokenizer,
        )

        self.model.eval()
        with torch.no_grad(), suppress_omni_text_inference_noise():
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)
            out = self.model.generate(**inputs, **gen_kwargs)
            sequences = _get_sequences(out)
            input_len = inputs["input_ids"].shape[-1]
            if sequences.shape[-1] > input_len:
                to_decode = sequences[:, input_len:]
            else:
                to_decode = sequences
            text = (
                self.processor.batch_decode(
                    to_decode,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                if to_decode.shape[-1] > 0
                else ""
            )

        return parse_vad_from_model_text(text)
