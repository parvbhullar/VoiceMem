"""Streaming API: feed audio chunks and get all perception results for the turn as soon as it ends.

    python examples/02_streaming.py speech.wav
"""
import asyncio
import os
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import soundfile as sf

from supermem import SuperMem

# Local E5: retrieval uses no network, so speculative prefetch fits in time (same config as the web demo)
vm = SuperMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},
    "slots": {"provider": "local"},
    "api_key": os.environ["OPENAI_API_KEY"],   # only used for fact extraction on the write side
    # A separate store: local E5 is 384-dim, and mixing with the default store (OpenAI, 1536-dim) fails with
    # shapes (n,384) and (1536,) not aligned
    "memory_root": str(Path(__file__).resolve().parent / "example_memory"),
})
WAV = sys.argv[1] if len(sys.argv) > 1 else str(
    Path(__file__).resolve().parent.parent / "assets/speech.wav")

# Local models load lazily: without warmup the first audio chunk waits twenty-odd seconds for model loading
vm.warmup(verbose=True)


async def main():
    audio, sr = sf.read(WAV, dtype="float32")
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)

    stream = vm.stream(
        src_rate=sr,
        vad_threshold=0.5,
        on_partial=lambda t: print(f"\r[partial] {t}", end="", flush=True),
    )

    step = int(sr * .032)

    for i in range(0, len(pcm), step):
        st = await stream.feed(pcm[i:i + step].tobytes())
        print(f"\n[state] {st.state}")

        FIELDS = [
            "result_leftbrain",
            "result_rightbrain",
            "speaker_id",
            "speaker_voiceprint",
            "emotion",
            "transcript",
            "entity",
            "slots",
            "text_embedding",
        ]

        if st.state == "turn_over":
            pprint({key: getattr(st, key) for key in FIELDS})


asyncio.run(main())
