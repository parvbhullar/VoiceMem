"""Store and search: audio in -> both brains; text in -> left brain only.

    export OPENAI_API_KEY=sk-...
    python examples/01_memory.py

Embedding and slots both use local E5, the same config as the web demo: the retrieval path uses 0 LLM calls and
0 network, ~10ms by itself. The default OpenAI embedding also works, but every search makes an HTTP request, and the
0-500ms speculative prefetch while the user speaks no longer fits (the 134ms in the README refers to this local setup).
"""
import os
from pathlib import Path

from supermem import SuperMem

# relative to the script, not the cwd -- found no matter which directory you run from
AUDIO = str(Path(__file__).resolve().parent.parent / "assets/input.wav")

LOCAL = {
    "embedding": {"provider": "local"},
    "slots": {"provider": "local"},
    "api_key": os.environ["OPENAI_API_KEY"],   # only used for fact extraction on the write side
    # A separate store. **Stores with different vector dimensions cannot be mixed** -- local E5 is 384-dim, OpenAI is
    # 1536-dim; pointing both at one directory fails with shapes (n,384) and (1536,) not aligned.
    # Without this line it would collide with the default store (most likely built with OpenAI dimensions).
    "memory_root": str(Path(__file__).resolve().parent / "example_memory"),
}
# top_k can also go into from_config (one dict for everything); misspelled keys now raise instead of being silently dropped.
# How many to fetch is passed to search().

vm = SuperMem.from_config({**LOCAL, "mode": "normal"})

# Local models load lazily: without warmup the first ingest waits an extra twenty-odd seconds (E5 / FunASR /
# the perception stack all load then). The web demo always does this, and so do we here.
vm.warmup(verbose=True)

# Store: an audio file
# Internally runs ASR / speaker ID / scene / emotion perception / embedding extraction
vm.ingest(audio=AUDIO)  # (Mandarin audio) "I'm vegetarian and allergic to nuts."

result = vm.search("What are my dietary restrictions?", top_k=5)

print(result.result_leftbrain, result.result_rightbrain)


# Store: left-brain factual text (no emotion)
vm = SuperMem.from_config({**LOCAL, "mode": "leftbrain_only"})

vm.ingest("I'm vegetarian and allergic to nuts.")

result = vm.search("What are my dietary restrictions?", top_k=5)

print(result.result_leftbrain, result.result_rightbrain)
