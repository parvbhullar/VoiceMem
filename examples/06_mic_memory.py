"""Listen only, no answers: mic -> transcription -> memory retrieval. No LLM reply, no TTS.

    export OPENAI_API_KEY=sk-...
    python examples/06_mic_memory.py

Say something and you will see three things happen in order:

    [hear]   live transcription, words appear while you are still talking
    [recall] the moment you finish, the relevant memories are **already in hand** -- retrieval ran in the
             background while you were talking (0-500ms speculative prefetch), taking no time after you finish
    [store]  this turn is written to the memory store and can be found next time

To see that it really remembers: say "I'm allergic to peanuts", then a few sentences later ask "What can't I eat?".

Ctrl-C to exit.
"""
import asyncio
import os
import queue
import sys
from pathlib import Path

import sounddevice as sd

from supermem import SuperMem

SR = 16000          # mic sample rate
BLOCK = 512         # samples per block: 32ms @16k, aligned with the VAD frame length

vm = SuperMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},   # memory vectors: local, 0 network
    "slots": {"provider": "local"},       # slot classification: local, 0 LLM
    "api_key": os.environ["OPENAI_API_KEY"],   # only used for write-side fact extraction
    # A separate store: local E5 is 384-dim, and mixing with the default store (OpenAI, 1536-dim) fails with
    # shapes (n,384) and (1536,) not aligned
    "memory_root": str(Path(__file__).resolve().parent / "example_memory"),
})


def show_partial(text):
    print(f"\r[hear] {text}", end="", flush=True)


async def main():
    vm.warmup()

    # sounddevice's callback runs on its own thread and cannot await. Pass through a queue and
    # let the event loop side take from it -- the callback only moves data and must never block, or audio is dropped.
    blocks: queue.Queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        blocks.put(bytes(indata))

    stream = vm.stream(src_rate=SR, on_partial=show_partial)

    with sd.RawInputStream(samplerate=SR, blocksize=BLOCK, dtype="int16",
                           channels=1, callback=on_audio):
        print("Start talking (Ctrl-C to exit)\n", flush=True)
        while True:
            pcm = await asyncio.to_thread(blocks.get)
            st = await stream.feed(pcm)
            if st.state != "turn_over":
                continue

            print(f"\r[hear] {st.transcript}")

            left = st.result_leftbrain or []
            right = st.result_rightbrain or []
            if left or right:
                print("[recall] Memories already retrieved the moment you finished:")
                for m in left:
                    print(f"       left brain   {m}")
                for m in right:
                    print(f"       right brain  {m}")
            else:
                print("[recall] No relevant memories yet (the store is empty; say a few more things)")

            # writing takes seconds, run it in a thread so the mic isn't blocked
            asyncio.create_task(asyncio.to_thread(vm.ingest, st.transcript))
            print("[store] Written, retrievable next time\n", flush=True)


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nBye.")
    sys.exit(0)
