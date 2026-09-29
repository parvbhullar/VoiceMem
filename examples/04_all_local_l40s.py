"""All open-source components, runs on a single L40S: ASR / VAD / memory vectors are local, the LLM goes through vLLM, TTS is local.

    # Start vLLM in another terminal (48G card, leave half the VRAM for TTS and the perception models)
    vllm serve Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.5

    pip install sounddevice voxcpm pywebrtc-audio
    bash scripts/download_models.sh          # FunASR / silero / E5
    python examples/04_all_local_l40s.py

Nothing leaves the machine: transcription is FunASR, end-of-speech is silero, memory vectors and slot classification are local E5,
fact extraction and replies both go to the local vLLM, and speech is VoxCPM2. Echo cancellation and barge-in live in _audio.py.
"""
import asyncio

from _audio import AudioIO, SR

from supermem import SuperMem
from supermem.tts import speak_stream, tts_stream

# The local vLLM's OpenAI-compatible endpoint. api_key can be anything, vLLM does not check it.
VLLM = {"model": "Qwen/Qwen3-8B", "api_key": "EMPTY",
        "base_url": "http://127.0.0.1:8000/v1"}

# TTS is local too. VoxCPM2 is 2B with good quality, but the first frame takes 0.5s+ and it competes with vLLM for the same card;
# for low latency switch to piper (provider "local", tens of MB, first frame in tens of ms,
# point SUPERMEM_TTS_MODEL at the voice .onnx).
TTS = {"tts": {"provider": "voxcpm"}}

# Embedding providers come from three kinds of sources; one config covers every channel (memory vectors, slot anchors,
# entity dedup, the right-brain judgement table):
#   local        Local E5 (multilingual-e5-small). 0 network, about 10ms -- real-time voice can only
#                use this kind; the speculative prefetch time budget cannot afford an HTTP request.
#   openai       OpenAI or any compatible endpoint. Honours base_url, so TEI / vLLM go through this too:
#                  {"provider": "openai", "config": {"model": "BAAI/bge-m3",
#                   "base_url": "http://127.0.0.1:8080/v1", "api_key": "EMPTY"}}
#   other names  Passed to mem0's EmbedderFactory: ollama / huggingface / gemini /
#                aws_bedrock / azure_openai / vertexai / together / lmstudio /
#                fastembed / langchain. config is passed through to mem0 unchanged:
#                  {"provider": "ollama", "config": {"model": "nomic-embed-text",
#                   "ollama_base_url": "http://127.0.0.1:11434"}}
#
# Two things to watch when switching embedders:
#   - Pick a model that matches your language. An English-only model like all-MiniLM-L6-v2 used for Chinese retrieval
#     **raises no error, it is just all wrong** (in a test, "where do I study" returned "allergic to nuts" first) -- the hardest kind of bug to track down.
#   - Old vectors in the store will have mismatched dimensions and are skipped with a warning; right-brain retrieval and
#     entity dedup stay empty until you re-embed: python3 tools/reembed.py <space> --apply
vm = SuperMem.from_config({
    "mode": "normal",
    "embedding": {"provider": "local"},                # memory vectors: local E5
    "slots":     {"provider": "local"},                # slot classification: local E5, 0 LLM
    "llm":   {"provider": "openai", "config": VLLM},   # write-side fact extraction -> vLLM
    "reply": {"provider": "openai", "config": VLLM},   # replies -> vLLM
})


async def warmup(stream):
    """All models load lazily: without warmup the first turn loads E5 / FunASR / silero / the perception stack
    / VoxCPM on the spot, and the user waits twenty-odd seconds after saying the first sentence at [ready]."""
    await asyncio.to_thread(vm.warmup)                   # E5 + FunASR + silero + perception
    await asyncio.to_thread(vm.search, "warmup")         # vector store
    await stream.feed(b"\x00" * 320)
    async for _ in tts_stream("Hello.", TTS):            # VoxCPM weights
        break


async def main():
    loop = asyncio.get_running_loop()
    stream = vm.stream(src_rate=SR)

    stop = asyncio.Event()                               # user interrupted -> this turn ends here
    audio = AudioIO(loop, on_barge_in=stop.set)

    print("[warmup] Loading models...", flush=True)
    await warmup(stream)

    audio.start()
    print("[ready] Start talking", flush=True)

    try:
        while True:
            st = await stream.feed(await audio.mic.get())
            if st.state != "turn_over":
                continue

            print(f"\nYou: {st.transcript}\nAssistant: ", end="", flush=True)

            # Memory was prefetched while you were talking, so reply right away; synthesise while generating, without waiting for the full text.
            stop.clear()
            audio.assistant_started()
            async for pcm in speak_stream(vm.reply_stream(st), TTS,
                                          on_delta=lambda d: print(d, end="", flush=True)):
                if stop.is_set():
                    print("\n[interrupted]", flush=True)
                    break
                audio.play(pcm)
            audio.assistant_done()

            # Storing memory takes seconds, so run it in a thread -- on this loop it would block reading the mic for the next turn.
            # reply_stream has already recorded what the assistant said (when interrupted, only the half the user actually heard).
            asyncio.create_task(asyncio.to_thread(vm.ingest, st.transcript, async_facts=True))
    finally:
        audio.close()


asyncio.run(main())
