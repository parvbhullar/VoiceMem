"""Hook up a Realtime voice model: supermem only handles memory, audio is fed in parallel to gpt-realtime / qwen-omni-realtime.

    pip install sounddevice pywebrtc-audio "websockets>=14"

    OPENAI_API_KEY=sk-...    python examples/05_realtime_gpt_qwen.py gpt
    DASHSCOPE_API_KEY=sk-... python examples/05_realtime_gpt_qwen.py qwen

Every mic chunk (already echo-cancelled) takes two paths: one into realtime for native speech out, one into
supermem for speculative prefetch. By the moment local VAD decides the user has finished, the memory is ready and
is sent along with response.create -- so memory adds no time before the reply.

Each path has its own consumer (uplink / main loop) and they are not queued together: feed()'s ASR runs on the
event loop, so if they were chained, any hiccup there would also make the realtime uplink audio stutter.

Both providers use the same event names (session.update / input_audio_buffer.append / response.create /
response.*audio.delta); only the session structure differs, see the two dicts below.
"""
import argparse
import asyncio
import base64
import json
import os

import numpy as np
import websockets
from _audio import AudioIO, SR

from supermem import SuperMem
from supermem.utils.audio.stream_io import resample

PERSONA = ("You are a voice assistant. Use what you remember, but don't recite it, and don't open with \"I remember you said\". "
           "Keep sentences short; say one or two sentences at a time and stop.")

GPT = {
    "url": "wss://api.openai.com/v1/realtime?model=gpt-realtime",
    "key_env": "OPENAI_API_KEY",
    "in_rate": 24000,
    # turn_detection lives under session.audio.input, **not at the top level** -- at the top level it is silently rejected.
    # create_response=False: we decide when to reply (after the local turn ends and memory is prefetched);
    # interrupt_response=True: as soon as the user speaks, the server cuts off the reply being played.
    "session": {"type": "realtime", "audio": {
        "input": {"turn_detection": {"type": "server_vad", "create_response": False,
                                     "interrupt_response": True}},
        "output": {"voice": "marin"}}},
    # The memory write side also needs a plain chat model for fact extraction; this path uses OpenAI's.
    "llm": {"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": None},
}

QWEN = {
    "url": "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-omni-flash-realtime",
    "key_env": "DASHSCOPE_API_KEY",
    "in_rate": 16000,
    "session": {"modalities": ["text", "audio"], "voice": "Chelsie",
                "input_audio_format": "pcm16", "output_audio_format": "pcm16",
                "turn_detection": {"type": "server_vad", "create_response": False,
                                   "interrupt_response": True}},
    # Fact extraction also stays with Alibaba: DashScope's OpenAI-compatible mode.
    "llm": {"model": "qwen-plus", "api_key": os.environ.get("DASHSCOPE_API_KEY"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
}

# Server VAD commits the audio buffer itself after a sentence, so our following commit hits an empty buffer.
# For very short utterances it doesn't auto-commit, so the manual commit must stay -- this error is expected.
# response_cancel_not_active: barge-in has two paths (local AEC and server VAD) backing each other up;
# whichever arrives first wins, and the slower one missing is normal.
_EXPECTED_ERRORS = ("input_audio_buffer_commit_empty", "response_cancel_not_active")


def to_wire(pcm: bytes, rate: int) -> str:
    """Mic chunk (16k, AEC's native rate) -> the sample rate this provider wants -> base64."""
    if rate != SR:
        f = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        out = resample(f, src=SR, dst=rate)
        pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes()
    return base64.b64encode(pcm).decode()


async def main(p, name):
    vm = SuperMem.from_config({
        "mode": "normal",
        "embedding": {"provider": "local"},     # retrieval uses no network, so speculative prefetch fits in time
        "slots":     {"provider": "local"},
        "llm": {"provider": "openai", "config": p["llm"]},
    })

    loop = asyncio.get_running_loop()
    to_ws: asyncio.Queue = asyncio.Queue()      # uplink to realtime
    to_vm: asyncio.Queue = asyncio.Queue()      # the same audio, for memory prefetch
    stream = vm.stream(src_rate=SR)
    turn = {"text": "", "reply": "", "live": False}

    print("[warmup] Loading models...", flush=True)
    await asyncio.to_thread(vm.warmup)           # E5 + FunASR + silero + perception
    await asyncio.to_thread(vm.search, "warmup") # vector store
    await stream.feed(b"\x00" * 320)

    key = os.environ[p["key_env"]]

    async with websockets.connect(
            p["url"], additional_headers={"Authorization": f"Bearer {key}"}) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": p["session"]}))

        def on_barge_in():
            """Local AEC heard human voice (playback buffer already cleared) -> tell the server to stop generating too."""
            if not turn["live"]:
                return
            turn["live"] = False
            print("\n[interrupted]", flush=True)
            asyncio.create_task(ws.send(json.dumps({"type": "response.cancel"})))

        audio = AudioIO(loop, on_barge_in=on_barge_in)

        async def split():
            """One audio stream, two paths."""
            while True:
                pcm = await audio.mic.get()
                to_ws.put_nowait(pcm)
                to_vm.put_nowait(pcm)

        async def uplink():
            """Audio uplink gets its own coroutine, never queued behind local ASR."""
            while True:
                pcm = await to_ws.get()
                await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                          "audio": to_wire(pcm, p["in_rate"])}))

        async def pump():
            """The event stream can have only one consumer; audio, text and turn wrap-up are all handled here -- so
            no blocking calls are allowed in here at all."""
            async for raw in ws:
                ev = json.loads(raw)
                t = ev.get("type", "")

                if t.endswith("audio.delta"):                  # gpt: response.output_audio.delta
                    if turn["live"]:
                        audio.play(base64.b64decode(ev["delta"]))
                elif t.endswith("audio_transcript.delta"):
                    if turn["live"]:
                        turn["reply"] += ev["delta"]
                        print(ev["delta"], end="", flush=True)
                elif t.endswith("input_audio_buffer.speech_started"):
                    # Server VAD heard human voice: it already cut the reply on its side, the local buffer must be cleared too,
                    # or those seconds play out anyway and it sounds like "interrupting doesn't work".
                    if turn["live"]:
                        turn["live"] = False
                        audio.stop_playing()
                elif t.endswith("response.done") or t.endswith("response.cancelled"):
                    turn["live"] = False
                    audio.assistant_done()
                    # Storing memory takes seconds, so run it in a thread; blocking here would freeze the whole session.
                    asyncio.create_task(asyncio.to_thread(
                        vm.ingest, turn["text"], agent_reply=turn["reply"], async_facts=True))
                    turn["reply"] = ""
                elif t == "error":
                    if (ev.get("error") or {}).get("code") not in _EXPECTED_ERRORS:
                        print(f"\n[{name}] {ev.get('error')}", flush=True)

        tasks = [asyncio.create_task(f()) for f in (pump, uplink, split)]

        audio.start()
        print(f"[ready] {name}, start talking", flush=True)

        try:
            while True:
                st = await stream.feed(await to_vm.get())
                if st.state != "turn_over":
                    continue

                turn.update(text=st.transcript, reply="", live=True)
                print(f"\nYou: {st.transcript}\nAssistant: ", end="", flush=True)

                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                # Memory goes through per-response instructions, not session.update -- the latter is a
                # session-level setting the model does not read this turn (memory was retrieved, yet it said it didn't catch that).
                await ws.send(json.dumps({"type": "response.create", "response": {
                    "instructions": f"{PERSONA}\n\n{st.memory_context}"}}))
                audio.assistant_started()
        finally:
            for t in tasks:
                t.cancel()
            audio.close()


ap = argparse.ArgumentParser()
ap.add_argument("provider", choices=["gpt", "qwen"], nargs="?", default="gpt")
args = ap.parse_args()

asyncio.run(main({"gpt": GPT, "qwen": QWEN}[args.provider], args.provider))
