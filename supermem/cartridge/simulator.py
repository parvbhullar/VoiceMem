"""A fake OpenAI-compatible engine for rehearsing without a GPU.

NOT A MEASUREMENT. It exists so the benchmark, the demo page and the tests
can run on a laptop. It fakes an engine's behaviour:

* prefix cache over 16-token blocks (like vLLM APC), optional;
* latency = fixed overhead + per-recomputed-token cost, so cached prefixes
  look fast for the same reason they are fast on a GPU;
* the "answer" is the context line with the most word overlap with the
  question, which is enough to show memory-on vs memory-off.

Every streamed chunk carries ``"simulated": true`` and ``/version`` answers
``simulator``; the benchmark report and the demo page both print a SIMULATED
banner whenever they see either.

    python -m supermem.cartridge.simulator --port 8002            # KV reuse on
    python -m supermem.cartridge.simulator --port 8001 --no-cache # full prefill
"""

import argparse
import asyncio
import hashlib
import json
import re
from collections import OrderedDict

BLOCK = 16


def _tokens(text: str) -> list[str]:
    # ~4 chars per token, deterministic; good enough to make prompt sizes realistic.
    return [text[i:i + 4] for i in range(0, len(text), 4)]


def _render(messages: list[dict]) -> str:
    return "".join(f"<|{m['role']}|>{m.get('content') or ''}<|end|>" for m in messages)


class PrefixCache:
    def __init__(self, capacity_blocks: int = 200_000) -> None:
        self.blocks: OrderedDict[str, None] = OrderedDict()
        self.capacity = capacity_blocks

    def lookup_and_insert(self, toks: list[str], salt: str = "") -> int:
        """Returns how many leading tokens were cached, then caches the rest."""
        h = hashlib.sha256(salt.encode())
        cached, hit = 0, True
        for i in range(0, len(toks) - len(toks) % BLOCK, BLOCK):
            h.update("".join(toks[i:i + BLOCK]).encode())
            key = h.hexdigest()
            if hit and key in self.blocks:
                cached += BLOCK
                self.blocks.move_to_end(key)
            else:
                hit = False
                self.blocks[key] = None
                if len(self.blocks) > self.capacity:
                    self.blocks.popitem(last=False)
        return cached


_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"what", "is", "my", "the", "i", "am", "on", "do", "to", "for", "in", "of", "a", "me",
         "kab", "hai", "mera", "mujhe", "kis", "se", "ko", "which", "should", "go", "can",
         "till", "how", "many", "before", "were", "usually", "see"}


def _answer(messages: list[dict]) -> str:
    question = messages[-1]["content"].split("(Caller says)\n")[-1]
    q = {w for w in _WORD.findall(question.lower()) if w not in _STOP}
    system = "\n".join(m["content"] for m in messages[:-1] if m["role"] == "system")
    # Only the cartridges count as knowledge; the persona text above them is not an answer.
    context = system[system.find("### "):] if "### " in system else ""
    best, best_score = "", 0
    for line in context.splitlines():
        s = len(q & set(_WORD.findall(line.lower())))
        if s > best_score:
            best, best_score = line.strip("- ").strip(), s
    if not best or "###" in best:
        return "Let me check that for you and call you back."
    return f"{best} Anything else I can help with?"


def make_app(cache: bool, base_ms: float, per_token_ms: float, decode_ms: float):
    from fastapi import FastAPI, Request
    from fastapi.responses import PlainTextResponse, StreamingResponse

    app = FastAPI()
    pc = PrefixCache()
    stats = {"prefill_s": 0.0, "count": 0}

    @app.get("/version")
    def version():
        return {"version": "simulator"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return (f"vllm:request_prefill_time_seconds_sum{{model_name=\"sim\"}} {stats['prefill_s']}\n"
                f"vllm:request_prefill_time_seconds_count{{model_name=\"sim\"}} {stats['count']}\n")

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        msgs = body["messages"]
        toks = _tokens(_render(msgs))
        cached = pc.lookup_and_insert(toks, body.get("cache_salt") or "") if cache else 0
        recomputed = len(toks) - cached
        prefill_s = (base_ms + per_token_ms * recomputed) / 1000
        text = _answer(msgs) if body.get("max_tokens", 96) > 1 else "."
        words = text.split(" ")[: max(1, body.get("max_tokens", 96))]

        async def gen():
            await asyncio.sleep(prefill_s)
            stats["prefill_s"] += prefill_s
            stats["count"] += 1
            for i, w in enumerate(words):
                chunk = {"choices": [{"index": 0, "delta": {"content": w if i == 0 else " " + w}}],
                         "simulated": True}
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(decode_ms / 1000)
            usage = {"prompt_tokens": len(toks), "completion_tokens": len(words),
                     "prompt_tokens_details": {"cached_tokens": cached}}
            yield f"data: {json.dumps({'choices': [], 'usage': usage, 'simulated': True})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--no-cache", action="store_true", help="behave like prefix caching OFF")
    p.add_argument("--base-ms", type=float, default=35.0)
    p.add_argument("--per-token-ms", type=float, default=0.05)
    p.add_argument("--decode-ms", type=float, default=12.0)
    a = p.parse_args(argv)
    import uvicorn
    print(f"SIMULATED engine on :{a.port} (prefix cache {'OFF' if a.no_cache else 'ON'}) -- "
          f"not a measurement", flush=True)
    uvicorn.run(make_app(not a.no_cache, a.base_ms, a.per_token_ms, a.decode_ms),
                host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
