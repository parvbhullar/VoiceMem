"""Measuring client for an OpenAI-compatible engine (vLLM, vLLM + LMCache,
NVIDIA Dynamo frontend).

Every number the benchmark reports comes from here, so it measures only what
it can observe and leaves the rest as ``None`` rather than guessing:

* TTFT / time-to-first-sentence: wall clock on the streamed response.
* prompt / cached tokens: the engine's own ``usage`` (vLLM needs
  ``--enable-prompt-tokens-details`` for ``cached_tokens``).
* prefill GPU time: delta of vLLM's ``vllm:request_prefill_time_seconds``
  histogram sum around the request. Only exact when requests run one at a
  time, which is how the benchmark runs them -- so the histogram count must
  have moved by exactly one; any other delta means another request overlapped
  and the value is left ``None`` (``extra["metrics_overlap"]``).
* cached tokens without ``--enable-prompt-tokens-details``: delta of
  ``vllm:prefix_cache_hits_total`` over the same single-request window
  (``extra["cached_source"] == "engine_counters"``).
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import AsyncIterator

import httpx

_SENTENCE_END = re.compile(r"[.!?]\s|[.!?]$|\n")


@dataclass
class TurnResult:
    arm: str
    text: str = ""
    ttft_ms: float | None = None          # request sent -> first content token
    ttfs_ms: float | None = None          # request sent -> first full sentence (TTS can start)
    total_ms: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None      # served from KV cache (GPU prefix cache or LMCache)
    completion_tokens: int | None = None
    prefill_gpu_ms: float | None = None   # engine-reported prefill time for this request
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def recomputed_tokens(self) -> int | None:
        if self.prompt_tokens is None:
            return None
        return self.prompt_tokens - (self.cached_tokens or 0)

    @property
    def recomputed_frac(self) -> float | None:
        if not self.prompt_tokens:
            return None
        return self.recomputed_tokens / self.prompt_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(recomputed_tokens=self.recomputed_tokens, recomputed_frac=self.recomputed_frac)
        return d


def _metric_sum(text: str, name: str) -> float | None:
    """Sum every sample of one Prometheus metric (all label sets)."""
    total, found = 0.0, False
    for line in text.splitlines():
        if line.startswith(name) and line[len(name):len(name) + 1] in (" ", "{"):
            try:
                total += float(line.rsplit(" ", 1)[1])
                found = True
            except ValueError:
                pass
    return total if found else None


class Engine:
    def __init__(self, base_url: str, model: str, arm: str, api_key: str = "EMPTY",
                 timeout: float = 120.0, scrape_metrics: bool = True,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.arm = arm
        self.api_key = api_key
        self.timeout = timeout
        self.scrape_metrics = scrape_metrics
        #: Take the "before" snapshot alongside the request instead of ahead of it, so it
        #: adds no round trip to the TTFT a live viewer sees. Still exact: prefill counters
        #: only move when a prefill finishes, and the count delta must be exactly one.
        self.snapshot_in_flight = False
        self._transport = transport          # tests pass an ASGI transport here
        root = re.sub(r"/v1$", "", self.base_url)
        self.metrics_url = f"{root}/metrics"
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout, headers={"Authorization": f"Bearer {self.api_key}"},
                transport=self._transport)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _counters(self) -> dict | None:
        """One /metrics snapshot of the counters a turn is measured from."""
        if not self.scrape_metrics:
            return None
        try:
            r = await self.client.get(self.metrics_url, timeout=5.0)
            if r.status_code != 200:
                return None
        except httpx.HTTPError:
            return None
        snap = {k: _metric_sum(r.text, name) for k, name in (
            ("prefill_s", "vllm:request_prefill_time_seconds_sum"),
            ("prefill_n", "vllm:request_prefill_time_seconds_count"),
            ("cache_hits", "vllm:prefix_cache_hits_total"))}
        return snap if snap["prefill_s"] is not None else None

    async def version(self) -> str | None:
        try:
            r = await self.client.get(re.sub(r"/v1$", "", self.base_url) + "/version", timeout=5.0)
            return r.json().get("version") if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    def _payload(self, messages: list[dict], max_tokens: int, cache_salt: str | None) -> dict:
        body = {
            "model": self.model, "messages": messages, "stream": True,
            "max_tokens": max_tokens, "temperature": 0,
            "stream_options": {"include_usage": True},
        }
        if cache_salt:
            # vLLM: requests with different salts never share prefix-cache blocks.
            body["cache_salt"] = cache_salt
        return body

    async def stream(self, messages: list[dict], max_tokens: int = 96,
                     cache_salt: str | None = None) -> AsyncIterator[tuple[str, object]]:
        """Yields ("delta", str) while generating, then exactly one ("done", TurnResult)."""
        res = TurnResult(arm=self.arm)
        before_task = asyncio.ensure_future(self._counters()) if self.snapshot_in_flight else None
        before = None if before_task else await self._counters()
        t0 = time.perf_counter()
        try:
            async with self.client.stream(
                    "POST", f"{self.base_url}/chat/completions",
                    json=self._payload(messages, max_tokens, cache_salt)) as r:
                if r.status_code != 200:
                    body = (await r.aread()).decode("utf-8", "replace")[:500]
                    raise RuntimeError(f"HTTP {r.status_code}: {body}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("simulated"):
                        res.extra["simulated"] = True
                    usage = chunk.get("usage")
                    if usage:
                        res.prompt_tokens = usage.get("prompt_tokens")
                        res.completion_tokens = usage.get("completion_tokens")
                        details = usage.get("prompt_tokens_details") or {}
                        res.cached_tokens = details.get("cached_tokens")
                    for choice in chunk.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if not delta:
                            continue
                        now = (time.perf_counter() - t0) * 1000
                        if res.ttft_ms is None:
                            res.ttft_ms = now
                        res.text += delta
                        if res.ttfs_ms is None and _SENTENCE_END.search(res.text):
                            res.ttfs_ms = now
                        yield "delta", delta
        except Exception as e:  # noqa: BLE001 -- recorded on the result, the run continues
            res.error = f"{type(e).__name__}: {e}"
        res.total_ms = (time.perf_counter() - t0) * 1000
        if res.ttfs_ms is None and res.text:
            res.ttfs_ms = res.total_ms
        if before_task is not None:
            before = await before_task
        after = await self._counters()
        if before is not None and after is not None:
            self._apply_counters(res, before, after)
        yield "done", res

    @staticmethod
    def _apply_counters(res: TurnResult, before: dict, after: dict) -> None:
        n0, n1 = before["prefill_n"], after["prefill_n"]
        # Without the count we cannot tell whether the window held only this request:
        # keep the old behaviour (sum delta) for engines that expose just the sum.
        if n0 is not None and n1 is not None and n1 - n0 != 1:
            res.extra["metrics_overlap"] = True
            return
        res.prefill_gpu_ms = (after["prefill_s"] - before["prefill_s"]) * 1000
        h0, h1 = before["cache_hits"], after["cache_hits"]
        if res.cached_tokens is None and n0 is not None and h0 is not None and h1 is not None:
            res.cached_tokens = int(h1 - h0)
            res.extra["cached_source"] = "engine_counters"

    async def complete(self, messages: list[dict], max_tokens: int = 96,
                       cache_salt: str | None = None) -> TurnResult:
        result = None
        async for kind, val in self.stream(messages, max_tokens, cache_salt):
            if kind == "done":
                result = val
        return result

    async def warm(self, messages: list[dict], cache_salt: str | None = None) -> TurnResult:
        """Prefill-only request (one output token). Used for pre-ring prefetch:
        the KV for the cartridges is computed while the phone is still ringing."""
        return await self.complete(messages, max_tokens=1, cache_salt=cache_salt)
