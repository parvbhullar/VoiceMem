"""Startup self-check: time each component independently, print a simple report to the console, and gate startup on it.

Usage (see the integration in openai_voice_demo/backend/main.py)::

    from supermem.startup_check import check_and_gate
    if not check_and_gate(vm):          # prints the report; returns True (start right away) if everything is within budget
        raise SystemExit("User cancelled startup after the self-check")

Design points:
  · Each component is probed once independently (construct + run one representative operation), measuring single-call latency without interference.
  · Missing dependency / missing model / missing API key -> recorded as SKIP (excluded from speed gating), not FAIL.
  · Each component type has an empirical default speed budget (ms), overridable via the env var
    ``SUPERMEM_STARTUP_BUDGET_<KEY>`` (e.g.
    ``SUPERMEM_STARTUP_BUDGET_SPEAKER_ENCODER=2000``).
  · All within budget -> start right away; any component slow (SLOW) or erroring (FAIL) -> ask whether to start anyway.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from math import sin, tau
from pathlib import Path
from typing import Any, Callable

# ── Status ───────────────────────────────────────────────────────────────────
OK, SLOW, SKIP, FAIL = "ok", "slow", "skip", "fail"
_SYMBOL = {OK: "✓ ok", SLOW: "⚠ slow", SKIP: "– skip", FAIL: "✗ fail"}


class SkipProbe(Exception):
    """The probe skipped itself (missing dependency/model/credentials; not a failure)."""


@dataclass
class ProbeResult:
    name: str
    status: str
    budget_ms: float
    elapsed_ms: float | None = None
    detail: str = ""


@dataclass
class StartupReport:
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def slow(self) -> list[ProbeResult]:
        return [r for r in self.results if r.status == SLOW]

    @property
    def failed(self) -> list[ProbeResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def needs_confirm(self) -> bool:
        """When any component is slow or failing, the user must confirm whether to start anyway."""
        return bool(self.slow or self.failed)

    def render(self) -> str:
        w = max((len(r.name) for r in self.results), default=4)
        lines = ["", "── SuperMem startup self-check ─────────────────────────────────",
                 f"  {'Component'.ljust(w)}   {'Status':<6}  {'Time':>9}  {'Budget':>9}"]
        for r in self.results:
            used = f"{r.elapsed_ms:.0f} ms" if r.elapsed_ms is not None else "—"
            budget = f"{r.budget_ms:.0f} ms"
            line = f"  {r.name.ljust(w)}   {_SYMBOL[r.status]:<6}  {used:>9}  {budget:>9}"
            if r.detail:
                line += f"   {r.detail}"
            lines.append(line)
        n_ok = sum(r.status == OK for r in self.results)
        n_slow, n_skip, n_fail = len(self.slow), sum(r.status == SKIP for r in self.results), len(self.failed)
        lines.append("  " + "─" * 54)
        lines.append(f"  Total {len(self.results)}: ok {n_ok} / slow {n_slow} / skipped {n_skip} / failed {n_fail}")
        lines.append("─────────────────────────────────────────────────────────")
        return "\n".join(lines)


# ── Default speed budgets (ms) ─────────────────────────────────────────────────────────
# One empirical value per component type; first inference of audio models includes loading, so budgets are generous. Overridable via env vars.
_DEFAULT_BUDGETS: dict[str, float] = {
    "preprocess_text":  400,    # text preprocessing (emotion fallback + voiceprint registry)
    "scene_classifier": 60,     # pure-python scene classification
    "emotion_vad":      500,    # prosodic V/A (RMS/ZCR, no LLM)
    "environment_ast":  8000,   # AST acoustic scene (first call includes model load/download)
    "speaker_encoder":  4000,   # 3D-Speaker voiceprint (worker subprocess + onnx)
    "dual_search":      300,    # dual-brain search steady-state round trip (~tens to 300ms with local embedding)
}


def _budget(key: str) -> float:
    env = os.environ.get(f"SUPERMEM_STARTUP_BUDGET_{key.upper()}")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _DEFAULT_BUDGETS.get(key, 1000)


# ── Synthesize a test audio clip (stdlib; only used to feed audio components one input) ────────────────────
def _synth_wav(path: Path, seconds: float = 0.5, rate: int = 16000, freq: float = 220.0) -> None:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            frames += struct.pack("<h", int(0.3 * 32767 * sin(tau * freq * i / rate)))
        w.writeframes(bytes(frames))


# ── Probe context ───────────────────────────────────────────────────────────────
@dataclass
class _Ctx:
    vm: Any
    wav_path: Path
    _vm_built: bool = False

    def get_vm(self):
        if self.vm is None and not self._vm_built:
            self._vm_built = True
            try:
                from supermem.core import SuperMem
                self.vm = SuperMem()
            except Exception as e:
                raise SkipProbe(f"SuperMem construction failed: {e}")
        if self.vm is None:
            raise SkipProbe("No SuperMem instance available")
        return self.vm


# ── Component probes (return detail or (detail, steady-state ms); raise SkipProbe to skip) ───────
# Convention: warm up once first (absorbing one-off costs like model loading / subprocess start / lazy init), then measure a second call
# and return that steady-state time along with detail -- so the report shows the "normal state", not a cold start.
def _measure(op):
    """op is a zero-arg callable: warm up once (untimed), then measure once; returns (result, steady-state ms)."""
    op()
    t0 = time.perf_counter(); res = op(); return res, (time.perf_counter() - t0) * 1000


def _probe_preprocess_text(ctx: _Ctx):
    vm = ctx.get_vm()
    p, ms = _measure(lambda: vm.preprocess("I'm a bit anxious today, work stress is really high"))
    return (f"emotion={p.emotion or '-'}", ms)


def _probe_scene_classifier(ctx: _Ctx):
    from supermem.utils.audio.environment.scene_classifier import classify_scene
    pairs = [("Speech", 0.91), ("Music", 0.42), ("Vehicle", 0.30)]
    r, ms = _measure(lambda: classify_scene(pairs))
    return (f"scene={r.tag.value}", ms)


def _probe_emotion_vad(ctx: _Ctx):
    try:
        from supermem.utils.audio.emotion.vad_audio import HeuristicWavVADEstimator
    except Exception as e:
        raise SkipProbe(f"Missing dependency: {e}")
    est = HeuristicWavVADEstimator()
    vad, ms = _measure(lambda: est.estimate(str(ctx.wav_path)))
    return (f"V={vad.valence:+.2f} A={vad.arousal:+.2f}", ms)


def _probe_environment_ast(ctx: _Ctx):
    try:
        from supermem.utils.audio.environment.environment_detector_ast import ASTEnvironmentDetector
    except Exception as e:
        raise SkipProbe(f"Missing dependency (transformers/torch): {e}")
    try:
        det = ASTEnvironmentDetector()
        _, ms = _measure(lambda: det.detect_full(ctx.wav_path))   # warm-up = first model load
    except (ImportError, OSError, FileNotFoundError) as e:
        raise SkipProbe(f"Model unavailable: {e}")
    return ("AST inference succeeded", ms)


def _probe_speaker_encoder(ctx: _Ctx):
    try:
        from supermem.utils.audio.voiceprint.speaker_encoder import SpeakerEncoder
    except Exception as e:
        raise SkipProbe(f"Missing dependency (sherpa-onnx): {e}")
    try:
        enc = SpeakerEncoder(device="cpu")
        vec, ms = _measure(lambda: enc.embed(ctx.wav_path))       # warm-up = start the worker subprocess
    except (ImportError, OSError, FileNotFoundError) as e:
        raise SkipProbe(f"Model unavailable: {e}")
    if vec is None:
        raise SkipProbe("worker returned no embedding (missing model/dependency)")
    return (f"dim={len(vec)}", ms)


def _probe_dual_search(ctx: _Ctx) -> str:
    # Search() recalls left brain (fact hits) + right brain (profile rb_hits) in one call -- dual-brain search.
    # Warm up once first (lazy loading of repo/graph/index all happens on the first call), then measure the second call, which is the
    # steady-state latency (should be within ~200ms with local embedding); otherwise we'd be measuring a cold start.
    vm = ctx.get_vm()
    try:
        vm.Search("test search")          # warm-up, untimed
    except Exception as e:
        raise SkipProbe(f"Search backend unavailable: {e}")
    t0 = time.perf_counter()
    r = vm.Search("test search")           # this timed call is the steady state
    ms = (time.perf_counter() - t0) * 1000
    detail = f"L{len(r.hits)}/R{len(getattr(r,'rb_hits',[]) or [])}"
    return (detail, ms)                 # self-reported steady-state time (excluding the warm-up above)


_PROBES: list[tuple[str, str, Callable[[_Ctx], str]]] = [
    # (display name, budget key, probe)
    ("Text preprocess",         "preprocess_text",  _probe_preprocess_text),
    ("Scene classifier",        "scene_classifier", _probe_scene_classifier),
    ("Prosodic emotion VAD",    "emotion_vad",      _probe_emotion_vad),
    ("Acoustic scene AST",      "environment_ast",  _probe_environment_ast),
    ("Speaker encoder",         "speaker_encoder",  _probe_speaker_encoder),
    ("Dual-brain search",       "dual_search",      _probe_dual_search),
]


def run_startup_check(vm: Any = None) -> StartupReport:
    """Time each component independently and return the report (no printing, no gating).

    vm: optional; reuses an already-built SuperMem (e.g. the demo's memory_bridge.vm). If omitted,
        a default instance is built lazily when needed; probes that depend on it are SKIP if construction fails.
    """
    report = StartupReport()
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "probe.wav"
        try:
            _synth_wav(wav)
        except Exception:
            wav = Path(td) / "missing.wav"   # audio probes will SKIP because of this
        ctx = _Ctx(vm=vm, wav_path=wav)
        for name, key, fn in _PROBES:
            budget = _budget(key)
            t0 = time.perf_counter()
            try:
                detail = fn(ctx)
                elapsed = (time.perf_counter() - t0) * 1000
                # A probe may return (detail, measured_ms) to self-report steady-state time (e.g. measured after warm-up);
                # in that case judge by the self-reported value, not the wall-clock time including warm-up.
                if isinstance(detail, tuple):
                    detail, elapsed = detail[0], float(detail[1])
                status = OK if elapsed <= budget else SLOW
                report.results.append(ProbeResult(name, status, budget, elapsed, detail))
            except SkipProbe as e:
                report.results.append(ProbeResult(name, SKIP, budget, None, str(e)))
            except Exception as e:  # noqa: BLE001 — unexpected exceptions are recorded as FAIL
                elapsed = (time.perf_counter() - t0) * 1000
                report.results.append(ProbeResult(name, FAIL, budget, elapsed, f"{type(e).__name__}: {e}"))
    return report


def check_and_gate(vm: Any = None, *, interactive: bool | None = None) -> bool:
    """Run the self-check, print the report, and decide whether to start based on the result.

    Returns True if startup may proceed. All within budget -> True right away; any slow/failing component -> ask the user
    (in a non-interactive terminal, proceed by default and print a warning, so unattended deployments don't hang).
    """
    report = run_startup_check(vm)
    print(report.render(), flush=True)

    if not report.needs_confirm:
        print("  ✓ All components meet their speed budgets; starting.\n", flush=True)
        return True

    slow_names = ", ".join(r.name for r in report.slow + report.failed)
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        print(f"  ⚠ Slow/failing components ({slow_names}); non-interactive terminal, continuing startup by default.\n", flush=True)
        return True

    try:
        answer = input(f"  ⚠ These components are slow/failing: {slow_names}. Start anyway? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Startup cancelled.", flush=True)
        return False
    go = answer in ("y", "yes")
    print(("  Continuing startup.\n" if go else "  Startup cancelled.\n"), flush=True)
    return go


# ── util self-check (SuperMem.test()): 4 tiers = unavailable / slow / normal / fast ─────────────
_UTIL_BUDGET = {   # ms, the "normal" upper bound for each util
    "embedding": 200, "slots": 400, "entity": 400, "emotion": 500,
    "voiceprint": 3000, "asr": 5000, "memory_engine": 300,
}
_TIER = {"fast": "⚡fast", "normal": "✓normal", "slow": "⚠slow", "unavailable": "✗unavailable"}


def _exercise(name, obj, wav):
    """Run one representative operation (utils without a cheap representative operation only count build time)."""
    if name == "embedding":       obj.embed_texts(["test"])
    elif name == "slots":         obj.classify("I'm a bit tired today")
    elif name == "emotion":       obj.detect(wav)
    elif name == "voiceprint":    obj.embed(wav)
    elif name == "memory_engine": obj.search("test", user_id="u")


def run_util_report(utils):
    """Load + run once each util this mode needs, measure latency, print a 4-tier table. Returns the list of results."""
    rows = []
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "probe.wav"
        try:
            _synth_wav(wav)
        except Exception:
            pass
        for name in utils.need:
            budget = _UTIL_BUDGET.get(name, 1000)
            t0 = time.perf_counter()
            try:
                _exercise(name, utils.get(name), wav)
                ms = (time.perf_counter() - t0) * 1000
                tier = "fast" if ms < budget * 0.25 else "normal" if ms <= budget else "slow"
                detail = ""
            except Exception as e:
                ms, tier, detail = None, "unavailable", f"{type(e).__name__}: {e}"
            rows.append({"util": name, "tier": tier, "ms": ms, "budget": budget, "detail": detail})

    print("\n── SuperMem utils self-check ────────────────────────────────")
    print(f"  {'util':<15}{'Status':<13}{'Latency':>9}{'Budget':>9}")
    for r in rows:
        used = f"{r['ms']:.0f} ms" if r["ms"] is not None else "—"
        line = f"  {r['util']:<15}{_TIER[r['tier']]:<13}{used:>9}{str(r['budget'])+'ms':>9}"
        if r["detail"]:
            line += "   " + r["detail"][:40]
        print(line)
    print("─────────────────────────────────────────────────────────")
    return rows


if __name__ == "__main__":  # python -m supermem.startup_check
    raise SystemExit(0 if check_and_gate() else 1)
