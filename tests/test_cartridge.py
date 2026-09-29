"""KV cartridges: contract, compiler, runtime layout, report, and the measuring
client end to end against the in-process simulator.

No GPU, no network, no model download (the token counter is forced into
estimate mode). Run directly::

    python tests/test_cartridge.py
"""
import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from evaluation.cartridges.dataset import build, score, turn_order  # noqa: E402
from supermem.cartridge.compiler import ContextCompiler, Fact, TokenCounter  # noqa: E402
from supermem.cartridge.contract import Cartridge, ordered  # noqa: E402
from supermem.cartridge.engine import Engine, _metric_sum  # noqa: E402
from supermem.cartridge.report import pct, showcase_rows, summarize  # noqa: E402
from supermem.cartridge.runtime import BLEND_SEPARATOR, ContextRuntime  # noqa: E402
from supermem.cartridge.simulator import make_app  # noqa: E402


class _EstimateCounter(TokenCounter):
    def __init__(self):  # never touch transformers in tests
        self.model = self.tokenizer_id = "test-model"
        self._tok = None


def _compiler(tenant="t1"):
    return ContextCompiler("test-model", tenant, _EstimateCounter())


class ContractTest(unittest.TestCase):
    def test_id_is_stable_across_recompiles(self):
        a = _compiler().compile_org("org", [("A", "policy text")], version=1)
        b = _compiler().compile_org("org", [("A", "policy text")], version=1)
        self.assertEqual(a.id, b.id)

    def test_id_changes_with_anything_that_changes_kv(self):
        base = _compiler().compile_org("org", [("A", "policy text")], version=1)
        self.assertNotEqual(base.id, _compiler().compile_org("org", [("A", "policy text!")], version=1).id)
        self.assertNotEqual(base.id, _compiler().compile_org("org", [("A", "policy text")], version=2).id)
        self.assertNotEqual(base.id, _compiler("t2").compile_org("org", [("A", "policy text")], version=1).id)

    def test_ordered_rejects_mixed_tenants(self):
        a = _compiler("t1").compile_org("org", [("A", "x")])
        b = _compiler("t2").compile_user("u", [Fact("s", "y")])
        with self.assertRaises(ValueError):
            ordered([a, b])

    def test_ordered_puts_shared_first(self):
        c = _compiler()
        rel = c.compile_rel("org", "u", {"k": "v"})
        user = c.compile_user("u", [Fact("s", "fact")])
        org = c.compile_org("org", [("A", "x")])
        self.assertEqual([x.kind for x in ordered([rel, user, org])], ["org", "user", "rel"])

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            Cartridge("turn", "t", {}, "b", 1, "m", "m", 1)


class CompilerTest(unittest.TestCase):
    def test_user_cartridge_is_canonical(self):
        c = _compiler()
        facts = [Fact("health", "Takes metformin.", "2026-04-01"),
                 Fact("identity", "Lives in Whitefield.", "2026-03-01"),
                 Fact("health", "takes metformin", "2026-04-02")]   # duplicate, different case/punct
        a = c.compile_user("u", facts, ["b trait", "a trait"], version=1)
        b = c.compile_user("u", list(reversed(facts)), ["a trait", "b trait"], version=1)
        self.assertEqual(a.body, b.body)
        self.assertEqual(a.body.lower().count("metformin"), 1)
        self.assertLess(a.body.index("[health]"), a.body.index("[identity]"))


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        c = _compiler()
        self.rt = ContextRuntime(org=c.compile_org("org", [("A", "policy")]), epoch="e1")
        self.rt.register("u", [c.compile_user("u", [Fact("s", "fact one")]),
                               c.compile_rel("org", "u", {"patient id": "SUN-1"})])

    def test_prefetch_shares_the_whole_cartridge_prefix(self):
        real = self.rt.messages("u", "hello", turn_memory="hit")
        warm = self.rt.prefetch_messages("u")
        self.assertEqual(real[0], warm[0])

    def test_volatile_memory_stays_out_of_the_prefix(self):
        a = self.rt.messages("u", "q1", turn_memory="hit one")
        b = self.rt.messages("u", "q2", turn_memory="hit two")
        self.assertEqual(a[0], b[0])
        self.assertIn("hit one", a[-1]["content"])

    def test_nomem_has_no_cartridges(self):
        sys_msg = self.rt.messages("u", "q", mode="nomem")[0]["content"]
        self.assertNotIn("###", sys_msg)

    def test_blend_separates_cartridges(self):
        sys_msg = self.rt.messages("u", "q", mode="blend")[0]["content"]
        self.assertEqual(sys_msg.count(BLEND_SEPARATOR), 2)

    def test_epoch_changes_the_first_tokens(self):
        other = ContextRuntime(org=self.rt.org, epoch="e2")
        other.register("u", self.rt.cartridges("u"))
        self.assertNotEqual(other.messages("u", "q")[0]["content"][:20],
                            self.rt.messages("u", "q")[0]["content"][:20])


class ReportTest(unittest.TestCase):
    def test_nearest_rank_percentiles(self):
        vals = list(range(1, 101))
        self.assertEqual(pct(vals, 50), 50)
        self.assertEqual(pct(vals, 95), 95)
        self.assertIsNone(pct([], 50))

    def test_missing_data_is_never_a_number(self):
        full = summarize("full", [{"ttft_ms": 800, "ttfs_ms": 900, "prefill_gpu_ms": None,
                                   "prompt_tokens": 100, "cached_tokens": None,
                                   "recomputed_tokens": 100, "recomputed_frac": 1.0, "extra": {}}])
        cart = summarize("cartridge", [{"ttft_ms": 200, "ttfs_ms": 300, "prefill_gpu_ms": None,
                                        "prompt_tokens": 100, "cached_tokens": None,
                                        "recomputed_tokens": 100, "recomputed_frac": 1.0, "extra": {}}])
        rows = {r[0]: r for r in showcase_rows(full, cart)}
        self.assertIn("PASS", rows["TTFT p50"][4])
        self.assertEqual(rows["Context recomputed"][2], "not measured")   # no cached_tokens from engine
        self.assertEqual(rows["GPU prefill time/turn"][4], "not measured")
        self.assertEqual(rows["Answer accuracy"][4], "not measured")

    def test_errors_are_counted_not_averaged(self):
        s = summarize("x", [{"error": "boom", "extra": {}},
                            {"ttft_ms": 10, "ttfs_ms": 10, "prefill_gpu_ms": 1, "prompt_tokens": 5,
                             "cached_tokens": 0, "recomputed_tokens": 5, "recomputed_frac": 1.0,
                             "extra": {"correct": True}}])
        self.assertEqual((s.n, s.errors, s.accuracy), (1, 1, 1.0))

    def test_metric_sum_over_labels(self):
        text = ('vllm:request_prefill_time_seconds_sum{model="a"} 1.5\n'
                'vllm:request_prefill_time_seconds_sum{model="b"} 0.5\n'
                'vllm:request_prefill_time_seconds_count{model="a"} 3\n')
        self.assertEqual(_metric_sum(text, "vllm:request_prefill_time_seconds_sum"), 2.0)
        self.assertIsNone(_metric_sum(text, "missing"))


class DatasetTest(unittest.TestCase):
    def test_deterministic_and_answerable(self):
        a, b = build(3, 2000, 500, seed=1), build(3, 2000, 500, seed=1)
        self.assertEqual(a.org_sections, b.org_sections)
        self.assertEqual([q.expect for q in a.callers[0].questions],
                         [q.expect for q in b.callers[0].questions])
        c = a.callers[0]
        context = "\n".join(t for _, t in a.org_sections) + " ".join(f.content for f in c.facts) \
            + " ".join(c.account.values())
        for q in c.questions:
            self.assertTrue(score(context, q.expect), f"answer for {q.text!r} not in context")

    def test_round_robin_order(self):
        w = build(3, 1000, 200)
        order = turn_order(w, 5)
        self.assertEqual([c.user_id for c, _, _ in order][:3], [c.user_id for c in w.callers])


class EngineEndToEndTest(unittest.TestCase):
    """The measuring client against the simulator over an in-process transport."""

    def _engine(self, cache):
        app = make_app(cache=cache, base_ms=1, per_token_ms=0.0, decode_ms=0)
        return Engine("http://sim/v1", "m", arm="t", transport=httpx.ASGITransport(app=app))

    def test_second_turn_hits_the_cartridge_prefix(self):
        async def go():
            c = _compiler()
            rt = ContextRuntime(org=c.compile_org("org", [("Fees", "Cardiology fee Rs 1200. " * 60)]))
            rt.register("u", [c.compile_user("u", [Fact("s", "Allergic to penicillin.")])])
            e = self._engine(cache=True)
            warm = await e.warm(rt.prefetch_messages("u"))
            turn = await e.complete(rt.messages("u", "What is the Cardiology fee?"))
            await e.close()
            return warm, turn
        warm, turn = asyncio.run(go())
        self.assertIsNone(turn.error)
        self.assertEqual(warm.cached_tokens, 0)
        self.assertGreater(turn.cached_tokens, 0.8 * turn.prompt_tokens)
        self.assertIsNotNone(turn.ttft_ms)
        self.assertIsNotNone(turn.prefill_gpu_ms)
        self.assertTrue(turn.extra.get("simulated"))

    def test_cache_salt_forces_full_prefill(self):
        async def go():
            e = self._engine(cache=True)
            msgs = [{"role": "system", "content": "x" * 800}, {"role": "user", "content": "q"}]
            await e.complete(msgs, cache_salt="a")
            r = await e.complete(msgs, cache_salt="b")
            await e.close()
            return r
        self.assertEqual(asyncio.run(go()).cached_tokens, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
