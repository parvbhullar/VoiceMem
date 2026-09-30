"""/memories background loader: the one-thread Jobs runner, driven with a fake SuperMem.

Every gate is released in tearDown and every wait has a deadline, so no test can hang::

    python tests/test_ingest_jobs.py
"""
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))

import ingest  # noqa: E402
from ingest import Chunk, Jobs  # noqa: E402

TERMINAL = ("done", "failed", "cancelled")


class FakeMem:
    def __init__(self, fail=(), soft_fail=(), gate=None, log=None):
        self.calls, self.fail, self.soft_fail, self.gate = [], set(fail), set(soft_fail), gate
        self.entered = threading.Event()        # set once ingest() has been called
        self.log = log if log is not None else []

    def ingest(self, text, speaker="user", agent_reply=None, observed_at=None, **kw):
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(5)
        self.calls.append((text, speaker, agent_reply, observed_at, kw))
        self.log.append(("ingest", text))
        if text in self.fail:
            raise RuntimeError("llm down")
        if text in self.soft_fail:
            return {"facts_count": 0, "memory_ids": [], "error": "no key"}
        return {"facts_count": 2, "memory_ids": ["m"]}


def _chunks(*texts: str) -> list[Chunk]:
    return [Chunk(t) for t in texts]


def _wait(jobs: Jobs, job_id: str, timeout: float = 5, states=TERMINAL) -> dict:
    """Poll get() until the job reaches one of ``states``; fail the test at the deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = jobs.get(job_id)
        if rec is not None and rec["state"] in states:
            return rec
        time.sleep(0.005)
    raise AssertionError(f"job {job_id} never reached {states}: {jobs.get(job_id)}")


class JobsTest(unittest.TestCase):
    def setUp(self):
        self.gates: list[threading.Event] = []

    def tearDown(self):
        for g in self.gates:
            g.set()

    def gate(self) -> threading.Event:
        g = threading.Event()
        self.gates.append(g)
        return g

    def test_counts(self):
        jobs = Jobs()
        jid = jobs.submit(FakeMem(), "a", _chunks("x", "y", "z"), filename="n.txt", kind="prose")
        rec = _wait(jobs, jid)
        self.assertEqual((rec["state"], rec["done"], rec["total"], rec["facts"]), ("done", 3, 3, 6))
        self.assertEqual((rec["space"], rec["filename"], rec["format"]), ("a", "n.txt", "prose"))
        self.assertEqual((rec["errors"], rec["error_count"]), ([], 0))
        self.assertLessEqual(rec["created_at"], rec["started_at"])
        self.assertLessEqual(rec["started_at"], rec["finished_at"])

    def test_ingest_kwargs_and_no_session_id(self):
        jobs, mem = Jobs(), FakeMem()
        jid = jobs.submit(mem, "a", [Chunk("hi", "Bob", "yo"), Chunk("ok")], "2026-01-02")
        _wait(jobs, jid)
        self.assertEqual(mem.calls, [("hi", "Bob", "yo", "2026-01-02", {}),
                                     ("ok", "user", None, "2026-01-02", {})])

    def test_hard_and_soft_failures_recorded_and_skipped(self):
        jobs = Jobs()
        mem = FakeMem(fail={"bad"}, soft_fail={"soft"})
        rec = _wait(jobs, jobs.submit(mem, "a", _chunks("x", "bad", "soft", "y")))
        self.assertEqual((rec["state"], rec["done"], rec["facts"], rec["error_count"]), ("done", 4, 4, 2))
        self.assertEqual([(e["index"], e["text"]) for e in rec["errors"]], [(1, "bad"), (2, "soft")])
        self.assertIn("llm down", rec["errors"][0]["error"])
        self.assertEqual(rec["errors"][1]["error"], "no key")

    def test_fail_fast_after_five_failures(self):
        texts = [f"t{i}" for i in range(7)]
        jobs, mem = Jobs(), FakeMem(fail=texts)
        rec = _wait(jobs, jobs.submit(mem, "a", _chunks(*texts)))
        self.assertEqual((rec["state"], rec["done"], rec["total"]), ("failed", ingest.FAIL_FAST, 7))
        self.assertEqual(len(mem.calls), ingest.FAIL_FAST)

    def test_all_chunks_failing_is_failed(self):
        jobs = Jobs()
        rec = _wait(jobs, jobs.submit(FakeMem(soft_fail={"x", "y", "z"}), "a", _chunks("x", "y", "z")))
        self.assertEqual((rec["state"], rec["done"], rec["error_count"]), ("failed", 3, 3))

    def test_one_job_at_a_time_in_submit_order(self):
        log: list = []
        gate = self.gate()
        first, second = FakeMem(gate=gate, log=log), FakeMem(log=log)
        jobs = Jobs()
        j1 = jobs.submit(first, "a", _chunks("one"))
        j2 = jobs.submit(second, "a", _chunks("two"))
        self.assertTrue(first.entered.wait(5))
        self.assertEqual(jobs.get(j1)["state"], "running")
        self.assertEqual(jobs.get(j2)["state"], "queued")
        gate.set()
        _wait(jobs, j2)
        self.assertEqual(log, [("ingest", "one"), ("ingest", "two")])
        self.assertLessEqual(jobs.get(j1)["finished_at"], jobs.get(j2)["started_at"])

    def test_busy(self):
        gate = self.gate()
        jobs = Jobs()
        j1 = jobs.submit(FakeMem(gate=gate), "a", _chunks("one"))
        j2 = jobs.submit(FakeMem(), "a", _chunks("two"))
        self.assertTrue(jobs.busy("a"))
        self.assertFalse(jobs.busy("b"))
        gate.set()
        _wait(jobs, j1)
        _wait(jobs, j2)
        self.assertFalse(jobs.busy("a"))

    def test_cancel_running_stops_before_next_chunk(self):
        gate = self.gate()
        mem = FakeMem(gate=gate)
        jobs = Jobs()
        jid = jobs.submit(mem, "a", _chunks("x", "y", "z"))
        self.assertTrue(mem.entered.wait(5))
        self.assertTrue(jobs.cancel(jid))
        gate.set()
        rec = _wait(jobs, jid)
        self.assertEqual((rec["state"], rec["done"], rec["total"]), ("cancelled", 1, 3))
        self.assertEqual(len(mem.calls), 1)

    def test_cancel_queued(self):
        gate = self.gate()
        first, second = FakeMem(gate=gate), FakeMem()
        jobs = Jobs()
        j1 = jobs.submit(first, "a", _chunks("one"))
        j2 = jobs.submit(second, "a", _chunks("two"))
        self.assertTrue(jobs.cancel(j2))
        gate.set()
        rec = _wait(jobs, j2)
        self.assertEqual((rec["state"], rec["done"]), ("cancelled", 0))
        self.assertEqual(second.calls, [])
        self.assertEqual(_wait(jobs, j1)["state"], "done")

    def test_cancel_queued_frees_the_brain_at_once(self):
        gate = self.gate()
        first, second = FakeMem(gate=gate), FakeMem()
        jobs = Jobs()
        j1 = jobs.submit(first, "a", _chunks("one"))
        j2 = jobs.submit(second, "b", _chunks("two"))
        self.assertTrue(first.entered.wait(5))
        self.assertTrue(jobs.cancel(j2))
        rec = jobs.get(j2)
        self.assertEqual(rec["state"], "cancelled")          # not "queued" until j1 ends
        self.assertIsNotNone(rec["finished_at"])
        self.assertFalse(jobs.busy("b"))
        gate.set()
        _wait(jobs, j1)
        j3 = jobs.submit(FakeMem(), "b", _chunks("three"))    # the worker skipped j2
        _wait(jobs, j3)
        self.assertEqual((jobs.get(j2)["state"], second.calls), ("cancelled", []))
        self.assertEqual(jobs._cancelled, set())

    def test_late_cancel_does_not_leak(self):
        gate = self.gate()
        mem = FakeMem(gate=gate)
        jobs = Jobs(finish=lambda m: jobs.cancel(jid))       # cancel after the last chunk
        jid = jobs.submit(mem, "a", _chunks("x"))
        gate.set()
        self.assertEqual(_wait(jobs, jid)["state"], "done")
        self.assertEqual(jobs._cancelled, set())

    def test_cancel_unknown(self):
        self.assertFalse(Jobs().cancel("nope"))

    def test_prepare_and_finish_wrap_the_job(self):
        log: list = []
        jobs = Jobs(prepare=lambda mem: log.append(("prepare", mem)),
                    finish=lambda mem: log.append(("finish", mem)))
        mem = FakeMem(log=log)
        _wait(jobs, jobs.submit(mem, "a", _chunks("x", "y")))
        self.assertEqual(log, [("prepare", mem), ("ingest", "x"), ("ingest", "y"), ("finish", mem)])

    def test_finish_runs_after_failure_and_cancel(self):
        finished: list = []
        jobs = Jobs(finish=finished.append)
        failing = FakeMem(fail={"x"})
        self.assertEqual(_wait(jobs, jobs.submit(failing, "a", _chunks("x")))["state"], "failed")
        gate = self.gate()
        gated = FakeMem(gate=gate)
        jid = jobs.submit(gated, "a", _chunks("x", "y"))
        self.assertTrue(gated.entered.wait(5))
        jobs.cancel(jid)
        gate.set()
        self.assertEqual(_wait(jobs, jid)["state"], "cancelled")
        self.assertEqual(finished, [failing, gated])

    def test_on_done_runs_once(self):
        calls: list = []
        jobs = Jobs()
        _wait(jobs, jobs.submit(FakeMem(), "a", _chunks("x", "y"), on_done=lambda: calls.append(1)))
        self.assertEqual(calls, [1])

    def test_raising_prepare_fails_job_but_worker_survives(self):
        seen: list = []

        def prepare(mem):
            seen.append(mem)
            if len(seen) == 1:
                raise RuntimeError("lock busy")

        jobs = Jobs(prepare=prepare)
        first, second = FakeMem(), FakeMem()
        j1 = jobs.submit(first, "a", _chunks("x"))
        j2 = jobs.submit(second, "a", _chunks("y"))
        rec = _wait(jobs, j1)
        self.assertEqual((rec["state"], rec["done"], rec["error_count"]), ("failed", 0, 0))
        self.assertEqual(rec["errors"][0]["index"], -1)          # hook errors don't count as chunks
        self.assertIn("lock busy", rec["errors"][0]["error"])
        self.assertEqual(first.calls, [])
        self.assertEqual(_wait(jobs, j2)["state"], "done")

    def test_list_newest_first_and_filtered(self):
        jobs = Jobs()
        ids = [jobs.submit(FakeMem(), space, _chunks("x")) for space in ("a", "b", "a", "a")]
        for jid in ids:
            _wait(jobs, jid)
        self.assertEqual([r["id"] for r in jobs.list("a")], [ids[3], ids[2], ids[0]])
        self.assertEqual([r["id"] for r in jobs.list("b")], [ids[1]])
        self.assertEqual(jobs.list("c"), [])

    def test_get_returns_copy(self):
        jobs = Jobs()
        jid = jobs.submit(FakeMem(soft_fail={"x"}), "a", _chunks("x", "y"))
        rec = _wait(jobs, jid)
        rec["state"] = "hacked"
        rec["errors"][0]["error"] = "hacked"
        rec["errors"].append({})
        again = jobs.get(jid)
        self.assertEqual(again["state"], "done")
        self.assertEqual(again["errors"], [{"index": 0, "text": "x", "error": "no key"}])
        self.assertIsNone(jobs.get("nope"))

    def test_old_finished_jobs_trimmed(self):
        old = ingest.KEEP_FINISHED
        ingest.KEEP_FINISHED = 2
        try:
            jobs = Jobs()
            ids = [_wait(jobs, jobs.submit(FakeMem(), "a", _chunks("x")))["id"] for _ in range(4)]
            ids.append(jobs.submit(FakeMem(), "a", _chunks("x")))
            _wait(jobs, ids[-1])
            self.assertEqual([r["id"] for r in jobs.list("a")], [ids[4], ids[3], ids[2]])
        finally:
            ingest.KEEP_FINISHED = old

    def test_errors_capped(self):
        texts = [f"t{i}" for i in range(60)]
        jobs = Jobs()
        rec = _wait(jobs, jobs.submit(FakeMem(fail=texts[1:]), "a", _chunks(*texts)))
        self.assertEqual((rec["state"], rec["done"], rec["error_count"]), ("done", 60, 59))
        self.assertEqual(len(rec["errors"]), ingest.MAX_ERRORS_KEPT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
