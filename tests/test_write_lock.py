"""Orchestrator._write_lock serialises the slow ingest write path per instance.

No models, no network: the orchestrator is built with __new__ and the real write is stubbed.
Run directly::

    python tests/test_write_lock.py
"""
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("SUPERMEM_MEMORYSPACE_ROOT", tempfile.mkdtemp(prefix="supermem-test-"))

from supermem.orchestrator import Orchestrator  # noqa: E402


def _bare() -> Orchestrator:
    o = Orchestrator.__new__(Orchestrator)
    o._write_lock = threading.RLock()
    return o


class WriteLockTest(unittest.TestCase):
    def test_finish_ingest_runs_one_at_a_time(self):
        o = _bare()
        state = {"now": 0, "max": 0}
        guard = threading.Lock()

        def fake_locked(ctx: dict) -> dict:
            with guard:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            with guard:
                state["now"] -= 1
            return {"facts_count": 0}

        o._finish_ingest_locked = fake_locked
        threads = [threading.Thread(target=o._finish_ingest, args=({},)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(state["max"], 1)

    def test_lock_is_reentrant(self):
        o = _bare()
        o._finish_ingest_locked = lambda ctx: {"facts_count": 0}
        out: list[dict] = []

        def run() -> None:
            with o._write_lock:
                out.append(o._finish_ingest({}))

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout=2)
        self.assertFalse(t.is_alive(), "nested _write_lock deadlocked")
        self.assertEqual(out, [{"facts_count": 0}])

    def test_retired_instance_refuses_queued_writes(self):
        o = _bare()
        o._finish_ingest_locked = lambda ctx: {"facts_count": 1}
        release, entered, out = threading.Event(), threading.Event(), []

        def holder() -> None:
            with o._write_lock:
                entered.set()
                release.wait(5)

        def queued() -> None:
            try:
                out.append(o._finish_ingest({}))
            except RuntimeError as e:
                out.append(str(e))

        h = threading.Thread(target=holder)
        h.start()
        self.assertTrue(entered.wait(5))
        q = threading.Thread(target=queued)
        q.start()                                   # waits on the lock like a voice bg thread
        time.sleep(0.05)
        r = threading.Thread(target=o.retire)
        r.start()
        release.set()
        for t in (h, q, r):
            t.join(5)
        # Whichever got the lock first, nothing may run after retire().
        self.assertTrue(o._retired)
        self.assertIn(out[0], ({"facts_count": 1}, "This brain was cleared or deleted."))
        with self.assertRaisesRegex(RuntimeError, "cleared or deleted"):
            o._finish_ingest({})
        with self.assertRaisesRegex(RuntimeError, "cleared or deleted"):
            o.Ingest("hello")

    def test_init_creates_write_lock(self):
        self.assertIn("_write_lock", Orchestrator.__init__.__code__.co_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
