"""run.py's brain wipe (delete/clear), session gating and profile, with every heavy dependency faked.

run.py loads models and binds a brain at import, so the functions under test are lifted out of
its source with ast and run in a namespace of fakes. The orchestrator is the real class, built
with __new__, so retire()/_check_live() are the production ones.
Run directly::

    python tests/test_space_wipe.py
"""
import ast
import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("SUPERMEM_MEMORYSPACE_ROOT", tempfile.mkdtemp(prefix="supermem-test-"))

from supermem.orchestrator import Orchestrator  # noqa: E402

FUNCS = {"space_dir", "space_exists", "resolve_space", "_wipe_space", "_wipe_files",
         "delete_space", "clear_space", "_counted", "_update_fact", "right_brain_tree"}


def _load(ns: dict) -> dict:
    tree = ast.parse((ROOT / "web" / "run.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS]
    assert {n.name for n in nodes} == FUNCS, FUNCS - {n.name for n in nodes}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "run.py", "exec"), ns)
    return ns


class FakeRepo:
    def __init__(self) -> None:
        self.updated: list = []

    def update_memory(self, mid, text, user_id=None):
        self.updated.append((mid, text))
        return True


class FakeMem:
    def __init__(self, root: Path) -> None:
        o = Orchestrator.__new__(Orchestrator)
        o._write_lock = threading.RLock()
        o._memory_root = str(root)
        o._user_id = "u"
        o._repo = FakeRepo()
        o._get_repo = lambda: o._repo
        self._o = o


class SpaceWipeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="rv-wipe-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.evicted: list = []
        self.used: list = []
        ns = {"Path": Path, "os": os, "shutil": shutil, "threading": threading,
              "_ROOT": self.tmp, "_SPACES": {}, "_SPACES_LOCK": threading.RLock(),
              "_CLEARING": set(), "_CARTRIDGES": {}, "_OWNER_NAME_CACHE": {},
              "_space_audio": lambda d: [], "_write_space_language": lambda s, l: None,
              "ACTIVE_SPACE": "live", "_LIVE_SESSIONS": 0, "_REMEMBER_TASKS": set(),
              "RB_ENTITIES_PER_SLOT": 6}
        ns["evict_client"] = lambda d: self.evicted.append(("live" in ns["_CLEARING"], d.name))
        ns["use_space"] = self.used.append
        ns["get_space"] = lambda safe: ns["_SPACES"].setdefault(
            safe, FakeMem(ns["space_dir"](safe)[0]))
        self.ns = _load(ns)

    def _open(self, name: str) -> FakeMem:
        d, _ = self.ns["space_dir"](name)
        d.mkdir(parents=True)
        (d / "x.sqlite").write_text("x")
        return self.ns["get_space"](name)

    def test_delete_waits_for_the_write_in_flight_and_retires_the_instance(self) -> None:
        mem = self._open("rv")
        d = Path(mem._o._memory_root)
        held, release = threading.Event(), threading.Event()

        def edit() -> None:                        # an LLM-backed edit holding the write lock
            with mem._o._write_lock:
                held.set()
                release.wait(5)

        t = threading.Thread(target=edit)
        t.start()
        self.assertTrue(held.wait(5))
        dl = threading.Thread(target=self.ns["delete_space"], args=("rv",))
        dl.start()
        time.sleep(0.1)
        self.assertTrue(d.exists(), "rmtree ran under an in-flight write")
        release.set()
        for th in (t, dl):
            th.join(5)
        self.assertFalse(d.exists())
        self.assertTrue(mem._o._retired)
        with self.assertRaisesRegex(RuntimeError, "cleared or deleted"):
            self.ns["_update_fact"](mem, "m1", "new text")
        self.assertEqual(mem._o._repo.updated, [])

    def test_clear_retires_old_instance_and_recreates_empty(self) -> None:
        old = self._open("rv")
        self.assertEqual(self.ns["clear_space"]("rv"), {"cleared": "rv"})
        d, _ = self.ns["space_dir"]("rv")
        self.assertTrue(d.is_dir())
        self.assertEqual(list(d.iterdir()), [])
        self.assertTrue(old._o._retired)
        with self.assertRaisesRegex(RuntimeError, "cleared or deleted"):
            old._o._finish_ingest({})               # a voice bg thread queued on the old lock
        self.assertEqual(self.ns["_CLEARING"], set())
        self.assertEqual(self.used, [])             # not the live brain

    def test_clear_live_marks_clearing_and_rebinds(self) -> None:
        self._open("live")
        self.ns["clear_space"]("live")
        self.assertEqual(self.evicted, [(True, "live")])
        self.assertEqual(self.used, ["live"])
        self.assertEqual(self.ns["_CLEARING"], set())

    def test_clear_live_refused_with_a_session_open(self) -> None:
        self._open("live")
        self.ns["_LIVE_SESSIONS"] = 1
        with self.assertRaises(PermissionError):
            self.ns["clear_space"]("live")
        self.assertEqual(self.ns["_CLEARING"], set())
        self.assertEqual(self.evicted, [])

    def test_session_opened_while_clearing_is_refused(self) -> None:
        ran, sent = [], []

        class Sock:
            async def send_json(self, m):
                sent.append(m)

            async def close(self, code=1000):
                sent.append(code)

        async def session(sock):
            ran.append(sock)

        run = self.ns["_counted"](session)
        self.ns["_CLEARING"].add("live")
        asyncio.run(run(Sock()))
        self.assertEqual(ran, [])
        self.assertEqual(sent[0]["type"], "error")
        self.assertEqual(sent[-1], 1013)
        self.assertEqual(self.ns["_LIVE_SESSIONS"], 0)
        self.ns["_CLEARING"].clear()
        asyncio.run(run(Sock()))
        self.assertEqual(len(ran), 1)

    def test_resolve_open_brain_does_not_wait_on_the_spaces_lock(self) -> None:
        mem = self._open("rv")
        got, held, release = [], threading.Event(), threading.Event()

        def wipe_elsewhere() -> None:              # e.g. a clear waiting on another brain's lock
            with self.ns["_SPACES_LOCK"]:
                held.set()
                release.wait(5)

        t = threading.Thread(target=wipe_elsewhere)
        t.start()
        self.assertTrue(held.wait(5))
        r = threading.Thread(target=lambda: got.append(self.ns["resolve_space"]("rv")))
        r.start()
        r.join(1)
        release.set()
        t.join(5)
        self.assertEqual(got, [mem])
        with self.assertRaises(FileNotFoundError):
            self.ns["resolve_space"]("nope")

    def test_profile_without_humanize_queues_no_rewrite(self) -> None:
        batches: list = []

        class Trait:
            slot, cluster, claim, evidence = "values", "c", "likes tea", []

        class Store:
            def all(self, uid, per_slot):
                return [Trait()]

        mem = FakeMem(self.tmp)
        mem._o._right = type("R", (), {"_traits": lambda self: Store()})()
        self.ns.update(vm=mem, rb_human_batch=batches.append, rb_human=lambda c: "HUMAN " + c)
        rows = self.ns["right_brain_tree"]("u", {}, mem, per_slot=10_000, humanize=False)
        self.assertEqual((batches, rows[0]["text"]), ([], "likes tea"))
        rows = self.ns["right_brain_tree"]("u", {}, mem)         # the brain map still rewrites
        self.assertEqual((batches, rows[0]["text"]), ([["likes tea"]], "HUMAN likes tea"))


class CreateSpaceRouteTest(unittest.TestCase):
    def test_create_runs_off_the_event_loop(self) -> None:
        sys.path.insert(0, str(ROOT / "web"))
        import utils
        from fastapi.testclient import TestClient

        on_loop: list = []

        def create(name, lang):                    # builds + warms a brain: seconds, blocking
            try:
                asyncio.get_running_loop()
                on_loop.append(True)
            except RuntimeError:
                on_loop.append(False)
            return {"id": name}

        app = utils.build_app("llm_tts", None, None,
                              spaces=(lambda: [], create, lambda n: n, lambda: "demo"))
        r = TestClient(app).post("/api/spaces", json={"name": "rv-x"})
        self.assertEqual((r.status_code, r.json()), (200, {"id": "rv-x"}))
        self.assertEqual(on_loop, [False])


if __name__ == "__main__":
    unittest.main(verbosity=2)
