"""/memories routes on a bare FastAPI app, every callback a fake.

No models, no network, no web/utils.py or web/run.py (utils builds an AsyncOpenAI at import).
Run directly (this repo has no pytest in the demo venv)::

    python tests/test_memories_api.py
"""
import sys
import time
import unittest
from datetime import date
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))

from memories_api import register_routes  # noqa: E402


def _row(mid: str, text: str, slot: str, entities: list, day: str, role: str = "user") -> dict:
    return {"id": mid, "text": text, "slot": slot, "entities": entities, "date": day, "role": role}


class FakeMem:
    """One brain: rows by id, a canned profile and a canned semantic ranking."""

    def __init__(self, *rows: dict, profile=None, ranked=None) -> None:
        self.rows = {r["id"]: dict(r) for r in rows}
        self.profile = profile or []
        self.ranked = ranked or []


class FakeJobs:
    """Records submit() calls; busy_spaces decides busy()."""

    def __init__(self) -> None:
        self.submitted: list[dict] = []
        self.recs: dict[str, dict] = {}
        self.busy_spaces: set[str] = set()
        self.cancelled: list[str] = []

    def submit(self, mem, space, chunks, observed_at=None, *, filename="", kind="", on_done=None):
        job_id = f"j{len(self.submitted) + 1}"
        self.submitted.append(dict(mem=mem, space=space, chunks=chunks, observed_at=observed_at,
                                   filename=filename, kind=kind, on_done=on_done))
        self.recs[job_id] = {"id": job_id, "space": space, "state": "queued"}
        return job_id

    def get(self, job_id):
        rec = self.recs.get(job_id)
        return dict(rec) if rec else None

    def list(self, space):
        return [dict(r) for r in self.recs.values() if r["space"] == space]

    def busy(self, space):
        return space in self.busy_spaces

    def cancel(self, job_id):
        if job_id not in self.recs:
            return False
        self.cancelled.append(job_id)
        return True


class MemoriesApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spaces = {
            "a": FakeMem(
                _row("m1", "Alice likes tea", "preferences", ["Alice", "tea"], "2026-01-02"),
                _row("m2", "Alice lives in Pune", "identity", ["Pune"], "2026-03-01"),
                _row("m3", "Sure, noted", "daily_life", [], "", role="assistant"),
                _row("m4", "Bob is Alice's brother", "relationships", ["bob"], "2025-12-31"),
                profile=[{"cluster": "food", "slot": "preferences", "raw": "likes tea",
                          "text": "Likes tea", "desc": "", "notes": []}],
                ranked=[("m4", 0.91234), ("m1", 0.5), ("gone", 0.4)]),
            "b": FakeMem(_row("m1", "B's own fact", "identity", [], "2026-01-01")),
        }
        self.updates: list = []
        self.cleared: list = []
        self.deleted_spaces: list = []
        self.changed: list = []
        self.planned: list = []
        self.semantic_qs: list = []
        self.clear_error: Exception | None = None
        self.delete_error: Exception | None = None
        self.plan_error: Exception | None = None
        self.jobs = FakeJobs()

        def resolve(space):
            if "!" in space:
                raise ValueError("Invalid brain name.")
            if space not in self.spaces:
                raise FileNotFoundError(space)
            return self.spaces[space]

        def semantic(mem, q):
            self.semantic_qs.append(q)
            return mem.ranked

        def update_fact(mem, mid, text):
            self.updates.append((mid, text))
            if mid not in mem.rows:
                return False
            mem.rows[mid]["text"] = text
            return True

        def clear_space(name):
            if self.clear_error:
                raise self.clear_error
            self.cleared.append(name)
            return {"cleared": name}

        def delete_space(name):
            if self.delete_error:
                raise self.delete_error
            self.deleted_spaces.append(name)

        def plan(filename, data, fmt, owner):
            self.planned.append((filename, data, fmt, owner))
            if self.plan_error:
                raise self.plan_error
            return "prose", ["chunk one", "chunk two"]

        app = FastAPI()
        register_routes(app, resolve=resolve, delete_space=delete_space,
                        clear_space=clear_space, facts=lambda mem: list(mem.rows.values()),
                        profile=lambda mem: mem.profile, semantic=semantic,
                        update_fact=update_fact,
                        delete_fact=lambda mem, mid: mem.rows.pop(mid, None) is not None,
                        jobs=self.jobs, plan=plan, on_change=self.changed.append)
        self.client = TestClient(app)

    def _ids(self, resp) -> list:
        self.assertEqual(resp.status_code, 200, resp.text)
        return [r["id"] for r in resp.json()["items"]]

    def _not_found(self, resp, detail: str) -> None:
        # The detail proves the route answered: a missing route is a 404 "Not Found" too.
        self.assertEqual((resp.status_code, resp.json()["detail"]), (404, detail))

    # ---- GET /api/mem ----

    def test_facts_newest_first_blank_dates_last(self) -> None:
        r = self.client.get("/api/mem", params={"space": "a"})
        self.assertEqual(self._ids(r), ["m2", "m1", "m4", "m3"])
        body = r.json()
        self.assertEqual(body["total"], 4)
        self.assertEqual((body["page"], body["size"]), (1, 50))

    def test_filter_options_come_from_the_whole_brain(self) -> None:
        body = self.client.get("/api/mem", params={"space": "a", "slot": "identity"}).json()
        self.assertEqual([r["id"] for r in body["items"]], ["m2"])
        self.assertEqual(body["slots"], ["daily_life", "identity", "preferences", "relationships"])
        self.assertEqual(body["entities"], ["Alice", "bob", "Pune", "tea"])

    def test_paging(self) -> None:
        r = self.client.get("/api/mem", params={"space": "a", "page": 2, "size": 2})
        self.assertEqual(self._ids(r), ["m4", "m3"])
        self.assertEqual(r.json()["total"], 4)

    def test_size_clamped(self) -> None:
        body = self.client.get("/api/mem", params={"space": "a", "size": 500}).json()
        self.assertEqual(body["size"], 200)
        body = self.client.get("/api/mem", params={"space": "a", "size": 0, "page": 0}).json()
        self.assertEqual((body["size"], body["page"]), (1, 1))

    def test_substring_case_insensitive(self) -> None:
        r = self.client.get("/api/mem", params={"space": "a", "q": "  PUNE "})
        self.assertEqual(self._ids(r), ["m2"])
        self.assertEqual(r.json()["total"], 1)

    def test_entity_and_role_filters(self) -> None:
        self.assertEqual(self._ids(self.client.get(
            "/api/mem", params={"space": "a", "entity": "Pune"})), ["m2"])
        self.assertEqual(self._ids(self.client.get(
            "/api/mem", params={"space": "a", "role": "assistant"})), ["m3"])

    def test_semantic_keeps_rank_order_with_scores(self) -> None:
        r = self.client.get("/api/mem", params={"space": "a", "q": " tea ", "mode": "semantic"})
        self.assertEqual(self._ids(r), ["m4", "m1"])        # "gone" is not in the brain
        self.assertEqual([x["score"] for x in r.json()["items"]], [0.912, 0.5])
        self.assertEqual(r.json()["total"], 2)
        self.assertEqual(self.semantic_qs, ["tea"])

    def test_semantic_without_query_lists_everything(self) -> None:
        r = self.client.get("/api/mem", params={"space": "a", "mode": "semantic"})
        self.assertEqual(self._ids(r), ["m2", "m1", "m4", "m3"])
        self.assertEqual(self.semantic_qs, [])

    def test_bad_mode_is_400(self) -> None:
        r = self.client.get("/api/mem", params={"space": "a", "mode": "bogus"})
        self.assertEqual(r.status_code, 400)

    def test_unknown_space_is_404_and_creates_nothing(self) -> None:
        self._not_found(self.client.get("/api/mem", params={"space": "zzz"}),
                        "No brain with that name.")
        self.assertEqual(set(self.spaces), {"a", "b"})

    def test_bad_space_name_is_400(self) -> None:
        self.assertEqual(self.client.get("/api/mem", params={"space": "a!"}).status_code, 400)

    # ---- profile ----

    def test_profile(self) -> None:
        r = self.client.get("/api/mem/profile", params={"space": "a"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"items": self.spaces["a"].profile})
        self._not_found(self.client.get("/api/mem/profile", params={"space": "zzz"}),
                        "No brain with that name.")

    # ---- edit / delete ----

    def test_edit_strips_and_signals_change(self) -> None:
        r = self.client.patch("/api/mem/m1", params={"space": "a"}, json={"text": " new "})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"id": "m1", "text": "new"})
        self.assertEqual(self.spaces["a"].rows["m1"]["text"], "new")
        self.assertEqual(self.spaces["b"].rows["m1"]["text"], "B's own fact")
        self.assertEqual(self.changed, ["a"])

    def test_edit_blank_is_400_without_update(self) -> None:
        r = self.client.patch("/api/mem/m1", params={"space": "a"}, json={"text": "   "})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.updates, [])
        self.assertEqual(self.changed, [])

    def test_edit_unknown_id_is_404(self) -> None:
        r = self.client.patch("/api/mem/nope", params={"space": "a"}, json={"text": "x"})
        self._not_found(r, "No such memory.")
        self.assertEqual(self.changed, [])

    def test_delete_only_touches_its_space(self) -> None:
        r = self.client.delete("/api/mem/m1", params={"space": "a"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"id": "m1", "deleted": True})
        self.assertNotIn("m1", self.spaces["a"].rows)
        self.assertIn("m1", self.spaces["b"].rows)
        self.assertEqual(self.changed, ["a"])

    def test_delete_unknown_id_is_404(self) -> None:
        r = self.client.delete("/api/mem/nope", params={"space": "a"})
        self._not_found(r, "No such memory.")
        self.assertEqual(self.changed, [])

    # ---- clear ----

    def test_clear(self) -> None:
        r = self.client.post("/api/mem/clear", params={"space": "a"}, json={"confirm": "a"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"cleared": "a"})
        self.assertEqual(self.cleared, ["a"])
        self.assertEqual(self.changed, ["a"])

    def test_clear_wrong_confirm_is_400(self) -> None:
        r = self.client.post("/api/mem/clear", params={"space": "a"}, json={"confirm": "A"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.cleared, [])

    def test_clear_busy_is_409(self) -> None:
        self.jobs.busy_spaces.add("a")
        r = self.client.post("/api/mem/clear", params={"space": "a"}, json={"confirm": "a"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self.cleared, [])

    def test_clear_refused_is_409_with_message(self) -> None:
        self.clear_error = PermissionError("A voice session is using this brain.")
        r = self.client.post("/api/mem/clear", params={"space": "a"}, json={"confirm": "a"})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(r.json()["detail"], "A voice session is using this brain.")
        self.assertEqual(self.changed, [])

    # ---- delete space ----

    def _delete_space(self, name: str, confirm: str):
        return self.client.request("DELETE", f"/api/spaces/{name}", json={"confirm": confirm})

    def test_delete_space(self) -> None:
        r = self._delete_space("b", "b")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"deleted": "b"})
        self.assertEqual(self.deleted_spaces, ["b"])
        self.assertEqual(self.changed, ["b"])

    def test_delete_space_wrong_confirm_is_400(self) -> None:
        self.assertEqual(self._delete_space("b", "a").status_code, 400)
        self.assertEqual(self.deleted_spaces, [])

    def test_delete_space_errors_map_to_status(self) -> None:
        for err, status in ((PermissionError("live"), 409), (RuntimeError("elsewhere"), 409),
                            (FileNotFoundError("b"), 404), (ValueError("bad"), 400)):
            with self.subTest(err=type(err).__name__):
                self.delete_error = err
                self.assertEqual(self._delete_space("b", "b").status_code, status)
        self.assertEqual(self.changed, [])

    def test_delete_space_busy_is_409(self) -> None:
        self.jobs.busy_spaces.add("b")
        self.assertEqual(self._delete_space("b", "b").status_code, 409)
        self.assertEqual(self.deleted_spaces, [])

    # ---- ingest ----

    def test_ingest_submits_a_job(self) -> None:
        r = self.client.post("/api/ingest", params={"space": "a", "filename": "notes.txt"},
                             content=b"Alice likes tea.")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json(), {"job_id": "j1", "total": 2, "format": "prose"})
        self.assertEqual(self.planned, [("notes.txt", b"Alice likes tea.", "auto", "")])
        job = self.jobs.submitted[0]
        self.assertIs(job["mem"], self.spaces["a"])
        self.assertEqual(job["space"], "a")
        self.assertEqual(job["chunks"], ["chunk one", "chunk two"])
        self.assertEqual(job["observed_at"], date.today().isoformat())
        self.assertEqual((job["filename"], job["kind"]), ("notes.txt", "prose"))

    def test_ingest_passes_format_owner_and_date(self) -> None:
        r = self.client.post("/api/ingest", params={"space": "a", "format": "transcript",
                                                    "owner": "Bob", "observed_at": "2026-01-02"},
                             content=b"Bob: hi")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.planned, [("", b"Bob: hi", "transcript", "Bob")])
        self.assertEqual(self.jobs.submitted[0]["observed_at"], "2026-01-02")
        self.assertEqual(self.jobs.submitted[0]["filename"], "pasted text")

    def test_ingest_bad_date_is_400(self) -> None:
        r = self.client.post("/api/ingest", params={"space": "a", "observed_at": "01/02/2026"},
                             content=b"x")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.jobs.submitted, [])

    def test_ingest_plan_error_is_400_with_message(self) -> None:
        self.plan_error = ValueError("The file is not UTF-8 text.")
        r = self.client.post("/api/ingest", params={"space": "a", "filename": "x.txt"},
                             content=b"\xff")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["detail"], "The file is not UTF-8 text.")
        self.assertEqual(self.jobs.submitted, [])

    def test_ingest_unknown_space_is_404(self) -> None:
        r = self.client.post("/api/ingest", params={"space": "zzz"}, content=b"x")
        self._not_found(r, "No brain with that name.")
        self.assertEqual((self.planned, self.jobs.submitted), ([], []))

    def test_ingest_on_done_signals_change(self) -> None:
        self.client.post("/api/ingest", params={"space": "a"}, content=b"x")
        self.assertEqual(self.changed, [])
        self.jobs.submitted[0]["on_done"]()
        self.assertEqual(self.changed, ["a"])

    def test_ingest_over_size_is_413_before_planning(self) -> None:
        import ingest
        big = b"a" * (ingest.MAX_BYTES + 1)
        r = self.client.post("/api/ingest", params={"space": "a"}, content=big)
        self.assertEqual(r.status_code, 413)
        self.assertIn("MB", r.json()["detail"])
        # No Content-Length (chunked): the running cap still stops it.
        r = self.client.post("/api/ingest", params={"space": "a"},
                             content=(big[i:i + (1 << 20)] for i in range(0, len(big), 1 << 20)))
        self.assertEqual(r.status_code, 413)
        self.assertEqual((self.planned, self.jobs.submitted), ([], []))

    def test_ingest_brain_replaced_while_planning_is_409(self) -> None:
        stale = self.spaces["a"]
        real_plan_calls = self.planned

        def swap(*a):
            self.spaces["a"] = FakeMem()                    # a clear re-created the brain
            real_plan_calls.append(a)
            return "prose", ["x"]

        app = FastAPI()
        register_routes(app, resolve=lambda sp: self.spaces[sp], delete_space=lambda n: None,
                        clear_space=lambda n: None, facts=lambda m: [], profile=lambda m: [],
                        semantic=lambda m, q: [], update_fact=lambda m, i, t: True,
                        delete_fact=lambda m, i: True, jobs=self.jobs, plan=swap)
        r = TestClient(app).post("/api/ingest", params={"space": "a"}, content=b"x")
        self.assertEqual(r.status_code, 409, r.text)
        self.assertEqual(self.jobs.submitted, [])
        self.assertIsNot(self.spaces["a"], stale)

    def test_cross_origin_writes_refused(self) -> None:
        evil = {"Origin": "https://evil.example", "Content-Type": "text/plain"}
        r = self.client.post("/api/ingest", params={"space": "a"}, content=b"x", headers=evil)
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.client.post("/api/ingest/j1/cancel", headers=evil).status_code, 403)
        self.assertEqual(self.jobs.submitted, [])
        same = {"Origin": "http://testserver"}
        r = self.client.post("/api/ingest", params={"space": "a"}, content=b"x", headers=same)
        self.assertEqual(r.status_code, 200, r.text)
        # Reads stay open (the page itself is same-origin anyway).
        self.assertEqual(self.client.get("/api/mem", params={"space": "a"}, headers=evil).status_code, 200)

    def test_write_on_a_retired_brain_is_409(self) -> None:
        app = FastAPI()

        def retired(*a):
            raise RuntimeError("This brain was cleared or deleted.")

        register_routes(app, resolve=lambda sp: self.spaces[sp], delete_space=lambda n: None,
                        clear_space=lambda n: None, facts=lambda m: [], profile=lambda m: [],
                        semantic=lambda m, q: [], update_fact=retired, delete_fact=retired,
                        jobs=self.jobs, plan=lambda *a: ("prose", []))
        c = TestClient(app)
        self.assertEqual(c.patch("/api/mem/m1", params={"space": "a"}, json={"text": "x"}).status_code, 409)
        self.assertEqual(c.delete("/api/mem/m1", params={"space": "a"}).status_code, 409)

    # ---- job list / status / cancel ----

    def test_job_routes(self) -> None:
        self.client.post("/api/ingest", params={"space": "a"}, content=b"x")
        self.client.post("/api/ingest", params={"space": "b"}, content=b"y")
        jobs = self.client.get("/api/ingest", params={"space": "a"}).json()["jobs"]
        self.assertEqual([j["id"] for j in jobs], ["j1"])
        self.assertEqual(self.client.get("/api/ingest/j2").json()["space"], "b")
        self._not_found(self.client.get("/api/ingest/nope"), "No such job.")
        r = self.client.post("/api/ingest/j1/cancel")
        self.assertEqual((r.status_code, r.json()["id"]), (200, "j1"))
        self.assertEqual(self.jobs.cancelled, ["j1"])
        self._not_found(self.client.post("/api/ingest/nope/cancel"), "No such job.")

    # ---- page ----

    @unittest.skipUnless((ROOT / "web" / "memories.html").exists(),
                         "web/memories.html not written yet")
    def test_page(self) -> None:
        r = self.client.get("/memories")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/html"))
        self.assertEqual(r.headers["cache-control"], "no-store")


class IngestEndToEndTest(unittest.TestCase):
    """Routes + the REAL Jobs and plan_chunks: pasted transcript -> mem.ingest calls."""

    def test_pasted_transcript_reaches_mem_ingest(self) -> None:
        from ingest import Jobs, plan_chunks

        class RecordingMem:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def ingest(self, text, *, speaker, agent_reply, observed_at):
                self.calls.append((text, speaker, agent_reply, observed_at))
                return {"facts_count": 1}

        mem, changed, jobs = RecordingMem(), [], Jobs()
        app = FastAPI()
        register_routes(app, resolve=lambda s: mem, delete_space=lambda n: None,
                        clear_space=lambda n: None, facts=lambda m: [],
                        profile=lambda m: [], semantic=lambda m, q: [],
                        update_fact=lambda m, i, t: False, delete_fact=lambda m, i: False,
                        jobs=jobs, plan=plan_chunks, on_change=changed.append)
        client = TestClient(app)
        text = ("Alice: I am vegetarian\nBot: Noted.\nBob: I live in Pune\n"
                "Alice: I love tea\nBot: Nice.\nBob: Me too")
        r = client.post("/api/ingest", params={"space": "a", "owner": "alice",
                                               "observed_at": "2026-01-02"},
                        content=text.encode())
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual((r.json()["format"], r.json()["total"]), ("transcript", 4))
        job_id = r.json()["job_id"]
        for _ in range(200):
            rec = client.get(f"/api/ingest/{job_id}").json()
            if rec["state"] not in ("queued", "running"):
                break
            time.sleep(0.01)
        self.assertEqual((rec["state"], rec["done"], rec["facts"]), ("done", 4, 4))
        self.assertEqual(mem.calls, [
            ("I am vegetarian", "user", "Noted.", "2026-01-02"),
            ("I live in Pune", "Bob", None, "2026-01-02"),
            ("I love tea", "user", "Nice.", "2026-01-02"),
            ("Me too", "Bob", None, "2026-01-02"),
        ])
        self.assertEqual(changed, ["a"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
