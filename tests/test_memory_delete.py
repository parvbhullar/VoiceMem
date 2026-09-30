"""Deleting a fact cascades into the cognitive graph; editing one refreshes it.

Real sqlite in a temp dir for the graph store; fakes for the vector store and annotator.
Run directly::

    python tests/test_memory_delete.py
"""
import os
import sqlite3
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("SUPERMEM_MEMORYSPACE_ROOT", tempfile.mkdtemp(prefix="supermem-test-"))

from supermem.leftbrain.cognitive_graph import EntityType, SlotV2  # noqa: E402
from supermem.leftbrain.cognitive_graph.store_v2 import CognitiveGraphStoreV2  # noqa: E402
from supermem.leftbrain.memory_repository import LeftBrainMemoryRepository  # noqa: E402

UID = "u"
SLOT = SlotV2.DAILY_LIFE


class GraphStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "cog.db"
        self.store = CognitiveGraphStoreV2(self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def _count(self, sql: str, *args) -> int:
        with sqlite3.connect(self.path) as c:
            return c.execute(sql, args).fetchone()[0]

    def _seed(self) -> str:
        ent = self.store.upsert_entity(UID, "Pune", EntityType.PLACE, slot=SLOT)
        for mid, text in (("m1", "Alice lives in Pune"), ("m2", "Bob visited Pune")):
            self.store.upsert_memory_record(UID, mid, SLOT, text)
            self.store.link_memory(mid, ent.id, UID)
            self.store.upsert_memory_tags(mid, UID, [(SLOT.value, 0.9)])
        return ent.id

    def test_upsert_memory_record_refreshes_content(self):
        self.store.upsert_memory_record(UID, "m1", SLOT, "old text")
        self.store.upsert_memory_record(UID, "m1", SLOT, "new text")
        with sqlite3.connect(self.path) as c:
            row = c.execute("SELECT content FROM memories WHERE id='m1'").fetchone()
        self.assertEqual(row[0], "new text")

    def test_delete_memory_removes_only_that_memory(self):
        ent_id = self._seed()
        self.store.delete_memory("m1")
        for table, col in (("memory_tags", "memory_id"), ("entity_memory_links", "memory_id"),
                           ("memories", "id")):
            self.assertEqual(self._count(f"SELECT COUNT(*) FROM {table} WHERE {col}='m1'"), 0, table)
            self.assertEqual(self._count(f"SELECT COUNT(*) FROM {table} WHERE {col}='m2'"), 1, table)
        self.assertEqual(self._count("SELECT COUNT(*) FROM entities WHERE id=?", ent_id), 1)

    def test_unlink_memory_drops_only_its_links(self):
        self._seed()
        self.store.unlink_memory("m2")
        self.assertEqual(self._count("SELECT COUNT(*) FROM entity_memory_links WHERE memory_id='m2'"), 0)
        self.assertEqual(self._count("SELECT COUNT(*) FROM entity_memory_links WHERE memory_id='m1'"), 1)
        self.assertEqual(self._count("SELECT COUNT(*) FROM memories WHERE id='m2'"), 1)


class _Vectors:
    def __init__(self, ok: bool = True):
        self.ok = ok

    def delete_memory(self, memory_id: str) -> bool:
        return self.ok

    def update_memory(self, memory_id: str, new_text: str, **kw) -> bool:
        return self.ok


class _Cognitive:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[tuple] = []

    def delete_memory(self, memory_id: str) -> None:
        self.calls.append(("delete_memory", memory_id))
        if self.fail:
            raise RuntimeError("boom")

    def unlink_memory(self, memory_id: str) -> None:
        self.calls.append(("unlink_memory", memory_id))

    def ingest_annotated_fact(self, user_id: str, annotated, memory_ids: list[str]) -> None:
        self.calls.append(("ingest_annotated_fact", memory_ids[0]))


class _Annotator:
    def __init__(self, entities=("alice",)):
        self.entities = list(entities)

    def annotate(self, texts: list[str]) -> list:
        return [SimpleNamespace(entities=self.entities)]


def _repo(vectors_ok: bool = True, cog_fail: bool = False) -> LeftBrainMemoryRepository:
    repo = LeftBrainMemoryRepository.__new__(LeftBrainMemoryRepository)
    repo._vector_store = _Vectors(vectors_ok)
    repo._cognitive_store = _Cognitive(cog_fail)
    repo._cognitive_annotator = _Annotator()
    repo.load_json_store = lambda: {"results": []}
    repo._write_json_store = lambda results: None
    return repo


class RepositoryWiringTest(unittest.TestCase):
    def test_delete_cascades_into_graph(self):
        repo = _repo()
        self.assertTrue(repo.delete_memory("m1"))
        self.assertEqual(repo._cognitive_store.calls, [("delete_memory", "m1")])

    def test_graph_failure_does_not_fail_delete(self):
        repo = _repo(cog_fail=True)
        with self.assertLogs("supermem.leftbrain.memory_repository", level="WARNING"):
            self.assertTrue(repo.delete_memory("m1"))

    def test_vector_miss_skips_graph(self):
        repo = _repo(vectors_ok=False)
        self.assertFalse(repo.delete_memory("m1"))
        self.assertEqual(repo._cognitive_store.calls, [])

    def test_update_relinks_before_ingest(self):
        repo = _repo()
        self.assertTrue(repo.update_memory("m1", "t", user_id=UID))
        self.assertEqual(repo._cognitive_store.calls,
                         [("unlink_memory", "m1"), ("ingest_annotated_fact", "m1")])

    def test_update_with_empty_annotation_keeps_links(self):
        # CognitiveAnnotator returns [AnnotatedFact(entities=[])] when the LLM call fails.
        repo = _repo()
        repo._cognitive_annotator = _Annotator(entities=())
        self.assertTrue(repo.update_memory("m1", "t", user_id=UID))
        self.assertEqual(repo._cognitive_store.calls, [("ingest_annotated_fact", "m1")])

    def test_update_without_user_leaves_graph(self):
        repo = _repo()
        self.assertTrue(repo.update_memory("m1", "t"))
        self.assertEqual(repo._cognitive_store.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
