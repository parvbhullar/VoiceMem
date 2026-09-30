"""evict_client drops a space's cached mem0 client, so a space re-created under the same
name starts empty and accepts writes (instead of serving deleted vectors from RAM and
failing every write with "attempt to write a readonly database").

Model-free and network-free: a hash embedder, embedded Qdrant in a temp dir, infer=False adds.
Run directly::

    python tests/test_space_evict.py
"""
import gc
import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("SUPERMEM_MEMORYSPACE_ROOT", tempfile.mkdtemp(prefix="supermem-test-"))

from supermem.leftbrain import mem0_backend_store as m0  # noqa: E402
from supermem.leftbrain.mem0_backend_store import Mem0BackendStore, evict_client  # noqa: E402


class _Emb:
    dimensions = 8
    model_name = "fake"

    def embed_texts(self, ts):
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in ts]


class SpaceEvictTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        evict_client(self.root / "s1")
        self._tmp.cleanup()

    def test_recreated_space_starts_empty_and_writable(self):
        (self.root / "s1").mkdir()
        s = Mem0BackendStore(_Emb(), memory_root=self.root / "s1")
        s.add_text("u", "Alice likes tea")
        self.assertTrue(s.list_ids(user_id="u"))

        evict_client(self.root / "s1")
        del s
        gc.collect()
        shutil.rmtree(self.root / "s1")
        (self.root / "s1").mkdir()

        s2 = Mem0BackendStore(_Emb(), memory_root=self.root / "s1")
        self.assertEqual(s2.list_ids(user_id="u"), [])
        s2.add_text("u", "Bob likes coffee")
        self.assertEqual(len(s2.list_ids(user_id="u")), 1)

    def test_evict_unopened_root_is_noop(self):
        before = dict(m0._MEM0_CLIENT_CACHE)
        evict_client(self.root / "never-opened")
        self.assertEqual(m0._MEM0_CLIENT_CACHE, before)
        self.assertFalse((self.root / "never-opened").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
