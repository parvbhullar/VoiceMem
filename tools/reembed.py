"""After switching embedders, recompute the vectors in the store whose dimension is now invalid.

When you need it: `_embed_text` now follows the injected embedder, while
`rb_traits` / `graph_entities` may still hold vectors computed by the previous embedder.
Mismatched dimensions are skipped (with a warning), so right-brain retrieval and entity dedup stop working until re-embedded.

Run: python3 tools/reembed.py <space> [--apply] [--local]
Without --apply it only counts; --local computes with the web demo's config (local E5),
otherwise the default embedder (OpenAI). **It must match the config you actually run with**,
or the dimensions will be wrong and nothing is fixed.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    space = sys.argv[1] if len(sys.argv) > 1 else "demo"
    apply = "--apply" in sys.argv
    db = Path("supermem_memoryspace") / space / f"{space}.sqlite"
    if not db.is_file():
        print(f"Not found: {db}")
        return

    # Build only the embedder, not a full SuperMem -- the latter opens the vector store and fights the running
    # service for qdrant's file lock ("Storage folder ... already accessed by another instance").
    # The migration changes vector columns in sqlite and has nothing to do with the vector store.
    if "--local" in sys.argv:                     # match the web demo config
        from supermem.leftbrain.local_e5_embedder import LocalE5Embedder
        e = LocalE5Embedder()
        embed = e.embed_query_text
    else:
        from supermem.leftbrain.local_memory_store import (
            OpenAILocalEmbedder, OpenAILocalEmbedderConfig,
        )
        e = OpenAILocalEmbedder(OpenAILocalEmbedderConfig())
        embed = lambda t: e.embed_texts([t])[0]
    want = len(embed("dimension probe"))
    print(f"{space}: current embedder outputs {want} dims")

    import json
    import numpy as np

    def vec_len(b):
        """The two tables store vectors differently: rb_traits as float32 binary, graph_entities as JSON."""
        if isinstance(b, (bytes, bytearray)):
            return len(np.frombuffer(b, dtype=np.float32))
        try:
            return len(json.loads(b))
        except Exception:
            return -1

    def pack(table, vec):
        return (np.asarray(vec, dtype=np.float32).tobytes() if table == "rb_traits"
                else json.dumps([float(x) for x in vec]))

    jobs = []          # (table, id column, rows to recompute)
    con = sqlite3.connect(db)
    for table, idc, txtc in (("rb_traits", "id", "claim"),
                             ("graph_entities", "id", "name")):
        try:
            rows = con.execute(
                f"SELECT {idc}, {txtc}, embedding FROM {table} "
                "WHERE embedding IS NOT NULL").fetchall()
        except sqlite3.OperationalError:
            continue
        stale = [(i, t) for i, t, b in rows if vec_len(b) != want]
        print(f"  {table:16} {len(rows):4} rows, {len(stale)} with stale dimensions")
        if stale:
            jobs.append((table, idc, stale))

    if not jobs:
        print("Nothing to recompute.")
        return
    if not apply:
        print("(no --apply given, counting only)")
        return

    total = 0
    for table, idc, stale in jobs:
        for n, (mid, text) in enumerate(stale, 1):
            con.execute(f"UPDATE {table} SET embedding=? WHERE {idc}=?",
                        (pack(table, embed(text)), mid))
            total += 1
            if n % 25 == 0:
                con.commit()
                print(f"  {table} …{n}/{len(stale)}")
        con.commit()
    print(f"\nRecompute done: {total} rows")


if __name__ == "__main__":
    main()
