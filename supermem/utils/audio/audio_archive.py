"""AudioArchive — records the original WAV file path for each memory.

Storage schema (SQLite)::

    audio_archive(
        memory_id   TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        audio_path  TEXT NOT NULL,
        archived_at TEXT NOT NULL,
        PRIMARY KEY (memory_id, user_id)
    )
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AudioArchive:
    """Persists the memory_id -> audio_path mapping to SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS audio_archive (
                memory_id   TEXT NOT NULL,
                user_id     TEXT NOT NULL,
                audio_path  TEXT NOT NULL,
                archived_at TEXT NOT NULL,
                PRIMARY KEY (memory_id, user_id)
            )
            """)

    def record(self, memory_ids: list[str], user_id: str, audio_path: str) -> None:
        """Store the association between a batch of memory_ids and an audio_path."""
        now = _now()
        with self._conn() as c:
            c.executemany(
                "INSERT OR IGNORE INTO audio_archive VALUES (?,?,?,?)",
                [(mid, user_id, audio_path, now) for mid in memory_ids],
            )

    def get_audio_path(self, memory_id: str, user_id: str) -> str | None:
        """Look up the original WAV path for a memory; returns None if absent."""
        with self._conn() as c:
            row = c.execute(
                "SELECT audio_path FROM audio_archive WHERE memory_id=? AND user_id=?",
                (memory_id, user_id),
            ).fetchone()
        return row[0] if row else None

    def get_by_audio(self, audio_path: str, user_id: str) -> list[str]:
        """Look up all memory_ids associated with a WAV file."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT memory_id FROM audio_archive WHERE audio_path=? AND user_id=?",
                (audio_path, user_id),
            ).fetchall()
        return [r[0] for r in rows]

    def cleanup_expired(self, retention_days: int = 30) -> list[str]:
        """Delete original WAV files older than the retention period; mapping rows in the
        database are left untouched.

        The memory_id -> audio_path mapping is kept forever; only the file itself is
        physically deleted once expired. `get_audio_path()` / `GetOriginalAudio()` then
        rely on ``Path.exists()`` to detect "expired", so no extra flag is needed. One
        audio_path may map to several memory_ids (multiple memories extracted from one
        recording), so paths are de-duplicated here and each file is deleted only once.

        Returns
        -------
        list[str]
            File paths actually deleted in this call (de-duplicated).
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT audio_path FROM audio_archive WHERE archived_at < ?",
                (cutoff,),
            ).fetchall()

        deleted: list[str] = []
        for (audio_path,) in rows:
            p = Path(audio_path)
            if p.exists():
                try:
                    p.unlink()
                    deleted.append(audio_path)
                except OSError:
                    pass
        return deleted
