"""RoutineStore — daily-life sound patterns: automatically build routine memories (audiomem 2.7).

Records scene-arrival events by "scene + approximate time slot" (once per scene change, not
per utterance; see core.py Ingest, which calls observe() only when scene_changed). The same
scene in the same time slot on the same day counts only once (the PRIMARY KEY de-duplicates
naturally). When a (scene, time slot) combination has appeared on routine_threshold distinct
days, it is judged to be a routine. The routine memory is created only on the observation
that first crosses the threshold; the same (scene, bucket) is never generated again (see the
existence check on the routines table).

Time slots are 3 hours wide (8 per day), leaving room for "around this time" without
requiring minute-level alignment.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

_BUCKET_LABELS = [
    "deep night (00:00-03:00)", "early morning (03:00-06:00)",
    "morning (06:00-09:00)", "late morning (09:00-12:00)",
    "early afternoon (12:00-15:00)", "afternoon (15:00-18:00)",
    "evening (18:00-21:00)", "night (21:00-24:00)",
]


def bucket_label(bucket: int) -> str:
    return _BUCKET_LABELS[bucket % 8]


class RoutineStore:
    """Maintains observations of "which scene regularly occurs in which time slot" and the established routines."""

    def __init__(self, db_path: Path, routine_threshold: int = 3) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._threshold = routine_threshold
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS scene_observations (
                user_id  TEXT NOT NULL,
                scene    TEXT NOT NULL,
                bucket   INTEGER NOT NULL,
                obs_date TEXT NOT NULL,
                ts       TEXT NOT NULL,
                PRIMARY KEY (user_id, scene, bucket, obs_date)
            );
            CREATE TABLE IF NOT EXISTS routines (
                user_id        TEXT NOT NULL,
                scene          TEXT NOT NULL,
                bucket         INTEGER NOT NULL,
                established_at TEXT NOT NULL,
                distinct_days  INTEGER NOT NULL,
                PRIMARY KEY (user_id, scene, bucket)
            );
            """)

    def observe(self, user_id: str, scene: str, dt: datetime) -> dict:
        """Record one scene-arrival observation.

        Returns
        -------
        dict
            ``{is_new_routine, distinct_days, bucket}``. ``is_new_routine=True``
            means this observation just pushed (scene, bucket) across routine_threshold
            distinct days and a new routine memory is warranted; each
            (user, scene, bucket) triggers this only once.
        """
        bucket = dt.hour // 3
        obs_date = dt.date().isoformat()
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO scene_observations VALUES (?,?,?,?,?)",
                (user_id, scene, bucket, obs_date, dt.isoformat()),
            )
            row = c.execute(
                "SELECT COUNT(DISTINCT obs_date) FROM scene_observations "
                "WHERE user_id=? AND scene=? AND bucket=?",
                (user_id, scene, bucket),
            ).fetchone()
            distinct_days = row[0] if row else 0

            already = c.execute(
                "SELECT 1 FROM routines WHERE user_id=? AND scene=? AND bucket=?",
                (user_id, scene, bucket),
            ).fetchone()

            is_new_routine = False
            if distinct_days >= self._threshold and not already:
                c.execute(
                    "INSERT INTO routines VALUES (?,?,?,?,?)",
                    (user_id, scene, bucket, dt.isoformat(), distinct_days),
                )
                is_new_routine = True

        return {
            "is_new_routine": is_new_routine,
            "distinct_days": distinct_days,
            "bucket": bucket,
        }

    def list_routines(self, user_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT scene, bucket, established_at, distinct_days "
                "FROM routines WHERE user_id=?",
                (user_id,),
            ).fetchall()
        return [
            {"scene": r[0], "bucket": r[1], "bucket_label": bucket_label(r[1]),
             "established_at": r[2], "distinct_days": r[3]}
            for r in rows
        ]
