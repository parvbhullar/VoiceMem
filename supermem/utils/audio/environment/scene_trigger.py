"""SceneTrigger — acoustically triggered reminder system.

The user says "remind me to call mom when I get on the bus"; the system stores a
scene-triggered reminder that fires automatically when the scene changes to transit.

Storage schema (SQLite)::

    scene_triggers(
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        trigger_scene TEXT NOT NULL,   -- trigger scene tag, e.g. "transit"
        message     TEXT NOT NULL,     -- reminder text
        created_at  TEXT NOT NULL,
        status      TEXT NOT NULL,     -- "pending" | "fired" | "cancelled"
        required_label TEXT            -- specific sound label (e.g. "bus"); NULL = coarse scene match is enough
    )

    scene_state(
        user_id     TEXT PRIMARY KEY,
        last_scene  TEXT,              -- last detected scene
        updated_at  TEXT
    )
"""
from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from supermem.utils.audio.environment.scene_classifier import SceneTag


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Natural-language scene description -> (scene_tag, specific AudioSet label that must be hit in raw_matches)
#
# required_label None means a coarse scene match is enough; when not None, check_and_fire()
# must also find it among the specific labels detected this time before actually firing.
# This avoids "bus" and "subway" both being coarsely bucketed as transit, so that a
# "remind me when I get on the bus" reminder is not wrongly fired on the subway.
# (Aliases are matched as lowercase substrings, in dict order.)
_SCENE_ALIASES: dict[str, tuple[SceneTag, str | None]] = {
    # transit -- narrowed down to the specific vehicle
    "bus": (SceneTag.TRANSIT, "bus"),
    "subway": (SceneTag.TRANSIT, "subway"), "metro": (SceneTag.TRANSIT, "subway"),
    "train": (SceneTag.TRANSIT, "train"),
    # these are generic; the user named no specific vehicle, so a coarse scene match is enough
    "commute": (SceneTag.TRANSIT, None), "on the way to work": (SceneTag.TRANSIT, None),
    "on my way home": (SceneTag.TRANSIT, None), "on the road": (SceneTag.TRANSIT, None),
    "in the car": (SceneTag.TRANSIT, None),
    # office
    "office": (SceneTag.OFFICE, None), "at work": (SceneTag.OFFICE, None),
    "my desk": (SceneTag.OFFICE, None),
    # home
    "home": (SceneTag.HOME, None),
    # café
    "coffee shop": (SceneTag.CAFE, None), "starbucks": (SceneTag.CAFE, None),
    "café": (SceneTag.CAFE, None), "cafe": (SceneTag.CAFE, None),
    # outdoor
    "outside": (SceneTag.OUTDOOR, None), "outdoors": (SceneTag.OUTDOOR, None),
    "for a walk": (SceneTag.OUTDOOR, None),
    # meeting
    "meeting": (SceneTag.MEETING, None), "conference room": (SceneTag.MEETING, None),
    # quiet
    "somewhere quiet": (SceneTag.QUIET, None), "alone": (SceneTag.QUIET, None),
}

# Trigger phrases: detect whether the user is creating a scene trigger
# ("remind me to" comes first so the extracted message does not start with "to").
_TRIGGER_PATTERNS = [
    "remind me to", "remind me",
]

_SCENE_ARRIVAL_PATTERNS = [
    "when i get", "when i'm", "when i am", "when i arrive", "when i reach",
    "once i get", "once i'm", "once i am", "as soon as i", "when i board",
]


@dataclass
class SceneTrigger:
    id: str
    user_id: str
    trigger_scene: str   # SceneTag.value
    message: str
    created_at: str
    status: str          # "pending" | "fired" | "cancelled"
    required_label: str | None = None   # specific sound label, e.g. "bus"; None = not narrowed


class SceneTriggerStore:
    """SQLite storage for scene-triggered reminders."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._path)
        c.row_factory = sqlite3.Row
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS scene_triggers (
                id            TEXT PRIMARY KEY,
                user_id       TEXT NOT NULL,
                trigger_scene TEXT NOT NULL,
                message       TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'pending',
                required_label TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_st_user ON scene_triggers(user_id, status);

            CREATE TABLE IF NOT EXISTS scene_state (
                user_id    TEXT PRIMARY KEY,
                last_scene TEXT,
                updated_at TEXT
            );
            """)

    def create(
        self, user_id: str, trigger_scene: str, message: str,
        required_label: str | None = None,
    ) -> SceneTrigger:
        t = SceneTrigger(
            id=str(uuid.uuid4()),
            user_id=user_id,
            trigger_scene=trigger_scene,
            message=message,
            created_at=_now(),
            status="pending",
            required_label=required_label,
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO scene_triggers VALUES (?,?,?,?,?,?,?)",
                (t.id, t.user_id, t.trigger_scene, t.message, t.created_at, t.status, t.required_label),
            )
        return t

    def get_pending(self, user_id: str, scene: str) -> list[SceneTrigger]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM scene_triggers WHERE user_id=? AND trigger_scene=? AND status='pending'",
                (user_id, scene),
            ).fetchall()
        return [_row_to_trigger(r) for r in rows]

    def fire(self, trigger_id: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE scene_triggers SET status='fired' WHERE id=?",
                (trigger_id,),
            )

    def cancel(self, trigger_id: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE scene_triggers SET status='cancelled' WHERE id=?",
                (trigger_id,),
            )

    def get_last_scene(self, user_id: str) -> str | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT last_scene FROM scene_state WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["last_scene"] if row else None

    def update_scene(self, user_id: str, scene: str) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO scene_state (user_id, last_scene, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       last_scene=excluded.last_scene,
                       updated_at=excluded.updated_at""",
                (user_id, scene, _now()),
            )


def _row_to_trigger(row: sqlite3.Row) -> SceneTrigger:
    return SceneTrigger(
        id=row["id"], user_id=row["user_id"],
        trigger_scene=row["trigger_scene"], message=row["message"],
        created_at=row["created_at"], status=row["status"],
        required_label=row["required_label"],
    )


# ── Trigger parsing (extract scene-trigger intent from the user utterance) ───────────────────────

def parse_trigger_intent(text: str) -> tuple[SceneTag | None, str, str | None]:
    """Parse a scene-trigger intent from the user utterance.

    Returns:
        (trigger_scene, reminder_message, required_label) if a trigger intent is found,
        otherwise (None, "", None). When required_label is set, it must also be verified
        against the specific detected sound labels before firing (see check_and_fire),
        so that "bus" and "subway" are not mixed up.

    Examples::
        "When I get on the bus, remind me to call mom" -> (TRANSIT, "call mom", "bus")
        "When I get to the office remind me to send the email" -> (OFFICE, "send the email", None)
        "I went for a run today" -> (None, "", None)
    """
    text_lower = text.lower()

    # Must contain both an arrival phrase and a reminder phrase
    has_arrival = any(p in text_lower for p in _SCENE_ARRIVAL_PATTERNS)
    has_remind  = any(p in text_lower for p in _TRIGGER_PATTERNS)
    if not (has_arrival and has_remind):
        return None, "", None

    # Identify the scene
    matched_scene: SceneTag | None = None
    required_label: str | None = None
    for alias, (scene, label) in _SCENE_ALIASES.items():
        if alias in text_lower:
            matched_scene = scene
            required_label = label
            break

    if matched_scene is None:
        return None, "", None

    # Extract the reminder text: the part after "remind me"
    for kw in _TRIGGER_PATTERNS:
        idx = text_lower.find(kw)
        if idx != -1:
            message = text[idx + len(kw):].strip()
            if message:
                return matched_scene, message, required_label

    return matched_scene, text, required_label  # use the whole sentence as the reminder text


# ── Scene change checker ───────────────────────────────────────────────────────

@dataclass
class TriggerFireResult:
    scene: str
    fired: list[SceneTrigger]   # reminders fired this time
    scene_changed: bool


def check_and_fire(
    store: SceneTriggerStore,
    user_id: str,
    new_scene: str,
    raw_labels: list[str] | None = None,
) -> TriggerFireResult:
    """Check whether the new scene fires any pending reminders; mark fired ones as fired.

    Called at the end of each Ingest with the currently detected scene_tag and the
    specific AudioSet labels that were hit when deciding that scene (raw_labels,
    e.g. ["Bus", "Vehicle"]).

    When trigger.required_label is set (e.g. "bus"), a substring match must be found in
    raw_labels before it actually fires, so "remind me on the bus" is not wrongly fired on
    the subway (subway and bus are both coarsely transit, but the specific labels differ).
    Triggers without required_label (e.g. a generic "commute") skip this check and fire on
    a coarse scene match, as before.
    """
    last_scene = store.get_last_scene(user_id)
    scene_changed = (last_scene != new_scene)

    raw_labels_lower = [l.lower() for l in (raw_labels or [])]

    fired: list[SceneTrigger] = []
    if scene_changed and new_scene not in (SceneTag.UNKNOWN.value, SceneTag.QUIET.value):
        pending = store.get_pending(user_id, new_scene)
        for trigger in pending:
            if trigger.required_label and not any(
                trigger.required_label in rl for rl in raw_labels_lower
            ):
                continue  # specific scene did not match (e.g. wants "bus" but detected "train"); do not fire yet
            store.fire(trigger.id)
            fired.append(trigger)
            print(f"  [scene_trigger] 🔔 {trigger.message} (scene={new_scene})", flush=True)

    store.update_scene(user_id, new_scene)
    return TriggerFireResult(scene=new_scene, fired=fired, scene_changed=scene_changed)
