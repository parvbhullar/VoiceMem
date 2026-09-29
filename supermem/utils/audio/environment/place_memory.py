"""PlaceMemoryStore — automatic clustering of familiar places (audiomem 2.11).

Almost the same structure as MusicMemoryStore (music_memory.py), reusing adaptive_centroid
for adaptive multi-centroid clustering, but this time clustering "specific places" instead
of "specific tunes": under the same coarse scene_tag (e.g. café), different actual cafés
have different background acoustics (reverb, noise floor, frequency response from the
building materials), and the AST ambient embedding can capture that difference. Repeat
visits to the same specific place are recognised as a "familiar place" instead of being
recorded as a new place each time. This is a prerequisite for audiomem 2.12 (proactively
hinting "you were here last time" in a familiar environment).

Unlike the three-level voiceprint decision, only two levels are used here: match (known
place) / new (new place). Misidentifying a place has none of the social consequences of
misidentifying a speaker, so candidates need no manual confirmation.

match_threshold=0.80 is, like MusicMemoryStore, an empirical default not calibrated on
real data (same reasoning: AST embeddings already give high similarity for "the same kind
of place", so the threshold must be on the high side to separate "the same specific place"
from "the same kind of place"). It should be recalibrated once real repeat-visit
recordings are available.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from supermem.utils.audio.voiceprint import Profile, SubCentroid, l2norm


@dataclass
class PlaceIdentifyResult:
    place_id: str
    score: float
    action: str                    # "match" | "new"
    visit_count: int               # cumulative visit count (including this one)
    previous_visit_at: str | None  # time of the previous recognised visit; None = first visit
    scene: str | None              # coarse scene tag associated with this place (if any)


class PlaceMemoryStore:
    """Maintains the place_id -> Profile library of "familiar places". Thread-safe."""

    def __init__(self, store_dir: Path, match_threshold: float = 0.80):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._match_thr = match_threshold
        self._lock = threading.Lock()
        self._profiles: dict[str, Profile] = {}
        # place_id → {scene, visit_count, first_seen, last_seen, created_at}
        self._meta: dict[str, dict] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _meta_path(self) -> Path:
        return self._dir / "place_meta.json"

    def _profile_path(self, place_id: str) -> Path:
        return self._dir / f"profile_{place_id}.npz"

    def _load(self) -> None:
        if not self._meta_path().exists():
            return
        data = json.loads(self._meta_path().read_text(encoding="utf-8"))
        self._meta = data.get("places", {})
        for pid in self._meta:
            path = self._profile_path(pid)
            if not path.exists():
                continue
            npz = np.load(path, allow_pickle=True)
            prof = Profile(
                w_max=float(npz.get("w_max", [40.0])[0]),
                t_split=float(npz.get("t_split", [0.55])[0]),
            )
            vecs = npz["vecs"]
            ws = npz["ws"]
            prof.subs = [SubCentroid(vecs[i], float(ws[i])) for i in range(len(ws))]
            self._profiles[pid] = prof

    def _save(self) -> None:
        data = {"places": self._meta}
        self._meta_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for pid, prof in self._profiles.items():
            if not prof.subs:
                continue
            vecs = np.stack([s.vec for s in prof.subs])
            ws = np.array([s.w for s in prof.subs])
            np.savez(
                self._profile_path(pid),
                vecs=vecs, ws=ws,
                w_max=np.array([prof.w_max]),
                t_split=np.array([prof.t_split]),
            )

    # ── Main API ─────────────────────────────────────────────────────────────

    def identify(
        self, vec: np.ndarray, scene: str | None = None, when: datetime | None = None,
    ) -> PlaceIdentifyResult:
        """Given the AST embedding of this recording, return the place identification result (match/new)."""
        vec = l2norm(np.asarray(vec, dtype=np.float64))
        now_str = (when or datetime.now()).isoformat()
        with self._lock:
            if not self._profiles:
                pid = self._new_place(scene, now_str)
                self._profiles[pid].update(vec, quality=1.0)
                self._meta[pid]["visit_count"] = 1
                self._save()
                return PlaceIdentifyResult(pid, 1.0, "new", 1, None, scene)

            scores = {pid: prof.score(vec) for pid, prof in self._profiles.items()}
            best_pid = max(scores, key=scores.__getitem__)
            best_score = scores[best_pid]

            if best_score >= self._match_thr:
                prev_last_seen = self._meta[best_pid].get("last_seen")
                self._profiles[best_pid].update(vec, quality=float(best_score))
                self._meta[best_pid]["visit_count"] += 1
                self._meta[best_pid]["last_seen"] = now_str
                self._save()
                return PlaceIdentifyResult(
                    best_pid, best_score, "match",
                    self._meta[best_pid]["visit_count"], prev_last_seen,
                    self._meta[best_pid].get("scene"),
                )

            pid = self._new_place(scene, now_str)
            self._profiles[pid].update(vec, quality=1.0)
            self._meta[pid]["visit_count"] = 1
            self._save()
            return PlaceIdentifyResult(pid, best_score, "new", 1, None, scene)

    def _new_place(self, scene: str | None, now_str: str) -> str:
        pid = f"place_{uuid.uuid4().hex[:8]}"
        self._profiles[pid] = Profile()
        self._meta[pid] = {
            "scene": scene,
            "visit_count": 0,
            "first_seen": now_str,
            "last_seen": now_str,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return pid

    # ── Helpers ─────────────────────────────────────────────────────────────

    def get_meta(self, place_id: str) -> dict:
        return dict(self._meta.get(place_id, {}))

    def list_places(self) -> list[dict]:
        with self._lock:
            return [{"place_id": pid, **self._meta.get(pid, {})} for pid in self._profiles]
