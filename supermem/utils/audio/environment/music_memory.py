"""MusicMemoryStore — background music / humming recognition memory (audiomem 2.5).

Reuses the adaptive multi-centroid mechanism from supermem/voiceprint/adaptive_centroid.py,
but this time clustering AST ambient-sound embeddings instead of voiceprints: the same
background music/humming heard repeatedly is recognised as a "familiar tune" rather than
being stored as something new every time.

Unlike the three-level voiceprint decision (match/candidate/new, see voiceprint_store.py),
there is no "wrong person, needs manual confirmation" consequence here: misidentifying a
song does not corrupt the profile or hurt the user, so only two levels are used:
match (merge into a known tune) / new (create a new tune profile).

match_threshold=0.80 is an empirical default not calibrated on real data (unlike the
voiceprint side, which had an EER analysis on MagicData). AST embeddings are intermediate
representations of a classifier, so sounds of the same class (e.g. "all piano pieces")
are already fairly similar; the threshold must be on the high side to separate "the same
song" from "the same kind of music". It should be recalibrated once real repeated-humming
data is available.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from supermem.utils.audio.voiceprint import Profile, SubCentroid, l2norm


@dataclass
class TuneIdentifyResult:
    tune_id: str
    score: float
    action: str        # "match" | "new"
    heard_count: int   # cumulative times this tune was recognised (including this one)


class MusicMemoryStore:
    """Maintains the tune_id -> Profile library of "familiar tunes". Thread-safe."""

    def __init__(self, store_dir: Path, match_threshold: float = 0.80):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._match_thr = match_threshold
        self._lock = threading.Lock()
        self._profiles: dict[str, Profile] = {}
        # tune_id → {labels, heard_count, created_at}
        self._meta: dict[str, dict] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _meta_path(self) -> Path:
        return self._dir / "music_meta.json"

    def _profile_path(self, tune_id: str) -> Path:
        return self._dir / f"profile_{tune_id}.npz"

    def _load(self) -> None:
        if not self._meta_path().exists():
            return
        data = json.loads(self._meta_path().read_text(encoding="utf-8"))
        self._meta = data.get("tunes", {})
        for tid in self._meta:
            path = self._profile_path(tid)
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
            self._profiles[tid] = prof

    def _save(self) -> None:
        data = {"tunes": self._meta}
        self._meta_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for tid, prof in self._profiles.items():
            if not prof.subs:
                continue
            vecs = np.stack([s.vec for s in prof.subs])
            ws = np.array([s.w for s in prof.subs])
            np.savez(
                self._profile_path(tid),
                vecs=vecs, ws=ws,
                w_max=np.array([prof.w_max]),
                t_split=np.array([prof.t_split]),
            )

    # ── Main API ─────────────────────────────────────────────────────────────

    def identify(self, vec: np.ndarray, labels: list[str] | None = None) -> TuneIdentifyResult:
        """Given the AST embedding of this music/humming clip, return the identification result (match/new)."""
        vec = l2norm(np.asarray(vec, dtype=np.float64))
        with self._lock:
            if not self._profiles:
                tid = self._new_tune(labels)
                self._profiles[tid].update(vec, quality=1.0)
                self._meta[tid]["heard_count"] = 1
                self._save()
                return TuneIdentifyResult(tid, 1.0, "new", 1)

            scores = {tid: prof.score(vec) for tid, prof in self._profiles.items()}
            best_tid = max(scores, key=scores.__getitem__)
            best_score = scores[best_tid]

            if best_score >= self._match_thr:
                self._profiles[best_tid].update(vec, quality=float(best_score))
                self._meta[best_tid]["heard_count"] += 1
                self._save()
                return TuneIdentifyResult(
                    best_tid, best_score, "match", self._meta[best_tid]["heard_count"]
                )

            tid = self._new_tune(labels)
            self._profiles[tid].update(vec, quality=1.0)
            self._meta[tid]["heard_count"] = 1
            self._save()
            return TuneIdentifyResult(tid, best_score, "new", 1)

    def _new_tune(self, labels: list[str] | None) -> str:
        tid = f"tune_{uuid.uuid4().hex[:8]}"
        self._profiles[tid] = Profile()
        self._meta[tid] = {
            "labels": labels or [],
            "heard_count": 0,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return tid

    # ── Helpers ─────────────────────────────────────────────────────────────

    def get_meta(self, tune_id: str) -> dict:
        return dict(self._meta.get(tune_id, {}))

    def list_tunes(self) -> list[dict]:
        with self._lock:
            return [{"tune_id": tid, **self._meta.get(tid, {})} for tid in self._profiles]
