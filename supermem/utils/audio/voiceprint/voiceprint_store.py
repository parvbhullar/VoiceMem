"""Voiceprint manager.

Three-way decision:
  score >= match_thr  → match: refine the profile directly
  cand_thr <= score < match_thr → candidate: first provisionally create a new profile, and also keep a
                                   candidate record; as the same person's voice keeps appearing, once the
                                   accumulated sub-centroids cross-score against the original profile above
                                   match_thr, find_resolvable_candidates() / merge_persons() automatically
                                   merge the two profiles back together
                                   (see core.py::_reconcile_speaker_candidates).
                                   If after the fork the original profile (top) stays with very few samples
                                   (<=2, meaning it lost to the fork every time afterwards and never got
                                   another chance to be updated by a real match) while the fork has already
                                   accumulated >=3 real samples on its own, the looser cand_thr is used
                                   instead -- otherwise such a candidate stays stuck forever on the original
                                   profile's single noisy sample and can never be reclaimed.
  score < cand_thr    → new: treat as a new person, create a new profile
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
class IdentifyResult:
    person_id: str   # match/new → real person_id; candidate → candidate_id
    score: float
    action: str      # "match" | "candidate" | "new"


class VoiceprintStore:
    """Maintains the person_id → Profile voiceprint store. Thread-safe."""

    def __init__(
        self,
        store_dir: Path,
        match_threshold: float = 0.50,
        candidate_threshold: float = 0.40,
        merge_threshold: float = 0.65,
    ):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._match_thr = match_threshold
        self._cand_thr = candidate_threshold
        # Used for automatic candidate reclamation/merging, deliberately higher than match_thr -- see the explanation in
        # find_resolvable_candidates: a merge is permanent and nearly irreversible, so the single-utterance match bar can't be used.
        self._merge_thr = merge_threshold
        self._lock = threading.Lock()
        self._profiles: dict[str, Profile] = {}
        # person_id → {obs_count, confidence, created_at}
        # Real-name binding does not live here: that is VoiceprintRegistry's job (voice_input.py),
        # stored in voiceprint_registry.json, with bind() triggered by a self-stated name such as "Hi, I am X".
        self._meta: dict[str, dict] = {}
        # candidate_id → {vec, top_match_person_id, score, context, ts}
        self._candidates: dict[str, dict] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────────

    def _meta_path(self) -> Path:
        return self._dir / "voiceprint_meta.json"

    def _profile_path(self, person_id: str) -> Path:
        return self._dir / f"profile_{person_id}.npz"

    def _load(self) -> None:
        if not self._meta_path().exists():
            return
        data = json.loads(self._meta_path().read_text(encoding="utf-8"))
        self._meta = data.get("persons", {})
        self._candidates = data.get("candidates", {})
        for pid in self._meta:
            path = self._profile_path(pid)
            if not path.exists():
                continue
            npz = np.load(path, allow_pickle=True)
            prof = Profile(
                w_max=float(npz.get("w_max", [40.0])[0]),
                t_split=float(npz.get("t_split", [0.55])[0]),
            )
            vecs = npz["vecs"]  # [k, D]
            ws   = npz["ws"]    # [k]
            prof.subs = [SubCentroid(vecs[i], float(ws[i])) for i in range(len(ws))]
            self._profiles[pid] = prof

    def _save(self) -> None:
        data = {"persons": self._meta, "candidates": self._candidates}
        self._meta_path().write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for pid, prof in self._profiles.items():
            if not prof.subs:
                continue
            vecs = np.stack([s.vec for s in prof.subs])
            ws   = np.array([s.w   for s in prof.subs])
            np.savez(
                self._profile_path(pid),
                vecs=vecs, ws=ws,
                w_max=np.array([prof.w_max]),
                t_split=np.array([prof.t_split]),
            )

    # ── Main interface ──────────────────────────────────────────────────────────────

    def identify(
        self, vec: np.ndarray, context: str = "", pinned_person_id: str | None = None,
    ) -> IdentifyResult:
        """Given a voiceprint vector, return an IdentifyResult.

        ``pinned_person_id``: once the caller (core.py) has established, within the same session, via a
        self-introduction "who said this utterance", it passes that person_id in. Reason: the split
        triggered by a self-introduction (``create_person``) builds a brand-new profile from only that one
        utterance's vector, a sample size of 1; yet every remaining utterance in the session still goes through
        the pure-voiceprint scoring below and picks the global top score -- which can almost never beat the other
        person's old profile with dozens of accumulated samples, so the newly split profile never gets another
        chance to be updated by a real match, making the split pointless (observed in the
        Nancy/Jennifer case: the split fired only once, on the self-introduction utterance; after that every one of
        Jennifer's utterances was assigned back to Nancy's profile, with scores even higher than when Nancy herself spoke).
        Here, as long as the pinned person's own score is still acceptable (>=cand_thr, not "completely
        dissimilar"), we prefer the person the session has already confirmed over the pure voiceprint top score --
        this breaks the "new profile is forever too thin" deadlock and also avoids continuing to update another
        person's profile with samples that don't belong to them.
        """
        vec = l2norm(np.asarray(vec, dtype=np.float64))
        with self._lock:
            if not self._profiles:
                pid = self._new_person()
                self._profiles[pid].update(vec, quality=1.0)
                self._meta[pid]["obs_count"] = 1
                self._save()
                return IdentifyResult(pid, 1.0, "new")

            scores = {pid: prof.score(vec) for pid, prof in self._profiles.items()}
            best_pid = max(scores, key=scores.__getitem__)
            best_score = scores[best_pid]

            if (
                pinned_person_id is not None
                and pinned_person_id in scores
                and pinned_person_id != best_pid
                and scores[pinned_person_id] >= self._cand_thr
            ):
                best_pid, best_score = pinned_person_id, scores[pinned_person_id]

            if best_score >= self._match_thr:
                self._profiles[best_pid].update(vec, quality=float(best_score))
                self._meta[best_pid]["obs_count"] += 1
                conf = self._meta[best_pid]["confidence"]
                self._meta[best_pid]["confidence"] = round(min(1.0, conf + 0.02), 4)
                self._save()
                return IdentifyResult(best_pid, best_score, "match")

            elif best_score >= self._cand_thr:
                # A candidate can no longer be returned as the person_id: the caller would treat it as an unstable
                # ID, and multi-person memories couldn't be attached. First create a separate voiceprint to guarantee
                # isolation; the candidate record remembers "this new voiceprint was nearly assigned to
                # top_match_person_id", and find_resolvable_candidates() checks it automatically later -- a single
                # voiceprint score on a short utterance is very noisy, but once the new voiceprint has accumulated a
                # few real matches of its own, scoring the two accumulated profiles against each other is far more
                # accurate than a single vector; only when they truly converge on the same person are they merged back
                # automatically, instead of waiting forever on the verify_candidate stub for a human/LLM confirmation nobody calls.
                pid = self._new_person()
                self._add_candidate(vec, best_pid, best_score, context, forked_person_id=pid)
                self._profiles[pid].update(vec, quality=1.0)
                self._meta[pid]["obs_count"] = 1
                self._save()
                return IdentifyResult(pid, best_score, "new")

            else:
                pid = self._new_person()
                self._profiles[pid].update(vec, quality=1.0)
                self._meta[pid]["obs_count"] = 1
                self._save()
                return IdentifyResult(pid, best_score, "new")

    def create_person(self, vec: np.ndarray, context: str = "") -> IdentifyResult:
        """Force-split the current voiceprint into a separate person, used on name conflicts."""
        vec = l2norm(np.asarray(vec, dtype=np.float64))
        with self._lock:
            pid = self._new_person()
            self._profiles[pid].update(vec, quality=1.0)
            self._meta[pid]["obs_count"] = 1
            self._meta[pid]["context"] = context[:120]
            self._save()
            return IdentifyResult(pid, 0.0, "new")

    def _new_person(self) -> str:
        pid = f"person_{uuid.uuid4().hex[:8]}"
        self._profiles[pid] = Profile()
        self._meta[pid] = {
            "obs_count": 0,
            "confidence": 0.5,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return pid

    def _add_candidate(
        self, vec: np.ndarray, top_match: str, score: float, context: str,
        forked_person_id: str,
    ) -> str:
        cid = f"cand_{uuid.uuid4().hex[:8]}"
        self._candidates[cid] = {
            "vec": vec.tolist(),
            "top_match_person_id": top_match,
            "forked_person_id": forked_person_id,
            "score": round(float(score), 4),
            "context": context,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()
        return cid

    # ── Automatic candidate reclamation ────────────────────────────────────────────────────────

    def find_resolvable_candidates(self, updated_pid: str) -> list[tuple[str, str, str, float]]:
        """Call right after ``updated_pid`` got a new real match: check whether any candidate record

        can now, thanks to this update, be judged "actually the same person". ``updated_pid`` may be the
        candidate record's ``top_match_person_id`` (the original voiceprint got really matched again), or the
        ``forked_person_id`` (the voiceprint split off back then accumulated a new real match of its own) --
        in both cases the two sides' latest profiles are scored against each other, rather than the single
        isolated old vector stored in the candidate record; with more samples the noise is naturally suppressed.

        Returns ``[(candidate_id, absorb_pid, into_pid, score), ...]``; read-only, modifies no
        state. Whether to actually merge and how to handle name conflicts is left to the caller (``core.py``
        holds the VoiceprintRegistry; this store knows nothing about names).
        """
        with self._lock:
            out: list[tuple[str, str, str, float]] = []
            for cid, c in self._candidates.items():
                top_pid = c.get("top_match_person_id")
                fork_pid = c.get("forked_person_id")
                if updated_pid not in (top_pid, fork_pid):
                    continue
                top_prof = self._profiles.get(top_pid)
                fork_prof = self._profiles.get(fork_pid)
                if top_prof is None or fork_prof is None or not top_prof.subs or not fork_prof.subs:
                    continue  # one side has already been merged away / deleted
                cross = max(
                    float(np.dot(l2norm(a.vec), l2norm(b.vec)))
                    for a in top_prof.subs for b in fork_prof.subs
                )
                # Normally merge_thr-level confidence is required -- higher than the single-utterance match_thr,
                # because a merge is a permanent, nearly irreversible operation (two profiles welded into one, names
                # overwriting each other), so it can't use the "does this one utterance sound alike" bar. We've observed
                # cross=0.542 (just barely over match_thr) being enough to permanently weld two genuinely different people
                # (Nancy/Jennifer, whose voices are hard to separate anyway) and skew the names, with no way to split them afterwards.
                #
                # But if top (the original profile hit when the fork happened) has stayed very thin
                # (obs_count<=2) while fork (the candidate profile split off) has already accumulated
                # at least 3 real samples on its own -- that means identify() assigned every later utterance to fork
                # (steadier score, more samples), and top has never again had a chance to be updated by a real match
                # since the fork; the noise of its 1-2 samples can never be diluted, and the cross score stays pinned
                # below merge_thr (Nancy case: max_cross stuck around 0.47). Only in this combination -- "top
                # structurally can't prove itself, fork already has real accumulation" -- do we switch to the looser
                # cand_thr, which is exactly the line for "worth creating a candidate, possibly the same person". The
                # relaxation is asymmetric and covers only this case, so it doesn't extend to the normal "candidate just
                # created, fork has only 1 sample" scenario that should keep waiting to accumulate, nor to the "both
                # sides have independently accumulated several samples" scenario, which is more likely two different people.
                top_obs = self._meta.get(top_pid, {}).get("obs_count", 0)
                fork_obs = self._meta.get(fork_pid, {}).get("obs_count", 0)
                thin_and_stuck = top_obs <= 2 and fork_obs >= 3
                threshold = self._cand_thr if thin_and_stuck else self._merge_thr
                if cross >= threshold:
                    out.append((cid, fork_pid, top_pid, cross))
            return out

    def merge_persons(self, absorb_pid: str, into_pid: str) -> None:
        """Merge ``absorb_pid``'s profile into ``into_pid``, delete ``absorb_pid``,
        and clear all candidate records pointing at either id (a conclusion has been reached; no need to wait for confirmation)."""
        with self._lock:
            absorb_prof = self._profiles.pop(absorb_pid, None)
            into_prof = self._profiles.get(into_pid)
            if absorb_prof is None or into_prof is None:
                return
            into_prof.subs.extend(absorb_prof.subs)
            into_prof.consolidate()
            while len(into_prof.subs) > into_prof.k_max:
                into_prof._merge_closest()

            absorb_meta = self._meta.pop(absorb_pid, {})
            into_meta = self._meta[into_pid]
            into_meta["obs_count"] = into_meta.get("obs_count", 0) + absorb_meta.get("obs_count", 0)
            into_meta["confidence"] = round(
                min(1.0, max(into_meta.get("confidence", 0.5), absorb_meta.get("confidence", 0.5)) + 0.02), 4
            )

            self._candidates = {
                cid: c for cid, c in self._candidates.items()
                if absorb_pid not in (c.get("top_match_person_id"), c.get("forked_person_id"))
                and into_pid not in (c.get("top_match_person_id"), c.get("forked_person_id"))
            }

            path = self._profile_path(absorb_pid)
            if path.exists():
                path.unlink()
            self._save()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def get_meta(self, person_id: str) -> dict:
        return dict(self._meta.get(person_id, {}))

    def list_persons(self) -> list[dict]:
        with self._lock:
            return [
                {"person_id": pid, **self._meta.get(pid, {})}
                for pid in self._profiles
            ]
