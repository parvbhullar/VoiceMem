"""
Capped cumulative average + adaptive multi-centroid (minimal demo)
==================================================================
Shows only one thing, "how a single person's profile is updated", and marks clearly where the two mechanisms mesh:
  · Capped cumulative average: with few samples it is a true cumulative average (stable, denoising); once full it degrades into an EMA (can adapt to change)
  · Adaptive multi-centroid: a single centroid while samples are few; only when a new sample is not similar enough to any existing centroid is a new sub-centroid split off
  · Quality weighting: the foundation throughout; bad samples get small weights
Dependencies: numpy
"""

import numpy as np


def l2norm(v):
    n = np.linalg.norm(v)
    return v / n if n else v


class SubCentroid:
    """A sub-centroid = the representative vector of this person under one state/channel."""
    def __init__(self, vec, w):
        self.vec = l2norm(vec)   # unit vector
        self.w = w               # cumulative weight (= effective sample count, gets capped)


class Profile:
    # ========== Parameters ==========
    def __init__(self, w_max=40.0, t_split=0.55, k_max=8):
        self.w_max = w_max        # weight cap: once full, single-centroid updates degrade into an EMA
        self.t_split = t_split    # split threshold: if a new sample vs. nearest centroid < this, start a new sub-centroid
        self.k_max = k_max        # max number of sub-centroids
        self.subs = []            # list of sub-centroids (empty = no samples yet)

    # ========== Mechanism A: capped cumulative average (how a single centroid is updated) ==========
    def _update_one(self, sub, x, w):
        """Quality-weighted cumulative average + renormalisation in cosine space + weight cap.
        Key: while W is not full it is a true cumulative average (new sample weight = w / (W+w), shrinking as W grows -> stable);
              once W is capped, the old weight stops growing and the new sample's share is fixed -> equivalent to an EMA (keeps up with change)."""
        sub.vec = l2norm(sub.vec * sub.w + x * w)
        sub.w = min(sub.w + w, self.w_max)        # <-- the cap is on this line; it turns the cumulative average into an EMA

    # ========== Mechanism B: adaptive multi-centroid (merge into an old centroid, or split a new one) ==========
    def update(self, x, quality=1.0):
        x = l2norm(x)
        w = float(quality)                         # quality weighting: the foundation

        # First sample -> directly create the first sub-centroid (single centroid at this point)
        if not self.subs:
            self.subs.append(SubCentroid(x, w))
            return

        # Find the most similar sub-centroid
        sims = [float(np.dot(x, s.vec)) for s in self.subs]
        j = int(np.argmax(sims))

        if sims[j] >= self.t_split:
            self._update_one(self.subs[j], x, w)   # similar enough -> merge in (mechanism A)
        else:
            self.subs.append(SubCentroid(x, w))    # none similar enough -> split off a new sub-centroid (a new state appeared)

        # Too many sub-centroids -> merge the two closest to bound the size
        if len(self.subs) > self.k_max:
            self._merge_closest()

    def _merge_closest(self):
        best = (-1.0, 0, 1)
        for i in range(len(self.subs)):
            for k in range(i + 1, len(self.subs)):
                s = float(np.dot(self.subs[i].vec, self.subs[k].vec))
                if s > best[0]:
                    best = (s, i, k)
        _, i, k = best
        a, b = self.subs[i], self.subs[k]
        merged = SubCentroid(a.vec * a.w + b.vec * b.w, min(a.w + b.w, self.w_max))
        self.subs = [s for idx, s in enumerate(self.subs) if idx not in (i, k)] + [merged]

    # ========== Mechanism C: merge nearby sub-centroids (clean up early fragments / self-clean after convergence) ==========
    def consolidate(self):
        """Merge any two sub-centroids whose similarity is >= t_split.
        Unimodal: orphan centroids are merged back once the main centroid converges -> collapses to 1.
        Bimodal: the two centroids are nearly orthogonal and fail the condition -> 2 are kept."""
        changed = True
        while changed and len(self.subs) > 1:
            changed = False
            for i in range(len(self.subs)):
                for k in range(i + 1, len(self.subs)):
                    if np.dot(self.subs[i].vec, self.subs[k].vec) >= self.t_split:
                        a, b = self.subs[i], self.subs[k]
                        merged = SubCentroid(a.vec * a.w + b.vec * b.w,
                                             min(a.w + b.w, self.w_max))
                        self.subs = [s for idx, s in enumerate(self.subs)
                                     if idx not in (i, k)] + [merged]
                        changed = True
                        break
                if changed:
                    break

    # ========== Matching: take the best match over all sub-centroids (preserves multimodality) ==========
    def score(self, x):
        x = l2norm(x)
        return max(float(np.dot(x, s.vec)) for s in self.subs) if self.subs else -1.0


# ============================================================
# Self-test: demonstrates "adaptive" -- unimodal grows only one centroid; bimodal splits into two automatically
# ============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    D = 64
    voiceA = l2norm(rng.standard_normal(D))           # normal voice
    voiceA_cold = l2norm(rng.standard_normal(D))      # voice with a cold (another mode)

    def noisy(v, s=0.1):
        return l2norm(v + rng.standard_normal(D) * s)

    # Scenario 1: feed only the normal voice (unimodal) -> should grow only 1 sub-centroid
    p1 = Profile()
    for _ in range(50):
        p1.update(noisy(voiceA), quality=rng.uniform(0.6, 1.0))
    print(f"Unimodal scenario (before cleanup): sub-centroids = {len(p1.subs)}")
    p1.consolidate()                                   # self-clean: merge orphan fragments back
    fit1 = float(np.dot(p1.subs[0].vec, voiceA))
    print(f"Unimodal scenario (after cleanup): sub-centroids = {len(p1.subs)} (expected 1), "
          f"fit with true voice = {fit1:.3f}, cumulative weight = {p1.subs[0].w:.1f} (capped at 40)")

    # Scenario 2: feed a mix of normal + cold voices (bimodal) -> should split into 2 sub-centroids automatically
    p2 = Profile()
    for _ in range(60):
        v = voiceA if rng.random() < 0.5 else voiceA_cold
        p2.update(noisy(v), quality=rng.uniform(0.6, 1.0))
    p2.consolidate()                                   # bimodal: the two centroids are nearly orthogonal, not merged by mistake
    print(f"\nBimodal scenario: sub-centroids = {len(p2.subs)} (expected 2)")
    for s in p2.subs:
        f_norm = float(np.dot(s.vec, voiceA))
        f_cold = float(np.dot(s.vec, voiceA_cold))
        which = "normal voice" if f_norm > f_cold else "cold voice"
        print(f"  sub-centroid -> closer to [{which}] (normal fit {f_norm:.2f} / cold fit {f_cold:.2f})")

    print("\nConclusion: the same code collapses unimodal data into 1 centroid (most accurate) and splits bimodal data into 2 (preserving states) -- "
          "no need to specify the number of clusters by hand.")
