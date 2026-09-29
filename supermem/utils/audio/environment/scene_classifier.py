"""SceneClassifier — maps AudioSet labels (from the AST detector) to high-level scene tags.

scene_tag values:
    office   — keyboards, printers, air conditioning, office noise
    outdoor  — wind, rain, birdsong, traffic noise
    transit  — public transport: bus/subway/train/plane
    café     — restaurant/café: cutlery, background voices, coffee machine
    meeting  — multi-person meeting: echo, reverb, multiple voices
    home     — household: TV/washing machine/cooking
    quiet    — quiet: almost no background sound
    unknown  — cannot be determined

Usage::
    from supermem.utils.audio.environment.scene_classifier import classify_scene, SceneTag
    tag, conf = classify_scene([("Computer keyboard", 0.72), ("Typing", 0.55)])
    # → SceneTag.OFFICE, 0.635
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SceneTag(str, Enum):
    OFFICE  = "office"
    OUTDOOR = "outdoor"
    TRANSIT = "transit"
    CAFE    = "café"
    MEETING = "meeting"
    HOME    = "home"
    QUIET   = "quiet"
    UNKNOWN = "unknown"


# AudioSet label keywords -> scene_tag mapping (all keywords lowercase)
_SCENE_KEYWORDS: dict[SceneTag, list[str]] = {
    SceneTag.OFFICE: [
        "keyboard", "typing", "typewriter", "printer", "photocopier",
        "air conditioning", "ventilation fan", "computer", "office",
        "mechanical fan", "white noise",
    ],
    SceneTag.OUTDOOR: [
        "wind", "rain", "thunder", "raindrop", "drizzle",
        "bird", "chirp", "tweet", "crow", "owl", "insect", "cricket",
        "stream", "river", "waterfall", "ocean", "wave",
        "traffic", "road", "highway", "nature",
    ],
    SceneTag.TRANSIT: [
        "bus", "train", "subway", "metro", "rail", "railroad",
        "aircraft", "airplane", "helicopter", "engine", "motor",
        "vehicle", "car", "truck", "automobile",
        "horn", "siren", "emergency vehicle",
    ],
    SceneTag.CAFE: [
        "restaurant", "cafe", "coffee", "espresso", "cutlery",
        "silverware", "dishes", "clatter", "glass", "clink",
        "background music", "crowd", "chatter", "babble",
    ],
    SceneTag.MEETING: [
        "echo", "reverberation", "reverb", "conference",
        "auditorium", "classroom", "lecture",
    ],
    SceneTag.HOME: [
        "television", "tv set", "washing machine", "microwave",
        "kettle", "boiling", "frying", "cooking", "vacuum cleaner",
        "door", "sink", "water tap", "refrigerator", "dishwasher",
        "toilet", "shower",
    ],
}

# Precomputed label -> scene lookup table (flattens the lists above)
_LABEL_TO_SCENE: dict[str, SceneTag] = {}
for _scene, _keywords in _SCENE_KEYWORDS.items():
    for _kw in _keywords:
        _LABEL_TO_SCENE[_kw] = _scene


@dataclass
class SceneResult:
    tag: SceneTag
    confidence: float
    raw_matches: list[tuple[str, float]]  # [(label, score), ...]


def classify_scene(
    audioset_results: list[tuple[str, float]],
    quiet_threshold: float = 0.10,
    min_evidence: float = 0.35,
    min_margin: float = 0.10,
) -> SceneResult:
    """AudioSet (label, score) list -> SceneResult.

    Args:
        audioset_results:  list of (label, score) pairs returned by ASTEnvironmentDetector.
                        An empty list returns quiet (a quiet environment was detected).
        quiet_threshold: if every score is below this value, the scene is judged quiet.
    """
    if not audioset_results:
        return SceneResult(tag=SceneTag.QUIET, confidence=1.0, raw_matches=[])

    max_score = max(s for _, s in audioset_results)
    if max_score < quiet_threshold:
        return SceneResult(tag=SceneTag.QUIET, confidence=1.0 - max_score, raw_matches=[])

    # Accumulate the scores of matching labels per scene
    scene_scores: dict[SceneTag, float] = {}
    matches_by_scene: dict[SceneTag, list[tuple[str, float]]] = {}

    for label, score in audioset_results:
        label_lower = label.lower()
        matched_scene: SceneTag | None = None

        # Keyword match (substring match, exact equality not required)
        for kw, scene in _LABEL_TO_SCENE.items():
            if kw in label_lower or label_lower in kw:
                matched_scene = scene
                break

        if matched_scene is not None:
            scene_scores[matched_scene] = scene_scores.get(matched_scene, 0.0) + score
            matches_by_scene.setdefault(matched_scene, []).append((label, score))

    if not scene_scores:
        return SceneResult(tag=SceneTag.UNKNOWN, confidence=0.0, raw_matches=audioset_results)

    ranked = sorted(scene_scores, key=scene_scores.__getitem__, reverse=True)
    best_scene = ranked[0]
    best_evidence = scene_scores[best_scene]
    runner_up = scene_scores[ranked[1]] if len(ranked) > 1 else 0.0

    # The original implementation reported confidence only as "share among mapped labels",
    # so a random 0.15 label that happened to be the only match became 1.00 confidence.
    # Speech-dominated clips with weak background trigger this a lot. Here both the
    # absolute evidence and the margin between classes are considered: with insufficient
    # or conflicting evidence we explicitly return unknown instead of inventing a scene.
    if best_evidence < min_evidence or best_evidence - runner_up < min_margin:
        return SceneResult(tag=SceneTag.UNKNOWN, confidence=round(best_evidence, 3), raw_matches=audioset_results)

    total = sum(scene_scores.values())
    dominance = best_evidence / total if total > 0 else 0.0
    confidence = best_evidence * dominance

    return SceneResult(
        tag=best_scene,
        confidence=round(confidence, 3),
        raw_matches=matches_by_scene.get(best_scene, []),
    )


# ── Response style hints ─────────────────────────────────────────────────────────

SCENE_RESPONSE_STYLE: dict[SceneTag, str] = {
    SceneTag.OFFICE:  "The user is currently in an office; give a concise, professional reply of no more than 3 sentences.",
    SceneTag.OUTDOOR: "The user is currently outdoors; give a short, easy-to-remember reply of 1-2 sentences.",
    SceneTag.TRANSIT: "The user is currently commuting; reply as briefly as possible, ideally in one sentence.",
    SceneTag.CAFE:    "The user is currently in a café and it is fairly noisy; keep the reply short.",
    SceneTag.MEETING: "The user is currently in a meeting; reply minimally or ask whether to talk later.",
    SceneTag.HOME:    "The user is currently at home and it is quiet; the reply can be somewhat more detailed.",
    SceneTag.QUIET:   "The environment is quiet; reply normally with as much detail as needed.",
    SceneTag.UNKNOWN: "",
}


def scene_to_response_directive(scene: SceneTag) -> str:
    """Return the scene response-style hint injected into rb_directive."""
    return SCENE_RESPONSE_STYLE.get(scene, "")


# ── Scene-bound memory: infer the scene from query text (audiomem 2.1) ────────────────────────

# Scene keywords mentioned in the user's question -> SceneTag. At Ingest time each memory
# is tagged scene:<tag> (coarse-grained, see the upsert_memory_tags call in core.py Ingest),
# so here we only need to infer the same coarse SceneTag to match against; there is no
# need for scene_trigger.py's required_label that narrows down to a specific vehicle.
_QUERY_SCENE_KEYWORDS: dict[SceneTag, list[str]] = {
    SceneTag.TRANSIT: [
        "on the bus", "on the subway", "on the metro", "on the train", "on the plane",
        "commute", "commuting", "in the car", "on the way to work", "on my way home",
    ],
    SceneTag.OFFICE: ["at the office", "in the office", "at work", "at my desk"],
    SceneTag.HOME:   ["at home", "back home"],
    SceneTag.CAFE:   ["coffee shop", "starbucks", "café", "cafe"],
    SceneTag.MEETING: ["meeting", "conference room"],
    SceneTag.OUTDOOR: ["outdoors", "outside", "on a walk", "going out"],
    SceneTag.QUIET:  ["somewhere quiet", "by myself"],
}


def infer_scene_from_text(text: str) -> SceneTag | None:
    """Infer the scene intent from the user's query, used to narrow retrieval by scene.

    For example "that thing I told you on the bus" -> SceneTag.TRANSIT; Search() uses it
    as scene_filter to match the scene:<tag> tags applied at Ingest. Keyword matching only;
    returns None on no match (no narrowing, falls back to full retrieval).
    """
    text = text.lower()
    for scene, keywords in _QUERY_SCENE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return scene
    return None
