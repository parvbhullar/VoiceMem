"""Vector utilities: cosine_sim, reused by core.py for semantic normalisation of slot labels, etc.

The old sub_slot mechanism (splitting via run_split_check / emerging new sub_slots via
emerge_slots, plus query routing route_sub_slot / write assignment assign_to_sub_slot) has
been removed entirely -- creating and routing new slots is now handled solely by
SubgraphManager (entity co-occurrence subgraph judgement).
"""

from __future__ import annotations

import math


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_sim(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)
