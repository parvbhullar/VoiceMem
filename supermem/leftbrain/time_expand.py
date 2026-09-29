"""Expand relative time words in a query into absolute dates before retrieval.

Why this is needed: extraction normalizes "physical exam next Wednesday at 3pm" into
"Jiaqi will have a physical exam on August 26, 2026 (Wednesday) at 3pm" -- the store
holds absolute dates. But the user asks "what do I have next week?", and that sentence
contains **not a single absolute date**, so the vectors don't line up; in testing,
none of three next-week appointments could be retrieved::

    "what do I have next week?"       appointments hit 0/3
    "what am I doing on August 26?"   appointments hit 3/3

The only difference is the phrasing. So before retrieval we expand "next week" in place
into the dates of those seven days and append them to the query -- only once the vector
carries literals like August 26 can it reach that stored memory.

Only the **text sent to retrieval** is changed; what the user said is untouched and
nothing is written into memory.

    expand_relative_dates("what do I have next week?")
    -> "what do I have next week? (August 24, 2026 August 25, 2026 ... August 30, 2026)"

If no relative time word is recognized the query is returned as-is, at zero cost
(pure regex, no model).
"""
from __future__ import annotations

import re
from datetime import date, timedelta

#: relative time word -> (start offset, number of days). Offset is in days relative to "today".
#: Week-related offsets are computed in _resolve from today's weekday; None is a placeholder.
_SPANS: dict[str, tuple[int | None, int]] = {
    "day before yesterday": (-2, 1),
    "yesterday":            (-1, 1),
    "today":                (0, 1),
    "tonight":              (0, 1),
    "tomorrow":             (1, 1),
    "day after tomorrow":   (2, 1),
    "these few days":       (0, 3),
    "past few days":        (-3, 4),
    "last few days":        (-3, 4),
    "next few days":        (0, 4),
    "coming days":          (0, 4),
    # weeks: offset computed from today's weekday
    "last week":            (None, 7),
    "this week":            (None, 7),
    "next week":            (None, 7),
}

#: week-related words -> week offset relative to "this Monday"
_WEEK_OFFSET = {
    "last week": -1,
    "this week": 0,
    "next week": 1,
}

#: Longer phrases must match first, otherwise "day after tomorrow" would be cut down to "tomorrow".
_WORDS = sorted(_SPANS, key=len, reverse=True)
_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in _WORDS) + r")\b",
    re.I,
)

#: Maximum number of dates expanded at once. Something like "the last three months"
#: would expand into a hundred dates, diluting the query's own meaning and making
#: retrieval worse.
_MAX_DAYS = 8


_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _normalize(word: str) -> str:
    return " ".join(word.lower().split())


def _resolve(word: str, today: date) -> list[date]:
    """Which days a relative time word covers."""
    if word in _WEEK_OFFSET:
        monday = today - timedelta(days=today.weekday())      # this Monday
        start = monday + timedelta(weeks=_WEEK_OFFSET[word])
        return [start + timedelta(days=i) for i in range(7)]
    offset, days = _SPANS[word]
    start = today + timedelta(days=offset or 0)
    return [start + timedelta(days=i) for i in range(days)]


def format_date(d: date) -> str:
    """"August 26, 2026" -- the same format extraction normalizes dates to."""
    return f"{_MONTHS[d.month - 1]} {d.day}, {d.year}"


def expand_relative_dates(query: str, today: date | None = None) -> str:
    """Append the absolute dates the query refers to. Returned as-is if there is no relative time word."""
    if not query:
        return query
    words = [_normalize(w) for w in _RE.findall(query)]
    if not words:
        return query

    today = today or date.today()
    days: list[date] = []
    for word in words:
        for d in _resolve(word, today):
            if d not in days:
                days.append(d)
    if not days or len(days) > _MAX_DAYS:
        return query

    days.sort()
    # Written as "August 26, 2026" to line up with how extraction normalizes dates --
    # with a different format the expansion would be wasted.
    stamps = " ".join(format_date(d) for d in days)
    return f"{query} ({stamps})"
