"""SpeakerIdentity — parsing self-introductions.

When the user says "my name is Annie" / "I'm Annie", extract that self-stated name from the text;
the caller (core.py) is responsible for binding the current voiceprint's person_id to that name
(VoiceprintRegistry.bind), so the next time the same voiceprint appears the name need not be repeated --
ingest_voice_input() will automatically replace "Speaker N" with the bound name.
"""
from __future__ import annotations

import re

# Latin-alphabet names (e.g. "my name is Annie")
#
# The capture groups of the "my name is X" / "i'm X" English patterns used to be unconstrained, so
# "I'm having a great time" would treat "having" as a name -- in English the "I'm <verb/preposition>..."
# construction is extremely common, and since almost every word is followed by whitespace,
# "followed by whitespace" is no signal at all for where a name ends.
# Now we require: the captured word starts with a capital letter (real names are usually proper nouns and
# ASR transcripts generally keep the capitalisation), and it is immediately followed by punctuation/"and"/end
# (no further words may follow), which rules out cases like "I'm OK with that" that are capitalised but
# clearly not a name and continue the sentence.
_LATIN_PATTERNS = [
    re.compile(r"\b(?i:my name is)\s+([A-Z][A-Za-z0-9_-]{0,19})(?=[,.!?]|\s+and\b|$)"),
    re.compile(r"\b(?i:i'?m)\s+([A-Z][A-Za-z0-9_-]{0,19})(?=[,.!?]|\s+and\b|$)"),
    re.compile(r"\b(?i:i am)\s+([A-Z][A-Za-z0-9_-]{0,19})(?=[,.!?]|\s+and\b|$)"),
]

# The short English sentence ``I'm X.`` only reaches here when X is a sentence-final proper-noun form, but ASR
# may capitalise ordinary words. Exclude known states, actions and conversational words so that a single
# misrecognition doesn't pollute the memory attribution of that voiceprint from then on.
_LATIN_NON_NAMES = {
    "actually", "afraid", "alone", "angry", "anxious", "busy", "confused",
    "excited", "fine", "going", "happy", "heading", "here", "hungry",
    "leaving", "lost", "nervous", "okay", "ok", "ready", "sad", "sorry",
    "tired", "walking", "worried",
}


def parse_self_identification(text: str) -> str | None:
    """Extract a self-introduced name from a sentence; return None if none is found.

    Examples::
        "My name is Annie, work has been exhausting lately" -> "Annie"
        "I'm Annie."                                        -> "Annie"
        "I'm having a great time"                           -> None
        "I'm OK with that"                                  -> None (sentence continues, not a name)
        "I'm tired."                                        -> None (a state, not a name)
    """
    for pattern in _LATIN_PATTERNS:
        m = pattern.search(text)
        if m:
            name = m.group(1).strip()
            if name and name.casefold() not in _LATIN_NON_NAMES:
                return name
    return None
