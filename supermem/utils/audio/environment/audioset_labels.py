"""Shared AudioSet label constants used by the environmental-sound detectors.

Both the AST detector (``environment_detector_ast.py``) and the CLAP detector
key off the same AudioSet ontology, so the speech-class index set and the
music/abnormal-sound keyword lists live here instead of being duplicated or
imported from one detector module into another.
"""
from __future__ import annotations

# AudioSet label indices for speech/voice classes — excluded from environment output
_SPEECH_LABEL_INDICES = {
    0,    # Speech
    1,    # Male speech, man speaking
    2,    # Female speech, woman speaking
    3,    # Child speech, kid speaking
    4,    # Conversation
    5,    # Narration, monologue
    6,    # Babbling
    7,    # Speech synthesizer
    8,    # Shout
    9,    # Bellow
    10,   # Whoop
    11,   # Yell
    12,   # Children shouting
    13,   # Screaming
    14,   # Whispering
    15,   # Laughter
    16,   # Baby laughter
    17,   # Giggling
    18,   # Snicker
    19,   # Belly laugh
    20,   # Chuckle, chortle
    21,   # Crying, sobbing
    22,   # Baby cry, infant cry
    23,   # Whimper
    24,   # Wail, moan
    25,   # Sigh
    26,   # Singing
    27,   # Choir
    28,   # Yodeling
    29,   # Chant
    30,   # Mantra
    31,   # Male singing
    32,   # Female singing
    33,   # Child singing
    34,   # Synthetic singing
    35,   # Rapping
    36,   # Humming
    37,   # Groan
    38,   # Grunt
    39,   # Whistling
    40,   # Breathing
    41,   # Wheeze
    42,   # Snoring
    43,   # Gasp
    44,   # Pant
    45,   # Snort
    46,   # Cough
    47,   # Throat clearing
    48,   # Sneeze
    49,   # Sniff
}

# Music/humming label keywords (lowercase). _SPEECH_LABEL_INDICES above filters
# Singing/Humming/Whistling out as "speech-like" (so the user's own voice is not
# mistaken for ambient sound), but background music/humming memory (audiomem 2.5)
# needs exactly these labels, so they get a separate keyword pass that is not
# affected by the scene-label filter.
_MUSIC_KEYWORDS = [
    "music", "singing", "humming", "whistling", "song", "singer",
    "musical instrument", "chant", "yodeling",
]

# Abnormal ambient-sound keywords (lowercase): breaking/alarms/screams (audiomem 2.6).
# Also bypasses the _SPEECH_LABEL_INDICES filter (Screaming/Yell would otherwise be
# dropped as "speech-like").
_ABNORMAL_KEYWORDS = [
    "glass", "shatter", "smash", "crash", "explosion", "gunshot", "gunfire",
    "alarm", "siren", "scream", "yell",
]
