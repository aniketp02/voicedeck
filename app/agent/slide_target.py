"""
Normalize LLM slide targets when users speak in 1-based ordinals ("fifth slide")
but the model echoes the spoken number as if it were a 0-based index.
"""

from __future__ import annotations

import re

# "go to the fifth slide" / "5th slide"
_ORDINAL_SLIDE_RE = re.compile(
    r"\b(\d+)(?:st|nd|rd|th)\s+slide\b",
    re.IGNORECASE,
)
_SLIDE_NUM_RE = re.compile(r"\bslide\s+(\d+)\b", re.IGNORECASE)
_GO_ORDINAL_RE = re.compile(
    r"\b(?:go to|open|show|jump to)\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)\s+slide\b",
    re.IGNORECASE,
)

_WORD_ORDINAL_SLIDES = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
)
_WORD_TO_N = {w: i + 1 for i, w in enumerate(_WORD_ORDINAL_SLIDES)}

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_SLIDE_NUMBER_WORD_RE = re.compile(
    r"\bslide\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)


def _human_slide_numbers_in_transcript(transcript: str) -> list[int]:
    """1-based slide positions explicitly mentioned by the user."""
    t = transcript.lower()
    found: list[int] = []

    for m in _ORDINAL_SLIDE_RE.finditer(t):
        found.append(int(m.group(1)))
    for m in _SLIDE_NUM_RE.finditer(t):
        found.append(int(m.group(1)))
    for m in _GO_ORDINAL_RE.finditer(t):
        found.append(int(m.group(1)))
    for word, n in _WORD_TO_N.items():
        if re.search(rf"\b{word}\s+slide\b", t):
            found.append(n)
    for m in _SLIDE_NUMBER_WORD_RE.finditer(t):
        w = m.group(1).lower()
        if w in _NUMBER_WORDS:
            found.append(_NUMBER_WORDS[w])

    return found


def normalize_slide_target(transcript: str, target: int, num_slides: int) -> int:
    """
    If the model returns the same integer the user said aloud as a slide *number*
    (1..N) but the app expects 0-based indices, convert: target -> target - 1.

    Example: user says "fifth slide", model returns 5 → use index 4.
    Does not change targets when the user asked with 0-based wording or when
    numbers do not line up (model already returned a valid 0-based index).
    """
    if num_slides <= 0:
        return target
    tl = transcript.lower()
    if "zero-based" in tl or "0-based" in tl:
        return target

    nums = _human_slide_numbers_in_transcript(transcript)
    if not nums:
        return target

    # Prefer the last explicit slide reference ("… then go to slide 5").
    for n in reversed(nums):
        if n == target and 1 <= n <= num_slides:
            return n - 1
    return target
