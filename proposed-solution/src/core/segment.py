"""Phrase segmentation sigma(s) = <p_1, ..., p_n>.

A description is split at sentence boundaries, then at clause boundaries where a comma or a
coordinating conjunction is followed by a new first-person clause ("..., I feel restless and I've
been ..."). Lists inside one clause ("tired, sad, and empty") are kept together, so each phrase is a
self-contained symptom statement that can be shown to the user as evidence.
"""

from __future__ import annotations

import re

_SENT = re.compile(r"(?<=[.!?])\s+|\n+|\s*;\s*")
_SUBJ = r"(?:i|i'm|im|i've|ive|i'd|i'll|my|me|it's|everything|nothing)\b"
_CLAUSE = re.compile(
    rf"\s*,?\s+(?:and|but|so|because|or|plus)\s+(?={_SUBJ})|\s*,\s+(?={_SUBJ})",
    flags=re.IGNORECASE,
)
_STRIP = " \t,.;:!?-–—\"'()"


def segment(text: str, min_tokens: int = 2) -> list[str]:
    """Return the ordered list of phrases of a description (never empty for non-empty text)."""
    phrases: list[str] = []
    for sent in _SENT.split(text.strip()):
        for clause in _CLAUSE.split(sent):
            clause = clause.strip(_STRIP)
            if not clause:
                continue
            if phrases and len(clause.split()) < min_tokens:
                phrases[-1] = f"{phrases[-1]} {clause}"  # merge fragments into the previous phrase
            else:
                phrases.append(clause)
    if not phrases and text.strip():
        phrases = [text.strip()]
    return phrases


def join(phrases: list[str]) -> str:
    return ". ".join(phrases)
