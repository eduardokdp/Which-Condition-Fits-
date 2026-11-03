"""Condition-name lexicon, name masking, and detection of self-reported diagnoses.

Two uses:
  * ``mask_condition_names`` replaces nosological names (ADHD, anxiety, depression, ...) with a
    neutral token so that the classifier must rely on symptom language, not on label names.
    Symptom words such as "anxious", "depressed", or "panic attack" are deliberately kept.
  * ``self_reported_conditions`` returns the conditions a writer states they have been diagnosed
    with or have ("I was diagnosed with ADHD", "I also have anxiety", "my depression ...").
    These statements provide the additional labels in the weak multi-label supervision.
"""

from __future__ import annotations

import re
from typing import Iterable

MASK_TOKEN = "_cond_"

# name patterns per condition: (regex, case_sensitive)
_NAMES: dict[str, list[tuple[str, bool]]] = {
    "adhd": [
        (r"attention[\s/-]*deficit(?:[\s/-]*hyperactivity)?(?:[\s/-]*disorder)?", False),
        (r"adhd", False),
        (r"ADD", True),  # only the upper-case acronym, never the verb "add"
    ],
    "anxiety": [
        (r"(?:generali[sz]ed\s+)?anxiety(?:\s+disorder)?", False),
        (r"panic\s+disorder", False),
        (r"gad", False),
    ],
    "depression": [
        (r"major\s+depressive\s+disorder", False),
        (r"(?:major\s+|clinical\s+)?depressive\s+disorder", False),
        (r"(?:major\s+|clinical\s+)?depression", False),
        (r"mdd", False),
    ],
}

_THIRD_PARTY = re.compile(
    r"\b(?:my|his|her|their|our)\s+(?:son|daughter|kid|kids|child|children|husband|wife|boyfriend|"
    r"girlfriend|partner|mom|mother|dad|father|brother|sister|friend|roommate|cousin|aunt|uncle|"
    r"teacher|boss|coworker)\b"
)
_NEGATION = re.compile(r"\b(?:not|never|no|without|don't|doesn't|didn't|isn't|wasn't|haven't|hasn't)\b|n't\b")
_HEDGE = re.compile(r"\b(?:think|maybe|might|wonder|wondering|if|whether|suspect|could|possibly|probably|undiagnosed|self[- ]diagnosed)\b")
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|me|im|ive)\b")
_DIAGNOSIS = re.compile(r"\b(?:diagnosed|dx'?d|diagnosis)\b")
_HAVE = re.compile(
    r"\bi\s+(?:also\s+|still\s+|really\s+|definitely\s+|actually\s+)?"
    r"(?:have|had|suffer\s+from|suffered\s+from|struggle\s+with|deal\s+with|live\s+with|got)\s+(?:\S+\s+){0,3}$"
)
_MY = re.compile(r"\bmy\s+(?:[a-z]+\s+){0,2}$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _compiled(condition: str) -> list[re.Pattern]:
    pats = _NAMES.get(condition, [(re.escape(condition), False)])
    out = []
    for pat, case_sensitive in pats:
        flags = 0 if case_sensitive else re.IGNORECASE
        out.append(re.compile(rf"(?<![\w/]){pat}(?!\w)", flags))
    return out


_CACHE: dict[str, list[re.Pattern]] = {}


def name_patterns(condition: str) -> list[re.Pattern]:
    if condition not in _CACHE:
        _CACHE[condition] = _compiled(condition)
    return _CACHE[condition]


def mask_condition_names(text: str, conditions: Iterable[str]) -> str:
    """Replace every condition name (and r/<name> mentions) by MASK_TOKEN."""
    text = re.sub(r"\br/", " ", text)
    for cond in conditions:
        for pat in name_patterns(cond):
            text = pat.sub(MASK_TOKEN, text)
    return text


def _is_self_report(prefix: str) -> bool:
    """Decide from the sentence text before a condition name whether it is a first-person report."""
    p = prefix.lower()
    if _THIRD_PARTY.search(p) or _NEGATION.search(p) or _HEDGE.search(p):
        return False
    if _DIAGNOSIS.search(p) and _FIRST_PERSON.search(p):
        return True
    if _HAVE.search(p):
        return True
    if _MY.search(p):
        return True
    return False


def self_reported_conditions(text: str, conditions: Iterable[str]) -> set[str]:
    """Conditions the writer explicitly reports having (high-precision patterns)."""
    found: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence:
            continue
        for cond in conditions:
            if cond in found:
                continue
            for pat in name_patterns(cond):
                m = pat.search(sentence)
                if m and _is_self_report(sentence[: m.start()]):
                    found.add(cond)
                    break
    return found
