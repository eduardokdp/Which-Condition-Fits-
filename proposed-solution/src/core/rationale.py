"""Occlusion-based extractive rationales.

For phrase p_j of s and condition d, the contribution is the probability drop when p_j is erased:
    Delta_d(p_j) = P(d | s) - P(d | s \\ p_j).
The evidence set E_d keeps the phrases with the largest drops, at most m = max(min_phrases,
ceil(max_fraction * n)) of them, and only those with Delta_d(p_j) > delta. A phrase may support
several conditions; such shared phrases are exactly the overlapping symptoms the project targets.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from src.core.segment import join


def occlusion_deltas(phrases: list[str], proba: Callable[[list[str]], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return P(.|s) of shape (k,) and Delta of shape (n, k)."""
    full = join(phrases)
    variants = [join(phrases[:j] + phrases[j + 1:]) for j in range(len(phrases))]
    P = proba([full, *variants])
    return P[0], P[0][None, :] - P[1:]


def select_evidence(delta_col: np.ndarray, delta: float, max_fraction: float, min_phrases: int) -> list[int]:
    n = len(delta_col)
    m = max(min_phrases, math.ceil(max_fraction * n))
    order = np.argsort(-delta_col, kind="stable")
    return sorted(int(j) for j in order[:m] if delta_col[j] > delta)


def comprehensiveness(phrases, evidence, proba, label) -> float:
    """ERASER comprehensiveness: P(d|s) - P(d|s without E_d). Higher is more faithful."""
    rest = [p for j, p in enumerate(phrases) if j not in set(evidence)]
    P = proba([join(phrases), join(rest)])
    return float(P[0, label] - P[1, label])


def sufficiency(phrases, evidence, proba, label) -> float:
    """ERASER sufficiency: P(d|s) - P(d|E_d). Lower is more faithful."""
    kept = [p for j, p in enumerate(phrases) if j in set(evidence)]
    P = proba([join(phrases), join(kept)])
    return float(P[0, label] - P[1, label])
