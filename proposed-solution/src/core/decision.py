"""Per-condition thresholds and the primary/secondary decision rule.

    d*   = argmax_d P(d | s)                                 (primary condition, always reported)
    D_2  = { d != d* : P(d | s) >= tau_d }                    (secondary conditions)
    tau_d = argmax_{tau in grid} F1_d(tau) on the validation split
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    n = int(round((stop - start) / step)) + 1
    return np.round(start + step * np.arange(n), 6)


def tune_thresholds(P: np.ndarray, Y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Pick, per label, the grid value with the best F1; ties go to the value closest to 0.5."""
    taus = np.empty(P.shape[1])
    for j in range(P.shape[1]):
        scores = np.array([f1_score(Y[:, j], (P[:, j] >= t).astype(int), zero_division=0) for t in grid])
        best = np.flatnonzero(np.isclose(scores, scores.max()))
        taus[j] = grid[best[np.argmin(np.abs(grid[best] - 0.5))]]
    return taus


def decide(p: np.ndarray, taus: np.ndarray) -> tuple[int, list[int]]:
    """Return the primary label index and the secondary label indices (sorted by probability)."""
    primary = int(np.argmax(p))  # np.argmax returns the first maximum: ties broken by order of D
    secondary = [int(j) for j in np.argsort(-p, kind="stable") if j != primary and p[j] >= taus[j]]
    return primary, secondary


def to_label_matrix(P: np.ndarray, taus: np.ndarray) -> np.ndarray:
    Yhat = np.zeros_like(P, dtype=int)
    for i, p in enumerate(P):
        primary, secondary = decide(p, taus)
        Yhat[i, [primary, *secondary]] = 1
    return Yhat
