"""Multi-label classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support


def bootstrap_ci(Y: np.ndarray, Yhat: np.ndarray, average: str = "micro", n_boot: int = 1000,
                 seed: int = 0, Yhat_ref: np.ndarray | None = None) -> list[float]:
    """Seeded percentile bootstrap 95% CI over test descriptions for an F1 score. With `Yhat_ref`,
    the CI is for the paired difference F1(Yhat) - F1(Yhat_ref) on the same resamples."""
    rng = np.random.default_rng(seed)
    n, vals = len(Y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = f1_score(Y[idx], Yhat[idx], average=average, zero_division=0)
        if Yhat_ref is not None:
            v -= f1_score(Y[idx], Yhat_ref[idx], average=average, zero_division=0)
        vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def multilabel_report(Y: np.ndarray, Yhat: np.ndarray, conditions: list[str], P: np.ndarray | None = None,
                      seed: int = 0) -> dict:
    prec, rec, f1, sup = precision_recall_fscore_support(Y, Yhat, zero_division=0)
    multi = Y.sum(1) > 1
    out = {
        "micro_f1": f1_score(Y, Yhat, average="micro", zero_division=0),
        "macro_f1": f1_score(Y, Yhat, average="macro", zero_division=0),
        "micro_f1_ci95": bootstrap_ci(Y, Yhat, "micro", seed=seed),
        "macro_f1_ci95": bootstrap_ci(Y, Yhat, "macro", seed=seed),
        "hamming_loss": hamming_loss(Y, Yhat),
        "subset_accuracy": float((Y == Yhat).all(1).mean()),
        # recall of the co-occurring labels on descriptions that truly carry more than one condition
        "multi_label_recall": float((Yhat[multi] & Y[multi]).sum() / max(Y[multi].sum(), 1)) if multi.any() else None,
        "per_condition": {d: {"precision": prec[j], "recall": rec[j], "f1": f1[j], "support": int(sup[j])}
                          for j, d in enumerate(conditions)},
    }
    if P is not None:
        out["primary_hit_rate"] = float(Y[np.arange(len(Y)), P.argmax(1)].mean())
    return _round(out)


def _round(x, nd: int = 4):
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x
