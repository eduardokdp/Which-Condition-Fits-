"""Milestone 4 metrics: macro-F1 (primary), micro-F1, ROC-AUC, precision, recall, Hamming loss,
recall on multi-condition posts, confusion matrices, and seeded (paired) bootstrap intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support, roc_auc_score

from src.core.decision import to_label_matrix


def decide(P: np.ndarray, taus: np.ndarray | None, multilabel: bool) -> np.ndarray:
    """CoRE's decision rule for multi-label methods; arg-max (exactly one condition) otherwise."""
    if multilabel:
        return to_label_matrix(P, taus)
    Yhat = np.zeros_like(P, dtype=int)
    Yhat[np.arange(len(P)), P.argmax(1)] = 1
    return Yhat


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def macro_auc(Y: np.ndarray, P: np.ndarray) -> float:
    return float(np.nanmean([_auc(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))


def multi_label_recall(Y: np.ndarray, Yhat: np.ndarray) -> float | None:
    """Share of the true conditions recovered, on posts that truly carry more than one condition."""
    multi = Y.sum(1) > 1
    return float((Yhat[multi] & Y[multi]).sum() / Y[multi].sum()) if multi.any() else None


def summary(Y: np.ndarray, Yhat: np.ndarray, P: np.ndarray) -> dict:
    """The scalar metrics stored for every cross-validation round."""
    return {"macro_f1": float(f1_score(Y, Yhat, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(Y, Yhat, average="micro", zero_division=0)),
            "macro_auc": macro_auc(Y, P),
            "hamming_loss": float(hamming_loss(Y, Yhat))}


def full_report(Y: np.ndarray, Yhat: np.ndarray, P: np.ndarray, conditions: list[str]) -> dict:
    prec, rec, f1, sup = precision_recall_fscore_support(Y, Yhat, zero_division=0)
    out = summary(Y, Yhat, P)
    out["multi_label_recall"] = multi_label_recall(Y, Yhat)
    out["mean_labels_predicted"] = float(Yhat.sum(1).mean())
    out["per_condition"] = {}
    for j, d in enumerate(conditions):
        y, yh = Y[:, j], Yhat[:, j]
        out["per_condition"][d] = {
            "precision": float(prec[j]), "recall": float(rec[j]), "f1": float(f1[j]),
            "auc": _auc(y, P[:, j]), "support": int(sup[j]),
            # confusion matrix in the course's order [[TP, FP], [FN, TN]]
            "confusion_tp_fp_fn_tn": [int((y & yh).sum()), int((~y.astype(bool) & yh.astype(bool)).sum()),
                                      int((y.astype(bool) & ~yh.astype(bool)).sum()), int(((1 - y) & (1 - yh)).sum())],
        }
    return out


def combination_confusion(Y: np.ndarray, Yhat: np.ndarray) -> dict:
    """Counts of (true label set, predicted label set) pairs, keys as bit strings such as '011'."""
    key = lambda M: ["".join(map(str, r)) for r in M]  # noqa: E731
    out: dict[str, dict[str, int]] = {}
    for t, p in zip(key(Y), key(Yhat)):
        out.setdefault(t, {}).setdefault(p, 0)
        out[t][p] += 1
    return {t: dict(sorted(v.items())) for t, v in sorted(out.items())}


def bootstrap(Y: np.ndarray, preds: dict[str, tuple[np.ndarray, np.ndarray]], ref: str,
              n_boot: int, seed: int) -> dict:
    """Percentile 95% intervals for macro-F1, micro-F1 and macro-AUC of each method, and for the
    paired difference ref - method, all computed on the same seeded resamples of the test posts."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    names = list(preds)
    vals = {m: {"macro_f1": [], "micro_f1": [], "macro_auc": []} for m in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Yb = Y[idx]
        for m in names:
            Yhat, P = preds[m]
            vals[m]["macro_f1"].append(f1_score(Yb, Yhat[idx], average="macro", zero_division=0))
            vals[m]["micro_f1"].append(f1_score(Yb, Yhat[idx], average="micro", zero_division=0))
            vals[m]["macro_auc"].append(macro_auc(Yb, P[idx]))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    out = {"n_boot": n_boot, "seed": seed, "ci95": {}, "paired_vs_" + ref: {}}
    for m in names:
        out["ci95"][m] = {k: ci(v) for k, v in vals[m].items()}
        if m != ref:
            d = {k: np.array(vals[ref][k]) - np.array(vals[m][k]) for k in vals[m]}
            out["paired_vs_" + ref][m] = {
                k: {"ci95": ci(v), "share_resamples_ref_better": float((v > 0).mean())} for k, v in d.items()}
    return out
