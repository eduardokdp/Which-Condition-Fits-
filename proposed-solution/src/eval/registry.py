"""CoRE and the Milestone 4 baselines behind one interface.

Every method receives the same input: the mask-aware TF-IDF matrix of the cleaned, masked text
(src/core/features.py), fitted on the fit part of the current round only. Each method returns one
score per condition. Multi-label methods then go through CoRE's decision rule (primary = highest
score, plus every other condition whose score reaches its tuned threshold); the multiclass baseline
returns exactly one condition, the one with the highest probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src.core.chain import BinaryRelevance, OrderAveragedChain


@dataclass(frozen=True)
class MethodInfo:
    key: str
    name: str          # name used in tables and figures
    multilabel: bool   # False: exactly one condition per post (no thresholds)
    uses_ngram: bool   # True: the n-gram range is a hyperparameter of this method


METHODS = {
    "core": MethodInfo("core", "CoRE", True, True),
    "multiclass_lr": MethodInfo("multiclass_lr", "Multiclass LR", False, True),
    "br_lr": MethodInfo("br_lr", "Binary relevance", True, True),
    "chain1": MethodInfo("chain1", "Chain (1 order)", True, True),
    "knn": MethodInfo("knn", "k-NN", True, False),
    "svm": MethodInfo("svm", "Linear SVM", True, False),
    "nb": MethodInfo("nb", "Naive Bayes", True, False),
}


class _PerLabel:
    """One binary scorer per condition (one-vs-rest)."""

    def __init__(self, make):
        self.make = make

    def fit(self, X, Y, primary=None):
        self.models_ = [self.make().fit(X, Y[:, j]) for j in range(Y.shape[1])]
        return self


class _SVM(_PerLabel):
    def scores(self, X):
        # decision values mapped to (0, 1) by the logistic function: a monotone transform, so AUC is
        # unchanged and the same threshold grid applies to every method
        return np.column_stack([expit(m.decision_function(X)) for m in self.models_])


class _NB(_PerLabel):
    def scores(self, X):
        return np.column_stack([m.predict_proba(X)[:, list(m.classes_).index(1)] for m in self.models_])


class _KNN:
    def __init__(self, k: int):
        self.k = k

    def fit(self, X, Y, primary=None):
        self.m_ = KNeighborsClassifier(n_neighbors=self.k, metric="cosine", algorithm="brute").fit(X, Y)
        return self

    def scores(self, X):
        # fraction of the k nearest training posts that carry each condition
        out = []
        for j, p in enumerate(self.m_.predict_proba(X)):
            cls = list(self.m_.classes_[j])
            out.append(p[:, cls.index(1)] if 1 in cls else np.zeros(X.shape[0]))
        return np.column_stack(out)


class _Chain:
    def __init__(self, cls, **kw):
        self.cls, self.kw = cls, kw

    def fit(self, X, Y, primary=None):
        self.m_ = self.cls(Y.shape[1], **self.kw).fit(X, Y)
        return self

    def scores(self, X):
        return self.m_.predict_proba(X)


class _Multiclass:
    def __init__(self, C: float, cfg_model: dict):
        self.C, self.mcfg = C, cfg_model

    def fit(self, X, Y, primary=None):
        # target: the condition of the community the post came from (the single-label framing)
        self.k_ = Y.shape[1]
        self.m_ = LogisticRegression(C=self.C, class_weight=self.mcfg["class_weight"],
                                     max_iter=self.mcfg["max_iter"]).fit(X, primary)
        return self

    def scores(self, X):
        P = np.zeros((X.shape[0], self.k_))
        P[:, self.m_.classes_] = self.m_.predict_proba(X)
        return P


def build(key: str, params: dict, cfg: dict):
    """Instantiate method `key` with hyperparameters `params` (one grid point)."""
    m, seed = cfg["model"], cfg["seed"]
    lr = dict(class_weight=m["class_weight"], max_iter=m["max_iter"], oof_folds=m["oof_folds"], seed=seed)
    if key == "core":
        return _Chain(OrderAveragedChain, C=params["C"], augment=params["augment"], **lr)
    if key == "chain1":  # Read et al. (2011): one fixed order, adhd -> anxiety -> depression
        return _Chain(OrderAveragedChain, C=params["C"], augment="oof", orders=[(0, 1, 2)], **lr)
    if key == "br_lr":
        return _Chain(BinaryRelevance, C=params["C"], **lr)
    if key == "multiclass_lr":
        return _Multiclass(params["C"], m)
    if key == "knn":
        return _KNN(params["k"])
    if key == "svm":
        return _SVM(lambda: LinearSVC(C=params["C"], class_weight="balanced", random_state=seed,
                                      max_iter=10000))
    if key == "nb":
        return _NB(lambda: ComplementNB(alpha=params["alpha"]))
    raise KeyError(key)
