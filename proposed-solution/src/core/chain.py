"""Order-averaged classifier chain (OACC).

For a label order pi = (pi_1, ..., pi_k), link j estimates
    P(d_{pi_j} | s, d_{pi_1}, ..., d_{pi_{j-1}}) = sigmoid(w^T [phi(s) ; q_{pi_1..pi_{j-1}}] + b),
where q are the probabilities produced by the earlier links. The final score of condition d is the
mean over all k! orders, which removes the dependence on an arbitrary chain order (k = 3 -> 6 chains).

Training uses cross-fitted (out-of-fold) predecessor probabilities so that each link sees inputs
with the same distribution at training and at inference time.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class OrderAveragedChain:
    def __init__(self, n_labels: int, C: float = 1.0, class_weight: str | None = "balanced",
                 max_iter: int = 2000, augment: str = "oof", oof_folds: int = 5, seed: int = 0,
                 orders: list[tuple[int, ...]] | None = None):
        if augment not in {"oof", "true"}:
            raise ValueError("augment must be 'oof' or 'true'")
        self.k = n_labels
        self.C, self.class_weight, self.max_iter = C, class_weight, max_iter
        self.augment, self.oof_folds, self.seed = augment, oof_folds, seed
        self.orders = orders or list(permutations(range(n_labels)))
        self.links_: list[list[LogisticRegression]] = []

    def _new_link(self) -> LogisticRegression:
        return LogisticRegression(C=self.C, class_weight=self.class_weight, solver="lbfgs",
                                  max_iter=self.max_iter)

    @staticmethod
    def _augment(X, Q: np.ndarray, prev: list[int]):
        if not prev:
            return X
        return sparse.hstack([X, sparse.csr_matrix(Q[:, prev])], format="csr")

    def fit(self, X, Y: np.ndarray) -> "OrderAveragedChain":
        Y = np.asarray(Y, dtype=int)
        for j in range(self.k):
            if len(np.unique(Y[:, j])) < 2:
                raise ValueError(f"Label column {j} has a single class in the training data.")
        self.links_ = []
        for order in self.orders:
            Q = Y.astype(float).copy() if self.augment == "true" else np.zeros_like(Y, dtype=float)
            links = []
            for pos, lab in enumerate(order):
                Xa = self._augment(X, Q, list(order[:pos]))
                clf = self._new_link()
                if self.augment == "oof" and pos < self.k - 1:
                    cv = StratifiedKFold(n_splits=self.oof_folds, shuffle=True, random_state=self.seed)
                    Q[:, lab] = cross_val_predict(self._new_link(), Xa, Y[:, lab], cv=cv,
                                                  method="predict_proba")[:, 1]
                clf.fit(Xa, Y[:, lab])
                links.append(clf)
            self.links_.append(links)
        return self

    def predict_proba(self, X) -> np.ndarray:
        n = X.shape[0]
        P = np.zeros((n, self.k))
        for order, links in zip(self.orders, self.links_):
            Q = np.zeros((n, self.k))
            for pos, lab in enumerate(order):
                Xa = self._augment(X, Q, list(order[:pos]))
                Q[:, lab] = links[pos].predict_proba(Xa)[:, 1]
            P += Q
        return P / len(self.orders)


class BinaryRelevance(OrderAveragedChain):
    """Ablation: independent per-label classifiers (no label dependence)."""

    def __init__(self, n_labels: int, **kw):
        kw.pop("orders", None)
        super().__init__(n_labels, **kw)

    def fit(self, X, Y):
        Y = np.asarray(Y, dtype=int)
        self.links_ = [[self._new_link().fit(X, Y[:, j]) for j in range(self.k)]]
        return self

    def predict_proba(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.links_[0]])
