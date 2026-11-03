"""Baselines used for comparison with CoRE.

* SingleLabel: multinomial logistic regression on the community label, predicting exactly one
  condition per description (the single-label framing used by most prior work).
* Binary relevance (src.core.chain.BinaryRelevance) is run through CoRE(chain="br") as an ablation
  that removes label dependence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.core.features import make_vectorizer


class SingleLabel:
    def __init__(self, conditions: list[str], features: dict, model: dict, community_to_condition: dict):
        self.conditions, self.f, self.m, self.c2d = conditions, features, model, community_to_condition

    def fit(self, train: pd.DataFrame) -> "SingleLabel":
        self.vec_ = make_vectorizer(self.f)  # same representation phi as CoRE
        X = self.vec_.fit_transform(train["text"])
        y = train["community"].map(lambda c: self.conditions.index(self.c2d[c])).to_numpy()
        self.clf_ = LogisticRegression(C=self.m["C"], class_weight=self.m["class_weight"],
                                       max_iter=self.m["max_iter"]).fit(X, y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        P = np.zeros((len(texts), len(self.conditions)))
        P[:, self.clf_.classes_] = self.clf_.predict_proba(self.vec_.transform(list(texts)))
        return P

    def predict_labels(self, texts) -> np.ndarray:
        P = self.predict_proba(texts)
        Yhat = np.zeros_like(P, dtype=int)
        Yhat[np.arange(len(P)), P.argmax(1)] = 1
        return Yhat
