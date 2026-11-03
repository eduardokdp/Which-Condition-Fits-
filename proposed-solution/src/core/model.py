"""CoRE: Co-occurring condition classification with Rationale Extraction.

f(s) = ((d*, E*), {(d_i, E_i)})  -- see Section 2 of the report.
Pipeline: clean + mask -> phi (TF-IDF) -> order-averaged chain -> thresholds/decision -> occlusion rationales.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.core.chain import BinaryRelevance, OrderAveragedChain
from src.core.decision import decide, threshold_grid, to_label_matrix, tune_thresholds
from src.core.features import make_vectorizer
from src.core.rationale import occlusion_deltas, select_evidence
from src.core.segment import segment
from src.dataprep.lexicon import mask_condition_names
from src.dataprep.prepare import clean_text


class CoRE:
    def __init__(self, conditions: list[str], features: dict, model: dict, decision: dict,
                 rationale: dict, seed: int = 0, chain: str = "oacc"):
        self.conditions = list(conditions)
        self.fcfg, self.mcfg, self.dcfg, self.rcfg = features, model, decision, rationale
        self.seed, self.chain = seed, chain
        self.vectorizer_: TfidfVectorizer | None = None
        self.clf_ = None
        self.thresholds_: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: dict, chain: str = "oacc") -> "CoRE":
        return cls(cfg["conditions"], cfg["features"], cfg["model"], cfg["decision"],
                   cfg["rationale"], seed=cfg["seed"], chain=chain)

    # ---------- representation ----------
    def _prep(self, texts) -> list[str]:
        return [mask_condition_names(clean_text(t), self.conditions) for t in texts]

    def _phi(self, texts):
        return self.vectorizer_.transform(self._prep(texts))

    # ---------- training ----------
    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "CoRE":
        self.vectorizer_ = make_vectorizer(self.fcfg)
        X = self.vectorizer_.fit_transform(self._prep(train["text"]))
        Y = train[self.conditions].to_numpy(dtype=int)
        kw = dict(C=self.mcfg["C"], class_weight=self.mcfg["class_weight"], max_iter=self.mcfg["max_iter"],
                  augment=self.mcfg.get("augment", "oof"), oof_folds=self.mcfg.get("oof_folds", 5),
                  seed=self.seed)
        model_cls = OrderAveragedChain if self.chain == "oacc" else BinaryRelevance
        self.clf_ = model_cls(len(self.conditions), **kw).fit(X, Y)
        g = self.dcfg["threshold_grid"]
        P_val = self.predict_proba(val["text"].tolist())
        self.thresholds_ = tune_thresholds(P_val, val[self.conditions].to_numpy(dtype=int),
                                           threshold_grid(g["start"], g["stop"], g["step"]))
        return self

    # ---------- inference ----------
    def predict_proba(self, texts) -> np.ndarray:
        return self.clf_.predict_proba(self._phi(list(texts)))

    def predict_labels(self, texts) -> np.ndarray:
        return to_label_matrix(self.predict_proba(texts), self.thresholds_)

    def explain(self, text: str) -> dict:
        """Full output f(s): primary and secondary conditions, each with its evidence phrases."""
        phrases = segment(clean_text(text))
        p, D = occlusion_deltas(phrases, self.predict_proba)
        primary, secondary = decide(p, self.thresholds_)
        r = self.rcfg

        def item(j: int) -> dict:
            ev = select_evidence(D[:, j], r["delta"], r["max_fraction"], r["min_phrases"])
            return {"condition": self.conditions[j], "probability": round(float(p[j]), 4),
                    "threshold": round(float(self.thresholds_[j]), 2), "evidence": ev,
                    "evidence_text": [phrases[e] for e in ev]}

        prim, secs = item(primary), [item(j) for j in secondary]
        # phrases cited as evidence for at least two predicted conditions (overlapping symptoms)
        cited = [e for x in [prim, *secs] for e in x["evidence"]]
        shared = sorted({e for e in cited if cited.count(e) > 1})
        return {
            "phrases": phrases,
            "probabilities": {d: round(float(p[j]), 4) for j, d in enumerate(self.conditions)},
            "occlusion_delta": {d: [round(float(x), 4) for x in D[:, j]] for j, d in enumerate(self.conditions)},
            "primary": prim,
            "secondary": secs,
            "shared_evidence": shared,
        }

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "CoRE":
        return joblib.load(path)


def format_output(out: dict) -> str:
    """Render f(s) in the notation of the report, e.g. ((adhd, {p1,p2}), {(anxiety, {p2})})."""
    def ev(e):
        return "{" + ", ".join(f"p{j + 1}" for j in e["evidence"]) + "}"
    sec = ", ".join(f"({x['condition']}, {ev(x)})" for x in out["secondary"])
    lines = [f"  p{j + 1} = \"{ph}\"" for j, ph in enumerate(out["phrases"])]
    # values are printed exactly as stored (4 decimals), never re-rounded
    lines.append("  P(d|s) = " + ", ".join(f"{k}: {v:.4f}" for k, v in out["probabilities"].items()))
    lines.append(f"  f(s) = (({out['primary']['condition']}, {ev(out['primary'])}), {{{sec}}})")
    return "\n".join(lines)

