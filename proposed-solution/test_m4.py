"""Milestone 4 protocol tests: no test post in any training or tuning step, the same folds for every
method, and identical results from the same seed. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.cmn.config import load_config
from src.dataprep.prepare import build_dataset
from src.eval import m4_metrics as M
from src.eval.protocol import build_protocol, rounds
from src.eval.registry import METHODS, build

CFG = load_config("config/m4_sample.yaml")


def _small_cfg(results_dir: str) -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["data"]["max_posts_per_community"] = 70
    ex = cfg["experiment"]
    ex["results_dir"], ex["n_jobs"], ex["bootstrap"] = results_dir, 1, 50
    ex["grids"] = {"core": {"C": [1], "ngram_range": [[1, 1]], "augment": ["oof"]},
                   "multiclass_lr": {"C": [1], "ngram_range": [[1, 1]]},
                   "knn": {"k": [5, 3]}, "svm": {"C": [1]}, "nb": {"alpha": [0.1]}}
    return cfg


class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = build_dataset(CFG)
        cls.proto = build_protocol(cls.df, CFG)

    def test_every_post_once_and_80_20(self):
        self.assertEqual(sorted(self.proto["id"]), sorted(self.df["id"]))
        share = (self.proto["part"] == "test").mean()
        self.assertAlmostEqual(share, CFG["experiment"]["test_size"], delta=0.01)

    def test_folds_only_on_dev(self):
        dev, test = self.proto[self.proto["part"] == "dev"], self.proto[self.proto["part"] == "test"]
        self.assertTrue((test["fold"] == 0).all())
        self.assertEqual(set(dev["fold"]), set(range(1, CFG["experiment"]["n_folds"] + 1)))

    def test_threshold_parts_never_touch_scoring_fold_or_test(self):
        for k in range(1, CFG["experiment"]["n_folds"] + 1):
            thr = self.proto[self.proto[f"thr_{k}"] == 1]
            self.assertTrue((thr["part"] == "dev").all())
            self.assertTrue((thr["fold"] != k).all())
        self.assertTrue((self.proto.loc[self.proto["thr_final"] == 1, "part"] == "dev").all())

    def test_rounds_are_disjoint_and_test_only_scored_at_the_end(self):
        test_ids = set(self.proto.loc[self.proto["part"] == "test", "id"])
        for name, fit, thr, score in rounds(self.proto, CFG["experiment"]["n_folds"]):
            fit, thr, score = set(fit), set(thr), set(score)
            self.assertFalse(fit & thr or fit & score or thr & score, name)
            self.assertFalse((fit | thr) & test_ids, name)
            if name == "final":
                self.assertEqual(score, test_ids)
            else:
                self.assertFalse(score & test_ids, name)

    def test_protocol_is_deterministic(self):
        pd.testing.assert_frame_equal(self.proto, build_protocol(self.df, CFG))


class TestMethods(unittest.TestCase):
    def test_scores_and_decisions(self):
        from src.core.features import make_vectorizer
        df = build_dataset(CFG).sample(240, random_state=0)
        X = make_vectorizer(CFG["features"]).fit_transform(df["text"])
        Y = df[CFG["conditions"]].to_numpy(dtype=int)
        prim = df["community"].map(lambda c: CFG["conditions"].index(c)).to_numpy()
        for key, params in [("core", {"C": 1, "augment": "oof"}), ("multiclass_lr", {"C": 1}),
                            ("br_lr", {"C": 1}), ("chain1", {"C": 1}), ("knn", {"k": 5}),
                            ("svm", {"C": 1}), ("nb", {"alpha": 0.1})]:
            P = build(key, params, CFG).fit(X, Y, primary=prim).scores(X)
            self.assertEqual(P.shape, Y.shape, key)
            self.assertTrue(((P >= 0) & (P <= 1)).all(), key)
            Yhat = M.decide(P, np.full(3, 0.5), METHODS[key].multilabel)
            self.assertTrue((Yhat.sum(1) >= 1).all(), key)
            if not METHODS[key].multilabel:
                self.assertTrue((Yhat.sum(1) == 1).all(), key)

    def test_metrics_on_perfect_prediction(self):
        Y = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1], [1, 1, 0]])
        r = M.summary(Y, Y, Y.astype(float))
        self.assertEqual((r["macro_f1"], r["macro_auc"], r["hamming_loss"]), (1.0, 1.0, 0.0))
        self.assertEqual(M.multi_label_recall(Y, Y), 1.0)


class TestExperiment(unittest.TestCase):
    """A small end-to-end run (140-ish posts per run) executed twice with the same seed."""

    def test_end_to_end_is_reproducible_and_leak_free(self):
        from src.eval.experiment import run
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            r1, r2 = run(_small_cfg(a)), run(_small_cfg(b))
            self.assertEqual(r1["test_predictions_sha256"], r2["test_predictions_sha256"])
            proto = pd.read_csv(Path(a) / "protocol.csv")
            cv = pd.read_csv(Path(a) / "cv_results.csv")
            # the same five folds for every method and configuration
            folds = cv.groupby(["method", "config"])["fold"].apply(lambda s: tuple(sorted(s)))
            self.assertEqual(len(set(folds)), 1)
            # the test predictions cover exactly the test posts
            preds = pd.read_csv(Path(a) / "test_predictions.csv")
            self.assertEqual(set(preds["id"]), set(proto.loc[proto["part"] == "test", "id"]))


if __name__ == "__main__":
    unittest.main()
