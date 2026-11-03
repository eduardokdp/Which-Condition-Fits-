"""Unit and reproducibility tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.cmn.config import load_config
from src.cmn.io import write_json, write_jsonl
from src.cmn.seed import set_seed
from src.core.decision import decide, threshold_grid, tune_thresholds
from src.core.features import MaskAwareAnalyzer
from src.core.model import CoRE
from src.core.rationale import select_evidence
from src.core.segment import segment
from src.dataprep.lexicon import mask_condition_names, self_reported_conditions
from src.dataprep.prepare import _strat_key, build_dataset, clean_text, is_bot, split_dataset

C = ["adhd", "anxiety", "depression"]
RUNNING_EXAMPLE = "I can't focus on anything, I feel restless and I've been more irritable than usual."


class TestSegment(unittest.TestCase):
    def test_running_example(self):
        self.assertEqual(segment(RUNNING_EXAMPLE),
                         ["I can't focus on anything", "I feel restless", "I've been more irritable than usual"])

    def test_list_inside_clause_is_kept(self):
        self.assertEqual(segment("I feel tired, sad, and empty."), ["I feel tired, sad, and empty"])

    def test_never_empty(self):
        self.assertEqual(segment("ok"), ["ok"])


class TestLexicon(unittest.TestCase):
    def test_self_reports(self):
        self.assertEqual(self_reported_conditions("I was diagnosed with ADHD last year.", C), {"adhd"})
        self.assertEqual(self_reported_conditions("I also have anxiety.", C), {"anxiety"})
        self.assertEqual(self_reported_conditions("My depression got worse.", C), {"depression"})

    def test_non_reports(self):
        for t in ["My brother has ADHD.", "My doctor says it is not depression.",
                  "I wonder if I have ADHD.", "I don't have anxiety."]:
            self.assertEqual(self_reported_conditions(t, C), set(), t)

    def test_mask(self):
        self.assertEqual(mask_condition_names("My ADHD and generalized anxiety disorder", C),
                         "My _cond_ and _cond_")
        self.assertIn("anxious", mask_condition_names("I feel anxious and depressed", C))


class TestPreprocessing(unittest.TestCase):
    def test_markdown_and_urls_removed(self):
        t = "**Bold** and *it*. [a link](https://x.org) &amp; more\n\n> quote\n- item"
        self.assertEqual(clean_text(t), "Bold and it. a link & more\nquote\nitem")

    def test_bot_filter(self):
        self.assertTrue(is_bot("AutoModerator") and is_bot("remindme_bot") and is_bot("RemindMeBot"))
        self.assertFalse(is_bot("abbott") or is_bot("robotics_fan") or is_bot("[deleted]"))

    def test_rare_label_combination_cannot_break_split(self):
        import pandas as pd
        df = pd.DataFrame({"adhd": [1] * 10 + [1], "anxiety": [0] * 10 + [1], "depression": [0] * 11})
        key = _strat_key(df, C)
        self.assertTrue((key.value_counts() >= 2).all())


class TestRepresentation(unittest.TestCase):
    def test_mask_is_never_a_feature_and_never_bridged(self):
        grams = MaskAwareAnalyzer((1, 2))(mask_condition_names("My ADHD is bad and I can't focus", C))
        self.assertFalse(any("_cond_" in g for g in grams))
        self.assertNotIn("my is", grams)  # no bigram across the removed name
        self.assertIn("i can't", grams)   # first-person pronoun and contraction are kept


class TestDecision(unittest.TestCase):
    def test_primary_always_reported(self):
        primary, secondary = decide(np.array([0.2, 0.1, 0.15]), np.array([0.5, 0.5, 0.5]))
        self.assertEqual((primary, secondary), (0, []))

    def test_secondary_above_threshold(self):
        primary, secondary = decide(np.array([0.7, 0.6, 0.2]), np.array([0.5, 0.5, 0.5]))
        self.assertEqual((primary, secondary), (0, [1]))

    def test_threshold_tuning(self):
        P = np.array([[0.9], [0.8], [0.3], [0.1]])
        Y = np.array([[1], [1], [0], [0]])
        tau = tune_thresholds(P, Y, threshold_grid(0.05, 0.95, 0.05))
        self.assertTrue(0.3 < tau[0] <= 0.8)

    def test_evidence_selection(self):
        self.assertEqual(select_evidence(np.array([0.2, 0.0, 0.05, -0.1]), 0.01, 0.5, 1), [0, 2])


class TestReproducibility(unittest.TestCase):
    """Same input, code, configuration and seed -> identical output."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config("config/quickstart.yaml")
        set_seed(cls.cfg["seed"])
        cls.splits = split_dataset(build_dataset(cls.cfg), cls.cfg)
        cls.model = cls._fit()

    @classmethod
    def _fit(cls):
        set_seed(cls.cfg["seed"])
        return CoRE.from_config(cls.cfg).fit(cls.splits["train"], cls.splits["val"])

    def test_identical_runs(self):
        other = self._fit()
        texts = self.splits["test"]["text"].tolist()
        np.testing.assert_array_equal(self.model.predict_proba(texts), other.predict_proba(texts))
        self.assertEqual(self.model.explain(RUNNING_EXAMPLE), other.explain(RUNNING_EXAMPLE))

    def test_split_is_deterministic(self):
        again = split_dataset(build_dataset(self.cfg), self.cfg)
        for k in ("train", "val", "test"):
            self.assertEqual(self.splits[k]["id"].tolist(), again[k]["id"].tolist())

    def test_output_structure(self):
        out = self.model.explain(RUNNING_EXAMPLE)
        self.assertEqual(len(out["phrases"]), 3)
        self.assertIn(out["primary"]["condition"], C)
        self.assertTrue(all(0 <= e < 3 for e in out["primary"]["evidence"]))


class TestPortableOutput(unittest.TestCase):
    """Output files must use LF line endings on every OS, or the predictions hash differs on Windows."""

    def test_writers_use_lf(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_json({"a": [1, 2]}, d / "x.json")
            write_jsonl([{"a": 1}, {"b": 2}], d / "x.jsonl")
            pd.DataFrame({"a": [1, 2]}).to_csv(d / "x.csv", index=False, lineterminator="\n")
            for f in ("x.json", "x.jsonl", "x.csv"):
                raw = (d / f).read_bytes()
                self.assertNotIn(b"\r\n", raw, f)
                self.assertIn(b"\n", raw, f)


if __name__ == "__main__":
    unittest.main()
