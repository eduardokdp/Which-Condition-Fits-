"""Milestone 4 evaluation protocol: one dev/test split and one set of folds, shared by every method.

    dataset --(80/20, stratified on the label combination)--> dev | test   (test used once, at the end)
    dev     --(5 stratified folds, computed once)-----------> round k: train = folds != k, score = fold k
    train   --(85/15, stratified)--------------------------> fit part | threshold part

The fit part trains the representation and the classifier; the threshold part is used only to tune
the per-condition decision thresholds. The scoring fold and the test set never take part in fitting
or tuning. The whole assignment is saved to protocol.csv so that it can be inspected and reused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.dataprep.prepare import _strat_key


def _split(df: pd.DataFrame, conditions: list[str], size: float, seed: int, min_count: int):
    return train_test_split(df, test_size=size, random_state=seed,
                            stratify=_strat_key(df, conditions, min_count=min_count))


def build_protocol(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return one row per post: id, part (dev/test), fold (1..n for dev, 0 for test), and one column
    per round (thr_1..thr_n, thr_final) that is 1 where the post is in that round's threshold part."""
    conditions, seed, ex = cfg["conditions"], cfg["seed"], cfg["experiment"]
    dev, test = _split(df, conditions, ex["test_size"], seed, min_count=6)

    proto = pd.DataFrame({"id": df["id"].to_numpy()}).set_index("id")
    proto["part"] = "dev"
    proto.loc[test["id"], "part"] = "test"
    proto["fold"] = 0

    skf = StratifiedKFold(n_splits=ex["n_folds"], shuffle=True, random_state=seed)
    dev = dev.reset_index(drop=True)
    for k, (_, score_idx) in enumerate(skf.split(dev, _strat_key(dev, conditions, min_count=ex["n_folds"])), 1):
        proto.loc[dev.loc[score_idx, "id"], "fold"] = k

    for k in range(1, ex["n_folds"] + 1):
        train = dev[proto.loc[dev["id"], "fold"].to_numpy() != k]
        _, thr = _split(train, conditions, ex["inner_val"], seed + k, min_count=2)
        proto[f"thr_{k}"] = 0
        proto.loc[thr["id"], f"thr_{k}"] = 1
    _, thr = _split(dev, conditions, ex["inner_val"], seed, min_count=2)
    proto["thr_final"] = 0
    proto.loc[thr["id"], "thr_final"] = 1
    return proto.reset_index()


def rounds(proto: pd.DataFrame, n_folds: int):
    """Yield (round name, fit ids, threshold ids, score ids) for the CV rounds and the final refit."""
    dev = proto[proto["part"] == "dev"]
    for k in range(1, n_folds + 1):
        train = dev[dev["fold"] != k]
        yield (f"fold{k}", train.loc[train[f"thr_{k}"] == 0, "id"].to_numpy(),
               train.loc[train[f"thr_{k}"] == 1, "id"].to_numpy(), dev.loc[dev["fold"] == k, "id"].to_numpy())
    yield ("final", dev.loc[dev["thr_final"] == 0, "id"].to_numpy(),
           dev.loc[dev["thr_final"] == 1, "id"].to_numpy(), proto.loc[proto["part"] == "test", "id"].to_numpy())


def length_terciles(df: pd.DataFrame, proto: pd.DataFrame) -> tuple[float, float]:
    """Token-length cut points (33rd and 67th percentiles) computed on the dev set only."""
    dev_ids = set(proto.loc[proto["part"] == "dev", "id"])
    n = df.loc[df["id"].isin(dev_ids), "text"].str.split().str.len()
    lo, hi = np.percentile(n, [100 / 3, 200 / 3])
    return float(lo), float(hi)
