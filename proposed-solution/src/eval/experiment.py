"""Milestone 4 experiment: hyperparameter grid with 5-fold cross-validation on the dev set, one refit
per method on the whole dev set, and a single evaluation on the held-out test set.

    python -m src.main experiment --config config/m4_sample.yaml

Writes to experiment.results_dir:
    data_card.json        dataset statistics and posts removed by each cleaning step
    protocol.csv          dev/test part, fold and threshold-part membership of every post
    cv_results.csv        every (method, configuration, fold) score on the dev folds
    cv_summary.csv        mean and standard deviation over folds; the selected configuration per method
    best_params.json      selected configuration per method
    test_predictions.csv  scores and decisions of every method on the test posts (hashed)
    test_metrics.json     full test report: metrics, CIs, paired differences, confusion matrices,
                          subgroups (RQ3) and timing (RQ4)
    test_metrics.csv      one row per method, for tables
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.cmn.config import resolve
from src.cmn.io import get_logger, sha256_file, write_json
from src.core.decision import threshold_grid, tune_thresholds
from src.core.features import make_vectorizer
from src.dataprep.prepare import build_dataset
from src.eval import m4_metrics as M
from src.eval.protocol import build_protocol, length_terciles, rounds
from src.eval.registry import METHODS, build

log = get_logger("core")
REF = "core"


def grid_points(grid: dict) -> list[dict]:
    """All combinations of a method's grid, in the order listed (simplest first)."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def config_id(params: dict) -> str:
    def fmt(v):
        return "-".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
    return ";".join(f"{k}={fmt(v)}" for k, v in params.items())


def _features(cfg, texts_fit, texts_other: list, ngram):
    fcfg = dict(cfg["features"], ngram_range=list(ngram))
    vec = make_vectorizer(fcfg)
    X_fit = vec.fit_transform(texts_fit)
    return vec, X_fit, [vec.transform(t) for t in texts_other]


def _run_one(key, params, cfg, X_fit, Y_fit, prim_fit, X_thr, Y_thr, X_score, timed=False):
    """Fit one method on the fit part, tune thresholds on the threshold part, score the scoring set."""
    info = METHODS[key]
    t0 = time.perf_counter()
    model = build(key, params, cfg).fit(X_fit, Y_fit, primary=prim_fit)
    taus = None
    if info.multilabel:
        g = cfg["decision"]["threshold_grid"]
        taus = tune_thresholds(model.scores(X_thr), Y_thr, threshold_grid(g["start"], g["stop"], g["step"]))
    t1 = time.perf_counter()
    P = model.scores(X_score)
    t2 = time.perf_counter()
    out = {"P": P, "taus": taus}
    if timed:
        out.update(fit_seconds=t1 - t0, predict_seconds=t2 - t1)
    return out


def run(cfg: dict) -> dict:
    ex, conditions, seed = cfg["experiment"], cfg["conditions"], cfg["seed"]
    out_dir = resolve(ex["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data and protocol ----------
    counts: dict = {}
    df = build_dataset(cfg, counts=counts)
    proto = build_protocol(df, cfg)
    proto.to_csv(out_dir / "protocol.csv", index=False, lineterminator="\n")
    df = df.set_index("id").loc[proto["id"]].reset_index()  # same row order as the protocol
    Y_all = df[conditions].to_numpy(dtype=int)
    c2d = cfg["data"]["community_to_condition"]
    prim_all = df["community"].map(lambda c: conditions.index(c2d[c])).to_numpy()
    row = {i: r for r, i in enumerate(df["id"])}
    texts = df["text"].tolist()
    write_json(_data_card(df, proto, conditions, counts, cfg), out_dir / "data_card.json")
    log.info("posts=%d dev=%d test=%d", len(df), (proto["part"] == "dev").sum(), (proto["part"] == "test").sum())

    # ---------- representations: one per (round, n-gram range), fitted on the fit part only ----------
    plan = list(rounds(proto, ex["n_folds"]))
    ngrams = sorted({tuple(p["ngram_range"]) for k, g in ex["grids"].items() if METHODS[k].uses_ngram
                     for p in grid_points(g)} | {tuple(cfg["features"]["ngram_range"])})
    feats, idx = {}, {}
    for name, fit_ids, thr_ids, score_ids in plan:
        idx[name] = tuple(np.array([row[i] for i in ids]) for ids in (fit_ids, thr_ids, score_ids))
        f, t, s = idx[name]
        for ng in ngrams:
            _, X_fit, (X_thr, X_score) = _features(cfg, [texts[i] for i in f],
                                                   [[texts[i] for i in t], [texts[i] for i in s]], ng)
            feats[(name, ng)] = (X_fit, X_thr, X_score)

    def args_for(key, params, name):
        ng = tuple(params["ngram_range"]) if METHODS[key].uses_ngram else tuple(cfg["features"]["ngram_range"])
        X_fit, X_thr, X_score = feats[(name, ng)]
        f, t, _ = idx[name]
        return (key, params, cfg, X_fit, Y_all[f], prim_all[f], X_thr, Y_all[t], X_score)

    # ---------- cross-validated grid on the dev folds ----------
    folds = [p[0] for p in plan if p[0] != "final"]
    tasks = [(key, params, name) for key, grid in ex["grids"].items() for params in grid_points(grid)
             for name in folds]
    log.info("cross-validation: %d fits (%d methods, %d folds)", len(tasks), len(ex["grids"]), len(folds))
    t0 = time.perf_counter()
    res = Parallel(n_jobs=ex.get("n_jobs", -1))(delayed(_run_one)(*args_for(*t)) for t in tasks)
    log.info("cross-validation done in %.0f s", time.perf_counter() - t0)

    rows = []
    for (key, params, name), r in zip(tasks, res):
        s = idx[name][2]
        Yhat = M.decide(r["P"], r["taus"], METHODS[key].multilabel)
        rows.append({"method": key, "config": config_id(params), "params": json.dumps(params),
                     "fold": name, **M.summary(Y_all[s], Yhat, r["P"])})
    cv = pd.DataFrame(rows)
    cv.to_csv(out_dir / "cv_results.csv", index=False, float_format="%.6f", lineterminator="\n")

    metrics = ["macro_f1", "micro_f1", "macro_auc", "hamming_loss"]
    order = {(k, config_id(p)): n for k, g in ex["grids"].items() for n, p in enumerate(grid_points(g))}
    summ = cv.groupby(["method", "config", "params"], sort=False)[metrics].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ = summ.reset_index()
    summ["grid_order"] = [order[(m, c)] for m, c in zip(summ["method"], summ["config"])]
    best = {}
    for key, g in summ.groupby("method", sort=False):
        # highest mean CV macro-F1; exact ties go to the configuration listed first (simplest)
        top = g.sort_values(["macro_f1_mean", "grid_order"], ascending=[False, True]).iloc[0]
        best[key] = json.loads(top["params"])
    summ["selected"] = [json.loads(p) == best[m] for m, p in zip(summ["method"], summ["params"])]
    summ.to_csv(out_dir / "cv_summary.csv", index=False, float_format="%.6f", lineterminator="\n")
    write_json(best, out_dir / "best_params.json")

    # ---------- refit the selected configuration on the dev set; evaluate once on the test set ----------
    keys = list(ex["grids"])
    fin = [_run_one(*args_for(k, best[k], "final"), timed=True) for k in keys]  # sequential: fair timing
    test_rows = idx["final"][2]
    Y = Y_all[test_rows]
    preds = {k: (M.decide(r["P"], r["taus"], METHODS[k].multilabel), r["P"]) for k, r in zip(keys, fin)}

    pred_rows = []
    for k in keys:
        Yhat, P = preds[k]
        for i, r in enumerate(test_rows):
            pred_rows.append({"id": df["id"][r], "method": k,
                              **{f"true_{d}": int(Y[i, j]) for j, d in enumerate(conditions)},
                              **{f"score_{d}": round(float(P[i, j]), 4) for j, d in enumerate(conditions)},
                              **{f"pred_{d}": int(Yhat[i, j]) for j, d in enumerate(conditions)}})
    pred_path = out_dir / "test_predictions.csv"
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False, float_format="%.4f", lineterminator="\n")
    digest = sha256_file(pred_path)

    lo, hi = length_terciles(df, proto)
    n_tok = df.loc[test_rows, "text"].str.split().str.len().to_numpy()
    groups = {"single_condition": Y.sum(1) == 1, "multi_condition": Y.sum(1) > 1,
              "short": n_tok <= lo, "medium": (n_tok > lo) & (n_tok <= hi), "long": n_tok > hi}

    report = {"dataset": ex["dataset_name"], "n_test": int(len(Y)), "primary_metric": ex["primary_metric"],
              "selected_params": best, "methods": {}, "subgroups": {"length_cutpoints_tokens": [lo, hi]},
              "test_predictions_sha256": digest}
    for k, r in zip(keys, fin):
        Yhat, P = preds[k]
        rep = M.full_report(Y, Yhat, P, conditions)
        rep["name"] = METHODS[k].name
        rep["thresholds"] = None if r["taus"] is None else [float(t) for t in r["taus"]]
        rep["fit_seconds"], rep["predict_seconds"] = round(r["fit_seconds"], 3), round(r["predict_seconds"], 4)
        rep["label_set_confusion"] = M.combination_confusion(Y, Yhat)
        report["methods"][k] = rep
        report["subgroups"][k] = {g: {"n": int(mask.sum()), **M.summary(Y[mask], Yhat[mask], P[mask]),
                                      "multi_label_recall": M.multi_label_recall(Y[mask], Yhat[mask])}
                                  for g, mask in groups.items()}
    report["bootstrap"] = M.bootstrap(Y, preds, REF, ex["bootstrap"], seed)
    write_json(report, out_dir / "test_metrics.json")

    flat = []
    for k in keys:
        r, ci = report["methods"][k], report["bootstrap"]["ci95"][k]
        flat.append({"method": METHODS[k].name, **{m: r[m] for m in ["macro_f1", "micro_f1", "macro_auc", "hamming_loss"]},
                     "macro_f1_ci_low": ci["macro_f1"][0], "macro_f1_ci_high": ci["macro_f1"][1],
                     "multi_label_recall": r["multi_label_recall"],
                     **{f"{d}_{s}": r["per_condition"][d][s] for d in conditions for s in ("precision", "recall")},
                     "fit_seconds": r["fit_seconds"], "params": config_id(best[k])})
    pd.DataFrame(flat).to_csv(out_dir / "test_metrics.csv", index=False, float_format="%.4f", lineterminator="\n")

    _check_expected(out_dir, digest)
    for k in keys:
        r = report["methods"][k]
        log.info("%-17s macro-F1 %.3f  micro-F1 %.3f  AUC %.3f  (%s)", METHODS[k].name, r["macro_f1"],
                 r["micro_f1"], r["macro_auc"], config_id(best[k]))
    return report


def _data_card(df, proto, conditions, counts, cfg) -> dict:
    def stats(part):
        Y = part[conditions].to_numpy(dtype=int)
        combo = part[conditions].astype(str).agg("".join, axis=1).value_counts().sort_index()
        return {"n": int(len(part)), "positives": {d: int(part[d].sum()) for d in conditions},
                "label_cardinality": float(Y.sum(1).mean()), "multi_label_share": float((Y.sum(1) > 1).mean()),
                "label_combinations": {k: int(v) for k, v in combo.items()},
                "posts_per_community": {k: int(v) for k, v in part["community"].value_counts().sort_index().items()},
                "median_tokens": float(part["text"].str.split().str.len().median())}
    parts = proto.set_index("id")["part"]
    vec = make_vectorizer(cfg["features"]).fit(df.loc[parts.loc[df["id"]].to_numpy() == "dev", "text"])
    return {"dataset": cfg["experiment"]["dataset_name"], "source": cfg["data"]["source"],
            "cleaning_counts": counts, "all": stats(df),
            "dev": stats(df[parts.loc[df["id"]].to_numpy() == "dev"]),
            "test": stats(df[parts.loc[df["id"]].to_numpy() == "test"]),
            "tfidf_features_dev": int(len(vec.vocabulary_)),
            "features": cfg["features"]}


def _check_expected(out_dir, digest: str) -> None:
    exp = out_dir / "EXPECTED_SHA256"
    if exp.exists():
        want = exp.read_text().split()[0]
        if want == digest:
            log.info("test_predictions.csv matches EXPECTED_SHA256 (%s...)", digest[:12])
        else:
            log.warning("test_predictions.csv sha256 %s differs from EXPECTED_SHA256 %s", digest[:12], want[:12])
    else:
        exp.write_text(digest + "  test_predictions.csv\n", encoding="utf-8", newline="\n")
        log.info("wrote EXPECTED_SHA256 (%s...)", digest[:12])
