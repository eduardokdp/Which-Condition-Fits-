"""Evaluate CoRE and the baselines on the held-out test split and write results to evaluation/."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cmn.config import resolve
from src.cmn.io import get_logger, sha256_file, write_json, write_jsonl
from src.core.model import CoRE
from src.core.rationale import comprehensiveness, sufficiency
from src.eval.baselines import SingleLabel
from src.eval.metrics import bootstrap_ci, multilabel_report

log = get_logger()


def faithfulness(model: CoRE, texts: list[str], seed: int = 0) -> tuple[dict, list[dict]]:
    """ERASER comprehensiveness and sufficiency of the occlusion rationales, next to the same scores
    for random rationales of equal size (seeded), so the numbers can be read against a reference."""
    rng = np.random.default_rng(seed)
    comp, suff, comp_rnd, suff_rnd, rows = [], [], [], [], []
    for i, t in enumerate(texts):
        out = model.explain(t)
        for item in [out["primary"], *out["secondary"]]:
            j = model.conditions.index(item["condition"])
            ev, n = item["evidence"], len(out["phrases"])
            if not ev or len(ev) == n:  # nothing to compare when no phrase or every phrase is evidence
                continue
            rnd = sorted(rng.choice(n, size=len(ev), replace=False).tolist())
            comp.append(comprehensiveness(out["phrases"], ev, model.predict_proba, j))
            suff.append(sufficiency(out["phrases"], ev, model.predict_proba, j))
            comp_rnd.append(comprehensiveness(out["phrases"], rnd, model.predict_proba, j))
            suff_rnd.append(sufficiency(out["phrases"], rnd, model.predict_proba, j))
        rows.append({"row": i, **out})
    mean = lambda v: round(float(np.mean(v)), 4) if v else None  # noqa: E731
    summary = {"comprehensiveness": mean(comp), "comprehensiveness_random": mean(comp_rnd),
               "sufficiency": mean(suff), "sufficiency_random": mean(suff_rnd), "n_rationales": len(comp)}
    return summary, rows


def evaluate(cfg: dict, splits: dict[str, pd.DataFrame], model: CoRE) -> dict:
    conditions = cfg["conditions"]
    test, train, val = splits["test"], splits["train"], splits["val"]
    Y = test[conditions].to_numpy(dtype=int)
    texts = test["text"].tolist()

    gt_dir = resolve(cfg["ground_truth_dir"])
    gt_dir.mkdir(parents=True, exist_ok=True)
    test[["id", "community", *conditions]].to_csv(gt_dir / "test_labels.csv", index=False, lineterminator="\n")

    seed = cfg["seed"]
    results, preds = {}, {}
    preds["CoRE (order-averaged chain)"] = (model.predict_labels(texts), model.predict_proba(texts))
    br = CoRE.from_config(cfg, chain="br").fit(train, val)
    preds["Ablation: binary relevance"] = (br.predict_labels(texts), br.predict_proba(texts))
    sl = SingleLabel(conditions, cfg["features"], cfg["model"], cfg["data"]["community_to_condition"]).fit(train)
    preds["Baseline: single-label"] = (sl.predict_labels(texts), sl.predict_proba(texts))
    for name, (Yhat, P) in preds.items():
        results[name] = multilabel_report(Y, Yhat, conditions, P, seed=seed)

    # paired bootstrap: is CoRE's micro-F1 difference from each baseline larger than sampling noise?
    core_hat = preds["CoRE (order-averaged chain)"][0]
    comparisons = {f"CoRE - {name}": {"micro_f1_diff": round(results["CoRE (order-averaged chain)"]["micro_f1"]
                                                             - results[name]["micro_f1"], 4),
                                      "ci95": bootstrap_ci(Y, core_hat, "micro", seed=seed, Yhat_ref=Yhat)}
                   for name, (Yhat, _) in preds.items() if name != "CoRE (order-averaged chain)"}

    n_explain = min(len(texts), cfg.get("n_explain", 200))
    faith, rows = faithfulness(model, test["post"].tolist()[:n_explain], seed=seed)
    for r, rid in zip(rows, test["id"].tolist()[:n_explain]):
        r["id"] = rid

    out_dir = resolve(cfg["eval_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    write_jsonl(rows, pred_path)
    report = {"run_name": cfg["run_name"], "seed": cfg["seed"], "n_test": len(test),
              "thresholds": {d: float(t) for d, t in zip(conditions, model.thresholds_)},
              "classification": results, "paired_comparisons": comparisons,
              "rationale_faithfulness": faith,
              "predictions_sha256": sha256_file(pred_path)}
    write_json(report, out_dir / "metrics.json")
    flat = pd.DataFrame([{"method": k, **{m: (f"{v[0]}-{v[1]}" if isinstance(v, list) else v)
                                          for m, v in r.items() if m != "per_condition"}}
                         for k, r in results.items()])
    flat.to_csv(out_dir / "metrics.csv", index=False, lineterminator="\n")
    log.info("\n%s", flat.to_string(index=False))
    log.info("paired comparisons (micro-F1): %s", comparisons)
    log.info("rationale faithfulness: %s", faith)
    log.info("predictions sha256: %s", report["predictions_sha256"])
    return report
