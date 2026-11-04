"""Draw the evaluation figures and a Markdown report for one run.

    python -m src.main all --config config/quickstart.yaml      # trains and evaluates first
    python scripts/make_figures.py --config config/quickstart.yaml

Writes PNG figures to evaluation/results/<run>/figures/{dark,light}/ and a report,
evaluation/results/<run>/REPORT.md, which uses the dark figures (open it in VS Code with
"Open Preview"). The light figures are for papers and slides. Needs matplotlib
(pip install -r requirements-viz.txt). Every number is read from the run's own outputs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cmn.config import load_config, resolve  # noqa: E402
from src.cmn.io import read_json  # noqa: E402
from src.core.model import CoRE  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
except ImportError:  # pragma: no cover
    sys.exit("matplotlib is not installed: pip install -r requirements-viz.txt")

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

QUICKSTART_DEMO = "I keep losing track of time, my heart races at work and I feel empty most days."
CORE = "CoRE (order-averaged chain)"
LABEL = {"adhd": "ADHD", "anxiety": "Anxiety", "depression": "Depression"}

# Colours: validated categorical slots (orange, aqua, violet) for the three conditions, blue for
# CoRE with the baselines in neutral grey, a blue ramp for counts, blue <-> red for occlusion drops.
THEMES = {
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", core="#3987e5", base="#6d6c67", mid="#383835", neg="#e66767",
                 pos="#3987e5", cond=["#d95926", "#199e70", "#9085e9"],
                 seq=["#1f2a3a", "#184f95", "#3987e5", "#86b6ef", "#cde2fb"]),
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", core="#2a78d6", base="#a9a8a1", mid="#f0efec", neg="#e34948",
                  pos="#2a78d6", cond=["#eb6834", "#1baf7a", "#6250d6"],
                  seq=["#eef4fc", "#b7d3f6", "#6da7ec", "#256abf", "#0d366b"]),
}


def style(t: dict) -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"], "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"], "xtick.color": t["muted"], "ytick.color": t["ink2"],
        "axes.titlecolor": t["ink"], "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.labelcolor": t["ink2"],
    })


def headline(fig, t, title: str, subtitle: str = "", pad_in: float = 0.45) -> None:
    """Figure title and wrapped subtitle placed in inches from the top edge; the plot area starts
    pad_in inches below the subtitle, which leaves room for the panel titles."""
    W, H = fig.get_size_inches()
    fig.text(0.012, 1 - 0.12 / H, title, ha="left", va="top", fontsize=14, weight="semibold", color=t["ink"])
    lines = textwrap.wrap(subtitle, int(W * 12.2)) if subtitle else []
    if lines:
        fig.text(0.012, 1 - 0.47 / H, "\n".join(lines), ha="left", va="top", fontsize=10.5,
                 color=t["ink2"], linespacing=1.35)
    head_in = 0.5 + 0.2 * len(lines)
    fig.subplots_adjust(top=1 - (head_in + pad_in) / H)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def fig_methods(m: dict, t: dict, path: Path) -> None:
    cls = m["classification"]
    methods = list(cls)[::-1]  # CoRE on top
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.16, wspace=0.08)
    for ax, key, name in zip(axes, ["micro_f1", "macro_f1"], ["Micro-F1", "Macro-F1"]):
        for y, meth in enumerate(methods):
            v, (lo, hi) = cls[meth][key], cls[meth][key + "_ci95"]
            c = t["core"] if meth == CORE else t["base"]
            ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([v], [y], s=90, color=c, edgecolor=t["surface"], linewidth=2, zorder=3)
            ax.text(hi + 0.006, y, f"{v:.3f}  [{lo:.3f}, {hi:.3f}]", va="center", fontsize=9.5,
                    color=t["ink"] if meth == CORE else t["ink2"])
        ax.set_title(name, pad=8)
        ax.set_xlim(0.52, 0.84)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("score (95% bootstrap CI)")
    axes[0].set_yticks(range(len(methods)), [x.replace(": ", ":\n").replace(" (", "\n(") for x in methods])
    pc = m["paired_comparisons"]
    overlap = all(v["ci95"][0] <= 0 <= v["ci95"][1] for v in pc.values())
    sub = (f"{m['n_test']} test posts. Paired micro-F1 differences: "
           + "; ".join(f"{k.replace('CoRE - ', 'vs ')} {v['micro_f1_diff']:+.3f} "
                       f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in pc.items())
           + (" - every interval includes 0." if overlap else "."))
    headline(fig, t, "CoRE against the two baselines", sub)
    save(fig, path)


def fig_other_metrics(m: dict, t: dict, path: Path) -> None:
    cls = m["classification"]
    methods = list(cls)
    short = {CORE: "CoRE", "Ablation: binary relevance": "Binary\nrelevance",
             "Baseline: single-label": "Single-\nlabel"}
    panels = [("hamming_loss", "Hamming loss", "lower is better"),
              ("subset_accuracy", "Subset accuracy", "higher is better"),
              ("multi_label_recall", "Recall on multi-label posts", "higher is better"),
              ("primary_hit_rate", "Primary condition correct", "higher is better")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.7))
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.17, wspace=0.28)
    for ax, (key, name, note) in zip(axes, panels):
        vals = [cls[x][key] for x in methods]
        cols = [t["core"] if x == CORE else t["base"] for x in methods]
        bars = ax.bar(range(3), vals, width=0.62, color=cols, edgecolor=t["surface"], linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9.5, color=t["ink"])
        ax.set_ylim(0, max(0.8, max(vals) * 1.25))
        ax.set_xticks(range(3), [short[x] for x in methods], fontsize=9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{name}\n", pad=2)
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=9, color=t["muted"])
    headline(fig, t, "Other multi-label metrics",
             "Single-label wins subset accuracy because most posts carry one label; "
             "the chain models trade it for recall on posts with several conditions.", pad_in=0.8)
    save(fig, path)


def fig_per_condition(m: dict, t: dict, conds: list[str], path: Path) -> None:
    pcd = m["classification"][CORE]["per_condition"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, wspace=0.12)
    for ax, key in zip(axes, ["precision", "recall", "f1"]):
        vals = [pcd[d][key] for d in conds]
        ys = np.arange(len(conds))[::-1]
        ax.barh(ys, vals, height=0.6, color=t["cond"][: len(conds)], edgecolor=t["surface"], linewidth=2)
        for y, v in zip(ys, vals):
            ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=10, color=t["ink"])
        ax.set_xlim(0, 1.0)
        ax.set_title({"f1": "F1"}.get(key, key.capitalize()), pad=8)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(conds))[::-1],
                       [f"{LABEL.get(d, d)}\n(n={pcd[d]['support']})" for d in conds])
    best = max(conds, key=lambda d: pcd[d]["f1"])
    low_p = min(conds, key=lambda d: pcd[d]["precision"])
    headline(fig, t, "CoRE per condition (test set)",
             f"{LABEL.get(best, best)} is recognised best (F1 {pcd[best]['f1']:.2f}); "
             f"{LABEL.get(low_p, low_p)} has the lowest precision ({pcd[low_p]['precision']:.2f}), "
             "so it is the most over-predicted label.")
    save(fig, path)


def fig_confusion(Y: np.ndarray, Yhat: np.ndarray, t: dict, conds: list[str], path: Path) -> None:
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    fig, axes = plt.subplots(1, len(conds), figsize=(11, 3.9))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, wspace=0.45)
    for k, (ax, d) in enumerate(zip(axes, conds)):
        y, p = Y[:, k], Yhat[:, k]
        cm = np.array([[np.sum((y == 1) & (p == 1)), np.sum((y == 1) & (p == 0))],
                       [np.sum((y == 0) & (p == 1)), np.sum((y == 0) & (p == 0))]])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=len(y))
        rows = cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                frac = cm[i, j] / max(rows[i, 0], 1)
                rgb = cmap(cm[i, j] / len(y))[:3]
                light_cell = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 0.55
                ax.text(j, i, f"{cm[i, j]}\n{frac:.0%} of row", ha="center", va="center", fontsize=10,
                        color="#0b0b0b" if light_cell else "#ffffff")
        ax.set_xticks([0, 1], ["predicted\nyes", "predicted\nno"], fontsize=9)
        ax.set_yticks([0, 1], ["actual\nyes", "actual\nno"], fontsize=9)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(LABEL.get(d, d), pad=8)
    headline(fig, t, "Where CoRE is right and wrong, per condition",
             f"Counts over {len(Y)} test posts; each cell also shows the share of its actual-class row.")
    save(fig, path)


def fig_probabilities(Y, P, thresholds, t, conds, path) -> None:
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, len(conds), figsize=(11, 3.9), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, wspace=0.1)
    peak = 0
    for k, (ax, d) in enumerate(zip(axes, conds)):
        pos, _ = np.histogram(P[Y[:, k] == 1, k], bins)
        neg, _ = np.histogram(P[Y[:, k] == 0, k], bins)
        peak = max(peak, pos.max(), neg.max())
        w = (bins[1] - bins[0]) * 0.86
        ax.bar(bins[:-1] + w / 2 + 0.003, pos, width=w, color=t["cond"][k], label="has condition")
        ax.bar(bins[:-1] + w / 2 + 0.003, -neg, width=w, color=t["base"], label="does not")
        ax.axhline(0, color=t["axis"], lw=1)
        tau = thresholds[k]
        ax.axvline(tau, color=t["ink"], lw=1.3, ls=(0, (4, 3)))
        ax.text(tau + 0.015, 0.97, f"threshold {tau:.2f}", transform=ax.get_xaxis_transform(),
                fontsize=9, color=t["ink"], va="top")
        ax.set_title(LABEL.get(d, d), pad=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"P({LABEL.get(d, d)} | s)")
        ax.text(0.01, 0.84, "posts with it", transform=ax.transAxes, fontsize=9, color=t["ink2"])
        ax.text(0.01, 0.06, "posts without it", transform=ax.transAxes, fontsize=9, color=t["ink2"])
    for ax in axes:
        ax.set_ylim(-peak * 1.15, peak * 1.25)
        ticks = ax.get_yticks()
        ax.set_yticks(ticks, [f"{abs(int(v))}" for v in ticks])
    axes[0].set_ylabel("test posts")
    headline(fig, t, "Predicted probabilities against the tuned thresholds",
             "Coloured bars (above the axis) are posts that carry the label, grey bars (below) are posts "
             "that do not; mass to the right of the dashed line is predicted positive.")
    save(fig, path)


def fig_faithfulness(m: dict, t: dict, path: Path) -> None:
    f = m["rationale_faithfulness"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.14, wspace=0.55)
    spec = [("comprehensiveness", "Comprehensiveness", "drop in P(d|s) when the evidence is deleted - higher is better"),
            ("sufficiency", "Sufficiency", "P(d|s) minus P(d|evidence only) - lower is better")]
    for ax, (key, name, note) in zip(axes, spec):
        vals = [f[key], f[key + "_random"]]
        ys = [1, 0]
        ax.barh(ys, vals, height=0.56, color=[t["core"], t["base"]], edgecolor=t["surface"], linewidth=2)
        ax.axvline(0, color=t["axis"], lw=1)
        span = max(abs(v) for v in vals)
        for y, v in zip(ys, vals):
            ax.text(v + (0.006 if v >= 0 else -0.006), y, f"{v:+.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=10, color=t["ink"])
        ax.set_xlim(min(0, min(vals)) - span * 0.45, max(0, max(vals)) + span * 0.45)
        ax.set_yticks(ys, ["CoRE evidence", "random phrases\n(same count)"])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(name, pad=20)
        ax.text(0, 1.04, note, transform=ax.transAxes, fontsize=9, color=t["muted"])
    headline(fig, t, "Are the evidence phrases faithful to the model? (ERASER)",
             f"{f['n_rationales']} rationales on the test set: deleting CoRE's evidence hurts the "
             f"prediction {f['comprehensiveness'] / max(f['comprehensiveness_random'], 1e-9):.1f}x more "
             "than deleting random phrases.", pad_in=0.75)
    save(fig, path)


def fig_evidence(model: CoRE, texts: list[str], t: dict, path: Path) -> None:
    outs = [model.explain(x) for x in texts]
    conds = model.conditions
    cmap = LinearSegmentedColormap.from_list("div", [t["neg"], t["mid"], t["pos"]])
    fig, axes = plt.subplots(len(outs), 1, figsize=(11, 2.3 + 1.25 * sum(len(o["phrases"]) for o in outs)))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.03, hspace=0.42)
    lim = max(max(abs(v) for o in outs for d in conds for v in o["occlusion_delta"][d]), 0.05)
    for n, (ax, o) in enumerate(zip(axes, outs)):
        D = np.array([o["occlusion_delta"][d] for d in conds]).T
        ax.imshow(D, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        predicted = {o["primary"]["condition"]: "primary"} | {s["condition"]: "secondary" for s in o["secondary"]}
        evid = {(e, conds.index(x["condition"])) for x in [o["primary"], *o["secondary"]] for e in x["evidence"]}
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                rgb = to_rgb(cmap((D[i, j] + lim) / (2 * lim)))
                light_cell = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] > 0.55
                mark = "  evidence" if (i, j) in evid else ""
                ax.text(j, i, f"{D[i, j]:+.3f}{mark}", ha="center", va="center", fontsize=10,
                        weight="semibold" if mark else "normal",
                        color="#0b0b0b" if light_cell else "#ffffff")
                if (i, j) in evid:
                    ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False,
                                               ec=t["ink"], lw=2))
        ax.set_yticks(range(len(o["phrases"])),
                      [f"p{i + 1}  " + "\n".join(textwrap.wrap(p, 44)) for i, p in enumerate(o["phrases"])],
                      fontsize=9.5)
        cols = []
        for j, d in enumerate(conds):
            tag = predicted.get(d, "not predicted")
            cols.append(f"{LABEL.get(d, d)}\nP = {o['probabilities'][d]:.3f} (threshold {model.thresholds_[j]:.2f})\n{tag}")
        ax.set_xticks(range(len(conds)), cols, fontsize=9)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.annotate(f"Example {n + 1}:  \"{texts[n]}\"", xy=(0.012, 1.0),
                    xycoords=("figure fraction", "axes fraction"), xytext=(0, 62), textcoords="offset points",
                    ha="left", va="bottom", fontsize=11, weight="semibold", color=t["ink"])
    headline(fig, t, "Which phrases drive each prediction (occlusion)",
             "Each cell is the drop in P(d|s) when that phrase is erased. Blue = the phrase supports the "
             "condition, red = it argues against it; boxed cells are the reported evidence.", pad_in=1.45)
    save(fig, path)


def fig_data(cfg: dict, t: dict, conds: list[str], path: Path) -> dict:
    pre = resolve(cfg["data"]["preprocessed_dir"]) / cfg["run_name"]
    counts = {}
    for s in ["train", "val", "test"]:
        df = pd.read_csv(pre / f"{s}.csv")
        counts[s] = {"n": len(df), **{d: int(df[d].sum()) for d in conds},
                     "multi": int((df[conds].sum(axis=1) >= 2).sum())}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.12, wspace=0.12)
    for ax, s in zip(axes, ["train", "val", "test"]):
        vals = [counts[s][d] for d in conds]
        ys = np.arange(len(conds))[::-1]
        ax.barh(ys, vals, height=0.6, color=t["cond"][: len(conds)], edgecolor=t["surface"], linewidth=2)
        for y, v in zip(ys, vals):
            ax.text(v + 4, y, f"{v}", va="center", fontsize=10, color=t["ink"])
        ax.set_title(f"{s}  (n = {counts[s]['n']}, multi-label {counts[s]['multi'] / counts[s]['n']:.0%})", pad=8)
        ax.set_xlim(0, max(counts["train"][d] for d in conds) * 1.2)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(len(conds))[::-1], [LABEL.get(d, d) for d in conds])
    headline(fig, t, "Posts carrying each label, per split",
             "A post can carry several labels (community plus self-reported diagnoses), so rows can sum to more than n.")
    save(fig, path)
    return counts


# ---------------------------------------------------------------------------------------------
def write_report(cfg, m, counts, fig_rel, path: Path) -> None:
    cls, f = m["classification"], m["rationale_faithfulness"]
    row = lambda k, r: (f"| {k} | {r['micro_f1']:.3f} ({r['micro_f1_ci95'][0]:.3f}-{r['micro_f1_ci95'][1]:.3f}) "  # noqa: E731
                        f"| {r['macro_f1']:.3f} ({r['macro_f1_ci95'][0]:.3f}-{r['macro_f1_ci95'][1]:.3f}) "
                        f"| {r['hamming_loss']:.3f} | {r['subset_accuracy']:.3f} | {r['multi_label_recall']:.3f} |")
    expected_file = resolve(cfg["eval_dir"]) / "EXPECTED_SHA256"
    status = ""
    if expected_file.exists():
        ok = expected_file.read_text().strip() == m["predictions_sha256"]
        status = ("**Reproducibility check: PASSED** - predictions match the reported run."
                  if ok else "**Reproducibility check: hash differs** from EXPECTED_SHA256; "
                             "compare library versions with requirements.txt.")
    test = counts["test"]
    md = f"""# CoRE results - `{cfg['run_name']}` run

{status}

`{m['predictions_sha256']}`

Seed {m['seed']} | {counts['train']['n']} / {counts['val']['n']} / {test['n']} posts (train / val / test) |
thresholds {", ".join(f"{LABEL.get(d, d)} {v:.2f}" for d, v in m['thresholds'].items())}

| Method | micro-F1 (95% CI) | macro-F1 (95% CI) | Hamming loss | subset accuracy | recall on multi-label posts |
|---|---|---|---|---|---|
""" + "\n".join(row(f"**{k}**" if k == CORE else k, r) for k, r in cls.items()) + f"""

Rationale faithfulness: comprehensiveness {f['comprehensiveness']:.3f} (random {f['comprehensiveness_random']:.3f}),
sufficiency {f['sufficiency']:+.3f} (random {f['sufficiency_random']:+.3f}), over {f['n_rationales']} rationales.

## Pipeline

![CoRE pipeline](../../../docs/figures/core_pipeline.png)

## 1. CoRE against the baselines

![Method comparison]({fig_rel}/1_method_comparison.png)

![Other metrics]({fig_rel}/2_other_metrics.png)

## 2. Per-condition performance

![Per condition]({fig_rel}/3_per_condition.png)

![Confusion matrices]({fig_rel}/4_confusion.png)

![Probability distributions]({fig_rel}/5_probabilities.png)

## 3. Evidence phrases

![Faithfulness]({fig_rel}/6_faithfulness.png)

![Occlusion evidence]({fig_rel}/7_evidence_examples.png)

## 4. Data

![Data]({fig_rel}/8_data.png)

---
Generated by `python scripts/make_figures.py --config {Path(cfg['_config_path']).relative_to(resolve('.')).as_posix()}`.
Light-background versions of every figure are in `figures/light/` for the report and slides.
Research prototype only - not a diagnostic tool.
"""
    path.write_text(md, encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/quickstart.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    conds = cfg["conditions"]
    eval_dir = resolve(cfg["eval_dir"])
    m = read_json(eval_dir / "metrics.json")
    model = CoRE.load(resolve(cfg["output_dir"]) / "model.joblib")
    test = pd.read_csv(resolve(cfg["data"]["preprocessed_dir"]) / cfg["run_name"] / "test.csv")
    Y = test[conds].to_numpy(dtype=int)
    P = model.predict_proba(test["text"].tolist())
    Yhat = model.predict_labels(test["text"].tolist())
    examples = [cfg.get("demo_text", QUICKSTART_DEMO), QUICKSTART_DEMO]

    counts = None
    for mode, t in THEMES.items():
        style(t)
        out = eval_dir / "figures" / mode
        fig_methods(m, t, out / "1_method_comparison.png")
        fig_other_metrics(m, t, out / "2_other_metrics.png")
        fig_per_condition(m, t, conds, out / "3_per_condition.png")
        fig_confusion(Y, Yhat, t, conds, out / "4_confusion.png")
        fig_probabilities(Y, P, model.thresholds_, t, conds, out / "5_probabilities.png")
        fig_faithfulness(m, t, out / "6_faithfulness.png")
        fig_evidence(model, examples, t, out / "7_evidence_examples.png")
        counts = fig_data(cfg, t, conds, out / "8_data.png")
    write_report(cfg, m, counts, "figures/dark", eval_dir / "REPORT.md")
    print(f"figures: {eval_dir / 'figures'}")
    print(f"report:  {eval_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
