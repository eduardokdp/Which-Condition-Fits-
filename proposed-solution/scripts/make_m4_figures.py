"""Milestone 4 figures, drawn from the files written by `python -m src.main experiment`.

    python scripts/make_m4_figures.py --results evaluation/results/m4_rmhd [--also evaluation/results/m4_sample]

Writes to <results>/figures/:
    rq2_sensitivity.pdf/.png   mean cross-validated macro-F1 (+/- 1 s.d. over the 5 folds) per setting
    label_set_confusion.pdf/.png   true vs predicted label set on the test posts, CoRE vs multiclass LR
Colours follow the method (never its rank); every series also has its own marker, so identity never
relies on colour alone (colour-blind readers, black-and-white print).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = {  # validated categorical palette (light surface), fixed order
    "core": ("CoRE", "#2a78d6", "o"), "multiclass_lr": ("Multiclass LR", "#eb6834", "s"),
    "br_lr": ("Binary relevance", "#1baf7a", "^"), "chain1": ("Chain (1 order)", "#eda100", "D"),
    "knn": ("k-NN", "#e87ba4", "v"), "svm": ("Linear SVM", "#008300", "P"), "nb": ("Naive Bayes", "#6250d6", "X"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Linux Libertine O", "Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.linewidth": 0.6, "lines.linewidth": 1.4, "lines.markersize": 3.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def _params(s: str) -> dict:
    return json.loads(s)


def _curve(ax, summ, method, x, fixed: dict | None = None, label=None, dashed=False):
    name, color, marker = STYLE[method]
    g = summ[summ["method"] == method].copy()
    g["p"] = g["params"].map(_params)
    for k, v in (fixed or {}).items():
        g = g[g["p"].map(lambda p: p[k] == v)]
    g["x"] = g["p"].map(lambda p: p[x])
    g = g.sort_values("x")
    m, s = g["macro_f1_mean"].to_numpy(), g["macro_f1_std"].to_numpy()
    ax.fill_between(g["x"], m - s, m + s, color=color, alpha=0.10, lw=0)
    ax.plot(g["x"], m, color=color, marker=marker, ls="--" if dashed else "-", label=label or name,
            markeredgecolor="white", markeredgewidth=0.4)
    sel = g[g["selected"]]
    if len(sel):
        ax.plot(sel["x"], sel["macro_f1_mean"], marker=marker, ms=7, mfc="none", mec=INK, mew=0.8, ls="none")


def _axes(ax, title, xlabel, log=False):
    ax.set_title(title, loc="left", color=INK, pad=3)
    ax.set_xlabel(xlabel)
    if log:
        ax.set_xscale("log")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def rq2(results: Path, out: Path) -> None:
    summ = pd.read_csv(results / "cv_summary.csv")
    best = json.loads((results / "best_params.json").read_text())
    ng = best["core"]["ngram_range"]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.48), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]},
                            sharey=True, constrained_layout=True)
    a = axs[0]
    _curve(a, summ, "core", "C", {"ngram_range": ng, "augment": "oof"}, "CoRE (OOF inputs)")
    _curve(a, summ, "core", "C", {"ngram_range": ng, "augment": "true"}, "CoRE (gold inputs)", dashed=True)
    for m in ("multiclass_lr", "br_lr", "chain1"):
        _curve(a, summ, m, "C", {"ngram_range": best[m]["ngram_range"]})
    _axes(a, "(a) Logistic models: regularisation C", "C (log scale)", log=True)
    a.set_ylabel("Mean CV macro-F1")
    _curve(axs[1], summ, "knn", "k")
    _axes(axs[1], "(b) k-NN: neighbours k", "k")
    _curve(axs[2], summ, "svm", "C")
    _axes(axs[2], "(c) Linear SVM: C", "C (log scale)", log=True)
    _curve(axs[3], summ, "nb", "alpha")
    _axes(axs[3], "(d) Naive Bayes: smoothing", "alpha (log scale)", log=True)
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="outside lower center", ncol=8, frameon=False, handlelength=1.8,
               columnspacing=0.9, handletextpad=0.4)
    _save(fig, out / "rq2_sensitivity")


def confusion(results: Path, out: Path) -> None:
    rep = json.loads((results / "test_metrics.json").read_text())
    names = {"100": "H", "010": "A", "001": "D", "110": "H+A", "101": "H+D", "011": "A+D", "111": "H+A+D"}
    keys = list(names)
    fig, axs = plt.subplots(1, 2, figsize=(3.33, 1.45), constrained_layout=True)
    mats = []
    for m in ("core", "multiclass_lr"):
        c = rep["methods"][m]["label_set_confusion"]
        M = np.array([[c.get(t, {}).get(p, 0) for p in keys] for t in keys], dtype=float)
        mats.append(M)
    vmax = max(M.max() for M in mats)
    for ax, M, m in zip(axs, mats, ("core", "multiclass_lr")):
        rows = M.sum(1) > 0
        Mr = M[rows]
        ax.imshow(Mr, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(Mr.shape[0]):
            for j in range(Mr.shape[1]):
                if Mr[i, j]:
                    ax.text(j, i, int(Mr[i, j]), ha="center", va="center", fontsize=5,
                            color="white" if Mr[i, j] > vmax * 0.55 else INK)
        ax.set_xticks(range(len(keys)), [names[k] for k in keys], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(rows.sum()), [names[k] for k, r in zip(keys, rows) if r])
        ax.set_title(STYLE[m][0], loc="left", color=INK, pad=3)
        ax.set_xlabel("Predicted label set")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_ylabel("True label set")
    _save(fig, out / "label_set_confusion")


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["evaluation/results/m4_sample"])
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    for r in args.results:
        res = root / r
        rq2(res, res / "figures")
        confusion(res, res / "figures")
        print(f"figures written to {res / 'figures'}")


if __name__ == "__main__":
    main()
