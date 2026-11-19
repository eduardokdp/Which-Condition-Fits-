"""LaTeX tables for the Milestone 4 report, generated from the experiment outputs so that the report
and the results cannot drift apart.

    python scripts/make_m4_tables.py --results evaluation/results/m4_sample --out ../experiments/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["core", "multiclass_lr", "br_lr", "chain1", "svm", "nb", "knn"]
NAMES = {"core": r"\core{}", "multiclass_lr": "Multiclass LR", "br_lr": "Binary relevance",
         "chain1": "Chain (1 order)", "svm": "Linear SVM", "nb": "Naive Bayes", "knn": "$k$-NN"}


def f3(x):
    return f"{x:.3f}".replace("0.", ".", 1) if x is not None else "--"


def data_table(dc: dict, out: Path) -> None:
    a, cc = dc["all"], dc["cleaning_counts"]
    pos = a["positives"]
    combo = a["label_combinations"]
    rows = [
        ("Posts (dev / test)", f"{a['n']:,} ({dc['dev']['n']} / {dc['test']['n']}); 350 per community"),
        ("Labels ADHD / anx. / dep.", f"{pos['adhd']} / {pos['anxiety']} / {pos['depression']} positive"),
        ("Label cardinality; multi-label", f"{a['label_cardinality']:.2f}; {100 * a['multi_label_share']:.1f}\\%"),
        ("Co-occurring sets", f"anx+dep {combo.get('011', 0)}, ADHD+anx {combo.get('110', 0)}, "
                              f"ADHD+dep {combo.get('101', 0)}, all {combo.get('111', 0)}"),
        ("Features (dev TF-IDF)", f"{dc['tfidf_features_dev']:,} word $n$-grams; median {a['median_tokens']:.0f} tokens"),
        ("Removed in cleaning", f"{cc['raw'] - cc['after_length_filter']} length, "
                                f"{cc['after_length_filter'] - cc['after_duplicate_text']} duplicate, "
                                f"{cc['after_duplicate_text'] - cc['after_cap']} over cap"),
        ("Source; licence", r"GitHub; ADHD 2024 (MIT), anx./dep. 2025 (none)"),
    ]
    body = "\n".join(f"{k} & {v} \\\\" for k, v in rows)
    out.write_text("\\begin{tabular}{@{}ll@{}}\n\\toprule\n" + body + "\n\\bottomrule\n\\end{tabular}\n",
                   encoding="utf-8", newline="\n")


def results_table(r: dict, out: Path) -> None:
    """RQ1 columns on all test posts, then RQ3 columns (macro-F1 per subgroup)."""
    lines = []
    paired = r["bootstrap"]["paired_vs_core"]
    groups = [("single_condition", "One"), ("multi_condition", "Several"), ("short", "Short"),
              ("medium", "Med."), ("long", "Long")]
    sg = r["subgroups"]
    best = {c: max(r["methods"][k][c] for k in ORDER) for c in ("macro_f1", "macro_auc", "micro_f1", "multi_label_recall")}
    best["hamming_loss"] = min(r["methods"][k]["hamming_loss"] for k in ORDER)
    gbest = {g: max(sg[k][g]["macro_f1"] for k in ORDER) for g, _ in groups}

    def b(v, ref):
        return f"\\textbf{{{f3(v)}}}" if v == ref else f3(v)

    for k in ORDER:
        m, ci = r["methods"][k], r["bootstrap"]["ci95"][k]["macro_f1"]
        if k == "core":
            diff = "--"
        else:
            lo, hi = paired[k]["macro_f1"]["ci95"]
            diff = f"{m_diff(r, k):+.3f}".replace("0.", ".") + ("$^\\ast$" if (lo > 0 or hi < 0) else "")
        cells = [f"{b(m['macro_f1'], best['macro_f1'])} {{\\scriptsize[{f3(ci[0])},{f3(ci[1])}]}}", diff,
                 b(m["macro_auc"], best["macro_auc"]), b(m["micro_f1"], best["micro_f1"]),
                 b(m["hamming_loss"], best["hamming_loss"]), b(m["multi_label_recall"], best["multi_label_recall"]),
                 f"{m['mean_labels_predicted']:.2f}"]
        cells += [b(sg[k][g]["macro_f1"], gbest[g]) for g, _ in groups]
        lines.append(f"{NAMES[k]} & " + " & ".join(cells) + " \\\\")
    n = {g: sg["core"][g]["n"] for g, _ in groups}
    head1 = ("& \\multicolumn{7}{c}{RQ1: all 210 test posts} & \\multicolumn{5}{c}{RQ3: macro-F1 by subgroup (posts)} \\\\\n"
             "\\cmidrule(lr){2-8}\\cmidrule(l){9-13}\n")
    head2 = ("Method & Macro-F1 [95\\% CI] & $\\Delta$ & AUC & Micro-F1 & Ham.$\\downarrow$ & ML-rec. & Lab. & "
             + " & ".join(f"{lab} {{\\scriptsize({n[g]})}}" for g, lab in groups) + " \\\\")
    out.write_text("\\begin{tabular}{@{}l@{\\hspace{4pt}}c" + "c" * 11 + "@{}}\n\\toprule\n" + head1 + head2
                   + "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n",
                   encoding="utf-8", newline="\n")


def m_diff(r: dict, k: str) -> float:
    return r["methods"]["core"]["macro_f1"] - r["methods"][k]["macro_f1"]


def subgroup_table(r: dict, out: Path) -> None:
    groups = [("single_condition", "Single"), ("multi_condition", "Multi"), ("short", "Short"),
              ("medium", "Medium"), ("long", "Long")]
    sg = r["subgroups"]
    n = " & ".join(f"{lab} {{\\scriptsize({sg['core'][g]['n']})}}" for g, lab in groups)
    lines = []
    for k in ["core", "multiclass_lr", "br_lr", "chain1", "svm"]:
        vals = [sg[k][g]["macro_f1"] for g, _ in groups]
        best = [max(sg[j][g]["macro_f1"] for j in ["core", "multiclass_lr", "br_lr", "chain1", "svm"]) for g, _ in groups]
        cells = [f"\\textbf{{{f3(v)}}}" if v == b else f3(v) for v, b in zip(vals, best)]
        lines.append(f"{NAMES[k]} & " + " & ".join(cells) + " \\\\")
    out.write_text("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nMethod & " + n + " \\\\\n\\midrule\n"
                   + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="evaluation/results/m4_sample")
    ap.add_argument("--out", default="../experiments/tables")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    res, out = root / args.results, (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    r = json.loads((res / "test_metrics.json").read_text())
    dc = json.loads((res / "data_card.json").read_text())
    data_table(dc, out / "tab_data.tex")
    results_table(r, out / "tab_results.tex")
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
