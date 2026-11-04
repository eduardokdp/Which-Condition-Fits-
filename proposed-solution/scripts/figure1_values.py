"""Print every number shown in Figure 1 of the report, computed from the trained quickstart model.

    python -m src.main all --config config/quickstart.yaml   # trains output/quickstart/model.joblib
    python scripts/figure1_values.py

Figure 1 is drawn with TikZ in proposed-solution/main.tex; its values are copied from this output.
All numbers are rounded once, from exact floating-point values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cmn.config import load_config, resolve  # noqa: E402
from src.core.decision import decide  # noqa: E402
from src.core.model import CoRE  # noqa: E402
from src.core.rationale import occlusion_deltas, select_evidence  # noqa: E402
from src.core.segment import join, segment  # noqa: E402
from src.dataprep.prepare import clean_text  # noqa: E402

cfg = load_config("config/quickstart.yaml")
model = CoRE.load(resolve(cfg["output_dir"]) / "model.joblib")
names, r = model.conditions, model.rcfg
phrases = segment(clean_text(cfg["demo_text"]))
P, D = occlusion_deltas(phrases, model.predict_proba)  # exact values
primary, secondary = decide(P, model.thresholds_)

print("phrases:", {f"p{j + 1}": ph for j, ph in enumerate(phrases)})
print("P(d|s):", {d: f"{p:.2f} ({p:.5f})" for d, p in zip(names, P)})
print("thresholds tau_d:", {d: f"{t:.2f}" for d, t in zip(names, model.thresholds_)})
for j in [primary, *secondary]:
    ev = select_evidence(D[:, j], r["delta"], r["max_fraction"], r["min_phrases"])
    print("primary" if j == primary else "secondary", names[j], "evidence", [f"p{e + 1}" for e in ev])
print("occlusion drops Delta_d(p_j) = P(d|s) - P(d|s without p_j):")
for j in range(len(phrases)):
    print(f"  p{j + 1}", {d: f"{D[j, k]:+.3f} ({D[j, k]:+.5f})" for k, d in enumerate(names)})

X = model._phi([join(phrases)])
chain, per_order = model.clf_, []
for order, links in zip(chain.orders, chain.links_):
    Q = np.zeros((1, chain.k))
    for pos, lab in enumerate(order):
        Q[:, lab] = links[pos].predict_proba(chain._augment(X, Q, list(order[:pos])))[:, 1]
    per_order.append(Q[0])
    print("chain order", tuple(names[i] for i in order), {d: f"{q:.2f}" for d, q in zip(names, Q[0])})
per_order = np.array(per_order)
print("range across the 6 orders:", {d: f"{per_order[:, j].min():.2f}-{per_order[:, j].max():.2f}"
                                     for j, d in enumerate(names)})
assert np.allclose(per_order.mean(0), P), "mean over orders must equal P(d|s)"
print("check: mean over orders equals P(d|s)")
