"""Per-seed box plots of the holdout eye-level metrics for the four conditions.

Each box is the spread over the five classifier seeds in
outputs/s05_classifier_micro5.json (the low-parameter classifier). The GAN
conditions are taken at the strongest dose (10x), which is the dose reported in
the results table.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "outputs", "s05_classifier_micro5.json")
OUT = os.path.join(ROOT, "figures", "fig_metrics_box.png")

DOSE = 10.0
CONDITIONS = [(("real_only", None), "real-only", "#4a9c4a"),
              (("oversample", None), "oversampled", "#7b52ab"),
              (("synth_clean", DOSE), "clean-GAN", "#2f7fbf"),
              (("synth_leak", DOSE), "contaminated-GAN", "#d1893c")]
PANELS = [("auc", "AUC (%)"),
          ("sensitivity", "Sensitivity (%)"),
          ("f1", "F1 score (%)")]

runs = defaultdict(list)
for r in json.load(open(SRC))["runs"]:
    runs[(r["arm"], r["dose"])].append(r["eye"])

fig, axes = plt.subplots(1, len(PANELS), figsize=(7.1, 3.4))

for ax, (key, ylabel) in zip(axes, PANELS):
    data = [[100 * s[key] for s in runs[c]] for c, _, _ in CONDITIONS]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True,
                    medianprops=dict(color="0.15", linewidth=1.4),
                    whiskerprops=dict(color="0.3"),
                    capprops=dict(color="0.3"),
                    flierprops=dict(marker="o", markersize=3,
                                    markerfacecolor="0.4",
                                    markeredgecolor="none"))
    for patch, (_, _, color) in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(color)
        patch.set_edgecolor("0.15")
        patch.set_alpha(0.85)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticklabels([lab for _, lab, _ in CONDITIONS], rotation=30,
                       ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)

    lines = ["%s: $\\mu$=%.2f, $\\sigma$=%.2f"
             % (lab, np.mean(v), np.std(v, ddof=1))
             for v, (_, lab, _) in zip(data, CONDITIONS)]
    ax.set_title("\n".join(lines), fontsize=6.5, color="0.25", loc="left",
                 linespacing=1.5)

fig.tight_layout()
fig.savefig(OUT, dpi=400)
print("wrote", os.path.normpath(OUT))
