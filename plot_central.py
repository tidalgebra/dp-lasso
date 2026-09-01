"""Central privacy in simulation: Thresholded Central-DP Lasso against Cai et al. (2021).

Panel (a) is the mechanism. Peeling divides its per-round budget across the coordinates
it selects, so its error grows with the sparsity level; our noise is added once to the
whole gradient and does not. Panel (b) is the same comparison against sample size at the
two sparsity levels where the paired intervals exclude zero.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

import figstyle

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.4, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "lines.markersize": 3.4, "legend.frameon": False,
    "pdf.fonttype": 42,
})
BLUE, ORANGE, GREY, RED = "#3B6FB6", "#D98B3A", "#777777", "#B03030"
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"
OURS = "Thresholded Central-DP Lasso (ours)"
CAI = "Cai et al. DP-IHT"
PLAIN = "Non-private"

raw = pd.read_csv(RESULTS / "central_confirm.csv")


def summarize(frame, key):
    table = frame.groupby(key)["l2_error"].agg(mean="mean", sd="std",
                                               count="count").reset_index()
    table["half"] = (t.ppf(0.975, np.maximum(table["count"] - 1, 1))
                     * table["sd"] / np.sqrt(table["count"]))
    return table.sort_values(key)


fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), constrained_layout=True)

focus = raw[raw.n == 400_000]
ax = axes[0]
for method, color, marker, label in ((OURS, BLUE, "o", "Central-DP Lasso (ours)"),
                                     (CAI, ORANGE, "s", "DP-IHT (Cai et al.)"),
                                     (PLAIN, GREY, "^", "Non-private proximal Lasso")):
    part = summarize(focus[focus.method == method], "s_star")
    ax.plot(part.s_star, part["mean"], color=color, marker=marker, label=label, zorder=3)
    ax.fill_between(part.s_star, part["mean"] - part["half"], part["mean"] + part["half"],
                    color=color, alpha=0.12, linewidth=0, zorder=2)
ax.axhline(1.0, color=RED, linestyle=":", linewidth=0.8, zorder=1)
levels = sorted(focus.s_star.unique())
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(levels, labels=[str(v) for v in levels])
ax.minorticks_off()
ax.set_xlim(levels[0] / 1.4, levels[-1] * 1.4)
ax.text(levels[0] / 1.3, 1.03, " zero estimator", color=RED, fontsize=6.3,
        va="bottom", ha="left")
ax.set_xlabel(r"Sparsity $s^*$")
ax.set_ylabel(r"$\|\hat\beta-\beta^*\|_2$")
ax.set_title(r"(a)  cost of the sparsity level", fontsize=7.4, loc="left")

ax = axes[1]
LEVEL = 20                       # the smallest level at which every interval excludes zero
for method, color, marker in ((OURS, BLUE, "o"), (CAI, ORANGE, "s"), (PLAIN, GREY, "^")):
    part = summarize(raw[(raw.method == method) & (raw.s_star == LEVEL)], "n")
    ax.plot(part.n, part["mean"], color=color, marker=marker, zorder=3)
    ax.fill_between(part.n, part["mean"] - part["half"], part["mean"] + part["half"],
                    color=color, alpha=0.12, linewidth=0, zorder=2)
ax.axhline(1.0, color=RED, linestyle=":", linewidth=0.8, zorder=1)
ticks = sorted(raw.n.unique())
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(ticks, labels=["100k", "200k", "400k"])
ax.minorticks_off()
ax.set_xlim(ticks[0] / 1.3, ticks[-1] * 1.3)
ax.set_xlabel(r"Sample size $n$")
ax.set_ylabel(r"$\|\hat\beta-\beta^*\|_2$")
ax.set_title(r"(b)  sample size", fontsize=7.4, loc="left")

for ax in axes:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=2.6, pad=1.5)

figstyle.figure_legend(fig, axes, target=1, loc="lower left", bbox=(0.02, 0.19))

out = FIGURES
out.mkdir(exist_ok=True)
for suffix in ("pdf", "png"):
    fig.savefig(out / f"central_simulation.{suffix}", dpi=220, bbox_inches="tight",
                facecolor="white")
summarize(raw, ["n", "s_star", "method"]).to_csv(
    out / "central_simulation_source_data.csv", index=False)
print("wrote", out / "central_simulation.pdf")
