"""Full-record local privacy: our procedure against Zhu et al. (2024).

A head-to-head between two procedures at fixed parameters. It is not a measurement of
either rate: both rates are upper bounds over a class, and one isotropic Gaussian design
is a single point in that class.
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
OURS, ZHU, PLAIN = ("Full-Record LDP Lasso (ours)", "Zhu et al. LDP-IHT",
                    "Non-private proximal")

raw = pd.read_csv(RESULTS / "exp1_confirm.csv")

# The comparator keeps its threshold-level sweep and is credited with its best cell.
best = []
for keys, group in raw[raw.method == ZHU].groupby(["s_star", "n"]):
    level = group.groupby("keep")["l2_error"].mean().idxmin()
    best.append(group[group.keep == level])
frame = pd.concat([raw[raw.method.isin([OURS, PLAIN])], *best], ignore_index=True)

table = frame.groupby(["s_star", "n", "method"])["l2_error"].agg(
    mean="mean", sd="std", count="count").reset_index()
table["half"] = (t.ppf(0.975, np.maximum(table["count"] - 1, 1))
                 * table["sd"] / np.sqrt(table["count"]))

fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), constrained_layout=True)
for ax, sparsity in zip(axes, sorted(table.s_star.unique())):
    part = table[table.s_star == sparsity]
    for method, color, marker, label in ((OURS, BLUE, "o", "Full-Record LDP Lasso (ours)"),
                                         (ZHU, ORANGE, "s", "LDP-IHT (Zhu et al.)"),
                                         (PLAIN, GREY, "^", "Non-private proximal Lasso")):
        arm = part[part.method == method].sort_values("n")
        ax.plot(arm.n, arm["mean"], color=color, marker=marker, label=label, zorder=3)
        ax.fill_between(arm.n, arm["mean"] - arm["half"], arm["mean"] + arm["half"],
                        color=color, alpha=0.11, linewidth=0, zorder=2)
    ax.axhline(1.0, color=RED, linestyle=":", linewidth=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = sorted(part.n.unique())
    ax.set_xticks(ticks, labels=["0.8M", "1.6M", "3.2M", "6.4M"])
    ax.minorticks_off()
    ax.set_xlim(ticks[0] / 1.35, ticks[-1] * 1.35)
    ax.set_xlabel(r"Sample size $n$")
    ax.set_title(rf"$s^*={sparsity}$", fontsize=7.4, loc="left")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=2.6, pad=1.5)
axes[0].set_ylabel(r"$\|\hat\beta-\beta^*\|_2$")
axes[0].text(0.97, 0.955, "zero estimator", transform=axes[0].transAxes,
             color=RED, fontsize=6.3, va="bottom", ha="right")

figstyle.figure_legend(fig, axes, target=0, loc="lower left", bbox=(0.01, 0.27))

out = FIGURES
out.mkdir(exist_ok=True)
for suffix in ("pdf", "png"):
    fig.savefig(out / f"full_record_confirm.{suffix}", dpi=220, bbox_inches="tight",
                facecolor="white")
table.to_csv(out / "full_record_confirm_source_data.csv", index=False)
print("wrote", out / "full_record_confirm.pdf")
