"""Central privacy on ACS records.

Panel (a) is the comparison against sample size, each arm at its own best threshold
level. Panel (b) is the mechanism: our noise does not grow with the number of retained
coordinates, so our error keeps falling as the level rises, while the peeling mechanism
splits its per-round budget across the selections and turns upward.
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
PLAIN = "Non-private, same procedure"

raw = pd.read_csv(RESULTS / "central_real_confirm.csv")
constant = raw[raw.method == "Constant predictor"].nmse.mean()


def summarize(frame, key):
    table = frame.groupby(key)["nmse"].agg(mean="mean", sd="std", count="count").reset_index()
    table["half"] = (t.ppf(0.975, np.maximum(table["count"] - 1, 1))
                     * table["sd"] / np.sqrt(table["count"]))
    return table


def best_level(method):
    rows = []
    for n, block in raw[raw.method == method].groupby("n"):
        level = block.groupby("level").nmse.mean().idxmin()
        rows.append(block[block.level == level])
    return summarize(pd.concat(rows), "n")


fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), constrained_layout=True)

ax = axes[0]
for method, color, marker, label in ((OURS, BLUE, "o", "Central-DP Lasso (ours)"),
                                     (CAI, ORANGE, "s", "DP-IHT (Cai et al.)"),
                                     (PLAIN, GREY, "^", "Non-private, same procedure")):
    part = best_level(method).sort_values("n")
    ax.plot(part.n, part["mean"], color=color, marker=marker, label=label, zorder=3)
    ax.fill_between(part.n, part["mean"] - part["half"], part["mean"] + part["half"],
                    color=color, alpha=0.12, linewidth=0, zorder=2)
ax.axhline(constant, color=RED, linestyle=":", linewidth=0.8, zorder=1)
ticks = sorted(raw.n.unique())
ax.set_xscale("log")
ax.set_xticks(ticks, labels=["25k", "50k", "100k", "200k"])
ax.minorticks_off()
ax.set_xlim(ticks[0] / 1.35, ticks[-1] * 1.35)
ax.text(ticks[0] / 1.25, constant, " constant predictor", color=RED, fontsize=6.3,
        va="bottom", ha="left")
ax.set_xlabel(r"Training size $n$")
ax.set_ylabel("Test NMSE")
ax.set_title(r"(a)  each arm at its best threshold level", fontsize=7.4, loc="left")

ax = axes[1]
focus = raw[raw.n == 100_000]
for method, color, marker, label in ((OURS, BLUE, "o", "Central-DP Lasso (ours)"),
                                     (CAI, ORANGE, "s", "DP-IHT (Cai et al.)")):
    part = summarize(focus[focus.method == method], "level").sort_values("level")
    ax.plot(part.level, part["mean"], color=color, marker=marker, label=label, zorder=3)
    ax.fill_between(part.level, part["mean"] - part["half"], part["mean"] + part["half"],
                    color=color, alpha=0.12, linewidth=0, zorder=2)
ax.axhline(constant, color=RED, linestyle=":", linewidth=0.8, zorder=1)
ax.set_xscale("log")
levels = sorted(focus.level.dropna().unique())
ax.set_xticks(levels, labels=[str(int(v)) for v in levels])
ax.minorticks_off()
ax.set_xlim(levels[0] / 1.4, levels[-1] * 1.4)
ax.set_xlabel("Threshold level")
ax.set_ylabel("Test NMSE")
ax.set_title(r"(b)  cost of retaining coordinates", fontsize=7.4, loc="left")

for ax in axes:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=2.6, pad=1.5)

figstyle.figure_legend(fig, axes, target=0, loc="upper right", bbox=(1.0, 0.92))

out = FIGURES
out.mkdir(exist_ok=True)
for suffix in ("pdf", "png"):
    fig.savefig(out / f"central_real.{suffix}", dpi=220, bbox_inches="tight",
                facecolor="white")
pd.concat([best_level(OURS).assign(series="ours"),
           best_level(CAI).assign(series="cai"),
           best_level(PLAIN).assign(series="non-private")]).to_csv(
    out / "central_real_source_data.csv", index=False)
print("wrote", out / "central_real.pdf")
