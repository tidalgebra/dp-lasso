"""Central privacy diagnostics: geometry, and the separation of the two error sources.

Panel (a) is the sparsity and conditioning tradeoff for the thresholded step, which
carries no privacy noise at all.

Panel (b) separates the privacy noise from the optimization error.
The same trajectory is run twice, once with the calibrated gradient noise and once with
it removed and every other choice preserved. The optimization component falls by a
factor 62 between one and eight iterations and is then flat, while the calibrated noise
scale rises by a factor 6.4 across the range, so the total error is non-monotone with an
interior minimum. That rules out the reading in which a private curve looks poor only
because the optimizer was stopped early.

Data predates the v3 rewrite and is still current: the central algorithm is unchanged
apart from its name, and panel (a) contains no privacy noise.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

import figstyle

figstyle.apply()
BLUE, ORANGE, GREY, RED = figstyle.BLUE, figstyle.ORANGE, figstyle.GREY, figstyle.RED
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA = ROOT / "data"

geometry = pd.read_csv(RESULTS / "figure2a_geometry.csv")

trajectory = pd.concat(
    [pd.read_csv(RESULTS / name)
     for name in ("confirm_figure2bc.csv", "confirm_figure2bc_extra.csv")],
    ignore_index=True)
trajectory = trajectory[trajectory.panel == "b"]


def summarize(frame, key, column="value"):
    table = frame.groupby(key)[column].agg(mean="mean", sd="std",
                                           count="count").reset_index()
    table["half"] = (t.ppf(0.975, np.maximum(table["count"] - 1, 1))
                     * table["sd"] / np.sqrt(table["count"]))
    return table.sort_values(key)


fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8), constrained_layout=True)

ax = axes[0]
# The sweep is six design correlations at three threshold levels; the level is the
# curve and the correlation the axis, so the gain from over-specifying the support is
# visible where the design is worst conditioned. Open markers mark the correlations at
# which the revised thresholding condition fails.
for ratio, color, marker in zip(sorted(geometry.hat_s_ratio.unique()),
                                (BLUE, ORANGE, GREY), ("o", "s", "^")):
    part = geometry[geometry.hat_s_ratio == ratio].sort_values("correlation")
    ax.plot(part.correlation, part["mean"], color=color, marker=marker,
            label=rf"$\hat s / s^* = {ratio:g}$", zorder=3)
    ax.fill_between(part.correlation, part.ci_low, part.ci_high, color=color,
                    alpha=0.12, linewidth=0, zorder=2)
    failed = part[~part.revised_condition.astype(bool)]
    ax.plot(failed.correlation, failed["mean"], linestyle="none", marker=marker,
            markerfacecolor="white", markeredgecolor=color, markersize=3.4, zorder=4)
ax.set_yscale("log")
ax.set_xlabel(r"Design correlation $\varrho$")
ax.set_ylabel(r"$\|\hat\beta-\beta^*\|_2$")
ax.set_title("(a)  sparsity and conditioning", fontsize=7.4, loc="left")
figstyle.finish(ax)
ax.legend(loc="upper left", fontsize=6.4, frameon=False, labelspacing=0.35,
          handlelength=1.9)

ax = axes[1]
for method, color, marker, label in (
        ("Central DP", BLUE, "o", "total"),
        ("Optimization only", GREY, "^", "optimization only")):
    part = summarize(trajectory[(trajectory.method == method)
                                & (trajectory.metric == "l2_error")], "K")
    ax.plot(part.K, part["mean"], color=color, marker=marker, label=label, zorder=3)
    ax.fill_between(part.K, part["mean"] - part["half"], part["mean"] + part["half"],
                    color=color, alpha=0.12, linewidth=0, zorder=2)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
counts = sorted(trajectory.K.unique())
ax.set_xticks(counts, labels=[str(v) for v in counts])
ax.minorticks_off()
ax.set_xlabel(r"Iterations $K$")
ax.set_ylabel(r"$\|\hat\beta-\beta^*\|_2$")
ax.set_title("(b)  privacy noise against optimization error", fontsize=7.4, loc="left")
figstyle.finish(ax)

noise = summarize(trajectory[trajectory.metric == "noise_sd"], "K")
twin = ax.twinx()
twin.plot(noise.K, noise["mean"], color=ORANGE, marker="s", linestyle="--",
          label="noise SD", zorder=3)
twin.set_yscale("log")
twin.set_ylabel("Gradient noise SD", color=ORANGE)
twin.tick_params(axis="y", colors=ORANGE, direction="out", length=2.6, pad=1.5)
twin.spines["right"].set_visible(True)
twin.spines["right"].set_color(ORANGE)
twin.spines["top"].set_visible(False)

handles = ax.get_legend_handles_labels()
extra = twin.get_legend_handles_labels()
ax.legend(handles[0] + extra[0], handles[1] + extra[1], loc="center right",
          fontsize=6.4, frameon=False, labelspacing=0.3, handlelength=1.6,
          borderaxespad=0.5)

out = FIGURES
out.mkdir(exist_ok=True)
for suffix in ("pdf", "png"):
    fig.savefig(out / f"central_diagnostic.{suffix}", dpi=220, bbox_inches="tight",
                facecolor="white")
pd.concat([
    geometry.assign(panel="a")[["panel", "correlation", "hat_s_ratio", "mean",
                                "ci_low", "ci_high"]],
    summarize(trajectory[trajectory.metric == "l2_error"], ["K", "method"])
        .assign(panel="b"),
    noise.assign(panel="b", method="Calibrated gradient noise SD"),
], ignore_index=True).to_csv(out / "central_diagnostic_source_data.csv", index=False)
print("wrote", out / "central_diagnostic.pdf")
