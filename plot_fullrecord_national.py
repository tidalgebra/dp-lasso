"""Full-record local privacy on the national ACS extract.

One panel: test NMSE against training size, the comparator credited with its best
truncation and threshold cell at each size. The marginal intervals already separate at
every size above 200,000 records, so a second panel for the paired difference would
restate what this one shows; the paired numbers stay in the record.

A third arm, the comparator's update run on our clipping radius, is computed and stored
but not drawn: it is not their algorithm, and the other two placement figures compare
against a published method rather than a hybrid. Its numbers are in the record.
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
OURS = "Full-Record LDP Lasso (ours)"
SPEC = "Zhu et al. LDP-IHT"

raw = pd.read_csv(RESULTS / "fullrecord_real_confirm.csv")
constant = raw[raw.method == "Constant predictor"].nmse.mean()


def interval(values):
    half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean(), half


def series():
    rows = []
    for n, block in raw.groupby("n"):
        ours = block[block.method == OURS].set_index("seed").nmse
        spec = block[block.method == SPEC]
        cell = spec.groupby(["tau", "keep"]).nmse.mean().idxmin()
        best = spec[(spec.tau == cell[0]) & (spec.keep == cell[1])].set_index("seed").nmse
        gap = (best - ours).dropna()
        mean_o, half_o = interval(ours.to_numpy())
        mean_s, half_s = interval(best.to_numpy())
        mean_g, half_g = interval(gap.to_numpy())
        plain = block[block.method == "Non-private proximal"].nmse
        mean_p, half_p = interval(plain.to_numpy())
        rows.append(dict(n=n, ours=mean_o, ours_half=half_o, spec=mean_s,
                         spec_half=half_s, plain=mean_p, plain_half=half_p,
                         gap=mean_g, gap_half=half_g, agree=int((gap > 0).sum()),
                         trials=len(gap), tau=cell[0], keep=cell[1]))
    return pd.DataFrame(rows).sort_values("n")


table = series()
ticks = table.n.tolist()

fig, ax = plt.subplots(figsize=(3.9, 2.9), constrained_layout=True)
for key, color, marker, label in (("ours", BLUE, "o", "Full-Record LDP Lasso (ours)"),
                                  ("spec", ORANGE, "s", "LDP-IHT (Zhu et al.)"),
                                  ("plain", GREY, "^", "Non-private proximal Lasso")):
    ax.plot(table.n, table[key], color=color, marker=marker, label=label, zorder=3)
    ax.fill_between(table.n, table[key] - table[f"{key}_half"],
                    table[key] + table[f"{key}_half"], color=color, alpha=0.12,
                    linewidth=0, zorder=2)
ax.axhline(constant, color=RED, linestyle=":", linewidth=0.8, zorder=1)
ax.text(0.97, 0.955, "constant predictor", transform=ax.transAxes, color=RED,
        fontsize=6.3, va="bottom", ha="right")
ax.set_ylim(table.plain.min() - 0.012, constant + 0.012)
ax.set_ylabel("Test NMSE")

ax.set_xscale("log")
ax.set_xticks(ticks, labels=["200k", "400k", "800k", "1.6M"])
ax.minorticks_off()
ax.set_xlim(ticks[0] / 1.35, ticks[-1] * 1.35)
ax.set_xlabel(r"Training size $n$")
figstyle.finish(ax)
figstyle.figure_legend(fig, [ax], loc="upper left", bbox=(0.01, 0.92))

out = FIGURES
out.mkdir(exist_ok=True)
for suffix in ("pdf", "png"):
    fig.savefig(out / f"fullrecord_national.{suffix}", dpi=220, bbox_inches="tight",
                facecolor="white")
table.to_csv(out / "fullrecord_national_source_data.csv", index=False)
print(table.round(4).to_string(index=False))
print("wrote", out / "fullrecord_national.pdf")
