"""Label-local comparison in the submitted paper's three-setting layout."""

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
TITLES = {"gaussian": "(a)  Gaussian design",
          "bounded_uniform": "(b)  Bounded uniform design",
          "rademacher": "(c)  Rademacher design"}
ARMS = (("Label-LDP Lasso (ours)", BLUE, "o"),
        ("Wang-Xu Label-LDP-IHT", ORANGE, "s"),
        ("Non-private", GREY, "^"))


def summarize(frame):
    table = frame.groupby(["setting", "n", "method"])["l2_error"].agg(
        mean="mean", sd="std", count="count").reset_index()
    half = t.ppf(0.975, np.maximum(table["count"] - 1, 1)) * table["sd"] / np.sqrt(table["count"])
    table["half"] = np.where(table["count"] > 1, half, 0.0)
    return table


def draw(frame, stem, subtitle):
    table = summarize(frame)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7), constrained_layout=True)
    for ax, setting in zip(axes, ("gaussian", "bounded_uniform", "rademacher")):
        part = table[table.setting == setting]
        for method, color, marker in ARMS:
            arm = part[part.method == method].sort_values("n")
            ax.plot(arm.n, arm["mean"], color=color, marker=marker, label=figstyle.NAME.get(method, method), zorder=3)
            ax.fill_between(arm.n, arm["mean"] - arm["half"], arm["mean"] + arm["half"],
                            color=color, alpha=0.10, linewidth=0, zorder=2)
        ax.axhline(1.0, color=RED, linestyle=":", linewidth=0.8, zorder=1)
        ax.set_xscale("log")
        ticks = sorted(part.n.unique())
        ax.set_xticks(ticks, labels=[f"{v // 1000}k" for v in ticks])
        ax.minorticks_off()
        ax.set_xlim(ticks[0] / 1.3, ticks[-1] * 1.3)
        ax.set_ylim(0.0, 1.42)
        ax.set_xlabel(r"Sample size $n$")
        ax.set_title(TITLES[setting], fontsize=7.4, loc="left")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.3, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(direction="out", length=2.6, pad=1.5)
    axes[0].set_ylabel(r"$\|\hat\beta-\beta^*\|_2$")
    # The setting is stated in the caption, not on the figure.

    axes[2].text(0.97, 0.735, "zero estimator", transform=axes[2].transAxes,
                 color=RED, fontsize=6.3, va="bottom", ha="right")
    figstyle.figure_legend(fig, axes, target=0, loc="upper right", bbox=(1.0, 1.0))

    out = FIGURES
    out.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    table.to_csv(out / f"{stem}_source_data.csv", index=False)
    return table


def paired(frame, label):
    ours = frame[frame.method.str.contains("Label-LDP Lasso (ours)")].set_index(["setting", "n", "seed"])["l2_error"]
    base = frame[frame.method.str.contains("Wang")].set_index(["setting", "n", "seed"])["l2_error"]
    gap = (base - ours).dropna().sort_index()
    print(f"\n{label}: paired reduction, Wang-Xu minus ours")
    print("%-17s %-7s %-9s %-9s %s" % ("setting", "n", "paired", "95% half", "seeds"))
    for (setting, n), group in gap.groupby(level=[0, 1]):
        v = group.to_numpy()
        half = t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        print("%-17s %-7d %+.4f   %-9.4f %d/%d"
              % (setting, n, v.mean(), half, int((v > 0).sum()), len(v)))


results = RESULTS
main = pd.read_csv(results / "label_v1format_confirm.csv")
draw(main, "label_v1format", "Label-local privacy under a matched release, "
                             r"$\varepsilon=0.95$, $s^*=5$, decaying coefficient profile")
paired(main, "decaying profile (seeds 6301-6308)")

flat = results / "label_v1format_confirm_flat.csv"
if flat.exists():
    frame = pd.read_csv(flat)
    draw(frame, "label_v1format_flat_profile",
         "Scope check: equal-magnitude coefficients, same layout and budget")
    paired(frame, "equal-magnitude profile (seeds 6201-6208)")
print("\nwrote figures to", FIGURES)
