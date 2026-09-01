"""Shared figure conventions: one legend below the panels, and the manuscript's names.

The legend is placed outside the axes so it can never sit on top of a curve, which also
lets every panel use the full plotting area.

`NAME` holds the labels exactly as the manuscript uses them after the first precise
definition. The dictionary keys are the `method` strings written into the result files;
those are never renamed, so old result files stay readable.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, GREY, RED = "#3B6FB6", "#D98B3A", "#5A8F5A", "#777777", "#B03030"

NAME = {
    "Label-LDP Lasso (ours)": "Label-LDP Lasso (ours)",
    "Full-Record LDP Lasso (ours)": "Full-Record LDP Lasso (ours)",
    "Thresholded Central-DP Lasso (ours)": "Central-DP Lasso (ours)",
    "Wang-Xu Label-LDP-IHT": "Label-LDP-IHT (Wang and Xu)",
    "Cai et al. DP-IHT": "DP-IHT (Cai et al.)",
    "Zhu et al. LDP-IHT": "LDP-IHT (Zhu et al.)",
    "Zhu et al. LDP-IHT, matched radius": "LDP-IHT update on our radius",
    "Non-private": "Non-private proximal Lasso",
    "Non-private proximal": "Non-private proximal Lasso",
    "Non-private, same procedure": "Non-private, same procedure",
    "Non-private ISTA": "Non-private proximal Lasso",
}

RCPARAMS = {
    "font.family": "DejaVu Sans", "font.size": 7.4, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.0, "lines.markersize": 3.4, "legend.frameon": False,
    "pdf.fonttype": 42,
}


def apply():
    plt.rcParams.update(RCPARAMS)


def finish(ax):
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.3, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=2.6, pad=1.5)


def figure_legend(fig, axes, ncol=None, target=0, loc="best", bbox=None):
    """One legend, drawn inside a panel rather than below the figure.

    A legend under the figure costs a band of vertical space in every panel. Placing it
    inside the panel with the most room keeps the plotting area, and `loc="best"` scores
    the candidate corners by how much data they would cover, so it lands in free space.
    Pass an explicit `loc` when a particular corner is known to be empty, and `bbox`
    to nudge it clear of a reference line that spans the panel.
    """
    panels = list(np.atleast_1d(axes).ravel())
    handles, labels = [], []
    for ax in panels:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if not handles:
        return
    panels[target].legend(handles, labels, loc=loc, ncol=ncol or 1, fontsize=6.4,
                          frameon=False, handlelength=1.9, borderaxespad=0.4,
                          labelspacing=0.35, bbox_to_anchor=bbox)
