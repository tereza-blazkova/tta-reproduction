#!/usr/bin/env python
"""fig_attainment_si — the student's record against the transcript (supplementary figure, stage 46).

Each model's paired difference from the attainment model (objective text + the student's attainment on
their other sessions), by the number of the student's other sessions in the training data: out of fold on clean rows
(a; the student's other training sessions) and on the TSL test set (b; the student's training
sessions). Top: log-loss reduction; bottom: AUROC gain (0-1 scale); up is better than knowing only
the student's record. Models, all the paper's XGBoost recipe: objective text only; the code model
(objective text + the 57 code rates, the paper's model); the codes and attainment in one model.
Paired 95% intervals (objectives resampled out of fold, students on test); hollow where the
interval spans zero.

Drawn at the text width and printed 1:1, with the main figures' sizes. Reads only
`figures/input/fig_attinc_models.csv` (`prepare_attainment_incremental_inputs.py`).

    python figures/scripts/fig_attainment_si.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style as st

W, H = st.TEXT_W, 2.55
LEFT, RIGHT, GAP_X = 0.50, 0.04, 0.42          # inches
TOP, BOTTOM, GAP_Y = 0.42, 0.36, 0.10
GROUPS = ["no other sessions", ">= 1", ">= 3", ">= 5"]
TICKS = ["none", "1+", "3+", "5+"]
VIEWS = {"oof": {"no other sessions": "out of fold, clean, no other sessions",
                 ">= 1": "out of fold, clean, >= 1 other sessions",
                 ">= 3": "out of fold, clean, >= 3 other sessions",
                 ">= 5": "out of fold, clean, >= 5 other sessions"},
         "test": {"no other sessions": "test, no training sessions",
                  ">= 1": "test, >= 1 training sessions",
                  ">= 3": "test, >= 3 training sessions",
                  ">= 5": "test, >= 5 training sessions"}}
#: model -> (label, fill, edge, marker, line style, x offset)
MODELS = {"objective model": ("Objective only", "#C9C9C5", "#7A7A76", "s", (0, (1, 1.5)), -0.2),
          "transcript model": ("Codes", "#2E2E2E", "#111111", "o", "-", 0.0),
          "both": ("Codes and attainment", "#8E8E8A", "#4A4A47", "D", (0, (3, 2)), 0.2)}
MS = {"s": 3.6, "o": 4.0, "D": 3.4}


def panel(ax, d: pd.DataFrame, view: dict, metric: str) -> None:
    est, lo, hi = {"ll": ("ll_vs_attainment", "ll_vs_attainment_lo", "ll_vs_attainment_hi"),
                   "auroc": ("auroc_vs_attainment", "auroc_vs_attainment_lo", "auroc_vs_attainment_hi")}[metric]
    scale = 1.0 if metric == "ll" else 0.01            # AUROC on its own 0-1 scale, as in Figure 2a
    ax.axhline(0, color=st.ATTAINMENT, lw=0.9, zorder=2)  # the attainment model
    xs = np.arange(len(GROUPS))
    for m, (_, fill, edge, mk, ls, dx) in MODELS.items():
        s = d[d.model == m].set_index("view").reindex([view[g] for g in GROUPS])
        e, l, h = (s[c].to_numpy() * scale for c in (est, lo, hi))
        ax.plot(xs + dx, e, color=edge, lw=0.7, ls=ls, zorder=3)
        for x, ei, li, hi_ in zip(xs + dx, e, l, h):
            ax.plot([x, x], [li, hi_], color=edge, lw=1.1, solid_capstyle="round", zorder=3)
            ax.plot([x], [ei], mk, ms=MS[mk], mfc="white" if li < 0 < hi_ else fill, mec=edge, mew=0.6,
                    zorder=4)
    ax.set_xticks(xs)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    # no grid: the SI figures have none (Fig. 5); the zero line is the reference


def main() -> None:
    st.apply_style(paper=True)
    d = pd.read_csv(st.INPUT / "fig_attinc_models.csv")
    fig = plt.figure(figsize=(W * st.SCALE, H * st.SCALE))
    pw = (W - LEFT - RIGHT - GAP_X) / 2
    ph = (H - TOP - BOTTOM - GAP_Y) / 2
    axes = {}
    for j, v in enumerate(("oof", "test")):
        for i, metric in enumerate(("ll", "auroc")):
            x0 = LEFT + j * (pw + GAP_X)
            y0 = BOTTOM + (1 - i) * (ph + GAP_Y)
            ax = fig.add_axes((x0 / W, y0 / H, pw / W, ph / H))
            panel(ax, d, VIEWS[v], metric)
            axes[v, metric] = ax
    for metric, label in (("ll", "Log loss"), ("auroc", "AUROC")):
        a, b = axes["oof", metric], axes["test", metric]
        lo = min(a.get_ylim()[0], b.get_ylim()[0])
        hi = max(a.get_ylim()[1], b.get_ylim()[1])
        for ax in (a, b):                                   # one scale per row, so a and b compare
            ax.set_ylim(lo, hi)
        a.set_ylabel(label, labelpad=3)
    for v in ("oof", "test"):
        axes[v, "ll"].tick_params(axis="x", labelbottom=False)
        axes[v, "auroc"].set_xticklabels(TICKS)
        axes[v, "auroc"].set_xlabel("Student's other sessions in the training data", labelpad=2)
    handles = [Line2D([], [], color=st.ATTAINMENT, lw=0.9, label="Attainment model (zero)")] + [
        Line2D([], [], color=edge, lw=0.7, ls=ls, marker=mk, ms=MS[mk], mfc=fill, mec=edge, mew=0.6, label=lab)
        for lab, fill, edge, mk, ls, _ in MODELS.values()]
    fig.legend(handles=handles, loc="center", ncol=4, frameon=False, handlelength=2.2, columnspacing=1.4,
               bbox_to_anchor=((LEFT + (W - LEFT - RIGHT) / 2) / W, 1 - 0.26 / H))
    for v, letter in zip(("oof", "test"), "ab"):
        st.letter_top_left(fig, axes[v, "ll"], letter)
    st.save(fig, "fig_attainment_si", width=W)


if __name__ == "__main__":
    main()
