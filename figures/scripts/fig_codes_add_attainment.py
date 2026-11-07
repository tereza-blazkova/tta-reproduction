#!/usr/bin/env python
"""fig_codes_add_attainment — per code, how much it differs by attainment and what it adds to the
attainment model (SI figure; replaces fig_codes_by_attainment).

All 57 codes, one row each, sorted by (b):
  a  how much more common the code is in the sessions of the top third of students by attainment than
     in those of the bottom third, in % of the bottom third's share (stage 48). Filled: significant
     after Benjamini-Hochberg across the 57 codes.
  b  the reduction in log loss when the code alone is added to the attainment model (objective and
     attainment), after one intercept per model (stage 55's crossfit), beyond the placebos' mean
     reduction, x 0.001; out of fold, clean rows (stage 59). Filled: better than the placebo after
     Benjamini-Hochberg across the 57 codes. Grey tick: the same without the intercept (stage 46).
  c  as b, on the released test set, TSL students with 3+ training sessions.
Bold: the ten most-attributed codes (mean |SHAP|, Fig. 2b), with their rank.

Drawn at the text width and printed 1:1. Reads only `figures/input/`
(`prepare_recalibration_intervals_inputs.py`, `prepare_attainment_incremental_inputs.py`,
`prepare_code_occurrence_inputs.py`, `prepare_beeswarm_inputs.py`).

    python figures/scripts/fig_codes_add_attainment.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

import fig_rq1 as rq1
import style as st

VIEWS = [("out of fold, clean rows", "Adds: out of fold,\nclean rows"),
         ("test, >= 3 training sessions", "Adds: test, students\nwith 3+ training sessions")]
TOP_SHAP = 10
LIM_OCC, TICKS_OCC = (-30, 38), [-20, 0, 20]
LIM_LL, TICKS_LL = (-4, 7.4), [-3, 0, 3, 6]
H = 6.75                                # inches: prints 1:1 with its caption on one page


def frame(ax, n, lim, ticks, labels):
    st.zebra(ax, list(range(n + 1))[::-1])      # one row above the plot first, so the top row is banded
    ax.axvline(0, color=st.INK_MUTED, lw=st.mark(0.8), zorder=1)
    ax.set_xlim(*lim)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    # no grid lines: the SI figures have none (Fig. 5); the zero line is the reference
    ax.set_ylim(-0.7, n - 0.3)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def dot(ax, y, est, lo, hi, role, filled):
    col, ink = rq1.ROLE_COLOR[role], rq1.ROLE_INK[role]
    ax.plot([lo, hi], [y, y], color=col, lw=st.mark(0.9), zorder=2)
    ax.plot([est], [y], "o", ms=st.mark(3.0), color=col, mfc=col if filled else "white", mec=ink,
            mew=st.mark(0.6), zorder=3)


def main() -> None:
    st.apply_style(paper=True)
    c = pd.read_csv(st.INPUT / "fig_recal_codes.csv")
    raw = pd.read_csv(st.INPUT / "fig_attinc_add.csv")
    occ = pd.read_csv(st.INPUT / "fig_occur_codes.csv").set_index("code")
    shap = pd.read_csv(st.INPUT / "fig_beeswarm_columns.csv").sort_values("mean_abs_shap", ascending=False)
    rank = {col.split("_", 1)[1].removesuffix("_rate"): i + 1 for i, col in enumerate(shap.column)}
    first = c[c.view == VIEWS[0][0]].set_index("code")
    if len(first) != 57 or set(first.index) != set(occ.index):
        raise SystemExit("codes in fig_recal_codes and fig_occur_codes differ")
    order = first.sort_values("vs_placebo").index.tolist()
    n = len(order)
    W = st.TEXT_W
    fig, axes = plt.subplots(1, 3, figsize=(W * st.SCALE, H * st.SCALE), sharey=True,
                             gridspec_kw=dict(wspace=0.08, left=0.29, right=0.99, top=0.965, bottom=0.075))

    ax = axes[0]
    frame(ax, n, LIM_OCC, TICKS_OCC, ["−20%", "0", "+20%"])
    for i, k in enumerate(order):
        o = occ.loc[k]
        dot(ax, i, o.strong_vs_weak, max(o.strong_vs_weak_lo, LIM_OCC[0]), min(o.strong_vs_weak_hi, LIM_OCC[1]),
            first.loc[k, "role"], bool(o.strong_vs_weak_bh))
    ax.set_xlabel("Difference in frequency", labelpad=2)

    for ax, (view, title) in zip(axes[1:], VIEWS):
        frame(ax, n, LIM_LL, TICKS_LL, ["−3", "0", "+3", "+6"])
        s = c[c.view == view].set_index("code")
        r = raw[raw.view == view].set_index("code")
        for i, k in enumerate(order):
            x = s.loc[k]
            ax.plot([1e3 * r.loc[k, "vs_placebo"]] * 2, [i - 0.32, i + 0.32], color=st.INK_MUTED,
                    lw=st.mark(0.9), zorder=1.5, solid_capstyle="butt")
            dot(ax, i, 1e3 * x.vs_placebo, 1e3 * x.vs_placebo_lo, 1e3 * x.vs_placebo_hi, x.role,
                bool(x.vs_placebo_bh) and x.vs_placebo > 0)
        ax.set_xlabel("Reduction in log loss\n(×0.001)", labelpad=2, linespacing=1.15)

    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([rq1.nice(k) + (f" (SHAP #{rank[k]})" if rank.get(k, 99) <= TOP_SHAP else "")
                             for k in order], fontsize=st.tsize("code"))
    for tl, k in zip(axes[0].get_yticklabels(), order):
        tl.set_color(rq1.ROLE_INK[first.loc[k, "role"]])
        if rank.get(k, 99) <= TOP_SHAP:
            tl.set_fontweight("bold")
    # no panel titles: the caption names them. The three letters on one line just above the
    # axes, a where the row names start, b and c at their axes' left edge.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y = axes[0].get_position().y1 + 3 / 72 / fig.get_figheight()
    for ax, letter in zip(axes, "abc"):
        x = (fig.transFigure.inverted().transform((ax.get_tightbbox(r).x0, 0))[0] if letter == "a"
             else ax.get_position().x0)
        fig.text(max(x, 0.0), y, letter, fontsize=st.tsize("panel_letter"), fontweight="bold",
                 ha="left", va="bottom", color=st.INK)
    st.save(fig, "fig_codes_add_attainment", width=W)


if __name__ == "__main__":
    main()
