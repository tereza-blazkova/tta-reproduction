#!/usr/bin/env python
"""fig_code_occurrence — which codes are more common for higher- or lower-attainment students, and what
each adds to the attainment model (stages 48 and 46).

Two blocks side by side: a, student signals (all 16); b, tutor moves, only the 16 of 41 whose occurrence differs
significantly (Benjamini-Hochberg across the 57 codes); three dots on the zero lines mark the gap.
One row per code, sorted by the occurrence difference, and two plots sharing the rows:

  left   how much more common: the code's share of the role's turns (student turns for a student
         signal, tutor turns for a tutor move) in the top third of students by attainment against the
         bottom third, in % of the bottom third's share (stage 48). Bars, 95% bootstrap intervals
         over students. Open: not significant after Benjamini-Hochberg across the 57 codes; among the
         codes shown, an interval excludes 0 exactly when the difference is significant (checked
         at run time).
  right  what it adds to the attainment model ("loss reduction"): the log-loss reduction when the code
         alone is added to the attainment model (objective + attainment), minus the reduction a placebo
         code brings (a code shuffled across sessions), x 0.001; out of fold, clean rows (stage 46).
         Open: not better than the placebo after Benjamini-Hochberg across the 57 codes; here
         intervals and the corrected test do not always agree, so the fill carries it.

Dashed: where codes switch from more common for higher- to more common for lower-attainment students.
Bold: the paper's ten most-attributed codes (mean |SHAP|, Fig. 2b), with their rank; always shown.

Reads only `figures/input/` (`prepare_code_occurrence_inputs.py`, `prepare_attainment_incremental_inputs.py`,
`prepare_beeswarm_inputs.py`).

    python figures/scripts/fig_code_occurrence.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import transforms

import fig_rq1 as rq1
import style as st

ROW = 0.102                      # inches per code
TITLE_IN = 0.49                  # title row (block title, two-line column headers, direction labels)
AXIS_IN = 0.45                   # x axis and its labels
OCC_W, LL_W = 1.15, 0.9          # widths of the occurrence and loss-reduction plots (inches)
#: (role, letter, title, left edge of the block, width of its code labels), inches
BLOCKS = [("student", "a", "Student signals", 0.0, 1.42), ("tutor", "b", "Tutor moves", 3.72, 1.52)]
W = BLOCKS[-1][3] + BLOCKS[-1][4] + 0.05 + OCC_W + 0.12 + LL_W + 0.04
RESOLVED_ONLY = {"tutor"}        # roles where codes whose occurrence does not differ are left out
TOP_SHAP = 10
LIM_OCC = (-29, 37)
LIM_LL = (-3.6, 10.5)
ZERO_LW = 0.8
GAP_ROWS = 1.6
#: add-one-in view shown on the right (out of fold, clean rows: no session shared across folds)
VIEW = "out of fold, clean rows"


def zero_and_divider(ax, lim, y_lo, y_hi, mid, gap):
    """Zero line (broken by three dots across a gap) and the dashed divider at `mid`."""
    zero = dict(color=st.INK_MUTED, lw=st.mark(ZERO_LW), zorder=1, solid_capstyle="butt")
    dash = dict(color=st.INK_MUTED, lw=st.mark(0.6), ls=(0, (3, 2)), zorder=1)
    span = lim[1] - lim[0]
    if gap:
        half = gap / 2 - 0.15
        ax.plot([0, 0], [y_lo, mid - half], **zero)
        ax.plot([0, 0], [mid + half, y_hi], **zero)
        ax.plot([0] * 3, [mid - 0.36, mid, mid + 0.36], "o", ms=st.mark(ZERO_LW * 1.3), mew=0, color=st.INK_MUTED,
                zorder=1)
        for a, b in ((lim[0], -0.025 * span), (0.025 * span, lim[1])):
            ax.plot([a, b], [mid, mid], **dash)
    else:
        ax.plot([0, 0], [y_lo, y_hi], **zero)
        ax.plot(lim, [mid, mid], **dash)


def style_axis(ax, lim, ticks, labels, y_lo, y_hi):
    ax.set_xlim(*lim)
    ax.set_ylim(y_lo, y_hi)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    for xg in ticks:
        if xg != 0:
            ax.axvline(xg, color=st.GRID, lw=st.STYLE["grid_lw"], zorder=0)


def occurrence_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(codes shown, add-one-in rows by code, rows per role incl. the gap) for the blocks."""
    c = pd.read_csv(st.INPUT / "fig_occur_codes.csv")
    shap = pd.read_csv(st.INPUT / "fig_beeswarm_columns.csv").sort_values("mean_abs_shap", ascending=False)
    rank = {col.split("_", 1)[1].removesuffix("_rate"): i + 1 for i, col in enumerate(shap.column)}
    c["rank"] = c.code.map(rank)
    shown = c[~c.role.isin(RESOLVED_ONLY) | c.strong_vs_weak_bh | (c["rank"] <= TOP_SHAP)]
    excl = (shown.strong_vs_weak_lo > 0) | (shown.strong_vs_weak_hi < 0)
    assert (excl == shown.strong_vs_weak_bh.astype(bool)).all(), "interval and BH disagree for a shown code"
    add = pd.read_csv(st.INPUT / "fig_attinc_add.csv")
    add = add[(add.view == VIEW) & add.role.isin(["tutor", "student"])].set_index("code")
    assert len(add) == 57
    rows = {role: (shown.role == role).sum() + GAP_ROWS * (role in RESOLVED_ONLY) for role in ("student", "tutor")}
    return shown, add, rows


def occurrence_block(fig, W: float, H: float, role: str, letter: str, title: str, left: float,
                     label_w: float, bottom: float, span: float, shown: pd.DataFrame, add: pd.DataFrame,
                     rows: dict, occ_w: float = OCC_W, ll_w: float = LL_W, row: float = ROW,
                     label_fs: float | None = None, ll_gap: float = 0.12, show_add: bool = True) -> None:
    """One role's block (inches on a W x H figure): code labels, how much more common the code is for
    higher-attainment students, and (with `show_add`) what it adds to the attainment model; title and headers above."""
    label_r = left + label_w
    OCC = (label_r + 0.05, occ_w)
    LL = (OCC[0] + occ_w + ll_gap, ll_w)
    label_fs = label_fs or st.tsize("tick") * 0.88
    g = shown[shown.role == role].sort_values("strong_vs_weak").reset_index(drop=True)
    n = len(g)
    gap = GAP_ROWS * (role in RESOLVED_ONLY)
    n_neg = int((g.strong_vs_weak < 0).sum())
    ypos = g.index.to_numpy() + gap * (g.index.to_numpy() >= n_neg)
    pad = (span - rows[role]) / 2                     # centre the shorter block on the taller one
    y_lo, y_hi = -0.7 - pad, n + gap - 0.3 + pad
    mid = n_neg - 0.5 + gap / 2
    h = span * row
    ax_o = fig.add_axes((OCC[0] / W, bottom / H, OCC[1] / W, h / H))
    col, ink = rq1.ROLE_COLOR[role], rq1.ROLE_INK[role]

    # left: occurrence, higher- vs lower-attainment students
    zero_and_divider(ax_o, LIM_OCC, y_lo, y_hi, mid, gap)
    for i, r in g.iterrows():
        y = ypos[i]
        ax_o.plot([max(r.strong_vs_weak_lo, LIM_OCC[0]), min(r.strong_vs_weak_hi, LIM_OCC[1])], [y, y],
                  color=col, lw=st.mark(1.0), zorder=2)
        sig = bool(r.strong_vs_weak_bh)                     # open: not significant (BH)
        ax_o.plot([r.strong_vs_weak], [y], "o", ms=st.mark(3.6), color=col, mfc=col if sig else "white", mec=ink,
                  mew=st.mark(0.6), zorder=3)
        topc = r["rank"] <= TOP_SHAP
        ax_o.text(label_r / W, y, rq1.nice(r.code) + (f" (SHAP #{int(r['rank'])})" if topc else ""),
                  transform=transforms.blended_transform_factory(fig.transFigure, ax_o.transData),
                  ha="right", va="center", fontsize=label_fs, color=rq1.ROLE_INK[role],
                  fontweight="bold" if topc else "normal")
    style_axis(ax_o, LIM_OCC, [-20, 0, 20], ["−20%", "0", "+20%"], y_lo, y_hi)
    ax_o.set_xlabel("Difference\nin frequency" if show_add else "Difference in frequency", labelpad=2,
                    linespacing=1.2)
    rows_top = list(ypos[::-1])
    st.zebra(ax_o, rows_top)                                   # inside each plot only
    tr = transforms.blended_transform_factory(ax_o.transData, ax_o.transAxes)
    inch = 1 / h
    # under the header "how much more common", which students each side of zero means (above the
    # plot, where nothing competes for the width the two labels need)
    dir_h = 1.3 * st.tsize("tick") / 72                    # inches: the direction labels' line
    for sign, txt in ((-1, "$\\leftarrow$ lower attainment"), (1, "higher attainment $\\rightarrow$")):
        ax_o.text(sign * 3, 1 + 0.05 * inch, txt, transform=tr, ha="right" if sign < 0 else "left",
                  va="bottom", fontsize=st.tsize("tick"), color=st.INK, clip_on=False)

    # right: log-loss gain over the attainment model, beyond the placebo
    if show_add:
        ax_l = fig.add_axes((LL[0] / W, bottom / H, LL[1] / W, h / H))
        zero_and_divider(ax_l, LIM_LL, y_lo, y_hi, mid, gap)
        st.zebra(ax_l, rows_top)
        s = add.reindex(g.code)
        for i, (code, r) in enumerate(s.iterrows()):
            y = ypos[i]
            est, lo, hi = 1e3 * r.vs_placebo, 1e3 * r.vs_placebo_lo, 1e3 * r.vs_placebo_hi
            ax_l.plot([max(lo, LIM_LL[0]), min(hi, LIM_LL[1])], [y, y], color=col, lw=st.mark(1.0), zorder=2)
            sig = bool(r.vs_placebo_bh) and r.vs_placebo > 0
            ax_l.plot([est], [y], "o", ms=st.mark(3.6), color=col, mfc=col if sig else "white", mec=ink,
                      mew=st.mark(0.6), zorder=3)
        style_axis(ax_l, LIM_LL, [0, 5, 10], ["0", "+5", "+10"], y_lo, y_hi)
        # an axis label like every other (open markers are explained in the caption)
        ax_l.set_xlabel("Loss reduction\n(×0.001)", labelpad=2, linespacing=1.2)

    # block title and the attainment-model plot's header
    y_title = (bottom + h + 0.08 + dir_h) / H
    fig.text(left / W, y_title, letter, fontsize=st.tsize("panel_letter"), color=st.INK, fontweight="bold",
             ha="left", va="baseline")
    # role title and column headers at the axis-label size, as the role headers of Figure 1
    title_dx = 1.4 * st.tsize("panel_letter") / 72          # inches: past the letter
    fig.text((left + title_dx) / W, y_title, title, fontsize=st.tsize("label"), color=rq1.ROLE_INK[role],
             ha="left", va="baseline")
    if show_add:
        fig.text(LL[0] / W, y_title, "Adds to the\nattainment\nmodel", fontsize=st.tsize("label"),
                 color=st.INK, ha="left", va="baseline", linespacing=1.15)


def main() -> None:
    st.apply_style()
    shown, add, rows = occurrence_data()
    span = max(rows.values())
    H = TITLE_IN + span * ROW + AXIS_IN
    fig = plt.figure(figsize=(W * st.SCALE, H * st.SCALE))
    for role, letter, title, left, label_w in BLOCKS:
        occurrence_block(fig, W, H, role, letter, title, left, label_w, AXIS_IN, span, shown, add, rows)
    st.save(fig, "fig_code_occurrence")


if __name__ == "__main__":
    main()
