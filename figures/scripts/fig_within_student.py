#!/usr/bin/env python
"""fig_within_student — what a within-student design can learn from, and what survives it (Fig. 3).

  a  across the text width, one column each: the twelve largest pooled code associations, then
     the block-level exchanges that clear BH pooled (stage 33, hard labels). Pooled, then within
     student one at a time, then (codes only) within student with every code in one model,
     joined by a line so its slope shows what each added control does to the association.
     Exchange names read first turn (arrow) on the first line, the reply on the second.
  b  training students by their number of sessions, the median dashed (axis stopped at 20
     sessions; 10 of 5,933 students have more).
  c  how far students with three or more sessions sit above or below their learning objectives:
     the mean of each student's answers' residuals against the objective's usual rate, in
     percentage points, beside the spread the outcomes give when shuffled among the answers to the
     same objective (prepare_student_attainment_inputs.py).

Drawn at the text width (5.95 in) and printed 1:1, like Figures 1 and 2. Reads only `figures/input/`.

    python figures/scripts/fig_within_student.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import fig_rq1 as rq1
import style as st

W = st.TEXT_W                           # the text width: prints 1:1, as Figures 1 and 2
#: the shared panel functions' marks (fig_rq1) are sized for wider canvases
MARK = 0.82
#: the codes and the exchanges, across the text width
B_X = 0.42                              # the plot; its y axis sits to the left
B_W = W - 0.02 - B_X
B_H = 1.35                              # the plot (1.50 until 2026-11-05, 10% lower at the user's request)
B_LAB = 0.74                            # under the plot: the angled names
B_HEAD = 0.20                           # over the plot: the two groups' titles
B_DX = 0.27                             # within a code's column: pooled, one at a time, all codes
EX_GAP = 0.7                            # columns between the codes and the exchanges
#: an exchange's column is wider than a code's: its two-line name needs the room between neighbours
EX_STEP = 1.4
LEG = 0.20                              # over the group titles: the legend, one row
#: the bottom row: b sessions per student, c students above or below their questions
BOT_H, BOT_AXIS, BOT_GAP = 0.55, 0.33, 0.10   # plot height, axis labels under it, gap above it
C_X, C_W = 0.42, 2.30                   # b plot
D_X = 3.45                              # c plot, to the right edge
SESS_MAX = 20                           # b's x axis ends here
BAR_GREY = "#B4B4B0"                    # b and c's bars: a light grey, so c's chance outline stands out


def column(ax, x: float, pooled, within, every, c: str, ink: str) -> None:
    """One code or exchange: its estimates left to right, joined, each with its interval.
    `pooled`, `within`, `every` are (estimate, lo, hi, significant); `every` may be None."""
    pts = [(x - B_DX, pooled, c), (x, within, c)] + ([(x + B_DX, every, c)] if every else [])
    ax.plot([p[0] for p in pts], [p[1][0] for p in pts], color=c, lw=st.mark(0.9), zorder=3,
            solid_capstyle="round")
    for px, (_, lo, hi, _), col in pts:
        ax.plot([px, px], [lo, hi], color=col, lw=st.mark(1.2), solid_capstyle="round", zorder=3)
    # pooled in the role's colour too: the marker's shape, not its colour, says which estimate it is
    ax.plot([x - B_DX], [pooled[0]], "s", ms=st.mark(3.4), mec=ink, mew=st.mark(0.7), zorder=4,
            color=c if pooled[3] else "white")
    ax.plot([x], [within[0]], "o", ms=st.mark(3.8), mec=ink, mew=st.mark(0.7), zorder=4,
            color=c if within[3] else "white")
    if every:
        ax.plot([x + B_DX], [every[0]], "D", ms=st.mark(3.4), mec=ink, mew=st.mark(0.7), zorder=4,
                color=c if every[3] else "white")


def exchange_label(ax, x: float, first: str, second: str, ink1: str, ink2: str) -> None:
    """Two lines at 45 degrees under column `x`: the first turn and its arrow in the first
    speaker's ink, the reply under it in the other role's. Two texts with the same two-line
    layout, each with the other's line blank, so the lines sit exactly where one text's would."""
    kw = dict(rotation=45, ha="right", va="center", rotation_mode="anchor", multialignment="right",
              fontsize=st.tsize("code"), linespacing=1.15,
              transform=ax.get_xaxis_transform(), clip_on=False)
    # the pair is centred on the column; its first line reaches up toward the axis, so the anchor
    # sits that far (half the pair's height along the diagonal) plus the ticks' 2 pt pad below it
    fs = st.tsize("code")
    drop = (2.0 + 1.15 * fs * 0.5 * 2 ** 0.5) / 72
    y = -drop / (ax.get_position().height * ax.figure.get_figheight())
    ax.text(x, y, f"{first} $\\rightarrow$\n ", color=ink1, **kw)
    ax.text(x, y, f" \n{second}", color=ink2, **kw)


def panel_b(ax, n: pd.DataFrame, joint: pd.DataFrame, seq: pd.DataFrame) -> list[float]:
    """The twelve codes, a gap, then the exchanges. Returns the exchanges' x positions."""
    jj = joint.set_index("code")
    xs_code = list(range(len(n)))
    for x, r in zip(xs_code, n.itertuples()):
        q = jj.loc[r.code]
        column(ax, x, (r.pp_per_sd_pooled, r.lo_pooled, r.hi_pooled, True),
               (r.pp_per_sd_within_student, r.lo_within_student, r.hi_within_student,
                r.bh_sig_within_student == 1),
               (q.estimate, q.lo, q.hi, bool(q.bh_sig)), rq1.ROLE_COLOR[r.role], rq1.ROLE_INK[r.role])
    x0 = len(n) - 1 + EX_GAP + EX_STEP / 2 + 0.5
    xs_ex = [x0 + k * EX_STEP for k in range(len(seq))]
    for x, r in zip(xs_ex, seq.itertuples()):
        column(ax, x, (r.pp_per_sd_pooled, r.lo_pooled, r.hi_pooled, True),
               (r.pp_per_sd_within_student, r.lo_within_student, r.hi_within_student,
                r.bh_sig_within_student == 1), None, rq1.ROLE_COLOR[r.role], rq1.ROLE_INK[r.role])
    # alternate columns banded within each group, inside the plot
    for xs in (xs_code, xs_ex):
        for k, x in enumerate(xs):
            if k % 2:
                half = 0.5 if xs is xs_code else EX_STEP / 2
                ax.axvspan(x - half, x + half, color=st.ZEBRA, lw=0, zorder=0)
    ax.axvline(len(n) - 0.5 + EX_GAP / 2, color=st.INK_MUTED, lw=st.mark(0.6), ls=(0, (2, 2)), zorder=1)
    ax.axhline(0, color=st.INK_MUTED, lw=st.mark(0.8), ls=(0, (3, 2)), zorder=2)
    ax.set_xlim(-0.5, xs_ex[-1] + EX_STEP / 2)
    ax.set_xticks(xs_code + xs_ex)
    ax.set_xticklabels([rq1.nice(c) for c in n.code] + [""] * len(xs_ex), rotation=45, ha="right",
                       rotation_mode="anchor", fontsize=st.tsize("code"))
    for t, role in zip(ax.get_xticklabels(), n.role):
        t.set_color(rq1.ROLE_INK[role])
    ax.tick_params(axis="x", length=0, pad=2)
    other = {"tutor": "student", "student": "tutor"}
    for x, r in zip(xs_ex, seq.itertuples()):
        first, second = r.code.split(" > ")
        exchange_label(ax, x, first, second, rq1.ROLE_INK[r.role], rq1.ROLE_INK[other[r.role]])
    ax.grid(axis="y", color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.set_ylabel("Difference in correct answers (%)\nper SD of the code's rate", labelpad=2, linespacing=1.15)
    return xs_ex


def sessions(ax, d: pd.DataFrame) -> None:
    """b: training students by their number of sessions, one colour, the median dashed and named."""
    ax.bar(d.n_sessions, d.n_students, width=0.8, color=BAR_GREY, lw=0, zorder=2)
    med = int(np.median(np.repeat(d.n_sessions.to_numpy(), d.n_students.to_numpy())))
    ax.axvline(med, color=st.INK, lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax.text(med + 0.45, 1500, f"median {med}", ha="left", va="center", fontsize=st.tsize("legend"),
            color=st.INK)
    ax.set_xlim(0.3, SESS_MAX + 0.7)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks([0, 500, 1000, 1500])
    ax.grid(axis="y", color=st.GRID, lw=st.STYLE["grid_lw"], zorder=0)
    ax.set_xlabel("Sessions per student (training set)", labelpad=2)
    ax.set_ylabel("Students", labelpad=3)


def attainment(ax, h: pd.DataFrame) -> None:
    """c: students by how far they sit above or below the questions they answer, the shuffled
    spread as an outline."""
    w = h.hi - h.lo
    ax.bar(h.lo, h.students, width=w, align="edge", color=BAR_GREY, lw=0, zorder=2)
    xs = np.r_[h.lo.to_numpy(), h.hi.iloc[-1]]
    ax.stairs(h.null.to_numpy(), xs, color=st.INK, lw=1.1, zorder=3)
    ax.axvline(0, color=st.INK_MUTED, lw=st.mark(0.8), ls=(0, (3, 2)), zorder=1)
    ticks = [-80, -60, -40, -20, 0, 20, 40]
    ax.set_xlim(h.lo.iloc[0], h.hi.iloc[-1])            # every student, the tails included
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:+d}%".replace("-", "−") if v else "0" for v in ticks])
    ax.grid(axis="y", color=st.GRID, lw=st.STYLE["grid_lw"], zorder=0)
    ax.set_xlabel("Correct answers above or below the objective's rate (%)", labelpad=2)
    ax.set_ylabel("Students", labelpad=3)
    ax.legend(handles=[Patch(color=BAR_GREY, label="Students"),
                       Line2D([], [], color=st.INK, lw=1.1, label="Chance: answers\nshuffled within\neach objective")],
              loc="upper left", frameon=False, handlelength=1.2, fontsize=st.tsize("legend"),
              borderaxespad=0.3)


def legend_b(fig, H: float, xc: float, y: float) -> None:
    key = dict(mec=st.INK, mew=st.mark(0.7), lw=0)
    # shape only, light grey with a black edge; hollow follows the Benjamini-Hochberg correction,
    # not the plotted 95% interval
    fig.legend([Line2D([], [], marker="s", ms=st.mark(3.4), mfc=rq1.KEY_FILL, **key),
                Line2D([], [], marker="o", ms=st.mark(3.8), mfc=rq1.KEY_FILL, **key),
                Line2D([], [], marker="D", ms=st.mark(3.4), mfc=rq1.KEY_FILL, **key),
                Line2D([], [], marker="o", ms=st.mark(3.8), mfc="white", **key)],
               ["Pooled", "Within student, one at a time", "Within student, all codes",
                "Not significant (BH)"],
               loc="center", ncol=4, frameon=False, bbox_to_anchor=(xc / W, y / H), handlelength=0.8,
               handletextpad=0.4, columnspacing=1.6, fontsize=st.tsize("legend"))


def main() -> None:
    st.apply_style(paper=True, mark=MARK)
    b_y = BOT_AXIS + BOT_H + BOT_GAP + B_LAB             # the top plot's bottom
    H = b_y + B_H + B_HEAD + LEG                         # the legend line ends here
    fig = plt.figure(figsize=(W * st.SCALE, H * st.SCALE))

    ax_b = fig.add_axes((B_X / W, b_y / H, B_W / W, B_H / H))
    # the all-codes-at-once estimate beside the one-code-at-a-time one (stage 32 via
    # prepare_explain_inputs.py): which associations are a code's own, and which its company's
    j = pd.read_csv(st.INPUT / "fig_explain_a_codes.csv")
    j = j[j.design == "within student, all 55 codes"]
    n = pd.read_csv(st.INPUT / "fig_rq1_b_named.csv")
    seq = pd.read_csv(st.INPUT / "fig_block_sequences.csv")
    xs_ex = panel_b(ax_b, n, j, seq)
    # the two groups' titles over the plot, each centred on its columns
    lim = ax_b.get_xlim()
    to_in = lambda x: B_X + (x - lim[0]) / (lim[1] - lim[0]) * B_W
    for (x0, x1), title in (((0, len(n) - 1), "Codes"),
                            ((xs_ex[0], xs_ex[-1]), "Sequences")):
        fig.text((to_in(x0) + to_in(x1)) / 2 / W, (b_y + B_H + 0.04) / H, title, ha="center",
                 va="bottom", fontsize=st.tsize("label"), color=st.INK)
    legend_b(fig, H, B_X + B_W / 2, H - LEG / 2)         # one row, over the group titles

    # b: sessions per student; c: students above or below their questions
    ax_s = fig.add_axes((C_X / W, BOT_AXIS / H, C_W / W, BOT_H / H))
    sessions(ax_s, pd.read_csv(st.INPUT / "fig_sessions_per_student.csv"))
    ax_r = fig.add_axes((D_X / W, BOT_AXIS / H, (W - 0.02 - D_X) / W, BOT_H / H))
    attainment(ax_r, pd.read_csv(st.INPUT / "fig_student_attainment.csv"))
    y_row = BOT_AXIS + BOT_H + BOT_GAP
    for x, y, letter in ((0.0, H - 0.02, "a"), (0.0, y_row, "b"), (D_X - 0.47, y_row, "c")):
        fig.text(x / W, y / H, letter, fontsize=st.tsize("panel_letter"), fontweight="bold",
                 va="top", color=st.INK)
    st.save(fig, "fig_within_student", width=W)


if __name__ == "__main__":
    main()
