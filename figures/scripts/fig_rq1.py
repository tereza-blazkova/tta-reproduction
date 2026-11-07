#!/usr/bin/env python
"""RQ1 — what survives comparing a student with themselves.

  a  every code twice: its association with the outcome estimated POOLED against the same
     estimate with the student absorbed. Points collapse toward the horizontal — 31 of 57
     codes clear a false-discovery correction pooled, 3 do within student.
  b  the ladder of designs. Pooled to within-student is SHRINKAGE: the estimates halve while
     the intervals barely widen. Within-student to within-session is POWER: the estimates
     hardly move while the intervals nearly triple, because only 8,296 responses sit in a
     session that taught more than one objective AND whose outcome varied.
  c  the compositional model, in units a practitioner can act on.

The distinction b draws is the one a reader must not miss: a null that arrives by shrinkage
says the association was between students; a null that arrives by widening says we cannot tell.

The twelve largest pooled associations by name (`panel_b`) are drawn in `fig_within_student`.

Reads only `figures/input/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as st  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROLE_COLOR = {"tutor": st.TUTOR, "student": st.STUDENT}
ROLE_INK = {"tutor": st.TUTOR_INK, "student": st.STUDENT_INK}
POOLED = "#9A9A95"
POOLED_INK = "#5A5A56"      # the pooled markers' border: the darker shade of the pooled grey
KEY_FILL = "#D4D4D0"        # legend keys: light grey with a black edge (shape only, no role colour)


def nice(code: str) -> str:
    return code.replace("_", " ")


def panel_a(ax, w: pd.DataFrame) -> None:
    lim = max(w.pp_per_sd_pooled.abs().max(), w.pp_per_sd_within_student.abs().max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=st.GRID, lw=0.9, zorder=1)
    ax.axhline(0, color=st.INK_MUTED, lw=0.7, zorder=2)
    for role in ("tutor", "student"):
        d = w[w.role == role]
        ax.scatter(d.pp_per_sd_pooled, d.pp_per_sd_within_student, s=13, lw=0, alpha=0.75,
                   color=ROLE_COLOR[role], zorder=3, label=f"{role} ({len(d)})")
    sig = w[w.bh_sig_within_student == 1]
    ax.scatter(sig.pp_per_sd_pooled, sig.pp_per_sd_within_student, s=38, facecolor="none",
               lw=0.9, edgecolor=st.INK, zorder=4)
    for _, r in sig.iterrows():
        ax.annotate(nice(r.code), xy=(r.pp_per_sd_pooled, r.pp_per_sd_within_student),
                    xytext=(5, -3), textcoords="offset points",
                    fontsize=st.tsize("note"), color=ROLE_INK[r.role])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("pooled (pp per SD)", labelpad=2)
    ax.set_ylabel("within student", labelpad=2)
    ax.grid(color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.legend(loc="upper left", handletextpad=0.3, borderpad=0.2)
    ax.set_title("Every code, both ways", pad=4, loc="left")


def panel_b(ax, n: pd.DataFrame, pooled_marker: str = "o", color: str | None = None,
            joint: pd.DataFrame | None = None, zebra_x0: float | None = None) -> None:
    """Pooled (grey, `pooled_marker`) above within student (circle) for each named row.

    `color` overrides the role colour of the within-student estimate, for rows that belong
    to no single role (an exchange between a tutor move and a student signal). `joint` adds a
    third estimate per row, below the other two: the same code within student with every other
    code in one model (columns code, estimate, lo, hi, bh_sig), drawn as a diamond.
    """
    d = n.iloc[::-1].reset_index(drop=True)
    up, mid, down = (0.27, 0.0, -0.27) if joint is not None else (0.18, -0.18, None)
    jj = joint.set_index("code") if joint is not None else None
    y = np.arange(len(d))
    for i, r in d.iterrows():
        ax.plot([r.lo_pooled, r.hi_pooled], [i + up] * 2, color=POOLED, lw=st.mark(1.3),
                solid_capstyle="round", zorder=3)
        ax.plot([r.pp_per_sd_pooled], [i + up], pooled_marker, ms=st.mark(3.4), color=POOLED,
                mec=POOLED_INK, mew=st.mark(0.6), zorder=4)
        sig = r.bh_sig_within_student == 1
        c = color or ROLE_COLOR[r.role]
        ink = ROLE_INK.get(r.role, st.INK) if color is None else st.INK
        ax.plot([r.lo_within_student, r.hi_within_student], [i + mid] * 2,
                color=c, lw=st.mark(1.3), solid_capstyle="round", zorder=3)
        ax.plot([r.pp_per_sd_within_student], [i + mid], "o", ms=st.mark(3.8),
                color=c if sig else "white", mec=ink, mew=st.mark(0.7), zorder=4)
        if jj is not None:
            q = jj.loc[r.code]
            ax.plot([q.lo, q.hi], [i + down] * 2, color=c, lw=st.mark(1.3), alpha=0.75,
                    solid_capstyle="round", zorder=3)
            ax.plot([q.estimate], [i + down], "D", ms=st.mark(3.4),
                    color=c if bool(q.bh_sig) else "white", mec=ink, mew=st.mark(0.7), zorder=4)
    ax.axvline(0, color=st.INK_MUTED, lw=st.mark(0.8), ls=(0, (3, 2)), zorder=2)
    if zebra_x0 is not None:                     # rows drawn bottom-up: read them from the top
        st.zebra(ax, list(y[::-1]), x0=zebra_x0)
    ax.set_yticks(y)
    ax.set_yticklabels([nice(c) for c in d.code])
    for t, role in zip(ax.get_yticklabels(), d.role):
        t.set_color(ROLE_INK[role])
    ax.set_xlabel("association with the outcome (pp per SD)", labelpad=2)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.grid(axis="x", color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.legend(handles=[Line2D([], [], color=POOLED, lw=st.mark(1.5), marker=pooled_marker, ms=st.mark(3.4),
                              mfc=POOLED, mec=POOLED_INK, mew=st.mark(0.6), label="pooled"),
                       Line2D([], [], color=st.INK_2, lw=st.mark(1.5), marker="o", ms=st.mark(3.8), mfc=KEY_FILL,
                              mec=st.INK, mew=st.mark(0.7),
                              label="within student, one code" if jj is not None
                              else "within student"),
                       *([Line2D([], [], color=st.INK_2, lw=st.mark(1.5), marker="D", ms=st.mark(3.4), mfc=KEY_FILL,
                                 mec=st.INK, mew=st.mark(0.7), label="within student, all codes")]
                         if jj is not None else []),
                       Line2D([], [], color="white", marker="o", ms=st.mark(3.8), lw=0,
                              mec=st.INK, mew=st.mark(0.7), label="hollow: not BH-significant")],
              loc="lower right", handlelength=1.1, labelspacing=0.22, borderpad=0.25)
    ax.set_title("The twelve largest pooled associations", pad=4, loc="left")


def panel_c(ax, lad: pd.DataFrame) -> None:
    """Which identity is the confound: absorb the student, the tutor, or both."""
    label = {"1 question": "question\nonly", "2 +student": "+ student",
             "3 +tutor": "+ tutor", "4 +student+tutor": "+ both"}
    d = lad.sort_values("design").reset_index(drop=True)
    x = np.arange(len(d))
    # teal wherever the TUTOR is absorbed, so the eye can see that it makes no difference
    cols = [st.INK_2, st.INK_2, st.TUTOR, st.TUTOR]
    ax.bar(x, d.n_sig, width=0.6, color=cols[: len(d)], lw=0, zorder=3)
    for i, r in d.iterrows():
        ax.text(i, r.n_sig + 1.0, f"{int(r.n_sig)}/57", ha="center",
                fontsize=st.tsize("note"), color=st.INK)
    ax.set_xticks(x)
    ax.set_xticklabels([label[s_] for s_ in d.design])
    ax.set_ylabel("codes surviving BH", labelpad=2)
    ax.set_ylim(0, 48)
    ax.grid(axis="y", color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.annotate("absorbing the tutor\nchanges nothing", xy=(2, d.n_sig.iloc[2] + 2.0),
                xytext=(1.9, 47), ha="center", va="top", fontsize=st.tsize("note"),
                color=st.TUTOR_INK,
                arrowprops=dict(arrowstyle="-", color=st.TUTOR_INK, lw=0.5, shrinkB=2))
    ax.annotate("absorbing the student\ncollapses everything", xy=(1, d.n_sig.iloc[1] + 1.5),
                xytext=(1.55, 17), ha="center", va="center", fontsize=st.tsize("note"),
                color=st.INK_2,
                arrowprops=dict(arrowstyle="-", color=st.INK_2, lw=0.5, shrinkB=2))
    ax.set_title("Which identity is the confound", pad=4, loc="left")


def panel_d(ax, c: pd.DataFrame, tsum: pd.DataFrame) -> None:
    """What ten percentage points of reallocated turns is associated with."""
    d = c.sort_values("estimate").reset_index(drop=True)
    y = np.arange(len(d))
    for i, r in d.iterrows():
        sig = bool(r.bh)
        ax.plot([r.lo, r.hi], [i, i], color=st.STUDENT if sig else "#D8D8D4",
                lw=1.5 if sig else 1.1, solid_capstyle="round", zorder=3)
        ax.plot([r.estimate], [i], "o", ms=3.6 if sig else 3.0,
                color=st.STUDENT if sig else "white",
                mec=st.STUDENT if sig else "#B6B6B2", mew=0.9, zorder=4)
    ax.axvline(0, color=st.INK_MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in d.code])
    for t, s_ in zip(ax.get_yticklabels(), d.bh):
        t.set_color(st.STUDENT_INK if s_ else st.INK_MUTED)
    ax.set_xlabel("pp on the outcome per 10 pp of the student's turns\n"
                  "reallocated from no visible move", labelpad=2)
    ax.grid(axis="x", color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.set_ylim(-0.8, len(d) - 0.2)
    # explicit, because autoscale clipped the widest interval against the left spine and a
    # clipped interval reads as a shorter one
    pad = 0.04 * (d.hi.max() - d.lo.min())
    ax.set_xlim(d.lo.min() - pad, d.hi.max() + pad)
    ax.set_title("In units a practitioner can act on", pad=4, loc="left")

    # The same model on the 41 tutor shares, stated rather than drawn. Drawn, 40 wide intervals
    # would dominate the panel; omitted, the panel would read as "only student behaviour
    # matters" when the tutor estimates are simply twice as imprecise.
    t = tsum.iloc[0]
    ax.text(0.015, 0.97,
            f"same model, {int(t.n_tutor)} tutor shares:\n"
            f"{int(t.n_bh_tutor)} clear the correction,\n"
            f"median interval $\\pm${t.median_hw_tutor:.1f} pp,\n"
            f"against $\\pm${t.median_hw_student:.1f} here",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=st.STYLE["note"], color=st.INK_MUTED, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E2E2DE", lw=0.6))


def main() -> None:
    st.apply_style()
    w = pd.read_csv(st.INPUT / "fig_rq1_a_scatter.csv")
    lad = pd.read_csv(st.INPUT / "fig_rq1_c_ladder_fe.csv")
    comp = pd.read_csv(st.INPUT / "fig_rq1_d_compositional.csv")
    tsum = pd.read_csv(st.INPUT / "fig_rq1_d_tutor_summary.csv")

    # The twelve named associations (panel_b) now sit in fig_within_student, so this figure
    # keeps the scatter, the ladder and the compositional model, lettered a, b, c.
    fig = plt.figure(figsize=(st.STYLE["fig_w"] * st.SCALE, 5.3 * st.SCALE))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05],
                  width_ratios=[1.0, 0.82], hspace=0.46, wspace=0.55,
                  left=0.21, right=0.96, top=0.94, bottom=0.10)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    panel_a(ax_a, w)
    panel_c(ax_b, lad)
    panel_d(ax_c, comp, tsum)
    st.panel_letter(ax_a, "a", dx=-0.26)
    st.panel_letter(ax_b, "b", dx=-0.26)
    st.panel_letter(ax_c, "c", dx=-0.205)
    st.save(fig, "fig_rq1_within_student")


if __name__ == "__main__":
    main()
