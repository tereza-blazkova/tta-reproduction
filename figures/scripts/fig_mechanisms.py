#!/usr/bin/env python
"""Mechanisms of prediction — what the model reads, and how far that explanation travels.

  a  the fourteen largest conversation columns by attribution, with the construct totals above.
     The single largest is a STUDENT signal (explains why), although tutor moves take half of
     all attribution between them.
  b  the same attribution on TSL against on Eedi. What the model prices is provider-specific
     even where its predictions are not.
  c  attribution under the 122B's own labels against under the distilled children. Two
     instruments that predict identically in-domain explain themselves differently.

**Attribution is not contribution**, and this figure cannot be read as if it were. What a block ADDS is the ladder panel of
`fig_main_measurement`; what the model PRICES is here. The two answer different questions and
the paper must not let them blur.

Reads only `figures/input/`.

    python figures/scripts/fig_mechanisms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as st  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROLE_COLOR = {"student": st.STUDENT, "tutor": st.TUTOR}
ROLE_INK = {"student": st.STUDENT_INK, "tutor": st.TUTOR_INK}
TOTAL_COLOR = {"tutor moves": st.TUTOR, "student signals": st.STUDENT,
               "learning objective": "#C9C9C4"}
N_TOP = 14
#: the construct names over the shares bar; "Objective" because the narrow bar has no room
LABEL_TOTAL = {"learning objective": "Objective"}


#: Two codes whose full names overrun a scatter label; shortened only where they are drawn
#: as annotations, never in the bar chart, where there is room for the real name.
SHORT = {"PROMPTING_ALTERNATIVE_REPRESENTATION": "prompting alt. representation",
         "PROMPTING_RELATED_CONCEPTS": "prompting related concepts",
         "SIMPLIFYING_TO_SUBPROBLEM": "simplifying to subproblem"}


def nice(code: str, short: bool = False) -> str:
    if short and code in SHORT:
        return SHORT[code]
    return code.lower().replace("_", " ")


def panel_totals(ax, tot: pd.DataFrame, labels_above: bool = False, align: dict | None = None, fs: float | None = None,
                 one_line: bool = False) -> None:
    """A slim stacked bar: where the model's attribution sits by construct.

    `align` moves a construct's name off its segment's centre: "end" ends it at the segment's right
    edge, "start" starts it at the left edge (for narrow segments whose names would collide)."""
    align = align or {}
    left = 0.0
    for _, r in tot.iterrows():
        ax.barh([0], [r.share], left=left, height=0.55, color=TOTAL_COLOR[r.construct],
                lw=0, zorder=3)
        ink = (st.TUTOR_INK if r.construct == "tutor moves" else
               st.STUDENT_INK if r.construct == "student signals" else st.INK)
        ax.text(left + r.share / 2, 0, f"{r.share:.0%}", ha="center", va="center",
                fontsize=fs or st.tsize("note"), color=st.INK)
        where = align.get(r.construct, "center")
        lx = {"center": left + r.share / 2, "end": left + r.share - 0.05, "start": left + 0.01,
              "end_flush": left + r.share}[where]
        ax.text(lx, 0.52 if labels_above else -0.52,
                (lambda s: s if one_line else s.replace(" ", "\n"))(
                    LABEL_TOTAL.get(r.construct, r.construct.capitalize())),
                ha={"center": "center", "end": "right", "start": "left", "end_flush": "right"}[where],
                va="bottom" if labels_above else "top", fontsize=fs or st.tsize("note"), color=ink)
        left += r.share
    ax.set_xlim(0, 1)
    top = 0.95 if one_line else 1.45
    ax.set_ylim(*((-0.50, top) if labels_above else (-top, 0.50)))
    ax.axis("off")


def panel_a(ax, a: pd.DataFrame) -> None:
    d = a.nlargest(N_TOP, "share").iloc[::-1]
    y = np.arange(len(d))
    ax.barh(y, d.share, height=0.66, color=[ROLE_COLOR[r] for r in d.role], lw=0, zorder=3)
    for i, (_, r) in enumerate(d.iterrows()):
        # direction as a drawn marker, not a font glyph: Helvetica has no triangle and
        # matplotlib would silently box-substitute it.
        m = "^" if r.direction == "up" else "v" if r.direction == "down" else "o"
        ax.plot([r.share + 0.0035], [i], marker=m, ms=3.0, color=ROLE_INK[r.role],
                lw=0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([nice(c) for c in d.code])
    for t, role in zip(ax.get_yticklabels(), d.role):
        t.set_color(ROLE_INK[role])
    ax.set_xlabel("share of the model's attribution", labelpad=2)
    ax.set_xlim(0, d.share.max() * 1.16)
    ax.grid(axis="x", color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.legend(handles=[Line2D([], [], marker="^", color=st.INK_2, lw=0, ms=3.2,
                              label="more raises P(correct)"),
                       Line2D([], [], marker="v", color=st.INK_2, lw=0, ms=3.2,
                              label="more lowers it")],
              loc="lower right", handletextpad=0.3, labelspacing=0.25)
    ax.set_title(f"What the model reads (top {N_TOP} of 57)", pad=4, loc="left")


def _scatter(ax, df, xcol, ycol, xlabel, ylabel, title, label_n=3):
    lo = min(df[xcol].min(), df[ycol].min()) * 0.75
    hi = max(df[xcol].max(), df[ycol].max()) * 1.35
    ax.plot([lo, hi], [lo, hi], color=st.INK_MUTED, lw=0.7, ls=(0, (4, 2)), zorder=2)
    for role in ("tutor", "student"):
        d = df[df.role == role]
        ax.scatter(d[xcol], d[ycol], s=12, color=ROLE_COLOR[role], alpha=0.85, lw=0, zorder=3)
    # limits first, so a label's position on the page is known while placing the next one
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # label the codes that move furthest off the diagonal — the point of the panel
    df = df.assign(_gap=(np.log10(df[ycol]) - np.log10(df[xcol])).abs())
    placed = []                         # (x, y) in points of each label already drawn
    pt = 72 / ax.figure.dpi
    for _, r in df.nlargest(label_n, "_gap").iterrows():
        # Put the label on the side of the diagonal its point already sits on, so it leans
        # away from the cloud instead of across it.
        # Vertically, and horizontally toward the middle of the panel, so a label near an
        # edge stays inside the axes instead of running into the tick labels.
        above = r[ycol] > r[xcol]
        left = np.log(r[xcol] / lo) / np.log(hi / lo) < 0.5
        dx, dy = (4 if left else -4), (5 if above else -6)
        px, py = (v * pt for v in ax.transData.transform((r[xcol], r[ycol])))
        # step away from any label already within a line of this one
        while any(abs(px + dx - qx) < 70 and abs(py + dy - qy) < 9 for qx, qy in placed):
            dy += 9 if above else -9
        placed.append((px + dx, py + dy))
        ax.annotate(nice(r.code, short=True), xy=(r[xcol], r[ycol]),
                    xytext=(dx, dy),
                    textcoords="offset points", fontsize=st.tsize("note"),
                    color=ROLE_INK[r.role], va="center",
                    ha="left" if left else "right")
    rho = df[[xcol, ycol]].corr(method="spearman").iloc[0, 1]
    ax.text(0.03, 0.95, f"ρ = {rho:.2f}", transform=ax.transAxes, va="top",
            fontsize=st.tsize("note"), color=st.INK_2)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.grid(color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.set_title(title, pad=4, loc="left")


def panel_d(ax, g: pd.DataFrame, n_show: int = 8) -> None:
    """How the model's valuation of a move shifts with what the student is doing."""
    states = [c for c in g.columns if c not in ("move", "mean_abs_shap", "range")]
    d = g.nlargest(n_show, "range").iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    marks = ["o", "s", "^", "D"]
    for j, s_ in enumerate(states):
        ax.scatter(d[s_], y, s=16, marker=marks[j % len(marks)], lw=0,
                   color=[st.STUDENT, st.TUTOR, st.TSL, st.EEDI][j % 4],
                   label=s_, zorder=3)
    for i in y:
        ax.plot([d[states].iloc[i].min(), d[states].iloc[i].max()], [i, i],
                color=st.GRID, lw=2.4, zorder=2, solid_capstyle="round")
    ax.axvline(0, color=st.INK_MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([nice(c) for c in d.move])
    for t in ax.get_yticklabels():
        t.set_color(st.TUTOR_INK)
    ax.set_xlabel("shift in the move's attribution when the student does more of this\n"
                  "(% of that move's own mean |SHAP|, doubly centred)", labelpad=2)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.grid(axis="x", color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.legend(loc="upper left", handletextpad=0.25, labelspacing=0.22, borderpad=0.25,
              ncol=2, columnspacing=0.8)
    ax.set_title("What the student is doing changes what the move is worth", pad=4, loc="left")


def main() -> None:
    """Main text: what the model reads. Supplementary: whether that transports (b, c)."""
    st.apply_style()
    a = pd.read_csv(st.INPUT / "fig_mech_a_attribution.csv")
    tot = pd.read_csv(st.INPUT / "fig_mech_a_totals.csv")
    b = pd.read_csv(st.INPUT / "fig_mech_b_transport.csv")
    c = pd.read_csv(st.INPUT / "fig_mech_c_instrument.csv")

    # ── main text: what the model reads ──
    # How that shifts with the student is a table now (fig_bystate.py), so this figure is the attribution alone. Nested gridspec because the totals bar
    # must sit ON the attribution panel as one unit.
    fig = plt.figure(figsize=(st.STYLE["fig_w"] * st.SCALE, 3.9 * st.SCALE))
    left = GridSpec(2, 1, figure=fig, height_ratios=[0.17, 1.0], hspace=0.10,
                    left=0.21, right=0.975, top=0.93, bottom=0.12)
    ax_tot = fig.add_subplot(left[0])
    ax_a = fig.add_subplot(left[1])
    panel_totals(ax_tot, tot)
    panel_a(ax_a, a)
    st.save(fig, "fig_mechanisms")

    # ── supplementary: does the explanation transport? ──
    fig, axes = plt.subplots(1, 2, figsize=(st.STYLE["fig_w"] * 0.72 * st.SCALE,
                                            st.STYLE["fig_h"] * 0.50 * st.SCALE))
    _scatter(axes[0], b, "share_tsl", "share_eedi",
             "attribution on TSL (test)", "on Eedi (test)",
             "Does the explanation transport?")
    _scatter(axes[1], c, "share_labels", "share_dst",
             "attribution under 122B labels", "under distilled children",
             "Does it depend on the instrument?")
    for ax, letter in zip(axes, "ab"):
        st.panel_letter(ax, letter)
    fig.subplots_adjust(wspace=0.52, left=0.12, right=0.97, top=0.88, bottom=0.19)
    st.save(fig, "fig_mechanisms_si")


if __name__ == "__main__":
    main()
