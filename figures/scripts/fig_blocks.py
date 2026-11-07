#!/usr/bin/env python
"""fig_blocks — what each block adds (a) next to what the model reads (b), one main-text figure.

  a  the ablation of `fig_ablation`: each block added over the objective and the other block,
     log-loss reduction above AUROC gain, on the two held-out test splits, for every learner.
  b  the fourteen largest conversation columns by mean |SHAP| as a beeswarm (the points of
     `fig_beeswarm_si`), each point one training response in its role's colour, light where the
     session's rate of the code is low and dark where it is high; the construct totals of
     `fig_mechanisms` above.

Side by side because they answer the two questions the paper keeps apart: what a block ADDS
and what the model PRICES. The panels are drawn by the two scripts' own functions, so this
figure cannot drift from them. Reads only `figures/input/`.

    python figures/scripts/fig_blocks.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import fig_ablation as abl
import fig_beeswarm_si as bee
import fig_bystate as bs
import fig_mechanisms as mech
import style as st

W, H = 6.5, 4.4                         # inches; `--table` widens W for panel c
W_TABLE = 9.4
A_X, A_W = 0.56, 2.10                   # panel a plots
B_X, B_W = 4.46, 1.86                   # panel b swarm; its labels sit to the left
N_TOP = mech.N_TOP
#: the session's rate of the code, low (light) to high (dark), in each role's own hue
SHADE = {role: LinearSegmentedColormap.from_list(role, ["#ECECE8", fill, ink])
         for role, fill, ink in (("tutor", st.TUTOR, st.TUTOR_INK),
                                 ("student", st.STUDENT, st.STUDENT_INK))}
TOP, BOTTOM, GAP = 0.54, 0.32, 0.18     # panel a: legend band above, tick labels below
LETTER_Y = H - 0.13                     # level with the first row of content, no strip above


def box(x, y, w, h):
    return (x / W, y / H, w / W, h / H)


def swarm(ax, cols: pd.DataFrame, pts: pd.DataFrame) -> None:
    """The top columns' SHAP values, one row each, coloured by role and shaded by the rate."""
    for i, r in enumerate(cols.itertuples()):
        s = pts[f"s__{r.column}"].to_numpy(float)
        q = pts[f"q__{r.column}"].to_numpy(float)
        ok = ~np.isnan(q)
        y = i + bee.swarm_offsets(s, 0.40, seed=i)
        area = 1.2 * st.mark(1.0) ** 2                  # points^2
        ax.scatter(s[~ok], y[~ok], s=area, color=st.INK_MUTED, lw=0, alpha=0.5, rasterized=True)
        ax.scatter(s[ok], y[ok], s=area, c=q[ok], cmap=SHADE[r.role], vmin=0, vmax=1, lw=0,
                   alpha=0.85, rasterized=True)
    ax.axvline(0, color=st.INK_MUTED, lw=st.mark(0.6), zorder=0)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([lab + ("$^\\dagger$" if res else "")
                        for lab, res in zip(cols.label, cols.residual)])
    for t, role in zip(ax.get_yticklabels(), cols.role):
        t.set_color(mech.ROLE_INK[role])
    ax.set_ylim(len(cols) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.set_xlabel("SHAP value (log-odds)", labelpad=2)


def shade_key(fig, xc: float, y: float) -> None:
    """Two small gradients, one per role, for what the shading means, centred on `xc`."""
    w, h = 0.42, 0.06
    x = xc - (2 * w + 0.06) / 2
    for j, role in enumerate(("tutor", "student")):
        k = fig.add_axes(box(x + j * (w + 0.06), y, w, h))
        k.imshow(np.linspace(0, 1, 64)[None, :], aspect="auto", cmap=SHADE[role])
        k.set_axis_off()
    fig.text((x - 0.05) / W, (y + h / 2) / H, "Low rate", ha="right",
             va="center", fontsize=st.tsize("note"), color=st.INK)
    fig.text((x + 2 * w + 0.06 + 0.05) / W, (y + h / 2) / H, "High rate", ha="left", va="center",
             fontsize=st.tsize("note"), color=st.INK)


def main(table: bool = False) -> None:
    """With `table`, the by-state SHAP grid (Table 3's) is drawn as panel c on the right."""
    global W, A_W
    W = W_TABLE if table else 6.5
    # the wide version has room for a fifth learner, in a wider panel a
    A_W = 2.34 if table else 2.10
    learners = abl.LEARNERS_SI if table else abl.LEARNERS
    st.apply_style()
    df = pd.read_csv(st.INPUT / "fig_main_a_transfer.csv")
    df = df[df.view.isin(abl.VIEWS)]
    fig = plt.figure(figsize=(W * st.SCALE, H * st.SCALE))

    # ── a: what each block adds, log loss above AUROC ──
    ah = (H - TOP - BOTTOM - GAP) / 2
    ax_ll = fig.add_axes(box(A_X, BOTTOM + ah + GAP, A_W, ah))
    ax_au = fig.add_axes(box(A_X, BOTTOM, A_W, ah))
    abl.panel(ax_ll, df, "ll", learners)
    abl.panel(ax_au, df, "auroc", learners)
    abl.share_range([ax_ll, ax_au])
    for ax, title, ylab in ((ax_ll, "Log loss", "log-loss reduction"),
                            (ax_au, "AUROC", "AUROC gain")):
        ax.set_title("", loc="left")
        ax.text(0.03, 0.95, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=st.tsize("label"), color=st.INK)
        ax.set_ylabel(ylab, labelpad=3)
    ax_ll.tick_params(axis="x", labelbottom=False)
    # the blocks inside the log-loss plot, where its top right is empty
    ax_ll.legend(handles=[Line2D([], [], color=abl.BLOCK_COLOR[b], lw=1.8, label=f"adding {b}")
                          for b in abl.BLOCKS],
                 loc="upper right", handlelength=1.2, handletextpad=0.4, labelspacing=0.3,
                 fontsize=st.tsize("note"))
    # the learners in one row above the plots, in the order their points sit in every group
    fig.legend(handles=[Line2D([], [], ls="", **abl.marker_kw(l, st.NEUTRAL),
                               label=abl.SHORT.get(l, l)) for l in learners],
               loc="center", ncol=len(learners), frameon=False,
               bbox_to_anchor=((A_X + A_W / 2) / W, 1 - 0.36 / H),
               handlelength=0.8, handletextpad=0.3, columnspacing=1.0,
               fontsize=st.tsize("legend"))

    # ── b: what the model reads ──
    tot = pd.read_csv(st.INPUT / "fig_mech_a_totals.csv")
    cols = pd.read_csv(st.INPUT / "fig_beeswarm_columns.csv").head(N_TOP)
    pts = pd.read_csv(st.INPUT / "fig_beeswarm_points.csv")
    ax_tot = fig.add_axes(box(B_X, H - 0.62, B_W, 0.50))
    mech.panel_totals(ax_tot, tot, labels_above=True)
    shade_key(fig, B_X + B_W / 2, H - 0.82)   # under the totals bar's centre
    ax_b = fig.add_axes(box(B_X, BOTTOM + 0.12, B_W, H - 0.96 - BOTTOM - 0.12))
    swarm(ax_b, cols, pts)

    letters = [(0.0, "a"), (B_X - 1.20, "b")]
    if table:
        # ── c: how the model's reading of a move shifts with the student's state ──
        g = pd.read_csv(st.INPUT / "fig_mech_d_by_state.csv")
        g = g.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        # rows as tall as panel b's and level with them from the bottom, so the grid is shorter
        # and the colour bar sits in the space it leaves, over the cells
        bot = BOTTOM + 0.12
        row = (H - 0.96 - BOTTOM - 0.12) / N_TOP
        top = bot + row * len(g)
        ax_m, ax_g = (fig.add_axes(box(x, bot, w, top - bot))
                      for x, w in ((7.86, 0.32), (8.22, 1.10)))
        # the colour bar over the grey column and the cells, as thick as panel b's totals bar
        # and on its line: its extent is read off the drawn bar, not restated
        tb = ax_tot.patches[0].get_window_extent().transformed(fig.transFigure.inverted())
        ax_cb = fig.add_axes((7.86 / W, tb.y0, 1.46 / W, tb.height))
        bs.draw(ax_m, ax_g, ax_cb, g, nto=False, orientation="horizontal",
                headers={"reasoning shown": "reasoning",
                         "confident answers": "confident\nanswers"})
        # no ticks: the two ends, written beside the bar, are all the scale a reader needs
        ax_cb.set_xticks([])
        for x, ha, v in ((0.0, "right", -bs.LIMIT), (1.0, "left", bs.LIMIT)):
            ax_cb.text(x + (-0.03 if ha == "right" else 0.03), 0.5,
                       f"{v:+.0f}".replace("-", "\u2212"),          # a true minus, as on axes
                       transform=ax_cb.transAxes, ha=ha, va="center",
                       fontsize=st.tsize("tick"), color=st.INK_2)
        # steeper headers: four narrow columns cannot hold the 30-degree ones side by side
        for t in ax_g.get_xticklabels():
            t.set_rotation(55)
        letters.append((6.48, "c"))
    for x, letter in letters:
        fig.text(x / W, LETTER_Y / H, letter, fontsize=st.tsize("panel_letter"),
                 fontweight="bold", va="top", color=st.INK)
    st.save(fig, "fig_blocks_table" if table else "fig_blocks")


def tight_range(ax) -> None:
    """The y range of what is drawn (intervals included), padded a little, with about four ticks."""
    from matplotlib.ticker import MaxNLocator
    ys = np.concatenate([np.asarray(l.get_ydata(), float) for l in ax.get_lines()])   # 0 line included
    lo, hi = min(ys.min(), 0.0), max(ys.max(), 0.0)
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))


MS_A = 0.85                             # marker size factor in Figure 2a, so markers never overlap


def abs_header(fig, ax_m, c_top_in: float) -> None:
    """The mean column's header, "Mean" over "|SHAP|", with the two bars drawn as lines: in this
    Helvetica the "|" glyph is identical to a capital I, so a typed "|SHAP|" reads "ISHAPI"."""
    ax_m.set_xticklabels([""])
    x = ax_m.get_position().x0 + ax_m.get_position().width / 2
    y0 = ax_m.get_position().y1 + 0.025 / fig.get_figheight()
    fs = st.tsize("tick")                    # as the angled column headers beside it
    shap = fig.text(x, y0, "SHAP", ha="center", va="bottom", fontsize=fs, color=st.INK)
    fig.text(x, y0 + 1.25 * fs / 72 / fig.get_figheight(), "Mean", ha="center", va="bottom",
             fontsize=fs, color=st.INK)
    fig.canvas.draw()
    bb = shap.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    pad = 0.015 / fig.get_figwidth()
    lo = bb.y0 - 0.15 * bb.height
    hi = bb.y1 + 0.02 * bb.height
    for xb in (bb.x0 - pad, bb.x1 + pad):
        fig.add_artist(Line2D([xb, xb], [lo, hi], transform=fig.transFigure, color=st.INK, lw=0.5,
                              solid_capstyle="butt"))


def main_table(name: str = "fig_blocks_table") -> None:
    """The paper's Figure 2, one row, as in the manuscript:

      a  what each block adds: log-loss reduction above AUROC gain, on the two test splits, for
         logistic regression, XGBoost and TabPFN;
      b  XGBoost's TreeSHAP values for its fourteen most-attributed columns (a beeswarm);
      c  logistic regression's weights for the same columns, row for row, b and c equally wide;
         above both, each model's attribution by construct, one bar under the other.

    The by-state grid that was panel c is the supplement's own figure (`fig_bystate.py`).

    Drawn at the text width and printed 1:1, with the printed sizes of Figures 1 and 3; the shared
    panel functions' marks (fig_ablation, fig_bystate), sized for wider canvases, at 82%.
    """
    W2 = st.TEXT_W
    learners = list(abl.LEARNERS)
    st.apply_style(paper=True, mark=0.82)
    fs_code = st.tsize("code")
    df = pd.read_csv(st.INPUT / "fig_main_a_transfer.csv")
    df = df[df.view.isin(abl.VIEWS)]
    # vertical layout, inches from the bottom
    ROW2 = 0.103                        # one beeswarm / table row
    XLAB = 0.35                         # x axes and their labels
    b_h = N_TOP * ROW2
    head = 0.69                         # above the plots: a's legends; the two shares bars, their names and b's key
    b_h += ROW2                         # one more row at the top of b and c: the model's name
    H2 = XLAB + b_h + head
    top2 = XLAB + b_h                   # top of the plotting area
    fig = plt.figure(figsize=(W2 * st.SCALE, H2 * st.SCALE))

    def bx(x, y, w, h):
        return (x / W2, y / H2, w / W2, h / H2)

    # ── a: what each block adds, log loss above AUROC ──
    A_X, A_W, A_GAP = 0.45, 1.52, 0.10                # a wider than b and c: the paper's headline
    a_top = H2 - 0.33                   # under the two legend rows
    ah = (a_top - XLAB - A_GAP) / 2
    ax_ll = fig.add_axes(bx(A_X, XLAB + ah + A_GAP, A_W, ah))
    ax_au = fig.add_axes(bx(A_X, XLAB, A_W, ah))
    # short y labels name the metric; the caption says both are drawn so that up is better
    for ax, metric, ylab in ((ax_ll, "ll", "Log loss"), (ax_au, "auroc", "AUROC")):
        # markers 0.10 apart within a block, 0.14 between the two blocks, 0.30 between the splits
        abl.panel(ax, df, metric, learners, step=0.84, block_dx=0.17, ldx=0.10, ms_scale=MS_A)
        tight_range(ax)
        ax.set_title("", loc="left")            # the metric is the y label instead
        ax.set_ylabel(ylab, labelpad=3)
    ax_ll.tick_params(axis="x", labelbottom=False)
    # two legend rows in the band above: the blocks (colour), then the learners (marker shape)
    leg_kw = dict(frameon=False, handletextpad=0.35, columnspacing=1.0, fontsize=st.tsize("legend"),
                  borderaxespad=0, borderpad=0)
    a_mid = (A_X + A_W / 2) / W2                     # both legend rows centre on a's x axis
    def blocks_legend(spacing: float):
        return fig.legend(handles=[Line2D([], [], color=abl.BLOCK_COLOR[b], lw=st.mark(1.8), label=b)
                                   for b in abl.BLOCKS],
                          loc="center", ncol=2, bbox_to_anchor=(a_mid, (H2 - 0.09) / H2),
                          handlelength=1.1, **(leg_kw | {"columnspacing": spacing}))
    leg_blocks = blocks_legend(leg_kw["columnspacing"])
    # the learners' keys in a light grey with a black edge: shape only, no block colour
    leg_learn = fig.legend(handles=[Line2D([], [], ls="", **(abl.marker_kw(l, "#D4D4D0")
                                                             | {"ms": st.mark(abl.SIZE.get(l, 5.0)) * MS_A}),
                                           label=abl.SHORT.get(l, l)) for l in learners],
                           loc="center", ncol=len(learners), bbox_to_anchor=(a_mid, (H2 - 0.24) / H2),
                           handlelength=0.8, **leg_kw)
    # justify the two rows: widen the gap between the two block entries so "tutor moves" ends where
    # "TabPFN" ends (both rows are centred, so the width to add is twice the gap at one end)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    gap_in = leg_learn.get_window_extent(rend).x1 - leg_blocks.get_window_extent(rend).x1   # pixels
    if gap_in > 0:
        leg_blocks.remove()
        em = st.tsize("legend") / 72 * fig.dpi                       # one font size, in pixels
        leg_blocks = blocks_legend(leg_kw["columnspacing"] + 2 * gap_in / em)

    # ── b and c: XGBoost's SHAP values and logistic regression's weights, row for row ──
    cols = pd.read_csv(st.INPUT / "fig_beeswarm_columns.csv").head(N_TOP)
    pts = pd.read_csv(st.INPUT / "fig_beeswarm_points.csv")
    lr = pd.read_csv(st.INPUT / "fig_learners_columns.csv").set_index("column")
    B_X2, RIGHT, C_GAP = 3.07, 0.05, 0.22           # b's plot starts here (code names to its left)
    PW = (W2 - RIGHT - B_X2 - C_GAP) / 2            # b and c share what is left, equally
    C_X2 = B_X2 + PW + C_GAP
    # each model's attribution by construct, one bar under the other across both plots, the model
    # named beside its bar and the constructs named once, under the bars
    BAR_H, BAR_TOP = 0.11, H2 - 0.07
    bars = [(pd.read_csv(st.INPUT / src), model) for src, model in
            (("fig_mech_a_totals.csv", "XGBoost"), ("fig_mech_a_totals_logreg.csv", "Logistic regression"))]
    for k, (tot, model) in enumerate(bars):
        y = BAR_TOP - BAR_H - k * (BAR_H + 0.04)
        ax = fig.add_axes(bx(B_X2, y, W2 - RIGHT - B_X2, BAR_H))
        left = 0.0
        for r in tot.itertuples():
            tutor = mech.TOTAL_COLOR[r.construct] == st.TUTOR   # white numbers on the dark segment
            ax.barh([0], [r.share], left=left, height=1.0, color=st.TUTOR if tutor else st.tint(mech.TOTAL_COLOR[r.construct], st.LIGHT),
                    edgecolor="none", lw=0)         # tutor moves in full teal, the others a light fill
            ax.text(left + r.share / 2, 0, f"{r.share:.0%}", ha="center", va="center",
                    fontsize=st.tsize("tick"), color="white" if tutor else st.INK)
            left += r.share
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        fig.text((B_X2 - 0.06) / W2, (y + BAR_H / 2) / H2, model, ha="right", va="center",
                 fontsize=st.tsize("tick"), color=st.INK)
    names_y = BAR_TOP - 2 * BAR_H - 0.04 - 0.08
    # each name under its segments: the mean of the two bars' segment centres
    centre = np.mean([(t.share.cumsum() - t.share / 2).to_numpy() for t, _ in bars], axis=0)
    for frac, construct, ink in zip(centre, ("Tutor moves", "Student signals", "Objective"),
                                    (st.TUTOR_INK, st.STUDENT_INK, st.INK)):
        fig.text((B_X2 + frac * (W2 - RIGHT - B_X2)) / W2, names_y / H2, construct, ha="center", va="center",
                 fontsize=st.tsize("tick"), color=ink)
    # b's shading key, under the names, over b's plot
    key_mid = top2 + 0.10
    kw, kh, kgap = 0.18, 0.05, 0.033
    kx = B_X2 + PW / 2 - (2 * kw + kgap) / 2 + 0.025
    k_right = kx + 2 * kw + kgap
    for j, role in enumerate(("tutor", "student")):
        k = fig.add_axes(bx(kx + j * (kw + kgap), key_mid - kh / 2, kw, kh))
        k.imshow(np.linspace(0, 1, 64)[None, :], aspect="auto", cmap=SHADE[role])
        k.set_axis_off()
    fig.text((kx - 0.033) / W2, key_mid / H2, "Low rate", ha="right", va="center",
             fontsize=st.tsize("tick"), color=st.INK)
    fig.text((k_right + 0.033) / W2, key_mid / H2, "High rate", ha="left", va="center",
             fontsize=st.tsize("tick"), color=st.INK)

    def headroom(ax, title: str) -> None:
        """One more row above the first code, and the model's name in it, at the top left."""
        ax.set_ylim(len(cols) - 0.5, -1.5)
        ax.text(0.02, -1.0, title, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=st.tsize("label"), color=st.INK, zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))   # over the grid and zero lines

    ax_b = fig.add_axes(bx(B_X2, XLAB, PW, b_h))
    swarm(ax_b, cols, pts)
    st.zebra(ax_b, range(len(cols)))
    for tl in ax_b.get_yticklabels():
        tl.set_fontsize(fs_code)
    ax_b.set_xlabel("SHAP value (log-odds)", labelpad=2)
    headroom(ax_b, "XGBoost SHAP values")

    ax_c = fig.add_axes(bx(C_X2, XLAB, PW, b_h))
    for i, r in enumerate(cols.itertuples()):
        w = lr.loc[r.column, "logreg_coef"]
        fill, ink = (st.TUTOR, st.TUTOR_INK) if r.role == "tutor" else (st.STUDENT, st.STUDENT_INK)
        ax_c.plot([0, w], [i, i], color=fill, lw=st.mark(1.6), solid_capstyle="butt", zorder=3)
        ax_c.plot([w], [i], "o", ms=st.mark(3.8), mfc=fill, mec=ink, mew=st.mark(0.7), zorder=4)
    ax_c.axvline(0, color=st.INK_MUTED, lw=st.mark(0.6), zorder=1)
    ax_c.set_yticks(range(len(cols)))
    ax_c.set_yticklabels([])
    ax_c.tick_params(axis="y", length=0)
    st.zebra(ax_c, range(len(cols)))
    ax_c.grid(axis="x", color=st.GRID, lw=st.STYLE["grid_lw"])
    ax_c.set_xlabel("Weight (log-odds per SD)", labelpad=2)
    headroom(ax_c, "Logistic regression weights")

    # letters at the top left of each panel
    for x, y, letter in ((0.0, H2 - 0.02, "a"), (2.01, H2 - 0.02, "b"),
                         (C_X2 - C_GAP + 0.03, XLAB + b_h, "c")):   # c beside its plot's top
        fig.text(x / W2, y / H2, letter, fontsize=st.tsize("panel_letter"), fontweight="bold",
                 va="top", color=st.INK)
    st.save(fig, name, width=W2)


if __name__ == "__main__":
    import sys
    if "--table" in sys.argv:
        main_table()
    else:
        main()
