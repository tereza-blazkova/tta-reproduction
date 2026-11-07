#!/usr/bin/env python
"""fig_bystate — what a tutor move is worth depends on what the student is doing.

Reads only `figures/input/fig_mech_d_by_state.csv` (from `results/shap-by-state/`).

Drawn as a table because the quantity
is a grid of numbers a reader wants to read off, not a shape to compare: one row per tutor move,
the move's mean |SHAP| beside it in grey, then one coloured cell per student state.

A cell is per-row attribution averaged within the top third of sessions for that student state,
doubly centred (row and column means removed, so neither a generally important move nor a
generally extreme state drives it), as a percentage of the move's own mean |SHAP|. Rows are the
twelve moves the model reads most, ordered by mean |SHAP|; restricting to them is what keeps the
denominator away from zero.

A statement about what the model prices, never about what a move does.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import style as st

#: struggling -> secure, so the colour runs across a reading of the student's state
STATES = ["not knowing", "hedging", "confident answers", "reasoning shown"]
LIMIT = 40.0          # symmetric colour range, in % of the move's mean |SHAP|


def nice(code: str) -> str:
    return code.replace("_", " ")


def draw(ax_m, ax, ax_cb, d: pd.DataFrame, nto: bool = True,
         orientation: str = "vertical", headers: dict | None = None, cell_fs: float | None = None) -> None:
    """The grid on three given axes: mean |SHAP|, the state cells, the colourbar.

    `nto` marks the published taxonomy's moves in the row labels; a figure whose caption says
    it instead can leave the labels short.
    """
    V = d[STATES].to_numpy(float)
    n = len(d)
    cell_fs = cell_fs or st.tsize("tick")         # the numbers in the cells and the mean column
    fig = ax.figure
    # ── mean |SHAP| (skipped when ax_m is None: a second grid beside one that already shows it) ──
    if ax_m is not None:
        _mean_column(ax_m, d, n, nto, cell_fs)

    # ── the state cells ──
    norm = TwoSlopeNorm(vmin=-LIMIT, vcenter=0, vmax=LIMIT)
    im = ax.imshow(V, cmap=st.DIVERGING_CMAP, norm=norm, aspect="auto")
    _cells(ax, V, n, norm, cell_fs, headers)
    _gutters([a for a in (ax_m, ax) if a is not None], n)
    if ax_cb is not None:
        _colourbar(fig, im, ax_cb, orientation)


def _mean_column(ax_m, d, n, nto, cell_fs) -> None:
    ax_m.imshow(np.ones((n, 1)), cmap="Greys", vmin=0, vmax=8, aspect="auto")  # light grey
    for i, v in enumerate(d.mean_abs_shap):
        ax_m.text(0, i, f"{v:.3f}", ha="center", va="center", fontsize=cell_fs, color=st.INK)
    ax_m.set_xticks([0]); ax_m.set_xticklabels(["Mean\nabsolute\nSHAP"], color=st.INK)   # this Helvetica draws "|" as I
    ax_m.xaxis.tick_top()
    ax_m.set_yticks(range(n))
    ax_m.set_yticklabels([nice(m) + (" (NTO)" if nto and o == "NTO" else "")
                          for m, o in zip(d.move, d.origin)], fontsize=st.tsize("code"))
    for t in ax_m.get_yticklabels():
        t.set_color(st.TUTOR_INK)


def _cells(ax, V, n, norm, cell_fs, headers) -> None:
    for i in range(n):
        for j in range(len(STATES)):
            v = V[i, j]
            lab = "0" if round(v) == 0 else f"{v:+.0f}".replace("-", "\u2212")
            r, g, b, _ = st.DIVERGING_CMAP(norm(v))
            # white only where dark text would fall below 4.5:1 (the deep blue end)
            dark = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2 < 0.17
            ax.text(j, i, lab, ha="center", va="center", fontsize=cell_fs,
                    color="white" if dark else st.INK)
    ax.set_xticks(range(len(STATES)))
    ax.set_xticklabels([(headers or {}).get(c, c) for c in STATES], rotation=30, ha="left",
                       rotation_mode="anchor",
                       color=st.STUDENT_INK)
    ax.xaxis.tick_top()
    ax.set_yticks([])


def _gutters(axes, n) -> None:
    for a in axes:
        a.tick_params(length=0)
        for sp in a.spines.values():
            sp.set_visible(False)
        # white gutters between cells
        a.set_xticks(np.arange(-0.5, a.get_xlim()[1], 1), minor=True)
        a.set_yticks(np.arange(-0.5, n, 1), minor=True)
        a.grid(which="minor", color="white", lw=st.mark(2.2))
        a.tick_params(which="minor", length=0)


def _colourbar(fig, im, ax_cb, orientation) -> None:
    cb = fig.colorbar(im, cax=ax_cb, orientation=orientation)
    if orientation == "horizontal":            # a bar above the grid, its label on top
        cb.ax.xaxis.set_label_position("top")
        cb.set_ticks([-40, -20, 0, 20, 40])
    cb.set_label("Change in attribution (% of mean absolute SHAP)", color=st.INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)


def main(src: str = "fig_mech_d_by_state.csv", out: str = "fig_bystate",
         sort: bool = True) -> None:
    """Draw one grid. The supplementary grids (logistic regression; XGBoost with each move's
    own-rate dependence removed) use the same function, so they share Table 3's format and
    colour scale and can be read against it cell by cell."""
    st.apply_style()
    d = pd.read_csv(st.INPUT / src)
    if sort:       # Table 3 orders by its own mean |SHAP|; the SI grids keep that order
        d = d.sort_values("mean_abs_shap", ascending=False)
    d = d.reset_index(drop=True)
    n = len(d)
    fig = plt.figure(figsize=(st.STYLE["fig_w"] * 0.80 * st.SCALE, (0.95 + 0.27 * n) * st.SCALE))
    # grey mean-|SHAP| column, then the state cells, then a colourbar
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, len(STATES), 0.18], wspace=0.06,
                          left=0.36, right=0.90, top=0.78, bottom=0.04)
    draw(*(fig.add_subplot(gs[0, i]) for i in range(3)), d, nto=False)   # short labels, as Fig. 2 had them
    st.save(fig, out)


def main_pair(out: str = "fig_bystate_pair") -> None:
    """Fig. 7 and its own-rate-removed version side by side: (a) as drawn, (b) with each move's
    dependence on its own rate removed. Same rows, same mean |SHAP| column (the two files share
    it), same colour scale, one colourbar."""
    st.apply_style(paper=True)          # the SI's sizes: drawn at the text width, printed 1:1
    a = (pd.read_csv(st.INPUT / "fig_mech_d_by_state.csv")
         .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    b = pd.read_csv(st.INPUT / "fig_si_bystate_ownrate.csv")
    if list(a.move) != list(b.move) or not np.allclose(a.mean_abs_shap, b.mean_abs_shap):
        raise SystemExit("the two grids no longer share rows and mean |SHAP|")
    n = len(a)
    fig = plt.figure(figsize=(st.TEXT_W * st.SCALE, (0.80 + 0.19 * n) * st.SCALE))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.9, len(STATES), 0.55, len(STATES), 0.18],
                          wspace=0.06, left=0.20, right=0.915, top=0.75, bottom=0.01)
    ax_m, ax_a, ax_b, ax_cb = (fig.add_subplot(gs[0, i]) for i in (0, 1, 3, 4))
    draw(ax_m, ax_a, None, a, nto=False)
    draw(None, ax_b, ax_cb, b, nto=False)
    # the letters on one line above the column headers: a where the row names start, b at its grid
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for ax, letter in ((ax_m, "a"), (ax_b, "b")):
        x0 = inv.transform((ax.get_tightbbox(r).x0 if letter == "a" else ax.get_window_extent(r).x0, 0))[0]
        fig.text(x0, 0.995, letter, fontsize=st.tsize("panel_letter"), color=st.INK,
                 fontweight="bold", va="top", ha="left")
    st.save(fig, out, width=st.TEXT_W)


if __name__ == "__main__":
    main()
    main_pair()
    main("fig_si_bystate_logreg.csv", "fig_bystate_logreg_si", sort=False)
    main("fig_si_bystate_ownrate.csv", "fig_bystate_ownrate_si", sort=False)
    main("fig_si_bystate_interaction.csv", "fig_bystate_interaction_si", sort=False)
