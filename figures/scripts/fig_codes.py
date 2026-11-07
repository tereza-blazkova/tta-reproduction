#!/usr/bin/env python
"""fig_codes — how often, how much and when each code occurs, and what one session looks like (Fig. 1).

Left column: (a) how all the codes, stacked by role, are distributed over b's first two columns,
beside the position colours' key, over (b) the student signals; then (d) one session, turn by turn. Right column: (c) the tutor moves. In (b)
and (c) one row per code, ordered within each role by the share of training sessions in which the
code occurs at least once, and three columns share the code labels:

* **Sessions with it** -- the share of the role's training sessions with at least one turn
  carrying the code (a session counts once).
* **Turns when present** -- in those sessions, the code's median share of the role's turns (the
  raw session rate).
* **Position in the session** -- how often the code falls in the first fifth, the middle three
  fifths and the last fifth of the session, relative to all of the role's turns there; each cell
  is as wide as its phase.

(d) is the session chosen by `prepare_session_length_inputs.py`'s rule (typical length, typical
mix of code groups): one row per code group, a mark at each turn whose code is in the group,
the fifths marked as in (b) and (c). Reads only `figures/input/fig_presence.csv`
(prepare_presence_inputs.py), `fig_timing.csv` (prepare_timing_inputs.py), whose twentieths are
summed into the three phases here, and `fig_example_session.csv`
(prepare_session_length_inputs.py). Drawn at the text width and printed 1:1.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import NullLocator

import style as st

W = st.TEXT_W                           # the manuscript's text width (prints 1:1)
ROW = 0.098                             # inches per code
TOP, HEAD, BOTTOM = 0.27, 0.03, 0.15    # inches: header band (the letter sits beside it), header to plot, tick labels
LAB = {"tutor": 1.10, "student": 0.89}  # inches: the code labels (longest + the 2 pt pad)
GAP, COL_GAP, END = 0.15, 0.16, 0.012   # inches: between panels, between the two columns, after "end"
P3 = 0.62                               # inches: position (room for start / middle / end)
P1 = (st.TEXT_W - END - LAB["tutor"] - LAB["student"] - 4 * GAP - COL_GAP - 2 * P3) / 4
P = (P1, P1, P3)                        # inches: sessions, turns, position
PHASES = (0.0, 0.2, 0.8, 1.0)           # start, middle, end: fifths of the session
STEP, NSTEP = 0.5, 9                    # colour steps: half an octave, 1x in the middle one
#: position colours, less often in the phase (dusty purple #80618F) to more often (muted ochre
#: #D5A044), through a light-grey midpoint: the poles of style.DIVERGING. Each half runs in OKLab from
#: the midpoint to its pole. Purple (OKLCH hue 314) and ochre (78) sit away from teal (189) and peach
#: (52); the ochre is lighter than the purple (L 0.74 vs 0.54), so its steps look a little weaker.
STEPS = ("#80618F", "#9A82A5", "#B5A5BB", "#D1C8D2", "#EDECE8", "#E8D9C2", "#E2C79B", "#DCB373", "#D5A044")
#: the bars and marks: the role colours mixed with this share of white, the edges in their dark shade
LIGHT = st.LIGHT
#: names shortened only where a row label would overrun the column
SHORT = {"prompting alternative representation": "prompting alt. representation"}
HEADERS = ("Sessions\nwith it (%)", "Turns when\npresent (%)", "Position in\nthe session")
X_MAX = {"tutor": (1.0, 0.15), "student": (1.0, 0.32)}
X_TICKS = {"tutor": ([0, 0.5, 1], [0, 0.05, 0.10, 0.15]), "student": ([0, 0.5, 1], [0, 0.1, 0.2, 0.3])}
#: (a) all the codes by b's first two columns, beside the key; (d) one session, under b
C_H = 0.40                              # (a) plot height
A_BINS = 10                             # (a) bars per histogram, across the column's x axis
D_LAB = 0.80                            # (d) its group names, inches
#: (d) rows, top to bottom: the tutor's blocks and the student's states (stage 33), each with its
#: organising / social group; the names as in Fig. 3's exchanges
D_ROWS = [("tutor", "eliciting", "eliciting"), ("tutor", "telling", "telling"),
          ("tutor", "evaluating", "evaluating"), ("tutor", "affective", "affective"),
          ("tutor", "organising", "organising"),
          ("student", "reasoning_shown", "reasoning"), ("student", "answering_confidently", "confident answer"),
          ("student", "hedging", "hedging"), ("student", "not_knowing", "not knowing"),
          ("student", "social", "social")]


def inks(role: str) -> tuple:
    """(fill, ink): a light tint of the role's colour, and its dark shade for edges and names."""
    return ((st.tint(st.TUTOR, LIGHT), st.TUTOR_INK) if role == "tutor"
            else (st.tint(st.STUDENT, LIGHT), st.STUDENT_INK))


def fy(y: float) -> float:
    """Inches from the top of the figure -> figure fraction."""
    return 1 - y / H


def heading(fig, x0: float, top: float, letter: str) -> None:
    fig.text(x0 / W, fy(top + 0.02), letter, fontsize=st.tsize("panel_letter"), fontweight="bold",
             va="top", color=st.INK)


def phases(tim: pd.DataFrame, role: str) -> pd.DataFrame:
    """log2 lift of each code in the three phases, from the twentieths."""
    t = tim[tim.role == role]
    n = t.pivot(index="code", columns="bin", values="n")
    nb = n.shape[1]
    edges = [round(p * nb) for p in PHASES]
    if not np.allclose(np.array(edges) / nb, PHASES):
        raise SystemExit(f"{nb} bins do not split into the phases {PHASES}")
    cnt = pd.concat([n.iloc[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    base = cnt.sum(axis=0) / cnt.to_numpy().sum()           # all of the role's labelled turns
    return np.log2(cnt.div(cnt.sum(axis=1), axis=0) / base)


def block(fig, role: str, x0: float, top: float, pres: pd.DataFrame, tim: pd.DataFrame, letter: str):
    """One role's codes; `top` is the heading's top, in inches from the top of the figure."""
    p = pres[pres.role == role].sort_values("share_sessions", ascending=False).reset_index(drop=True)
    k = len(p)
    fill, ink = inks(role)
    y_top = H - top - TOP                                    # the plots' top, inches from the bottom
    labels = [SHORT.get(lab, lab) + ("$^\\dagger$" if res else "") for lab, res in zip(p.label, p.residual)]
    xs, x = [], x0 + LAB[role]
    for w in P:
        xs.append(x)
        x += w + GAP
    axs = [fig.add_axes((xi / W, (y_top - k * ROW) / H, w / W, k * ROW / H)) for xi, w in zip(xs, P)]
    for j, (ax, col) in enumerate(zip(axs[:2], ("share_sessions", "turns_q50"))):
        ax.barh(range(k), p[col], height=0.64, color=fill, edgecolor=ink, lw=0.3, zorder=3)
        st.zebra(ax, range(k))
        ax.set_xlim(0, X_MAX[role][j])
        ax.set_xticks(X_TICKS[role][j])
        ax.set_xticklabels([f"{v * 100:g}" for v in X_TICKS[role][j]])
        ax.grid(axis="x", color=st.GRID, lw=st.STYLE["grid_lw"], zorder=1)
        ax.set_axisbelow(True)
    # position: one row of three cells per code, each as wide as its phase
    z = phases(tim, role).loc[p.code].to_numpy()
    half = STEP * NSTEP / 2
    cmap = ListedColormap(list(STEPS))
    norm = BoundaryNorm(np.linspace(-half, half, NSTEP + 1), NSTEP)
    ax = axs[2]
    im = ax.pcolormesh(PHASES, np.arange(k + 1) - 0.5, np.clip(z, -half, half), cmap=cmap, norm=norm,
                       edgecolor="white", lw=0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([(a + b) / 2 for a, b in zip(PHASES[:-1], PHASES[1:])])
    ax.set_xticklabels(["start", "middle", "end"])
    ax.tick_params(axis="x", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for ax in axs:
        ax.set_ylim(k - 0.5, -0.5)
        ax.set_yticks(range(k))
        ax.tick_params(axis="y", length=0, labelleft=False)
    axs[0].tick_params(axis="y", labelleft=True, pad=2)
    axs[0].set_yticklabels(labels, color=ink, fontsize=st.tsize("code"))
    for ax, head in zip(axs, HEADERS):
        ax.text(0.5, 1 + HEAD / (k * ROW), head, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=st.tsize("label"), color=st.INK, linespacing=1.1)
    heading(fig, x0, top, letter)
    return im, xs


def key(fig, im, x0: float, y0: float, w: float) -> None:
    """The position colours: one row of steps, as tall as a code's row; `y0` from the bottom."""
    cax = fig.add_axes((x0 / W, y0 / H, w / W, ROW / H))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", drawedges=False)
    cb.set_ticks([-2, 0, 2])
    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.set_ticklabels(["¼×", "1×", "4×"])
    cb.ax.set_title("Frequency in the\nphase, relative to\nthe role's turns", fontsize=st.tsize("tick"),
                    pad=3, color=st.INK, linespacing=1.1)
    cb.ax.tick_params(labelsize=st.tsize("tick"), length=1.5)
    cb.outline.set_visible(False)


def marginals(fig, x: list, y0: float, p: pd.DataFrame) -> None:
    """(a) All the codes by each of b's first two columns, on the column's own x axis, stacked by role."""
    limits = X_MAX["student"]
    hists = {}
    for j, c in enumerate(("share_sessions", "turns_q50")):
        if (p[c] > limits[j]).any():
            raise SystemExit(f"a code's {c} falls beyond the column's x axis")
        edges = np.linspace(0, limits[j], A_BINS + 1)
        hists[j] = edges, {r: np.histogram(p.loc[p.role == r, c], bins=edges)[0] for r in ("tutor", "student")}
    for j, xi in enumerate(x):
        edges, h = hists[j]
        y_max = (h["tutor"] + h["student"]).max()        # each histogram on its own count scale
        ax = fig.add_axes((xi / W, y0 / H, P[j] / W, C_H / H))
        bottom = np.zeros(A_BINS)
        for role in ("tutor", "student"):
            fill, ink = inks(role)
            ax.bar(edges[:-1], h[role], width=np.diff(edges), bottom=bottom, align="edge", color=fill,
                   edgecolor=ink, lw=0.3, zorder=3)
            bottom = bottom + h[role]
        ax.set_xlim(0, limits[j])
        ax.set_xticks(X_TICKS["student"][j])
        ax.set_xticklabels([f"{v * 100:g}" for v in X_TICKS["student"][j]])
        ax.set_ylim(0, y_max)
        ax.set_yticks([0, y_max])
        ax.set_yticklabels(["", y_max])                  # the corner's zero is the x axis's
        ax.grid(axis="y", color=st.GRID, lw=st.STYLE["grid_lw"], zorder=0)
        ax.set_axisbelow(True)
        if j == 0:
            ax.set_ylabel("Codes", labelpad=2)


def example(ax, e: pd.DataFrame) -> None:
    """(d) One session: a mark at each turn whose code is in the row's group."""
    n = int(e.n_turns.iloc[0])
    for i, (role, group, _) in enumerate(D_ROWS):
        t = e[(e.role == role) & (e.group == group)].turn.to_numpy()
        ax.broken_barh([(x - 0.45, 0.9) for x in t], (i - 0.36, 0.72), facecolors=inks(role)[0], lw=0,
                       zorder=3)
    for f in (0.2, 0.8):                                 # the fifths of (b) and (c)
        ax.axvline(f * n - 0.5, color=st.INK_MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for x, name in ((0.1 * n, "start"), (0.5 * n, "middle"), (0.9 * n, "end")):
        ax.text(x - 0.5, -0.75, name, ha="center", va="bottom", fontsize=st.tsize("tick"), color=st.INK)
    ax.axhline(4.5, color=st.GRID, lw=0.6, zorder=1)    # tutor groups above, student groups below
    ax.set_ylim(len(D_ROWS) - 0.5, -0.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_yticks(range(len(D_ROWS)))
    ax.set_yticklabels([lab for _, _, lab in D_ROWS], fontsize=st.tsize("code"))
    for t, (role, _, _) in zip(ax.get_yticklabels(), D_ROWS):
        t.set_color(inks(role)[1])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel(f"Turn (one session, {n} turns)", labelpad=2)


H = TOP + 41 * ROW + BOTTOM


def main() -> None:
    st.apply_style(paper=True)                           # printed sizes, 1:1 at the text width
    pres = pd.read_csv(st.INPUT / "fig_presence.csv")
    tim = pd.read_csv(st.INPUT / "fig_timing.csv")
    if (pres.role == "tutor").sum() != 41:
        raise SystemExit("the figure height assumes 41 tutor moves")
    fig = plt.figure(figsize=(W * st.SCALE, H * st.SCALE))
    col_end = LAB["student"] + sum(P) + 2 * GAP          # the left column's right edge
    x_t = col_end + COL_GAP                              # the right column: the tutor moves
    # a: above the student signals, a histogram over each of b's first two columns and the key over
    # its third; the key's bar sits on the histograms' x axis, so the tick labels share a line
    a_plot_top = 0.12
    a_bot = H - a_plot_top - C_H
    b_top = a_plot_top + C_H + 0.13                      # its letter, under a's tick labels
    im, xb = block(fig, "student", 0.0, b_top, pres, tim, "b")
    marginals(fig, xb[:2], a_bot, pres)
    key(fig, im, xb[2], a_bot, P[2])
    _, xt = block(fig, "tutor", x_t, 0.0, pres, tim, "c")
    if abs(xt[2] + P[2] + END - W) > 1e-9:
        raise SystemExit(f"the columns end at {xt[2] + P[2]:.3f} in, not {W - END:.3f}")
    k = int((pres.role == "student").sum())
    s_end = b_top + TOP + k * ROW + BOTTOM               # under the student signals' tick labels
    # d: one session, across the left column, down to the tutor moves' axis
    d_top = s_end + 0.08                                 # its letter
    d_plot_top = d_top + 0.19                            # under the start / middle / end names
    d_bot = BOTTOM + 0.18                                # its axis label ends level with c's ticks
    ax_d = fig.add_axes((D_LAB / W, d_bot / H, (col_end - D_LAB) / W, (H - d_plot_top - d_bot) / H))
    example(ax_d, pd.read_csv(st.INPUT / "fig_example_session.csv"))
    heading(fig, 0.0, 0.0, "a")
    heading(fig, 0.0, d_top, "d")
    if H - d_plot_top - d_bot < 10 * 0.088:              # a row at least as tall as its 6 pt name
        raise SystemExit("no room for the example session's ten rows")
    st.save(fig, "fig_codes", width=W)


if __name__ == "__main__":
    main()
