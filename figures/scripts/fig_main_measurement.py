#!/usr/bin/env python
"""Main figure — measurement quality and generalizability are the same axis.

Four panels, one argument:

  a  what each feature block ADDS, by provider, in LOG LOSS. Log loss and not AUROC because
     the Eedi contrast is decisive on log loss and ambiguous on AUROC — both blocks' AUROC
     intervals span zero there (n=868), so an AUROC panel would show a reversal the data does
     not establish and would invite the reader to see significance in the point estimates.
     The AUROC view is in the caption and in `results/ladder-v3/`.
  b  how reproducibly each code can be annotated at all: cross-annotator kappa over three LLM
     annotators coding the same turns with the identical prompt. The student scheme is
     markedly more reproducible than the tutor taxonomy.
  c  what a language-model-free instrument costs. Distilling the annotation into CPU
     classifiers is free in-domain and collapses off-provider — and log loss cannot see it.
  d  the two axes against each other, one point per code: a code that annotators cannot agree
     on is a code a cheap classifier cannot reproduce off-provider either.

Colour carries meaning: teal = tutor move, peach = student signal, and the providers keep
their own colours (`style.py`). Nothing is coloured decoratively.

Reads only `figures/input/`.

    python figures/scripts/fig_main_measurement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as st  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BLOCK_COLOR = {"student signals": st.STUDENT, "tutor moves": st.TUTOR}
BLOCK_INK = {"student signals": st.STUDENT_INK, "tutor moves": st.TUTOR_INK}
ROLE_COLOR = {"student": st.STUDENT, "tutor": st.TUTOR}
ROLE_INK = {"student": st.STUDENT_INK, "tutor": st.TUTOR_INK}
#: inches; fig_attainment_blocks uses the same, so the two print at one height
H5 = 2.15


def panel_a(ax, df: pd.DataFrame) -> None:
    """Grouped horizontal intervals: Δ log loss per block, one row group per view."""
    views = ["TSL (out-of-fold)", "TSL (test)", "Eedi (test)"]
    blocks = ["tutor moves", "student signals"]
    ytick, ylab = [], []
    for i, view in enumerate(views):
        base = len(views) - 1 - i
        for j, block in enumerate(blocks):
            r = df[(df.view == view) & (df.block == block)].iloc[0]
            y = base + (0.18 if j == 0 else -0.18)
            spans_zero = r.ll_lo < 0 < r.ll_hi
            ax.plot([r.ll_lo, r.ll_hi], [y, y], color=BLOCK_COLOR[block], lw=1.6,
                    solid_capstyle="round", zorder=3,
                    alpha=0.45 if spans_zero else 1.0)
            ax.plot([r.d_ll], [y], "o", ms=3.4, color=BLOCK_COLOR[block],
                    mec="white", mew=0.5, zorder=4,
                    alpha=0.45 if spans_zero else 1.0)
        ytick.append(base)
        ylab.append(view)
    ax.axvline(0, color=st.INK_MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.set_yticks(ytick)
    ax.set_yticklabels(ylab)
    ax.set_xlabel("Δ log loss when the block is added", labelpad=2)
    ax.set_ylim(-0.62, len(views) - 0.40)
    ax.grid(axis="x", color=st.GRID, lw=st.STYLE["grid_lw"])
    # Faded = the interval spans zero. Said once in words, so the encoding is not a puzzle.
    r = df[(df.view == "Eedi (test)") & (df.block == "tutor moves")].iloc[0]
    ax.annotate("faded: interval\nspans zero", xy=(r.d_ll, 0.18), xytext=(0.0085, 0.60),
                fontsize=st.tsize("note"), color=st.TUTOR_INK, va="center", ha="center",
                arrowprops=dict(arrowstyle="-", color=st.TUTOR_INK, lw=0.5,
                                shrinkA=1, shrinkB=3))
    # Helvetica has no U+2192/U+2190 glyph, so the direction is said in words. matplotlib
    # would otherwise draw a missing-glyph box and only warn.
    ax.text(0.015, -0.175, "lower = better", transform=ax.transAxes, ha="left",
            fontsize=st.tsize("note"), color=st.INK_MUTED)
    # a's colour key: the figure's semantic pair, stated in the panel a reader meets first
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLOCK_COLOR[b], lw=1.8, marker="o", ms=3.2,
                              mec="white", mew=0.4, label=b) for b in blocks],
              loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=1,
              handlelength=1.1, labelspacing=0.25)
    ax.set_title("What each block adds", pad=4, loc="left")


def panel_b(ax, df: pd.DataFrame) -> None:
    """Per-code kappa as two jittered strips; the vertical line is each strip's median.

    No text inside the panel: the caption says the line is the median.
    """
    rng = np.random.default_rng(3)
    for i, role in enumerate(("tutor", "student")):
        d = df[df.role == role]
        y = i + rng.uniform(-0.17, 0.17, len(d))
        ax.scatter(d.kappa_mean, y, s=11, color=ROLE_COLOR[role], alpha=0.85,
                   lw=0, zorder=3)
        med = d.kappa_mean.median()
        ax.plot([med, med], [i - 0.3, i + 0.3], color=ROLE_INK[role], lw=1.4, zorder=4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["tutor\nmoves", "student\nsignals"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("cross-annotator κ", labelpad=2)
    ax.set_box_aspect(1)


def panel_c(ax, df: pd.DataFrame) -> None:
    """Slope chart: AUROC per instrument within each provider; the legend names the provider."""
    views = ["TSL (test)", "Eedi (test)"]
    xs = {"122B labels": 0, "distilled children": 1}
    for view, col in zip(views, (st.TSL, st.EEDI)):
        d = df[df.view == view].set_index("instrument")
        ys = [d.loc[i, "auroc"] for i in xs]
        ax.plot(list(xs.values()), ys, "-o", color=col, lw=1.4, ms=3.6,
                mec="white", mew=0.5, zorder=3,
                label=f"{view.split(' ')[0]} (Δ{ys[1] - ys[0]:+.1f})".replace("-", "\u2212"))
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels(["122B\nlabels", "distilled\nchildren"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("AUROC", labelpad=2)
    ax.set_ylim(51.9, 63.4)
    ax.legend(loc="lower left", handlelength=1.4, labelspacing=0.3)
    ax.set_box_aspect(1)


def panel_d(ax, df: pd.DataFrame) -> None:
    """κ against off-provider feature fidelity, one point per code."""
    for role in ("tutor", "student"):
        d = df[df.role == role]
        ax.scatter(d.kappa_mean, d.fidelity_r_eedi, s=13, color=ROLE_COLOR[role],
                   alpha=0.85, lw=0, zorder=3,
                   label=role)
    x, y = df.kappa_mean.to_numpy(), df.fidelity_r_eedi.to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    b1, b0 = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, b0 + b1 * xx, color=st.INK_MUTED, lw=0.8, ls=(0, (4, 2)), zorder=2)
    ax.text(0.97, 0.05, f"r = {r:+.2f}", transform=ax.transAxes, ha="right",
            fontsize=st.tsize("note"), color=st.INK_2)
    ax.set_xlabel("cross-annotator κ", labelpad=2)
    ax.set_ylabel("distilled-rate fidelity on Eedi (r)", labelpad=2)
    ax.legend(loc="upper left", markerscale=1.1, handletextpad=0.2)
    ax.set_box_aspect(1)


def main() -> None:
    """Two figures, not one.

    Panel a answers what each feature block adds and belongs in the results; panels b-d are
    about how well a code can be measured at all, which is supporting material. They were one
    four-panel figure while the paper was organised around research questions, and the main
    text then had to cite half of it.
    """
    st.apply_style(paper=True)          # the SI's sizes: drawn at the text width, printed 1:1
    b = pd.read_csv(st.INPUT / "fig_main_b_kappa.csv")
    c = pd.read_csv(st.INPUT / "fig_main_c_instrument.csv")
    d = pd.read_csv(st.INPUT / "fig_main_d_convergence.csv")

    # the main-text ablation is drawn by fig_ablation.py (both metrics, every learner);
    # this script now owns only the supplementary measurement figure

    # ── supplementary: the measurement panels ──
    fig, axes = plt.subplots(1, 3, figsize=(st.TEXT_W * st.SCALE, H5 * st.SCALE))
    fig.subplots_adjust(wspace=0.55, left=0.085, right=0.995, top=0.90, bottom=0.17)
    panel_b(axes[0], b)
    panel_c(axes[1], c)
    panel_d(axes[2], d)
    for ax, letter in zip(axes, "abc"):
        st.letter_top_left(fig, ax, letter)
    st.save(fig, "fig_measurement_si", width=st.TEXT_W)


if __name__ == "__main__":
    main()
