#!/usr/bin/env python
"""fig_ablation — what each feature block adds, for each learner, on the two test splits.

Reads only `figures/input/fig_main_a_transfer.csv` (written by `prepare_main_figure_inputs.py`
from `results/ladder-v3/`).

Each point is one block added over the objective and the other block, so it is the block's own
contribution with everything else held fixed. Both panels are drawn so that up means the block
helps (said in the caption): log loss as a reduction (−Δ), AUROC as a gain. Out-of-fold TSL is not shown — the
comparison here is held-out performance, on the provider the model was trained on and on one it
never saw.

Learners: XGBoost and logistic regression from their ladders, the foundation models from
dropping each block from their full model (the same contrast). A missing slot is a hatched
"not yet run" band; a learner appears automatically once the input
carries rows with its name. A slot is drawn as a band, not as a marker at zero, because a marker
at zero would read as a measured null.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style as st

VIEWS = ["TSL (test)", "Eedi (test)"]
#: the axis says only the provider; that both are test splits is in the caption
VIEW_LABEL = {"TSL (test)": "TSL", "Eedi (test)": "Eedi"}
H = 0.51                                # figure height, as a share of the style's
AX_TOP, AX_BOTTOM = 0.75, 0.12          # the plots; the band above them holds the legend
TITLE_UP = 0.42                         # inches from the top of a plot to its title
BLOCKS = ["student signals", "tutor moves"]
BLOCK_COLOR = {"tutor moves": st.TUTOR, "student signals": st.STUDENT}
#: marker edges in the darker shade of the block's colour (the colour of its code names)
BLOCK_INK = {"tutor moves": st.TUTOR_INK, "student signals": st.STUDENT_INK}
#: share of white in the markers' fill: light enough to read as a light shape on its interval,
#: not so light that it reads as hollow (hollow: the 95% interval spans zero)
MARKER_TINT = 0.5
#: a light band behind the off-provider split, so TSL and Eedi read as two groups
EEDI_BG = "#F1F1EE"
#: simplest to most complex: trained on our data (linear, then boosted), then pretrained in-context
LEARNERS = ["logistic regression", "XGBoost", "TabPFN 3.5"]
LEARNERS_SI = LEARNERS
MARKER = {"logistic regression": "X", "XGBoost": "o", "TabPFN 3.5": "^"}
#: every learner filled: hollow means the interval spans zero (as "not significant" in Figure 3),
#: so a hollow learner marker would read as a null; shape and point order tell the learners apart
HOLLOW: set[str] = set()
SIZE = {"XGBoost": 5.8, "logistic regression": 5.4, "TabPFN 3.5": 6.8}
SHORT = {"logistic regression": "Logistic", "TabPFN 3.5": "TabPFN"}


def marker_kw(learner: str, color: str, edge: str = st.INK) -> dict:
    """How a learner's point is drawn, in a block's colour with an edge in its darker shade (or
    neutral, for the legend)."""
    kw = dict(marker=MARKER[learner], ms=st.mark(SIZE.get(learner, 5.0)), color=color)
    if learner in HOLLOW:
        return kw | dict(mfc="white", mec=color, mew=st.mark(1.0))
    return kw | dict(mec=edge, mew=st.mark(0.6))

BLOCK_DX = {"student signals": -0.19, "tutor moves": 0.19}


def _dx(learners: list[str], ldx: float = 0.07) -> dict[str, float]:
    """Offsets within a block's group, `ldx` apart and centred."""
    k = len(learners)
    return {l: ldx * (i - (k - 1) / 2) for i, l in enumerate(learners)}


def panel(ax, df: pd.DataFrame, metric: str, learners: list[str] = LEARNERS, step: float = 1.0,
          block_dx: float | None = None, ldx: float = 0.07, ms_scale: float = 1.0) -> None:
    """metric: 'll' (drawn as a reduction) or 'auroc' (drawn as a gain). `step`: distance between
    the two test splits' groups on the x axis (1.0 leaves wide gaps; smaller packs them).
    `block_dx`: offset of the tutor and student groups from their split's centre (default
    BLOCK_DX); keep the gap between the blocks smaller than the gap between the splits. `ldx`:
    spacing of the learners within a group; `ms_scale`: marker size factor (so they do not overlap)."""
    bdx = BLOCK_DX if block_dx is None else {"student signals": -block_dx, "tutor moves": block_dx}
    for view in VIEWS:
        for block in BLOCKS:
            for learner in learners:
                x = step * VIEWS.index(view) + bdx[block] + _dx(learners, ldx)[learner]
                r = df[(df.view == view) & (df.block == block) & (df.learner == learner)]
                if r.empty:
                    ax.axvspan(x - 0.026, x + 0.026, color="#D9D9D5", alpha=0.55, lw=0,
                               hatch="////", zorder=1)
                    continue
                r = r.iloc[0]
                if metric == "ll":     # a reduction: flip sign and the interval with it
                    est, lo, hi = -r.d_ll, -r.ll_hi, -r.ll_lo
                else:
                    # on AUROC's own 0-1 scale, like log loss, not in points
                    est, lo, hi = r.d_auroc / 100, r.lo / 100, r.hi / 100
                ax.plot([x, x], [lo, hi], color=BLOCK_COLOR[block], lw=st.mark(1.7),
                        solid_capstyle="round", zorder=3)
                # a light tint of the block's colour inside, its dark shade as the edge; hollow
                # (white inside) where the 95% interval spans zero
                mk = marker_kw(learner, st.tint(BLOCK_COLOR[block], MARKER_TINT), BLOCK_INK[block])
                if lo < 0 < hi:
                    mk["mfc"] = "white"
                mk["ms"] *= ms_scale
                ax.plot([x], [est], ls="", **mk, zorder=4)
    ax.axhline(0, color=st.INK_MUTED, lw=st.mark(0.8), ls=(0, (3, 2)), zorder=2)
    ax.set_xticks([step * i for i in range(len(VIEWS))])
    ax.set_xticklabels([VIEW_LABEL[v] for v in VIEWS])
    half = 0.5 if step == 1.0 else bdx["tutor moves"] + max(_dx(learners, ldx).values()) + 0.12
    ax.set_xlim(-half, step * (len(VIEWS) - 1) + half)
    if "Eedi (test)" in VIEWS:
        i = VIEWS.index("Eedi (test)")
        ax.axvspan(step * i - step / 2, step * i + half, color=EEDI_BG, lw=0, zorder=0)
    ax.grid(axis="y", color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.tick_params(axis="x", length=plt.rcParams["xtick.major.size"] * 1.4)   # a tick at each split's centre
    if metric == "ll":
        ax.set_ylabel("log-loss reduction when the block is added", labelpad=3)
        ax.set_title("Log loss", pad=TITLE_UP * 72, loc="left")
    else:
        ax.set_ylabel("AUROC gain when the block is added", labelpad=3)
        ax.set_title("AUROC", pad=TITLE_UP * 72, loc="left")


def share_range(axes) -> None:
    """One y-range and one set of ticks for every panel, so the panels can be read side by side.

    Symmetric about zero and coarse (about six intervals), so the axis carries few numbers."""
    from matplotlib.ticker import MaxNLocator
    m = max(max(abs(v) for v in ax.get_ylim()) for ax in axes)
    t = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10], symmetric=True).tick_values(-m, m)
    for ax in axes:
        ax.set_ylim(t[0], t[-1])
        ax.set_yticks(t)


def main(learners: list[str] = LEARNERS, name: str = "fig_ablation") -> None:
    st.apply_style()
    df = pd.read_csv(st.INPUT / "fig_main_a_transfer.csv")
    if "learner" not in df.columns:      # the ladder written before learners were compared
        df["learner"] = "XGBoost"
    df = df[df.view.isin(VIEWS)]

    fig, axes = plt.subplots(1, 2, figsize=(st.STYLE["fig_w"] * st.SCALE,
                                            st.STYLE["fig_h"] * H * st.SCALE))
    panel(axes[0], df, "ll", learners)
    panel(axes[1], df, "auroc", learners)
    share_range(axes)
    fig.subplots_adjust(wspace=0.38, left=0.09, right=0.985, top=AX_TOP, bottom=AX_BOTTOM)
    ax_h = (AX_TOP - AX_BOTTOM) * st.STYLE["fig_h"] * H * st.SCALE      # inches
    for ax, letter in zip(axes, "ab"):
        st.panel_letter(ax, letter, dy=1 + (TITLE_UP - 0.04) / ax_h)

    handles = ([Line2D([], [], color=BLOCK_COLOR[b], lw=1.8, label=f"adding {b}")
                for b in BLOCKS]
               # learners in their fixed order, each as a marker if run and a slot if not
               + [Line2D([], [], ls="", **marker_kw(l, st.NEUTRAL), label=l)
                  if (df.learner == l).any()
                  else Patch(facecolor="#D9D9D5", alpha=0.55, hatch="////", lw=0,
                             label=f"{l} (not yet run)")
                  for l in learners])
    # between the panel titles and the plots
    fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, (AX_TOP + 1) / 2 - 0.035), handlelength=1.3,
               columnspacing=1.2)
    st.save(fig, name)


if __name__ == "__main__":
    main()
    main(LEARNERS_SI, "fig_ablation_si")
