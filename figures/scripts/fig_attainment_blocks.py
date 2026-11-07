#!/usr/bin/env python
"""SI figure: what each block adds once the model knows attainment (stage 57). Figure 2a's marks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import fig_ablation as abl
import fig_blocks as fb
import style as st

LEARNERS = ["logistic regression", "XGBoost", "TabPFN 3.5"]
MS = 0.72                               # marker size factor, so markers within a group never overlap
W, H = st.TEXT_W, 1.66                 # inches: Fig. 5's size (H5), so the two print at one height


def main() -> None:
    st.apply_style(paper=True, mark=0.82)  # the SI's sizes, printed 1:1
    df = pd.read_csv(st.INPUT / "fig_attainment_blocks.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W * st.SCALE, H * st.SCALE))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.82, bottom=0.11, wspace=0.22)
    for ax, metric, ylab, letter in zip(axes, ("ll", "auroc"), ("Log-loss reduction", "AUROC gain"), "ab"):
        abl.panel(ax, df, metric, LEARNERS, step=0.84, block_dx=0.17, ldx=0.10, ms_scale=MS)
        fb.tight_range(ax)
        ax.set_title("")
        ax.set_title("", loc="left")
        ax.set_ylabel(ylab, labelpad=3)
        ax.grid(False)                   # no grid in the SI figures, as in Fig. 5
    for ax, letter in zip(axes, "ab"):
        st.letter_top_left(fig, ax, letter)
    handles = [Line2D([], [], color=abl.BLOCK_COLOR[b], lw=st.mark(1.8), label=b) for b in abl.BLOCKS]
    handles += [Line2D([], [], ls="", **(abl.marker_kw(l, "#D4D4D0") | {"ms": st.mark(abl.SIZE.get(l, 5.0)) * MS}),
                       label=abl.SHORT.get(l, l)) for l in LEARNERS]
    fig.legend(handles=handles, loc="center", ncol=5, bbox_to_anchor=(0.54, 0.96), handlelength=1.1,
               frameon=False, handletextpad=0.35, columnspacing=1.2, fontsize=st.tsize("legend"),
               borderaxespad=0, borderpad=0)
    st.save(fig, "fig_attainment_blocks", width=W)


if __name__ == "__main__":
    main()
