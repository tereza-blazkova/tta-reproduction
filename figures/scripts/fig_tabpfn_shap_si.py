#!/usr/bin/env python
"""fig_tabpfn_shap_si — TabPFN's SHAP values beside XGBoost's, on the same test responses.

Fig. 2b's fourteen columns, one row each, as Fig. 2b draws them: XGBoost's exact TreeSHAP values
(left) and TabPFN 3.5's (right), both on the same 2,000 TSL test responses (stage 49). TabPFN
learns in context, so it cannot be explained on its own training rows the way Fig. 2b explains
XGBoost; both are shown on test rows here so the two are compared on the same responses.
TabPFN's values are estimated by permutation sampling (16 orderings per response, each with its
own background response), so they carry some noise that XGBoost's exact values do not.

Reads only `figures/input/fig_learners_{points,columns}.csv` (prepare_learner_attribution_inputs.py)
and `fig_beeswarm_columns.csv` (for Fig. 2b's rows). Drawn at the text width and printed 1:1.

    python figures/scripts/fig_tabpfn_shap_si.py
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fig_blocks as fb
import style as st

W = st.TEXT_W
ROW, XLAB, TOP = 0.103, 0.35, 0.30      # inches: per column, x axes below, titles and key above
LAB = 1.25                              # the column names
GAP, RIGHT = 0.25, 0.05


def panel(ax, b: pd.DataFrame, pts: pd.DataFrame, learner: str, title: str, names: bool) -> None:
    p = pd.DataFrame({k: pts[f"s__{learner}__{c}" if k.startswith("s__") else f"q__{c}"]
                      for c in b.column for k in (f"s__{c}", f"q__{c}")})
    fb.swarm(ax, b, p)
    st.zebra(ax, range(len(b)))
    ax.grid(False)                                    # no grid in the SI figures (Fig. 5)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(st.tsize("code"))
    if not names:
        ax.set_yticklabels([])
    ax.set_ylim(len(b) - 0.5, -0.5)                  # no title row: the caption names the panels
    ax.set_xlabel("SHAP value (log-odds)", labelpad=2)


def main(other: str = "tabpfn35") -> None:
    st.apply_style(paper=True, mark=0.82)
    pts = pd.read_csv(st.INPUT / "fig_learners_points.csv")
    if f"s__{other}__v3s_explains_why_rate" not in pts:
        raise SystemExit(f"no {other} attributions in fig_learners_points.csv yet")
    b = pd.read_csv(st.INPUT / "fig_beeswarm_columns.csv").head(fb.N_TOP)
    h = len(b) * ROW
    H = XLAB + h + TOP
    fig = plt.figure(figsize=(W * st.SCALE, H * st.SCALE))
    pw = (W - LAB - GAP - RIGHT) / 2
    axs = []
    for k, (learner, title) in enumerate((("xgb", "XGBoost SHAP values"),
                                          (other, {"tabpfn35": "TabPFN SHAP values"}.get(other, other)))):
        x = LAB + k * (pw + GAP)
        ax = fig.add_axes((x / W, XLAB / H, pw / W, h / H))
        panel(ax, b, pts, learner, title, names=(k == 0))
        axs.append(ax)
    lo = min(a.get_xlim()[0] for a in axs)
    hi = max(a.get_xlim()[1] for a in axs)
    for a in axs:                                    # one scale, so the spreads compare
        a.set_xlim(lo, hi)
    # the shading key above the left panel, its "Low rate" starting at the panel's y axis
    key_mid, kw, kh, kgap, pad = XLAB + h + 0.13, 0.18, 0.05, 0.033, 0.033
    low = fig.text(LAB / W, key_mid / H, "Low rate", ha="left", va="center",
                   fontsize=st.tsize("tick"))
    bb = low.get_window_extent(fig.canvas.get_renderer())
    kx = LAB + bb.width / fig.dpi + pad
    for j, role in enumerate(("tutor", "student")):
        k = fig.add_axes(((kx + j * (kw + kgap)) / W, (key_mid - kh / 2) / H, kw / W, kh / H))
        k.imshow(np.linspace(0, 1, 64)[None, :], aspect="auto", cmap=fb.SHADE[role])
        k.set_axis_off()
    fig.text((kx + 2 * kw + kgap + pad) / W, key_mid / H, "High rate", ha="left", va="center",
             fontsize=st.tsize("tick"))
    st.save(fig, "fig_tabpfn_shap_si" if other == "tabpfn35" else f"fig_{other}_shap_check", width=W)


if __name__ == "__main__":
    main(*sys.argv[1:])
