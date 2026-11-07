#!/usr/bin/env python
"""fig_beeswarm_si — every code-rate column's SHAP values, one row per column.

Reads only `figures/input/fig_beeswarm_{points,columns}.csv` (prepare_beeswarm_inputs.py). Rows
are ordered by mean |SHAP| over 8,000 training responses; each point is one response, placed at
its SHAP value (log-odds of a correct answer) and coloured by how high the session's rate of that
code is relative to other sessions. Labels are coloured by role, as everywhere in the paper.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import style as st

LOW, HIGH = st.DIVERGING[0], st.DIVERGING[2]      # rate rank: low to high, the paper's diverging pair
CMAP = LinearSegmentedColormap.from_list("rate", [LOW, "#D9D9D5", HIGH])


def swarm_offsets(x: np.ndarray, width: float, bins: int = 120, seed: int = 0) -> np.ndarray:
    """Vertical jitter scaled by local density, so a row's thickness shows where values pile up."""
    h, edges = np.histogram(x, bins=bins)
    dens = h[np.clip(np.digitize(x, edges) - 1, 0, bins - 1)] / max(h.max(), 1)
    return np.random.default_rng(seed).uniform(-1, 1, len(x)) * width * np.sqrt(dens)


def main() -> None:
    st.apply_style()
    cols = pd.read_csv(st.INPUT / "fig_beeswarm_columns.csv")
    pts = pd.read_csv(st.INPUT / "fig_beeswarm_points.csv")
    n = len(cols)
    fig, ax = plt.subplots(figsize=(st.STYLE["fig_w"] * st.SCALE, 0.135 * n * st.SCALE + 0.9))
    for i, r in enumerate(cols.itertuples()):
        s = pts[f"s__{r.column}"].to_numpy(float)
        q = pts[f"q__{r.column}"].to_numpy(float)
        ok = ~np.isnan(q)
        y = i + swarm_offsets(s, 0.38, seed=i)
        ax.scatter(s[~ok], y[~ok], s=1.0, color=st.INK_MUTED, lw=0, alpha=0.5, rasterized=True)
        ax.scatter(s[ok], y[ok], s=1.0, c=q[ok], cmap=CMAP, vmin=0, vmax=1, lw=0, alpha=0.8,
                   rasterized=True)
    ax.axvline(0, color=st.INK_MUTED, lw=0.6, zorder=0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab + ("$^\\dagger$" if res else "")
                        for lab, res in zip(cols.label, cols.residual)])
    for t, role in zip(ax.get_yticklabels(), cols.role):
        t.set_color(st.TUTOR_INK if role == "tutor" else st.STUDENT_INK)
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=st.GRID, lw=st.STYLE["grid_lw"])
    ax.set_xlabel("SHAP value: contribution to the log-odds of a correct answer", labelpad=3)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, aspect=40)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("session's rate of the code (rank)", labelpad=2)
    cb.outline.set_linewidth(0.4)
    fig.subplots_adjust(left=0.27, right=0.92, top=0.995, bottom=0.9 / (0.135 * n + 0.9) * 0.75)
    st.zebra(ax, range(n))                                      # inside the plot only
    st.save(fig, "fig_beeswarm_si")


if __name__ == "__main__":
    main()
