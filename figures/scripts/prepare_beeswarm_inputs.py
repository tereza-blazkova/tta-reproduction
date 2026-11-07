#!/usr/bin/env python
"""Inputs for fig_beeswarm_si: per-response SHAP for all 57 code-rate columns.

Reads `results/shap-rows/shap_rows.npz` (stage 30: 8,000 training responses, the same draw as
stage 07). The beeswarm shows the first 2,000 of them -- enough to show each column's spread,
small enough to commit -- while the column order (mean |SHAP|) is taken over all 8,000, so it
matches the main attribution figure.

Writes two files to `figures/input/`:
* `fig_beeswarm_points.csv` -- one row per response: `s__<column>` the SHAP value (log-odds) and
  `q__<column>` the rate's rank within its column (0 = lowest, 1 = highest; empty if missing).
* `fig_beeswarm_columns.csv` -- one row per column, in plotting order: label, role, residual,
  mean |SHAP|.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "figures" / "input"
N_SHOWN = 2000


def main() -> None:
    z = np.load(ROOT / "results" / "shap-rows" / "shap_rows.npz", allow_pickle=False)
    S, R, cols = z["shap"], z["rates"], [str(c) for c in z["columns"]]
    meta = pd.read_csv(ROOT / "tables" / "table_columns.csv")
    meta["column"] = [("v3t_" if r == "tutor" else "v3s_") + c.lower() + "_rate"
                      for r, c in zip(meta.role, meta.code)]
    meta = meta.set_index("column").loc[cols]
    order = pd.DataFrame({"column": cols, "label": meta.label.to_numpy(),
                          "role": meta.role.to_numpy(), "residual": meta.residual.to_numpy(),
                          "mean_abs_shap": np.abs(S).mean(0)})
    order = order.sort_values("mean_abs_shap", ascending=False)
    order.to_csv(IN / "fig_beeswarm_columns.csv", index=False)

    pts = {}
    for j, c in enumerate(cols):
        q = pd.Series(R[:, j]).rank(pct=True)          # rank over all 8,000, NaN stays NaN
        pts[f"s__{c}"] = np.round(S[:N_SHOWN, j], 4)
        pts[f"q__{c}"] = np.round(q.to_numpy()[:N_SHOWN], 2)
    pd.DataFrame(pts).to_csv(IN / "fig_beeswarm_points.csv", index=False)
    print(f"wrote {IN / 'fig_beeswarm_points.csv'} ({N_SHOWN} rows) and fig_beeswarm_columns.csv")


if __name__ == "__main__":
    main()
