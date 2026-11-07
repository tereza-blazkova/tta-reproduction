#!/usr/bin/env python
"""Inputs for the Fig. 2c candidates: each learner's attribution on the same test responses.

Reads stage 49's per-learner files, `results/attribution-learners/<learner>[.<i>of<n>]/*.npz`
(2,000 TSL test responses, the same for XGBoost, logistic regression and TabPFN 3.5). A learner
whose shards are not all there is skipped, so the candidates that do not need it can be drawn.

Attribution per learner: XGBoost's TreeSHAP and logistic regression's closed form (both exact),
and TabPFN's permutation-sampled values. For the column summaries the sampled values' noise is
taken out: a column's mean |attribution| is reported beside the root mean square with the
sampling variance subtracted, which is unbiased where the plain mean |.| of a noisy value is not.

Writes to `figures/input/`:
* `fig_learners_points.csv` -- one row per response: `s__<learner>__<column>` and `q__<column>`
  (the rate's rank within its column), for the 57 code columns.
* `fig_learners_columns.csv` -- one row per column (57 codes and the objective): label, role,
  residual, logistic regression's weight (log-odds per SD of the rate), and per learner the mean
  |attribution|, the noise-free root mean square, its share of the learner's total, and the
  Spearman correlation between the rate and its attribution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "figures" / "input"
RES = ROOT / "results" / "attribution-learners"
LEARNERS = ("xgb", "logreg", "tabpfn35")
ROWS = 2000


def load(learner: str) -> dict | None:
    parts = [p for p in sorted(RES.glob(f"{learner}*/{learner}*.npz")) if "limit" not in p.name]
    if not parts:
        return None
    z = [dict(np.load(p, allow_pickle=True)) for p in parts]          # our own files (string ids)
    pos = np.concatenate([q["pos"] for q in z])
    if sorted(pos) != list(range(ROWS)):
        print(f"{learner}: {len(pos)} of {ROWS} rows so far, skipped")
        return None
    o = np.argsort(pos)
    out = {k: np.concatenate([q[k] for q in z])[o] for k in ("pos", "response_id", "phi", "se", "rates")}
    out["players"] = [str(p) for p in z[0]["players"]]
    if "exact" in z[0]:                                  # TreeSHAP / closed form: no sampling noise
        out["value"] = np.concatenate([q["exact"] for q in z])[o]
        out["noise"] = np.zeros_like(out["value"])
    else:
        out["value"], out["noise"] = out["phi"], out["se"]
    if "coef" in z[0]:
        out["coef"] = z[0]["coef"]
    return out


def main() -> None:
    got = {l: d for l in LEARNERS if (d := load(l)) is not None}
    if not got:
        raise SystemExit("no learner has all its rows")
    ref = next(iter(got.values()))
    for l, d in got.items():
        if not (d["response_id"] == ref["response_id"]).all():
            raise SystemExit(f"{l} explains different rows")
    players, R = ref["players"], ref["rates"]
    codes = players[:-1]
    meta = pd.read_csv(ROOT / "tables" / "table_columns.csv")
    meta["column"] = [("v3t_" if r == "tutor" else "v3s_") + c.lower() + "_rate" for r, c in zip(meta.role, meta.code)]
    meta = meta.set_index("column")
    pts = {}
    for j, c in enumerate(codes):
        q = pd.Series(R[:, j]).rank(pct=True)
        pts[f"q__{c}"] = ((q - q.min()) / (q.max() - q.min())).round(4)
        for l, d in got.items():
            pts[f"s__{l}__{c}"] = np.round(d["value"][:, j], 5)
    pd.DataFrame(pts).to_csv(IN / "fig_learners_points.csv", index=False)
    rows = []
    for j, c in enumerate(players):
        r = {"column": c, "label": meta.label.get(c, "objective"), "role": meta.role.get(c, "objective"),
             "residual": bool(meta.residual.get(c, False))}
        if "logreg" in got and j < len(codes):
            r["logreg_coef"] = got["logreg"]["coef"][j]
        for l, d in got.items():
            v, e = d["value"][:, j], d["noise"][:, j]
            r[f"{l}_mean_abs"] = np.abs(v).mean()
            r[f"{l}_rms"] = np.sqrt(max((v ** 2).mean() - np.nanmean(e ** 2), 0.0))
            if j < len(codes):
                ok = ~np.isnan(R[:, j])
                r[f"{l}_spearman"] = pd.Series(R[ok, j]).corr(pd.Series(v[ok]), method="spearman")
        rows.append(r)
    out = pd.DataFrame(rows)
    for l in got:
        out[f"{l}_share"] = out[f"{l}_mean_abs"] / out[f"{l}_mean_abs"].sum()
    out.to_csv(IN / "fig_learners_columns.csv", index=False)
    print(f"wrote fig_learners_points.csv and fig_learners_columns.csv for {', '.join(got)}")


if __name__ == "__main__":
    main()
