#!/usr/bin/env python
"""Distil the RQ1 design comparison into the CSVs the main RQ1 figure reads."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
IN = ROOT / "figures" / "input"
RES = ROOT / "results"

DESIGNS = {"1 pooled": "pooled", "2 within student": "within_student",
           "3 within session (matched)": "within_session", "4 within session (PLACEBO)": "placebo"}


def main() -> None:
    IN.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(RES / "rq1-within-session" / "rq1_designs.csv")
    d["short"] = d.design.map(DESIGNS)
    d["role"] = np.where(d.column.str.startswith("v3t_"), "tutor", "student")
    d["code"] = (d.column.str.replace(r"^v3[ts]_", "", regex=True)
                   .str.replace(r"_rate$", "", regex=True))

    wide = d.pivot_table(index=["code", "role"],
                         columns="short", values=["pp_per_sd", "lo", "hi", "bh_sig"],
                         aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(IN / "fig_rq1_a_scatter.csv", index=False)

    # the named codes: largest pooled associations, which is where the collapse is visible
    named = wide.reindex(wide.pp_per_sd_pooled.abs().sort_values(ascending=False).index).head(12)
    named.to_csv(IN / "fig_rq1_b_named.csv", index=False)

    # the fixed-effect ladder: question / +student / +tutor / +both. This replaces the
    # within-session ladder, which the narrowed scope drops: it was underpowered, and the
    # question that matters for a tutoring claim is whether an association survives absorbing
    # the TUTOR as well as the student.
    fe = pd.read_csv(RES / "tutor-and-composition" / "fe_ladder.csv")
    fe_lad = (fe.groupby("design")
                .apply(lambda g: pd.Series({
                    "n_sig": int(g.bh.sum()), "n": len(g),
                    "median_abs_b": g.pp_per_sd.abs().median(),
                    "median_ci_half": ((g.hi - g.lo) / 2).median()}), include_groups=False)
                .reset_index())
    fe_lad.to_csv(IN / "fig_rq1_c_ladder_fe.csv", index=False)

    comp = pd.read_csv(RES / "tutor-and-composition" / "compositional.csv", index_col=0)
    comp.index.name = "code"
    comp.reset_index().to_csv(IN / "fig_rq1_d_compositional.csv", index=False)

    # The same reallocation model on the 41 tutor shares, as one summary row. Panel d shows the
    # student reallocations and annotates this beside them: without it a reader cannot tell
    # whether the tutor scheme yields nothing because moves do not matter or because 41 shares
    # of the same turn budget are each estimated too imprecisely to say. It is the latter, and
    # the figure has to admit it.
    ct = pd.read_csv(RES / "tutor-and-composition" / "compositional_tutor.csv", index_col=0)
    hw_t, hw_s = (ct.hi - ct.lo) / 2, (comp.hi - comp.lo) / 2
    pd.DataFrame([{"n_tutor": len(ct), "n_bh_tutor": int(ct.bh.sum()),
                   "median_hw_tutor": round(float(hw_t.median()), 2),
                   "n_bound3_tutor": int((hw_t < 3).sum()),
                   "n_student": len(comp), "n_bh_student": int(comp.bh.sum()),
                   "median_hw_student": round(float(hw_s.median()), 2),
                   "n_bound3_student": int((hw_s < 3).sum())}]
                 ).to_csv(IN / "fig_rq1_d_tutor_summary.csv", index=False)

    # the design ladder: what survives, and how wide the intervals get
    lad = (d.groupby("short")
             .apply(lambda g: pd.Series({
                 "n_sig": int(g.bh_sig.sum()), "n": len(g),
                 "median_abs_b": g.pp_per_sd.abs().median(),
                 "median_ci_half": ((g.hi - g.lo) / 2).median()}), include_groups=False)
             .reindex(["pooled", "within_student", "within_session", "placebo"]).reset_index())
    lad.to_csv(IN / "fig_rq1_c_ladder.csv", index=False)
    print(f"wrote 3 CSVs | survivors: " +
          ", ".join(f"{r.short} {r.n_sig}/{r.n}" for _, r in lad.iterrows()))


if __name__ == "__main__":
    main()
