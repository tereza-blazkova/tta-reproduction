#!/usr/bin/env python
"""Inputs for fig_explain: what survives with everything held fixed, and explaining vs prompting.

Reads three runs and writes three CSVs to `figures/input/`:
* `fig_explain_a_codes.csv` -- the twelve codes of fig_rq1 panel b (largest pooled
  associations), each estimated three ways: pooled and within student one code at a time
  (`results/rq1-within-session/`, stage 15), and within student with all 55 codes in one model
  (`results/joint-within/`, stage 32). pp per SD, 95% CI, BH flag within its own design.
* `fig_explain_b_terms.csv` -- within-student estimates from `results/explain-vs-prompt/`
  (stage 31): explaining and prompting alone and together, and the three turn kinds.
* `fig_explain_c_heldout.csv` -- held-out AUROC of one standardised rate, TSL and Eedi, with the
  paired explaining-minus-prompting difference and its session-bootstrap CI (stage 31).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "figures" / "input"
RES = ROOT / "results"


def code(c: pd.Series) -> pd.Series:
    return c.str.replace(r"^v3[ts]_", "", regex=True).str.replace(r"_rate$", "", regex=True)


def main() -> None:
    named = pd.read_csv(IN / "fig_rq1_b_named.csv")[["code", "role"]]
    r15 = pd.read_csv(RES / "rq1-within-session" / "rq1_designs.csv")
    r15["code"] = code(r15.column)
    r32 = pd.read_csv(RES / "joint-within" / "joint_within.csv")
    r32 = r32[r32.design == "within student"].assign(code=lambda d: code(d.column))
    parts = []
    for design, src, est, lo, hi in (
            ("pooled", r15[r15.design == "1 pooled"], "pp_per_sd", "lo", "hi"),
            ("within student, one code", r15[r15.design == "2 within student"],
             "pp_per_sd", "lo", "hi"),
            ("within student, all 55 codes", r32, "pp_per_sd", "lo_sd", "hi_sd")):
        s = src.set_index("code").loc[named.code]
        parts.append(pd.DataFrame({"code": named.code, "role": named.role, "design": design,
                                   "estimate": s[est].to_numpy(), "lo": s[lo].to_numpy(),
                                   "hi": s[hi].to_numpy(), "bh_sig": s.bh_sig.to_numpy()}))
    pd.concat(parts).to_csv(IN / "fig_explain_a_codes.csv", index=False)

    a = pd.read_csv(RES / "explain-vs-prompt" / "associations.csv")
    a = a[a.design == "within student"]
    pick = [("1a explaining alone", "v3s_explains_why_rate", "student explaining", "alone", "student"),
            ("1b prompting (5 codes) alone", "prompting_all", "tutor prompting", "alone", "tutor"),
            ("1d explaining + prompting (5 codes)", "v3s_explains_why_rate", "student explaining",
             "together", "student"),
            ("1d explaining + prompting (5 codes)", "prompting_all", "tutor prompting", "together",
             "tutor"),
            ("2a turn kinds, any prompt", "all_landed", "prompt answered by an explanation",
             "turn kinds", "both"),
            ("2a turn kinds, any prompt", "all_not_landed", "prompt not answered by one",
             "turn kinds", "tutor"),
            ("2a turn kinds, any prompt", "all_unprompted", "explanation without a prompt",
             "turn kinds", "student")]
    rows = []
    for model, term, label, group, role in pick:
        r = a[(a.model == model) & (a.term == term)].iloc[0]
        rows.append({"label": label, "group": group, "role": role, "estimate": r.estimate,
                     "lo": r.lo, "hi": r.hi, "p": r.p})
    pd.DataFrame(rows).to_csv(IN / "fig_explain_b_terms.csv", index=False)

    h = pd.read_csv(RES / "explain-vs-prompt" / "held_out.csv")
    keep = h[h.model.isin(["explaining", "prompting (5 codes)", "prompting self-explanation"])]
    keep = keep[["view", "model", "AUROC"]].copy()
    diff = h[h.model == "explaining vs prompting (5 codes)"].set_index("view")
    keep["diff"] = keep.view.map(diff.dAUROC)
    keep["diff_lo"] = keep.view.map(diff.dAUROC_lo)
    keep["diff_hi"] = keep.view.map(diff.dAUROC_hi)
    keep.to_csv(IN / "fig_explain_c_heldout.csv", index=False)
    print("wrote fig_explain_{a_codes,b_terms,c_heldout}.csv")


if __name__ == "__main__":
    main()
