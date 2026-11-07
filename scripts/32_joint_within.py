#!/usr/bin/env python
"""Stage 32 — every tutor move and student signal in ONE within-student model.

Stage 15 tests each code on its own; stage 18 tests each role's shares together. This asks the
strictest version of "what survives": each code's association with the outcome holding every
other tutor move and student signal fixed, with student and question fixed effects absorbed
and SEs clustered on the student (`ttapub.within`).

Each role's shares sum to one, so the most common code of each role is left out as the
reference (TUTOR_NO_MOVE, STUDENT_NO_MOVE, as stage 18): a coefficient is the association of
more of this code *instead of* a turn with no visible move by that role. Reported per 10
percentage points of the role's turns (the practitioner unit of stage 18) and per SD (the unit
of stage 15), with Benjamini-Hochberg across the 55 codes; the pooled model (question fixed
effects only) beside it.

    python scripts/32_joint_within.py --run joint-within
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, data, runlog, within

REFERENCES = [agg.column("TUTOR_NO_MOVE", "tutor"), agg.column("STUDENT_NO_MOVE", "student")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="joint-within")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.FEATURES_TRAIN])

    meta = data.load_metadata()
    tr = (data.load_train().merge(meta[["response_id", "student_id", "question_id"]],
                                  on="response_id").reset_index(drop=True))
    feats = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    cols = [c for c in agg.code_columns(feats.columns) if c not in REFERENCES]
    X = tr[["session_id"]].merge(feats, on="session_id", how="left")[cols].astype(float)
    X = X.fillna(X.mean())
    y = tr.is_correct.to_numpy(float)
    cluster = tr.student_id.to_numpy()
    sd = X.std(ddof=0)
    print(f"rows {len(tr):,} | students {tr.student_id.nunique():,} | codes {len(cols)}")

    rows = []
    for design, absorb in (("within student", {"student": tr.student_id,
                                                "question": tr.question_id}),
                           ("pooled", {"question": tr.question_id})):
        M, n_abs = within.absorb_matrix(pd.concat([pd.Series(y, name="_y"), X * 10], axis=1),
                                        absorb)      # x10: a coefficient per 10 pp of turns
        t = within.ols_demeaned(M["_y"].to_numpy(), M[cols], cluster, n_abs).table()
        for c in cols:
            r = t.loc[c]
            per_sd = sd[c] * 10                       # 1 SD of the share, in units of 10 pp
            rows.append({"design": design, "column": c,
                         "role": "tutor" if c.startswith("v3t_") else "student",
                         "pp_per_10pp": 100 * r.estimate, "lo_10pp": 100 * r.lo,
                         "hi_10pp": 100 * r.hi, "pp_per_sd": 100 * r.estimate * per_sd,
                         "lo_sd": 100 * r.lo * per_sd, "hi_sd": 100 * r.hi * per_sd, "p": r.p})
    out = pd.DataFrame(rows)
    out["bh_sig"] = False
    for design in out.design.unique():
        m = out.design == design
        out.loc[m, "bh_sig"] = within.benjamini_hochberg(out.loc[m, "p"].to_numpy(), a.alpha)
    out.to_csv(d / "joint_within.csv", index=False)

    res = {"run": a.run, "n_codes": len(cols), "references": REFERENCES, "alpha": a.alpha}
    for design in ("pooled", "within student"):
        s = out[out.design == design]
        res[design] = {"n_bh_sig": int(s.bh_sig.sum()), "n_p05": int((s.p < 0.05).sum()),
                       "survivors": s[s.bh_sig].set_index("column")[
                           ["pp_per_sd", "lo_sd", "hi_sd", "pp_per_10pp", "p"]].to_dict("index")}
        print(f"\n=== {design}: {int(s.bh_sig.sum())}/{len(s)} survive BH, "
              f"{int((s.p < 0.05).sum())} at p<0.05 ===")
        show = s.sort_values("p").head(10)
        print(show[["column", "pp_per_sd", "lo_sd", "hi_sd", "pp_per_10pp", "p", "bh_sig"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    runlog.finish(d, res)


if __name__ == "__main__":
    main()
