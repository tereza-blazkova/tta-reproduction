#!/usr/bin/env python
"""Stage 35 — how much of each code's association is who the student is?

The pooled association of a code with the outcome (stage 15, design 1) mixes two things: students
who show more of the code may simply be stronger students, and the same student may do better in
sessions where they show more of it. Stage 15's within-student design removes the first; this
stage measures both, side by side, for every code.

* **Between (Mundlak).** Per code, one linear probability model of the observed outcome on the
  student's mean rate of the code (over their responses) and the session's deviation from that
  mean, question fixed effects absorbed, SEs clustered on the student. The coefficient on the
  student mean is the part that travels with the student.
* **Within.** The same code with student and question fixed effects absorbed — exactly stage
  15's design 2 (the deviation, once student-demeaned, is the rate student-demeaned), so this
  column is the within-student estimate of Figure 3b. The Mundlak deviation coefficient is kept
  as `within_mundlak`; it absorbs no student effect in the outcome and runs slightly larger.
  Units throughout: pp per SD of the session rate, so the parts read on one axis.
* **Other sessions.** For students with two or more sessions: does the code's rate in THIS
  session predict the student's accuracy in their OTHER sessions (leave-one-session-out mean,
  in pp)? A session cannot change what happens after a different session, so a non-zero slope
  is the code marking a stable trait of the student.

BH at 0.05 across the 57 codes for each estimate. Association, not effect, throughout.

    python scripts/35_between_within.py --run between-within
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, data, runlog, within


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="between-within")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.FEATURES_TRAIN])

    meta = data.load_metadata()
    tr = (data.load_train().merge(meta[["response_id", "student_id", "question_id"]],
                                  on="response_id").reset_index(drop=True))
    sess = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    cols = agg.code_columns(sess.columns)
    X = tr[["session_id"]].merge(sess, on="session_id", how="left")[cols].astype(float)
    X = X.fillna(X.mean())
    Z = (X - X.mean()) / X.std(ddof=0)
    y = tr.is_correct.to_numpy(float)
    stu = tr.student_id.to_numpy()
    q = tr.question_id.to_numpy()

    # the student's accuracy over their OTHER sessions (leave-one-session-out)
    g = tr.groupby(["student_id", "session_id"]).is_correct.agg(["sum", "count"]).reset_index()
    tot = g.groupby("student_id")[["sum", "count"]].transform("sum")
    g["other_acc"] = (tot["sum"] - g["sum"]) / (tot["count"] - g["count"]).replace(0, np.nan)
    other = tr[["student_id", "session_id"]].merge(
        g[["student_id", "session_id", "other_acc"]], on=["student_id", "session_id"],
        how="left").other_acc.to_numpy()
    ok = ~np.isnan(other)

    rows = []
    for c in cols:
        z = Z[c]
        m = z.groupby(stu).transform("mean")
        D = pd.DataFrame({"between": m.to_numpy(), "within": (z - m).to_numpy()})
        r = within.fe_ols(y, D, {"question": q}, stu).table()
        f = within.fe_ols(y, D[["within"]], {"student": stu, "question": q}, stu).table()
        s = within.fe_ols(100 * other[ok], pd.DataFrame({"rate": z.to_numpy()[ok]}),
                          {"question": q[ok]}, stu[ok]).table()
        row = {"column": c}
        for k, t, src in (("between", r, "between"), ("within", f, "within"),
                          ("within_mundlak", r, "within")):
            row |= {k: 100 * t.loc[src, "estimate"], f"{k}_lo": 100 * t.loc[src, "lo"],
                    f"{k}_hi": 100 * t.loc[src, "hi"], f"{k}_p": t.loc[src, "p"]}
        row |= {"other_sessions": s.loc["rate", "estimate"], "other_lo": s.loc["rate", "lo"],
                "other_hi": s.loc["rate", "hi"], "other_p": s.loc["rate", "p"]}
        rows.append(row)
    res = pd.DataFrame(rows)
    for k in ("between", "within", "other"):
        res[f"{k}_bh"] = within.benjamini_hochberg(res[f"{k}_p"].to_numpy(), 0.05)
    res.to_csv(d / "between_within.csv", index=False)

    slope = float(np.sum(res.between * res.within) / np.sum(res.between ** 2))
    ratio = float((res.within.abs() / res.between.abs()).median())
    summ = {"rows": int(len(tr)), "students": int(tr.student_id.nunique()),
            "students_with_other_sessions": int(tr.student_id[ok].nunique()),
            "within_on_between_slope": round(slope, 3),
            "median_abs_within_over_between": round(ratio, 3),
            "bh_between": int(res.between_bh.sum()), "bh_within": int(res.within_bh.sum()),
            "bh_other_sessions": int(res.other_bh.sum())}
    pd.set_option("display.width", 200)
    print(res.reindex(res.between.abs().sort_values(ascending=False).index).head(15)
          [["column", "between", "within", "other_sessions", "between_bh", "within_bh", "other_bh"]]
          .round(2).to_string(index=False))
    print(summ)
    runlog.finish(d, {"run": a.run} | summ)


if __name__ == "__main__":
    main()
