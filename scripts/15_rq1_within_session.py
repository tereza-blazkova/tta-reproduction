#!/usr/bin/env python
"""Stage 15 — the within-SESSION design, and the placebo that validates it.

Three nested comparisons of the same 57 codes, each removing more of the confounding:

  1. **pooled**        question fixed effects only — compares different students.
  2. **within student** + student FE on session-wide rates — removes who the student is, but
     not how they were doing that day, and not which tutor they had (34% of students saw more
     than one).
  3. **within session** + session FE on objective-localised rates — compares two objectives
     taught in the SAME session: same student, same tutor, same day, same mood. Everything
     stable about the pairing is absorbed, and what is left is which objective got which
     exchanges.

Design 3 is only estimable because stage 14 made the features vary within a session; with
session-wide rates a session fixed effect would absorb the predictor entirely.

**The placebo decides whether design 3 means anything.** Re-running it on a *different* segment
of the same session must give a null. If the placebo scores like the matched span, the
localisation is cutting noise and any within-session result is an artifact of the cut.

    python scripts/15_rq1_within_session.py --run rq1-within-session
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, data, runlog, within


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    d = df.astype(float)
    d = d.fillna(d.mean())
    return ((d - d.mean()) / d.std(ddof=0).replace(0, np.nan)).fillna(0.0)


def run_design(y, X: pd.DataFrame, absorb: dict, cluster, cols, label: str) -> pd.DataFrame:
    M = pd.concat([pd.Series(y, name="_y"), X], axis=1)
    Md, n_abs = within.absorb_matrix(M, absorb)
    yd = Md["_y"].to_numpy()
    rows = []
    for c in cols:
        r = within.ols_demeaned(yd, Md[[c]], cluster, n_abs)
        t = r.table().loc[c]
        rows.append({"column": c, "design": label, "pp_per_sd": 100 * t.estimate,
                     "lo": 100 * t.lo, "hi": 100 * t.hi, "p": t.p})
    out = pd.DataFrame(rows)
    out["bh_sig"] = within.benjamini_hochberg(out.p.to_numpy(), 0.05)
    print(f"  {label:26s} {int(out.bh_sig.sum()):2d}/{len(out)} survive BH | "
          f"median |b| {out.pp_per_sd.abs().median():.3f} | "
          f"median CI half-width {((out.hi - out.lo) / 2).median():.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rq1-within-session")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a))

    meta = data.load_metadata()
    tr = (data.load_train().merge(meta[["response_id", "student_id", "question_id"]],
                                  on="response_id").reset_index(drop=True))
    sess = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    locf = pd.read_parquet(config.DERIVED / "features_localised_train.parquet")
    locf["response_id"] = locf.response_id.astype(str)
    cols = agg.code_columns(sess.columns)
    # the two tables carry the SAME column names (session-wide vs objective-localised rates of
    # the same codes); namespacing the localised set keeps a merge from silently suffixing them
    locf = locf.rename(columns={c: "loc_" + c for c in cols})
    X = tr[["response_id", "session_id"]].merge(sess, on="session_id", how="left") \
                                         .merge(locf, on="response_id", how="left")
    y = tr.is_correct.to_numpy(float)

    # how much identifying variation does the within-session design actually have?
    nlo = tr.groupby("session_id").learning_objective_id.nunique()
    varies = tr.groupby("session_id").is_correct.nunique() > 1
    ident = tr.session_id.isin(set(nlo[nlo > 1].index) & set(varies[varies].index))
    print(f"rows {len(tr):,} | identifying rows for the session design "
          f"(multi-objective AND outcome varies): {ident.sum():,}")

    res = []
    print("\nper-code associations with the outcome (57 codes, BH at 0.05):")
    res.append(run_design(y, standardise(X[cols]), {"question": tr.question_id},
                          tr.student_id, cols, "1 pooled"))
    res.append(run_design(y, standardise(X[cols]),
                          {"student": tr.student_id, "question": tr.question_id},
                          tr.student_id, cols, "2 within student"))
    if config.ANNOTATOR:
        # the localised rates (stage 14) exist only for the 122B's labels; pairing them with
        # another annotator's session rates would mix two instruments in one record
        print(f"TTAPUB_ANNOTATOR={config.ANNOTATOR}: designs 3-4 skipped (localised rates are "
              f"the 122B's)")
        out = pd.concat(res, ignore_index=True)
        out.to_csv(d / "rq1_designs.csv", index=False)
        runlog.finish(d, {"run": a.run, "annotator": config.ANNOTATOR,
                          "by_design": out.groupby("design").bh_sig.sum().to_dict()})
        return
    loc = standardise(X[["loc_" + c for c in cols]]).rename(
        columns={("loc_" + c): c for c in cols})
    res.append(run_design(y, loc, {"session": tr.session_id, "question": tr.question_id},
                          tr.student_id, cols, "3 within session (matched)"))
    pl = standardise(X[["pl_" + c for c in cols]]).rename(
        columns={("pl_" + c): c for c in cols})
    res.append(run_design(y, pl, {"session": tr.session_id, "question": tr.question_id},
                          tr.student_id, cols, "4 within session (PLACEBO)"))

    out = pd.concat(res, ignore_index=True)
    out.to_csv(d / "rq1_designs.csv", index=False)
    print(f"\nwrote {d / 'rq1_designs.csv'}")
    runlog.finish(d, {"run": a.run, "identifying_rows": int(ident.sum()),
                      "by_design": out.groupby("design").bh_sig.sum().to_dict()})


if __name__ == "__main__":
    main()
