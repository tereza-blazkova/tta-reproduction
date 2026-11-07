#!/usr/bin/env python
"""Stage 18 — two questions the narrowed paper needs.

**A. Is an association about the move, or about which tutor the student drew?**
A nested ladder of fixed effects on the same codes: question only (pooled), + student,
+ tutor, + both. 732 tutors on TSL with no missing identifier, and 34% of students see more
than one tutor, so student and tutor are separately identified. If an association survives
absorbing the tutor it is about the move; if it vanishes it was about the person making it.
This is metadata integration in the sense the call asks for, and it is the sharpest test of a
tutoring claim this corpus allows.

**B. What does the headline mean in units a practitioner can act on?**
The 16 student shares sum to one, so a model with all of them is unidentified and a model with
one at a time (stage 15) answers a different question. Including 15 with the most common code
as the reference makes every coefficient a *reallocation*: what is associated with 10 percentage
points of a student's turns moving from no visible move to explaining. That is the form the
educational-implications argument needs.

**Run B on both schemes, and the contrast is the paper's thesis in one table.** The tutor side
is the same model over the 41 tutor shares with `TUTOR_NO_MOVE` as reference, and it asks the
question a practitioner would actually ask --- *if a tutor spent ten percent more of their turns
doing this instead, what then?* If the student scheme yields a reallocation that clears the
correction and the tutor scheme yields none, that is "student state matters for findings about
tutoring" stated in the units a reader cares about, without an interaction term and without
appeal to attribution. Reporting only the student side would leave the comparison implicit.

    python scripts/18_tutor_and_composition.py --run tutor-and-composition
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, data, runlog, within

REFERENCE = "STUDENT_NO_MOVE"      # the most common student code; coefficients read against it
REFERENCE_TUTOR = "TUTOR_NO_MOVE"  # its counterpart on the tutor axis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="tutor-and-composition")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.FEATURES_TRAIN])

    meta = data.load_metadata()
    tr = (data.load_train().merge(meta[["response_id", "student_id", "question_id", "tutor_id"]],
                                  on="response_id").reset_index(drop=True))
    feats = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    X = tr[["session_id"]].merge(feats, on="session_id", how="left")
    cols = agg.code_columns(feats.columns)
    y = tr.is_correct.to_numpy(float)
    print(f"rows {len(tr):,} | students {tr.student_id.nunique():,} | "
          f"tutors {tr.tutor_id.nunique():,} | "
          f"students with >1 tutor {(tr.groupby('student_id').tutor_id.nunique()>1).mean():.0%}")

    # ---------- A. the fixed-effect ladder ----------
    std = X[cols].astype(float)
    std = std.fillna(std.mean())
    std = ((std - std.mean()) / std.std(ddof=0).replace(0, np.nan)).fillna(0.0)
    M = pd.concat([pd.Series(y, name="_y"), std], axis=1)
    designs = {
        "1 question": {"question": tr.question_id},
        "2 +student": {"question": tr.question_id, "student": tr.student_id},
        "3 +tutor": {"question": tr.question_id, "tutor": tr.tutor_id},
        "4 +student+tutor": {"question": tr.question_id, "student": tr.student_id,
                             "tutor": tr.tutor_id},
    }
    rows = []
    for name, absorb in designs.items():
        Md, na = within.absorb_matrix(M, absorb)
        yd = Md["_y"].to_numpy()
        for c in cols:
            r = within.ols_demeaned(yd, Md[[c]], tr.student_id.to_numpy(), na)
            t = r.table().loc[c]
            rows.append({"column": c, "design": name, "pp_per_sd": 100 * t.estimate,
                         "lo": 100 * t.lo, "hi": 100 * t.hi, "p": t.p})
        # BH within this design, written back by position in the global list -- an earlier
        # version built a fresh frame with its own RangeIndex and wrote the flags back by
        # that index, which silently mislabelled every design after the first.
        idx = [i for i, x in enumerate(rows) if x["design"] == name]
        sig = within.benjamini_hochberg(np.array([rows[i]["p"] for i in idx]), 0.05)
        for i, v in zip(idx, sig):
            rows[i]["bh"] = bool(v)
        est = np.abs([rows[i]["pp_per_sd"] for i in idx])
        print(f"  {name:18s} {int(sig.sum()):2d}/57 survive BH | median |b| {np.median(est):.3f}")
    lad = pd.DataFrame(rows)
    lad.to_csv(d / "fe_ladder.csv", index=False)

    piv = lad.pivot_table(index="column", columns="design", values="pp_per_sd")
    key = ["v3s_explains_why_rate", "v3t_prompting_next_step_rate", "v3t_praising_outcome_rate"]
    print("\n=== the three RQ1 survivors across the ladder (pp per SD) ===")
    print(piv.loc[[k for k in key if k in piv.index]].to_string(float_format=lambda v: f"{v:.2f}"))
    surv = lad[(lad.design == "4 +student+tutor") & lad.bh]
    print(f"\nsurviving student AND tutor absorption: {len(surv)} codes "
          f"({', '.join(surv.column.str.replace('v3[ts]_','',regex=True).str.replace('_rate','',regex=True))})")

    # ---------- B. the compositional model ----------
    comp = {}
    for role, reference, dest in (("student", REFERENCE, "compositional.csv"),
                                  ("tutor", REFERENCE_TUTOR, "compositional_tutor.csv")):
        ref = agg.column(reference, role)
        comp_cols = [c for c in agg.code_columns(feats.columns, role) if c != ref]
        raw = X[comp_cols].astype(float)
        raw = raw.fillna(raw.mean())
        Mc = pd.concat([pd.Series(y, name="_y"), raw], axis=1)
        Md, na = within.absorb_matrix(Mc, {"question": tr.question_id,
                                           "student": tr.student_id})
        r = within.ols_demeaned(Md["_y"].to_numpy(), Md[comp_cols], tr.student_id.to_numpy(), na)
        t = r.table()
        # rates are shares, so a 0.10 change is ten percentage points of that role's turns
        t = (t * 100 * 0.10).assign(p=r.pvalues())
        t["bh"] = within.benjamini_hochberg(t.p.to_numpy(), 0.05)
        t.index = [i.replace(f"v3{role[0]}_", "").replace("_rate", "") for i in t.index]
        t = t.sort_values("estimate", ascending=False)
        t.to_csv(d / dest)
        comp[role] = {"reference": reference, "n_columns": len(comp_cols),
                      "n_bh": int(t.bh.sum()),
                      "survivors": {i: round(float(v), 2)
                                    for i, v in t.estimate[t.bh].items()},
                      "largest_non_survivor_abs": round(float(t.estimate[~t.bh].abs().max()), 2)}
        print(f"\n=== reallocating 10% of a {role}'s turns FROM {reference.lower()} "
              f"TO each code (pp on the outcome, within student) ===")
        print(t[["estimate", "lo", "hi", "bh"]].head(8).to_string(float_format=lambda v: f"{v:.2f}"))
        print(f"  ... {len(comp_cols)} codes, {int(t.bh.sum())} clear BH")
    print(f"\n=== the comparison ===\n  student scheme {comp['student']['n_bh']} of "
          f"{comp['student']['n_columns']} reallocations clear BH; tutor scheme "
          f"{comp['tutor']['n_bh']} of {comp['tutor']['n_columns']}")
    runlog.finish(d, {"run": a.run, "n_survive_student_and_tutor": int(len(surv)),
                      "compositional_reference": REFERENCE,
                      "compositional": comp})


if __name__ == "__main__":
    main()
