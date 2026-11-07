#!/usr/bin/env python
"""Stage 12 (RQ2) — which adjacent-turn exchanges are associated with the outcome?

For every pair that cleared stage 11's frequency floor, a within-student regression of the
outcome on that pair's session rate, **adjusting for the rates of its two components**. The
adjustment is what makes the estimate about the *sequence*: without it, "bare answer followed
by evaluative feedback" would simply inherit whatever bare answers and evaluative feedback
carry on their own.

Student and question fixed effects are absorbed, standard errors cluster on the student, and
rates are standardised so a coefficient reads as **percentage points per SD of exposure**.
Benjamini-Hochberg controls the false discovery rate across the pairs tested, which matters at
this width: 421 tests at alpha 0.05 would be expected to throw ~21 false positives unadjusted.

⚠️ These are associations within a student, not effects. A tutor chooses the next move in
response to what the student just did, so a sequence's coefficient still carries whatever the
tutor was reacting to. The design removes stable differences between students; it does not make
the tutor's choice random.

    python scripts/12_rq2_sequences.py --run rq2-sequences
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, data, runlog, within


def component_cols(pair: str, kind: str) -> tuple[str, str]:
    a, b = pair.split(" > ")
    if kind == "upt":          # tutor move -> student signal
        return agg.column(a, "tutor"), agg.column(b, "student")
    return agg.column(a, "student"), agg.column(b, "tutor")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rq2-sequences")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--validate", action="store_true",
                    help="prove the absorbed fit equals explicit dummies on a subsample first")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.FEATURES_TRAIN])

    meta = data.load_metadata()
    tr = (data.load_train()
          .merge(meta[["response_id", "student_id", "question_id"]], on="response_id")
          .reset_index(drop=True))
    seq = pd.read_parquet(config.DERIVED / "sequences_v3_train.parquet")
    feats = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    X = (tr[["session_id"]].merge(seq, on="session_id", how="left")
                           .merge(feats, on="session_id", how="left"))
    pair_cols = [c for c in seq.columns if c.startswith("seq_")]
    code_cols = agg.code_columns(feats.columns)
    print(f"rows {len(tr):,} | students {tr.student_id.nunique():,} | "
          f"questions {tr.question_id.nunique():,} | pairs {len(pair_cols)}")

    use = X[pair_cols + code_cols].astype(float)
    use = use.fillna(use.mean())
    # standardise so every coefficient is percentage points per SD of exposure
    use = (use - use.mean()) / use.std(ddof=0).replace(0, np.nan)
    use = use.fillna(0.0)
    y = tr.is_correct.to_numpy(float)

    if a.validate:
        # whole students, not random rows: the design is within-student, so a row sample
        # with one response per student tests arithmetic that the real model never does
        rng = np.random.default_rng(0)
        pick = set(rng.choice(tr.student_id.unique(), 700, replace=False))
        sub = np.where(tr.student_id.isin(pick))[0]
        c0 = pair_cols[0]
        ca, cb = component_cols(c0.split("_", 2)[2], c0.split("_")[1])
        within.check_against_dummies(
            y[sub], use.iloc[sub][[c0, ca, cb]],
            {"student": tr.student_id.to_numpy()[sub],
             "question": tr.question_id.to_numpy()[sub]},
            tr.student_id.to_numpy()[sub])

    M = pd.concat([pd.Series(y, name="_y"), use], axis=1)
    cluster = tr.student_id.to_numpy()
    print("absorbing fixed effects once for all models...", flush=True)
    # Two designs, the same pairs. POOLED absorbs the question only — the objective control the
    # manuscript specifies — so it compares sessions across different students. WITHIN adds the
    # student, so it compares a student's own sessions. The gap between them IS the result:
    # whatever shrinks was a stable difference between students, not something that happened
    # in the session.
    Md, n_abs = within.absorb_matrix(M, {"student": tr.student_id, "question": tr.question_id})
    Mp, n_abs_p = within.absorb_matrix(M, {"question": tr.question_id})
    yd, yp = Md["_y"].to_numpy(), Mp["_y"].to_numpy()

    rows = []
    for i, col in enumerate(pair_cols, 1):
        kind, pair = col.split("_")[1], col.split("_", 2)[2]
        ca, cb = component_cols(pair, kind)
        if ca not in Md.columns or cb not in Md.columns:
            continue
        r = within.ols_demeaned(yd, Md[[col, ca, cb]], cluster, n_abs)
        rp = within.ols_demeaned(yp, Mp[[col, ca, cb]], cluster, n_abs_p)
        t, tp = r.table().loc[col], rp.table().loc[col]
        rows.append({"pair": pair, "kind": "uptake" if kind == "upt" else "response",
                     "from": pair.split(" > ")[0], "to": pair.split(" > ")[1],
                     "pp_per_sd": 100 * t.estimate, "lo": 100 * t.lo, "hi": 100 * t.hi,
                     "se": 100 * t.se, "p": t.p,
                     "pooled_pp_per_sd": 100 * tp.estimate, "pooled_lo": 100 * tp.lo,
                     "pooled_hi": 100 * tp.hi, "pooled_p": tp.p,
                     "n": r.n, "n_students": r.n_clusters})
        if i % 100 == 0:
            print(f"  {i}/{len(pair_cols)}", flush=True)

    out = pd.DataFrame(rows)
    out["bh_sig"] = within.benjamini_hochberg(out.p.to_numpy(), a.alpha)
    out["bh_sig_pooled"] = within.benjamini_hochberg(out.pooled_p.to_numpy(), a.alpha)
    inv = pd.read_csv(config.DERIVED / "sequence_inventory_train.csv")
    out = out.merge(inv[["pair", "kind", "count", "lift"]], on=["pair", "kind"], how="left")
    out = out.sort_values("pp_per_sd")
    out.to_csv(d / "rq2_sequences.csv", index=False)

    sig = out[out.bh_sig]
    sigp = out[out.bh_sig_pooled]
    print(f"\n{len(out)} pairs tested")
    print(f"  POOLED (question FE only):      {len(sigp)} survive BH at FDR {a.alpha}")
    print(f"  WITHIN-STUDENT (+ student FE):  {len(sig)} survive BH")
    keep = (out.pooled_pp_per_sd.abs() > 0.01)
    shrink = 1 - (out.loc[keep, "pp_per_sd"].abs() / out.loc[keep, "pooled_pp_per_sd"].abs())
    print(f"  median shrinkage of |estimate| when the student is absorbed: "
          f"{shrink.median():.0%}")
    show = out.reindex(out.pooled_pp_per_sd.abs().sort_values(ascending=False).index).head(10)
    print("\n=== the ten largest POOLED associations, beside their within-student estimate ===")
    print(show[["kind", "pair", "pooled_pp_per_sd", "pp_per_sd", "lo", "hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    runlog.finish(d, {"run": a.run, "n_tested": len(out), "n_bh_sig": int(len(sig)),
                      "n_bh_sig_pooled": int(len(sigp)),
                      "median_shrinkage": float(shrink.median()),
                      "alpha": a.alpha, "n": int(out.n.iloc[0]),
                      "n_students": int(out.n_students.iloc[0])})


if __name__ == "__main__":
    main()
