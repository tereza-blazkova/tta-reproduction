#!/usr/bin/env python
"""Stage 37 — which codes mark a strong student, and which go with a student beating their own
usual?

Two questions about every code, kept apart because the pooled association mixes them:

  **marker** (x)  Do students who beat their questions' rates have more of this code in their
                  sessions? Stage 35's question-adjusted ranking: attainment from one random half
                  of a student's sessions, code shares from the other half, top against bottom
                  third, 20 splits used both ways, permutation null and Benjamini-Hochberg.
                  Read here from `results/student-heterogeneity-residual`.
  **mover**  (y)  When a student's session has more of this code than that student usually has,
                  do they do better than their usual on that session's questions? A linear
                  probability model of the outcome on the code's session rate (per SD) with
                  student and question fixed effects: the student effect is their usual rate,
                  the question effect makes it objective-aware (a question belongs to one
                  objective, so harder objectives and harder questions are both absorbed).
                  Standard errors clustered by student, Benjamini-Hochberg across the 57 codes.
                  This is stage 15's within-student design, recomputed and checked against it.

Robustness of the mover axis, each reported per code:
  * **tutor**: tutor fixed effects added, so a student switching tutors cannot carry it;
  * **held out**: the same model on the released test set's TSL students with two or more test
    sessions, rates scaled with the training mean and SD. A mover should keep its sign;
  * **reliability**: a session rate is measured with error, which flattens a within-student
    slope toward zero. Each code's within-student split-half reliability (stage 16's block10
    estimate, recomputed from the labels) bounds the true slope: interval / sqrt(reliability).
    A code is called *not a mover* only when that bound excludes 2 pp per SD, the size the paper
    already treats as practically interesting. Otherwise its null is uninformative.

The marker axis has its own robustness in stage 35 (within-main-tutor null, 4+ sessions arm);
a continuous version computed here (`marker_loso`: the student's rate in their *other* sessions
against this session's outcome, question fixed effects, this session's own rate held fixed, so
no session contributes both its codes and its outcome) is reported beside it as a check.

Classes, stated before the run:
  * mover: within-student slope clears BH, and the held-out slope has the same sign;
  * marker only: marker clears BH, the code is not a mover, and the reliability bound on its
    within-student slope excludes +-2 pp per SD;
  * marker, mover unresolved: marker clears BH, neither a mover nor bounded;
  * neither.
A mover is an association with a better-than-usual session, not an effect of the code: a tutor
says "that's right" because the student was right. Its direction is not identified here.

    python scripts/37_marker_vs_mover.py --run marker-vs-mover
"""
from __future__ import annotations

import argparse
import importlib
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, data, runlog, within
from ttapub import taxonomy_v3 as T

MARKER_RUN = config.RESULTS / "student-heterogeneity-residual"
STAGE15_ESTIMATES = config.FIGURES / "input" / "fig_rq1_a_scatter.csv"
RELIABILITY_RUN = config.RESULTS / "reliability"
BOUND_PP = 2.0


def standardise(df: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Stage 15's scaling (fill with the mean, z-score); `ref` supplies the mean and SD."""
    ref = df if ref is None else ref
    mu, sd = ref.astype(float).mean(), ref.astype(float).std(ddof=0).replace(0, np.nan)
    d = df.astype(float).fillna(mu)
    return ((d - mu) / sd).fillna(0.0)


def per_code(y, Z: pd.DataFrame, absorb: dict, cluster, label: str) -> pd.DataFrame:
    M = pd.concat([pd.Series(y, name="_y", index=Z.index), Z], axis=1)
    Md, n_abs = within.absorb_matrix(M, absorb)
    rows = []
    for c in Z.columns:
        t = within.ols_demeaned(Md["_y"].to_numpy(), Md[[c]], cluster, n_abs).table().loc[c]
        rows.append({"column": c, f"{label}": 100 * t.estimate, f"{label}_lo": 100 * t.lo,
                     f"{label}_hi": 100 * t.hi, f"{label}_p": t.p})
    out = pd.DataFrame(rows).set_index("column")
    out[f"{label}_bh"] = within.benjamini_hochberg(out[f"{label}_p"].to_numpy(), 0.05)
    print(f"  {label:14s} {int(out[f'{label}_bh'].sum()):2d}/{len(out)} clear BH", flush=True)
    return out


def label_reliability(codes_by_role: dict, student_of: dict) -> pd.DataFrame:
    """Stage 16's block10 within-student reliability, from the labels' own turn order."""
    s16 = importlib.import_module("16_reliability")
    out = []
    for role, codes in codes_by_role.items():
        files = sorted(config.LABELS_V3.glob(f"v3_full_train_{role}_shard*.csv"))
        df = (pd.concat([pd.read_csv(f, dtype={"session_id": str}, usecols=["session_id", "turn", "code"])
                         for f in files], ignore_index=True)
              .drop_duplicates(["session_id", "turn"]).sort_values(["session_id", "turn"]))
        df["r"] = df.groupby("session_id", sort=False).cumcount()
        df["n_role"] = df.groupby("session_id", sort=False).r.transform("size")
        r = s16.reliability(df, codes, "block10", student_of)
        out.append(r.assign(role=role))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="marker-vs-mover")
    a = ap.parse_args()
    shards = sorted(config.LABELS_V3.glob("v3_full_train_*_shard*.csv"))
    d = runlog.start(a.run, vars(a), [config.TRAIN_LABELS, config.TEST_LABELS, config.METADATA,
                                       config.FEATURES_TRAIN, config.FEATURES_TEST,
                                       MARKER_RUN / "per_code.csv", STAGE15_ESTIMATES, *shards])
    meta = data.load_metadata()
    tr = (data.load_train().merge(meta[["response_id", "student_id", "question_id", "tutor_id"]],
                                  on="response_id").reset_index(drop=True))
    feat = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    cols = agg.code_columns(feat.columns)
    X = tr[["session_id"]].merge(feat, on="session_id", how="left")[cols]
    y = tr.is_correct.to_numpy(float)
    Z = standardise(X)

    print("mover axis, training (35,072 responses)", flush=True)
    mover = per_code(y, Z, {"student": tr.student_id, "question": tr.question_id}, tr.student_id,
                     "mover")
    # gate: the same design as stage 15 must give stage 15's numbers. They were computed on
    # another machine and the alternating projections stop at a tolerance, so agreement is
    # required to 0.001 pp per SD (the paper reports two decimals), not bit for bit.
    s15 = pd.read_csv(STAGE15_ESTIMATES)
    s15.index = [agg.column(c.upper(), r) for c, r in zip(s15.code, s15.role)]
    gap = (mover.mover - s15.pp_per_sd_within_student.reindex(mover.index)).abs().max()
    same_bh = (mover.mover_bh == s15.bh_sig_within_student.reindex(mover.index).astype(bool)).all()
    print(f"  gate: max |difference| from stage 15's within-student estimates {gap:.2e} pp per SD; "
          f"same BH set: {same_bh}", flush=True)
    if not (gap < 1e-3 and same_bh):
        raise SystemExit("the mover axis does not reproduce stage 15")
    tutor = per_code(y, Z, {"student": tr.student_id, "question": tr.question_id,
                            "tutor": tr.tutor_id.fillna("none")}, tr.student_id, "mover_tutor")

    print("mover axis, held out (released test, TSL, students with 2+ test sessions)", flush=True)
    te = data.load_test()
    te = te[te.data_source == "tsl"].merge(meta[["response_id", "tutor_id"]], on="response_id")
    te = te[te.groupby("student_id").session_id.transform("nunique") >= 2].reset_index(drop=True)
    ft = pd.read_csv(config.FEATURES_TEST, dtype={"session_id": str})
    Zt = standardise(te[["session_id"]].merge(ft, on="session_id", how="left")[cols], ref=X)
    held = per_code(te.is_correct.to_numpy(float), Zt,
                    {"student": te.student_id, "question": te.question_id}, te.student_id,
                    "mover_test")
    print(f"  {len(te):,} test responses, {te.student_id.nunique():,} students", flush=True)

    print("reliability (block10, within student)", flush=True)
    student_of = dict(zip(tr.session_id, tr.student_id))
    rel = label_reliability({r: T.codes(r) for r in ("tutor", "student")}, student_of)
    rel.index = [agg.column(c, r) for c, r in zip(rel.code, rel.role)]
    med = float(rel.rel_within.median())
    ref = json.loads((RELIABILITY_RUN / "metrics.json").read_text())["median_within_block10"]
    print(f"  median within-student reliability {med:.3f} (stage 16: {ref:.3f}); by role "
          f"{rel.groupby('role').rel_within.median().round(3).to_dict()}", flush=True)
    if abs(med - ref) > 0.02:
        raise SystemExit("label-order reliability does not reproduce stage 16")

    print("marker axis (stage 35, question-adjusted ranking) and its continuous check", flush=True)
    mk = pd.read_csv(MARKER_RUN / "per_code.csv")
    mk.index = [agg.column(c, r) for c, r in zip(mk.code, mk.role)]
    # continuous check: the student's rate in their OTHER sessions, this session's rate held fixed
    sess = (tr.groupby("session_id").agg(st=("student_id", "first")).join(
        feat.set_index("session_id")[cols]))
    tot = sess.groupby("st")[cols].transform("sum")
    k = sess.groupby("st")[cols].transform("count")
    loso = (tot - sess[cols].fillna(0)) / (k - sess[cols].notna().astype(int)).replace(0, np.nan)
    keep = tr.student_id.map(tr.groupby("student_id").session_id.nunique()).to_numpy() >= 2
    L = loso.reindex(tr.session_id[keep]).reset_index(drop=True)
    Zk = Z[keep].reset_index(drop=True)
    yk, qk, sk = y[keep], tr.question_id[keep].to_numpy(), tr.student_id[keep].to_numpy()
    Lz = standardise(L)
    rows = []
    for c in cols:
        r = within.fe_ols(yk, pd.DataFrame({"usual": Lz[c], "today": Zk[c]}), {"question": qk}, sk)
        t = r.table().loc["usual"]
        rows.append({"column": c, "marker_loso": 100 * t.estimate, "marker_loso_lo": 100 * t.lo,
                     "marker_loso_hi": 100 * t.hi, "marker_loso_p": t.p})
    loso_tab = pd.DataFrame(rows).set_index("column")
    loso_tab["marker_loso_bh"] = within.benjamini_hochberg(loso_tab.marker_loso_p.to_numpy(), 0.05)

    name = {agg.column(k, r): (k, r) for r in ("tutor", "student") for k in T.codes(r)}
    out = pd.DataFrame({"code": [name[c][0] for c in cols], "role": [name[c][1] for c in cols]},
                       index=cols)
    out["marker_pct"] = 100 * (2 ** mk.log2_ratio.reindex(cols) - 1)
    out["marker_lo_pct"] = 100 * (2 ** mk.ci_lo.reindex(cols) - 1)
    out["marker_hi_pct"] = 100 * (2 ** mk.ci_hi.reindex(cols) - 1)
    out["marker_bh"] = mk.bh_sig.reindex(cols).astype(bool)
    out["marker_bh_within_tutor"] = mk.bh_sig_within_tutor.reindex(cols).astype(bool)
    out = out.join(mover).join(tutor).join(held).join(loso_tab)
    out["rel_within"] = rel.rel_within.reindex(cols)
    s = np.sqrt(out.rel_within.clip(lower=0.05))
    out["mover_bound_lo"], out["mover_bound_hi"] = out.mover_lo / s, out.mover_hi / s
    out["bounded"] = (out.mover_bound_lo > -BOUND_PP) & (out.mover_bound_hi < BOUND_PP)
    out["held_same_sign"] = np.sign(out.mover_test) == np.sign(out.mover)
    is_mover = out.mover_bh & out.held_same_sign
    out["class"] = np.select(
        [is_mover, out.marker_bh & out.bounded, out.marker_bh],
        ["mover", "marker only", "marker, mover unresolved"], "neither")
    out.to_csv(d / "codes.csv", index_label="column")

    for cl, g in out.groupby("class"):
        print(f"  {cl:26s} {len(g):2d}: {', '.join(g.code.str.lower())}", flush=True)
    agree = out[["marker_pct", "marker_loso"]].corr(method="spearman").iloc[0, 1]
    held_r = out[["mover", "mover_test"]].corr().iloc[0, 1]
    tut_r = out[["mover", "mover_tutor"]].corr().iloc[0, 1]
    print(f"  marker vs its continuous check: Spearman {agree:.2f}; mover train vs held out r "
          f"{held_r:.2f}; mover with tutor FE r {tut_r:.2f}", flush=True)
    runlog.finish(d, {
        "run": a.run, "gate_stage15_max_gap": float(gap),
        "reliability_median_within": med, "reliability_median_stage16": ref,
        "reliability_median_by_role": rel.groupby("role").rel_within.median().to_dict(),
        "bound_pp_per_sd": BOUND_PP, "test_rows": int(len(te)),
        "test_students": int(te.student_id.nunique()),
        "classes": out["class"].value_counts().to_dict(),
        "members": {cl: sorted(g.code.str.lower()) for cl, g in out.groupby("class")},
        "marker_bh": int(out.marker_bh.sum()), "mover_bh": int(out.mover_bh.sum()),
        "mover_tutor_bh": int(out.mover_tutor_bh.sum()), "mover_test_bh": int(out.mover_test_bh.sum()),
        "spearman_marker_vs_loso": float(agree), "r_mover_train_vs_test": float(held_r),
        "r_mover_vs_tutor_fe": float(tut_r),
    })


if __name__ == "__main__":
    main()
