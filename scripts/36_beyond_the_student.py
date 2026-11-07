#!/usr/bin/env python
"""Stage 36 — is the model's transcript signal more than who the student is?

Stages 15, 32 and 35 show that the pooled code associations are carried by differences between
students: strong students talk differently and get different moves. That leaves open what the
code-rate model has learned. It might recognise a strong student from how they talk and nothing
else, which is a statement about the student and not about the session. This stage tests that on
the released test set, with the models fitted on all training responses (`arm-*-pred`).

For each test response, the model's transcript signal is s = logit p(arm) - logit p(objective
only): what the conversation adds to the objective's prediction, for the student, tutor and both
arms.

**A. Who the student is, measured on other data.** A student's attainment is their
question-adjusted residual on their *training* responses (the outcome minus the question's
correctness rate, stage 35's `expected_correct`), shrunk toward 0 by a pseudo-count chosen by
leave-one-session-out prediction on training. So a student is "strong" when they beat their
questions' rates, not when they happen to get easy questions. Nothing from a test response enters
its student's attainment. The rows are the TSL test responses whose student has training responses.
Four logistic models on the objective-only logit, cross-fitted over five folds grouped by student:

    objective | + attainment | + transcript | + attainment + transcript

  * transcript beyond the student: (+ attainment + transcript) vs (+ attainment);
  * student beyond the transcript: (+ attainment + transcript) vs (+ transcript);
  * the coefficient on s with and without attainment, and its share retained.

The attainment score is a noisy measure of the trait (stage 35: split-half r 0.16 per half of a
student's sessions; the median test student has two training responses), so holding it fixed
under-adjusts. A is therefore repeated on students with at least 1, 2, 3, 5 and 10 training
responses (`--min-train`): as attainment is measured more precisely, the transcript's increment over
it shows how much of the transcript signal is the trait. B removes the trait completely.

**B. Within student.** Test students with two or more test sessions. A linear probability model
of the outcome on s (per SD), with question fixed effects (pooled) and with question and student
fixed effects (within), standard errors clustered by student. Eedi questions are mostly
singletons, so on Eedi the objective-only logit replaces the question fixed effects.

Reading, stated before the run. A model that only recognises strong students keeps none of its
transcript coefficient once attainment is held fixed and has a within-student slope near zero.
Whatever survives is session-specific: the session went better than this student's usual. That
can be learning in the session or the student's state on the day; nothing here separates the two.

    python scripts/36_beyond_the_student.py --run beyond-the-student
"""
from __future__ import annotations

import argparse
import importlib
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ttapub import config, data, metrics, runlog, within

ARMS = ("student", "tutor", "both")
#: Each cross-fit's student -> fold, frozen from the recorded run (see `student_folds`).
FOLDS = config.RESULTS / "beyond-the-student" / "student_folds.csv"
PRED = {a: config.RESULTS / f"arm-{a}-pred" / "test.csv" for a in ("lo", *ARMS)}
M_GRID = np.r_[np.arange(1, 20), np.arange(20, 101, 5)]
# C=inf is sklearn 1.8's spelling of an unpenalised fit, and it warns on every call
warnings.filterwarnings("ignore", message="Setting penalty=None")


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def student_attainment(tr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Shrunk mean question-adjusted residual per training student, and how the shrinkage was set.

    The pseudo-count m is the one that best predicts a held-out session's mean residual from the
    student's other sessions (response-weighted squared error)."""
    sess = tr.groupby("session_id").agg(st=("student_id", "first"), r=("res", "sum"),
                                        n=("res", "size"))
    tot = sess.groupby("st")[["r", "n"]].transform("sum")
    R, N = (tot.r - sess.r).to_numpy(), (tot.n - sess.n).to_numpy()
    ok = N > 0
    target = (sess.r / sess.n).to_numpy()[ok]
    w = sess.n.to_numpy()[ok]
    sse = [float(np.sum(w * (target - R[ok] / (N[ok] + m)) ** 2)) for m in M_GRID]
    m = float(M_GRID[int(np.argmin(sse))])
    st = tr.groupby("student_id").res.agg(["sum", "size"])
    out = pd.DataFrame({"attainment": st["sum"] / (st["size"] + m), "n_train": st["size"]})
    return out, {"pseudocount": m, "loso_sessions": int(ok.sum()),
                 "sse_at_0_shrinkage_ratio": sse[0] / min(sse)}


def student_folds(groups: np.ndarray) -> pd.Series:
    """GroupKFold(5) over students: each student's fold. Only `--refold` calls this.

    GroupKFold breaks ties between equally sized students with numpy's sort, whose order depends
    on the CPU (AVX-512 x86 and AVX2 x86 differ; so may ARM). The recorded run's assignment is
    therefore frozen in `FOLDS` and loaded, never recomputed, as the objective folds are."""
    fold = np.empty(len(groups), int)
    for k, (_, te_i) in enumerate(GroupKFold(5).split(groups, groups=groups)):
        fold[te_i] = k
    out = pd.Series(fold, index=groups)
    return out[~out.index.duplicated()]


def crossfit(X: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Out-of-fold logistic predictions over the given per-row folds (students never straddle)."""
    p = np.zeros(len(y))
    for k in np.unique(fold):
        tr_i, te_i = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        p[te_i] = LogisticRegression(C=np.inf, max_iter=1000).fit(X[tr_i], y[tr_i]).predict_proba(X[te_i])[:, 1]
    return p


def coefs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return LogisticRegression(C=np.inf, max_iter=1000).fit(X, y).coef_[0]


def design_a(df: pd.DataFrame, boot: int, rng, fold_of: pd.Series) -> dict:
    """Cross-fitted increments and coefficient retention for one set of rows."""
    y, g = df.y.to_numpy(int), df.student_id.to_numpy()
    z0, a = df.z_lo.to_numpy(), df.attainment.to_numpy()
    fold = fold_of.reindex(g).to_numpy()
    if np.isnan(fold.astype(float)).any():
        raise SystemExit("a student has no frozen fold: the rows differ from the recorded run (see --refold)")
    out = {"n": int(len(df)), "students": int(df.student_id.nunique()), "arms": {}}
    p0 = crossfit(z0[:, None], y, fold)
    pa = crossfit(np.c_[z0, a], y, fold)
    out["objective"] = metrics.basic(y, p0)
    out["attainment"] = metrics.basic(y, pa) | {"vs_objective": metrics.boot_delta(y, pa, p0, g, B=boot)}
    # bootstrap students once, shared by every arm's coefficient intervals
    ug, inv = np.unique(g, return_inverse=True)
    rows_of = [np.flatnonzero(inv == k) for k in range(len(ug))]
    draws = [np.concatenate([rows_of[k] for k in rng.integers(0, len(ug), len(ug))]) for _ in range(boot)]
    for arm in ARMS:
        s = df[f"s_{arm}"].to_numpy()
        ps = crossfit(np.c_[z0, s], y, fold)
        pb = crossfit(np.c_[z0, a, s], y, fold)
        b_s, b_both = coefs(np.c_[z0, s], y), coefs(np.c_[z0, a, s], y)
        b_a = coefs(np.c_[z0, a], y)
        ret, ret_a = [], []
        for ix in draws:
            r1, r2 = coefs(np.c_[z0, s][ix], y[ix]), coefs(np.c_[z0, a, s][ix], y[ix])
            ra = coefs(np.c_[z0, a][ix], y[ix])
            ret.append(r2[2] / r1[1])
            ret_a.append(r2[1] / ra[1])
        q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        out["arms"][arm] = {
            "corr_s_attainment": float(np.corrcoef(s, a)[0, 1]),
            "transcript": metrics.basic(y, ps) | {"vs_objective": metrics.boot_delta(y, ps, p0, g, B=boot)},
            "attainment_and_transcript": metrics.basic(y, pb),
            "transcript_beyond_attainment": metrics.boot_delta(y, pb, pa, g, B=boot),
            "attainment_beyond_transcript": metrics.boot_delta(y, pb, ps, g, B=boot),
            "coef_s_alone": float(b_s[1]), "coef_s_with_attainment": float(b_both[2]),
            "s_share_retained": float(b_both[2] / b_s[1]), "s_share_retained_ci": q(ret),
            "coef_attainment_alone": float(b_a[1]), "coef_attainment_with_s": float(b_both[1]),
            "attainment_share_retained": float(b_both[1] / b_a[1]), "attainment_share_retained_ci": q(ret_a),
        }
        r = out["arms"][arm]
        print(f"  {arm:7s} corr(s, attainment) {r['corr_s_attainment']:+.3f} | transcript beyond attainment "
              f"dLL {r['transcript_beyond_attainment']['d_ll']:+.4f} {r['transcript_beyond_attainment']['d_ll_ci']} "
              f"dAUROC {r['transcript_beyond_attainment']['d_auroc']:+.2f} | s retained "
              f"{r['s_share_retained']:.2f} {r['s_share_retained_ci']}", flush=True)
    return out


def design_b(df: pd.DataFrame, source: str) -> dict:
    ns = df.groupby("student_id").session_id.transform("nunique")
    d = df[ns >= 2].reset_index(drop=True)
    out = {"n": int(len(d)), "students": int(d.student_id.nunique()),
           "sessions": int(d.session_id.nunique()), "arms": {}}
    for arm in ARMS:
        s = d[f"s_{arm}"]
        X = pd.DataFrame({"s": (s - s.mean()) / s.std(ddof=0)})
        if source == "tsl":
            pooled_abs, within_abs = {"question": d.question_id}, {"question": d.question_id,
                                                                    "student": d.student_id}
        else:
            X["z_lo"] = d.z_lo
            pooled_abs, within_abs = {"all": np.zeros(len(d), int)}, {"student": d.student_id}
        res = {}
        for name, ab in (("pooled", pooled_abs), ("within_student", within_abs)):
            t = within.fe_ols(d.y.to_numpy(float), X, ab, d.student_id).table().loc["s"]
            res[name] = {"pp_per_sd": 100 * float(t.estimate), "lo": 100 * float(t.lo),
                         "hi": 100 * float(t.hi), "p": float(t.p)}
        res["within_over_pooled"] = res["within_student"]["pp_per_sd"] / res["pooled"]["pp_per_sd"]
        out["arms"][arm] = res
        print(f"  {source} {arm:7s} pooled {res['pooled']['pp_per_sd']:+.2f} pp/SD "
              f"[{res['pooled']['lo']:+.2f}, {res['pooled']['hi']:+.2f}] | within student "
              f"{res['within_student']['pp_per_sd']:+.2f} [{res['within_student']['lo']:+.2f}, "
              f"{res['within_student']['hi']:+.2f}] | ratio {res['within_over_pooled']:.2f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="beyond-the-student")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-train", default="1,2,3,5,10",
                    help="design A is repeated on students with at least this many training responses")
    ap.add_argument("--refold", action="store_true",
                    help="recompute the student folds with GroupKFold and rewrite FOLDS (CPU-dependent)")
    a = ap.parse_args()
    ladder = [int(k) for k in a.min_train.split(",")]
    d = runlog.start(a.run, vars(a), [config.TRAIN_LABELS, config.TEST_LABELS, config.METADATA,
                                       *PRED.values(), *([] if a.refold else [FOLDS])])
    rng = np.random.default_rng(a.seed)
    s35 = importlib.import_module("35_student_heterogeneity")

    meta = data.load_metadata()
    tr = (data.load_train()
          .merge(meta[["response_id", "student_id", "question_id"]], on="response_id")
          .reset_index(drop=True))
    tr["res"] = tr.is_correct - s35.expected_correct(tr)
    attain, shrink = student_attainment(tr)
    print(f"attainment: {len(attain):,} training students, pseudo-count {shrink['pseudocount']:.0f} "
          f"responses", flush=True)

    te = data.load_test()
    for arm, f in PRED.items():
        p = pd.read_csv(f, dtype={"response_id": str}).set_index("response_id")
        if not np.array_equal(p.y.reindex(te.response_id).to_numpy(), te.is_correct.to_numpy(float)):
            raise SystemExit(f"{f}: labels do not match the test split")
        te[f"z_{arm}"] = logit(p.p.reindex(te.response_id).to_numpy())
    for arm in ARMS:
        te[f"s_{arm}"] = te[f"z_{arm}"] - te.z_lo
    te["y"] = te.is_correct
    te = te.merge(attain, left_on="student_id", right_index=True, how="left")

    tsl = te[te.data_source == "tsl"]
    seen = tsl[tsl.attainment.notna()].reset_index(drop=True)
    A, frozen, made = {}, (None if a.refold else pd.read_csv(FOLDS)), []
    for k in ladder:
        rows_k = seen[seen.n_train >= k].reset_index(drop=True)
        print(f"A  >= {k} training responses: {len(rows_k):,} TSL test responses from "
              f"{rows_k.student_id.nunique():,} students", flush=True)
        if a.refold:
            fold_of = student_folds(rows_k.student_id.to_numpy())
            made.append(pd.DataFrame({"min_train": k, "student_id": fold_of.index, "fold": fold_of.to_numpy()}))
        else:
            f = frozen[frozen.min_train == k]
            fold_of = pd.Series(f.fold.to_numpy(), index=f.student_id.to_numpy())
        A[k] = design_a(rows_k, a.boot, rng, fold_of)
    if a.refold:
        pd.concat(made).to_csv(FOLDS, index=False)
        print(f"wrote {FOLDS}", flush=True)
    print("B  within student on test", flush=True)
    B = {src: design_b(te[te.data_source == src].reset_index(drop=True), src) for src in ("tsl", "eedi")}

    rows = []
    for k, res in A.items():
        for arm, r in res["arms"].items():
            rows.append({"design": "A", "view": f">= {k} training responses", "arm": arm,
                         "n": res["n"], "students": res["students"],
                         "corr_s_attainment": r["corr_s_attainment"],
                         "d_ll_transcript_beyond_attainment": r["transcript_beyond_attainment"]["d_ll"],
                         "d_auroc_transcript_beyond_attainment": r["transcript_beyond_attainment"]["d_auroc"],
                         "d_ll_attainment_beyond_transcript": r["attainment_beyond_transcript"]["d_ll"],
                         "d_auroc_attainment_beyond_transcript": r["attainment_beyond_transcript"]["d_auroc"],
                         "s_share_retained": r["s_share_retained"],
                         "attainment_share_retained": r["attainment_share_retained"]})
    for src, res in B.items():
        for arm, r in res["arms"].items():
            rows.append({"design": "B", "view": src, "arm": arm, "n": res["n"],
                         "pooled_pp_per_sd": r["pooled"]["pp_per_sd"],
                         "within_pp_per_sd": r["within_student"]["pp_per_sd"],
                         "within_lo": r["within_student"]["lo"], "within_hi": r["within_student"]["hi"],
                         "within_over_pooled": r["within_over_pooled"]})
    pd.DataFrame(rows).to_csv(d / "summary.csv", index=False)
    runlog.finish(d, {"run": a.run, "attainment_shrinkage": shrink,
                      "A_by_min_train": {str(k): v for k, v in A.items()}, "B_within_student": B})


if __name__ == "__main__":
    main()
