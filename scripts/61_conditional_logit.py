#!/usr/bin/env python
"""Stage 61 — the within-student result as a conditional logit, and who it drops.

Stage 15's within-student design is a linear probability model with student and question fixed
effects, which keeps every response. This stage checks that its result is not an artefact of the
linear model. A conditional (fixed-effects) logit conditions on each student's number of
correct answers, so the student effect drops out exactly instead of being estimated. Students
whose outcome never varies (all correct or all wrong) carry no information and are dropped.

Three things, per code (57 codes, one at a time, Benjamini--Hochberg at 0.05 across codes):

  1. ``lpm all``   stage 15's design 2 (student + question FE, student-clustered SE), all rows.
  2. ``lpm kept``  the same, on the students the logit keeps, so a change between 1 and 3 can be
                   split into "different students" and "different model".
  3. ``clogit``    conditional logit grouped by student, on the kept students. Question
                   difficulty enters as one covariate, the logit of the question's correct rate
                   among other students (shrunk to the overall rate with weight ``M_Q``),
                   because 1,015 question dummies are not estimable inside a conditional logit.
                   Standard errors are a student-clustered sandwich: a code's rate is constant
                   within a session, so a student's responses are not independent given the
                   model, and model-based errors (also written) are too small.
  2b. ``lpm kept, difficulty covariate``  the linear model with that same covariate in place
                   of question FE, so 2b against 3 isolates logit against linear.

Each code is standardised over all training rows, exactly as in stage 15, so ``clogit``'s
coefficient is a log-odds change per standard deviation and the LPM's a probability change.

Also written: the kept and dropped students side by side (sessions, responses, share correct,
and every code's mean rate standardised).

    python scripts/61_conditional_logit.py --run within-clogit --jobs 32
"""
from __future__ import annotations

import argparse
import importlib
from concurrent.futures import ProcessPoolExecutor

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy import optimize, stats

from ttapub import agg, config, data, runlog, within

s15 = importlib.import_module("15_rq1_within_session")
M_Q = 10.0  # prior weight of the overall rate in a question's leave-own-student-out rate

_G: dict = {}


def question_logit(tr: pd.DataFrame) -> np.ndarray:
    """Logit of each question's correct rate among OTHER students, shrunk toward the mean."""
    y = tr.is_correct.astype(float)
    g = y.mean()
    q_sum = y.groupby(tr.question_id).transform("sum")
    q_n = y.groupby(tr.question_id).transform("size")
    qs_sum = y.groupby([tr.question_id, tr.student_id]).transform("sum")
    qs_n = y.groupby([tr.question_id, tr.student_id]).transform("size")
    r = (q_sum - qs_sum + M_Q * g) / (q_n - qs_n + M_Q)
    return np.log(r / (1 - r)).to_numpy()


def pad(groups: np.ndarray, *cols: np.ndarray):
    """Rows into a (students x max responses) layout, with a mask for the padding."""
    _, inv, size = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    pos = np.empty(len(groups), int)
    starts = np.r_[0, np.cumsum(size)[:-1]]
    pos[order] = np.arange(len(groups)) - np.repeat(starts, size)
    shape = (len(size), size.max())
    out = []
    for c in cols:
        a = np.zeros(shape)
        a[inv, pos] = c
        out.append(a)
    mask = np.zeros(shape, bool)
    mask[inv, pos] = True
    return out, mask


def group_ll(beta, X, y, mask, k):
    """Each student's conditional log likelihood. X: (G, n, p); y, mask: (G, n); k: (G,).

    The denominator is the elementary symmetric polynomial of degree k in exp(eta), built one
    response at a time in log space, so no student's term can underflow.
    """
    eta = np.where(mask, X @ beta, -np.inf)  # padding has weight exp(-inf) = 0
    L = np.full((len(k), k.max() + 1), -np.inf)
    L[:, 0] = 0.0
    for j in range(eta.shape[1]):
        L[:, 1:] = np.logaddexp(L[:, 1:], eta[:, j:j + 1] + L[:, :-1])
    num = (np.where(mask, eta, 0.0) * y).sum(axis=1)
    return num - L[np.arange(len(k)), k]


def negll(beta, X, y, mask, k):
    return float(-group_ll(beta, X, y, mask, k).sum())


def hessian(f, x, h=1e-4):
    p = len(x)
    H = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            ei, ej = np.eye(p)[i] * h, np.eye(p)[j] * h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def fit_code(c: str) -> dict:
    g = _G
    X = np.stack([g["Z"][c], g["q"]], axis=-1)
    args = (X, g["y"], g["mask"], g["k"])
    f = lambda b: negll(b, *args)  # noqa: E731
    r = optimize.minimize(f, np.zeros(2), method="BFGS", options={"gtol": 1e-6})
    Hinv = np.linalg.inv(hessian(f, r.x))
    # student-clustered sandwich: a code's rate is constant within a session, so responses of
    # one student are not independent given the model, and the model-based SE is too small
    h = 1e-5
    S = np.stack([(group_ll(r.x + h * e, *args) - group_ll(r.x - h * e, *args)) / (2 * h)
                  for e in np.eye(2)], axis=1)
    V = Hinv @ (S.T @ S) @ Hinv * len(S) / (len(S) - 1)
    se, se_model = float(np.sqrt(V[0, 0])), float(np.sqrt(Hinv[0, 0]))
    b = float(r.x[0])
    return {"column": c, "logodds_per_sd": b, "se": se, "se_model": se_model,
            "lo": b - 1.96 * se, "hi": b + 1.96 * se,
            "p": float(2 * stats.norm.sf(abs(b / se))),
            "p_model": float(2 * stats.norm.sf(abs(b / se_model))),
            "or_per_sd": float(np.exp(b)), "q_coef": float(r.x[1]),
            "converged": bool(r.success), "grad_norm": float(np.abs(S.sum(axis=0)).max())}


def _init(G):
    _G.update(G)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="within-clogit")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.FEATURES_TRAIN])

    meta = data.load_metadata()
    tr = (data.load_train().merge(meta[["response_id", "student_id", "question_id"]],
                                  on="response_id").reset_index(drop=True))
    sess = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    cols = agg.code_columns(sess.columns)
    X = tr[["session_id"]].merge(sess, on="session_id", how="left")
    Z = s15.standardise(X[cols])  # over ALL rows, as stage 15
    y = tr.is_correct.to_numpy(float)

    varies = tr.groupby("student_id").is_correct.transform("nunique").to_numpy() > 1
    print(f"rows {len(tr):,}, students {tr.student_id.nunique():,}; kept by the logit "
          f"{varies.sum():,} rows / {tr.student_id[varies].nunique():,} students")

    res = [s15.run_design(y, Z, {"student": tr.student_id, "question": tr.question_id},
                          tr.student_id, cols, "lpm all")]
    k_ = varies
    res.append(s15.run_design(y[k_], Z[k_].reset_index(drop=True),
                              {"student": tr.student_id[k_].reset_index(drop=True),
                               "question": tr.question_id[k_].reset_index(drop=True)},
                              tr.student_id[k_].reset_index(drop=True), cols, "lpm kept"))

    q = question_logit(tr)[k_]
    sid = tr.student_id[k_].reset_index(drop=True)
    Mq = pd.concat([pd.Series(y[k_], name="_y"), pd.Series(q, name="_q"),
                    Z[k_].reset_index(drop=True)], axis=1)
    Md, n_abs = within.absorb_matrix(Mq, {"student": sid})
    rows = []
    for c in cols:
        t = within.ols_demeaned(Md["_y"].to_numpy(), Md[[c, "_q"]], sid, n_abs).table().loc[c]
        rows.append({"column": c, "design": "lpm kept, difficulty covariate",
                     "pp_per_sd": 100 * t.estimate, "lo": 100 * t.lo, "hi": 100 * t.hi,
                     "p": t.p})
    lq = pd.DataFrame(rows)
    lq["bh_sig"] = within.benjamini_hochberg(lq.p.to_numpy(), 0.05)
    print(f"  {'lpm kept, difficulty cov.':26s} {int(lq.bh_sig.sum()):2d}/{len(lq)} survive BH")
    res.append(lq)
    grp = tr.student_id.to_numpy()[k_]
    (yy, qq, *zz), mask = pad(grp, y[k_], q, *[Z[c].to_numpy()[k_] for c in cols])
    G = {"y": yy, "q": qq, "mask": mask, "k": yy.sum(axis=1).astype(int),
         "Z": dict(zip(cols, zz))}
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(G,)) as ex:
        cl = pd.DataFrame(list(ex.map(fit_code, cols)))
    cl["design"] = "clogit"
    cl["bh_sig"] = within.benjamini_hochberg(cl.p.to_numpy(), 0.05)
    cl["bh_sig_model_se"] = within.benjamini_hochberg(cl.p_model.to_numpy(), 0.05)
    print(f"  {'clogit':26s} {int(cl.bh_sig.sum()):2d}/{len(cl)} survive BH "
          f"({int(cl.bh_sig_model_se.sum())} with model-based SE) | "
          f"max |score| at optimum {cl.grad_norm.max():.2g}")

    lpm = pd.concat(res, ignore_index=True)
    lpm.to_csv(d / "lpm.csv", index=False)
    cl.to_csv(d / "clogit.csv", index=False)

    # kept vs dropped students
    st = tr.assign(kept=varies).groupby("student_id").agg(
        kept=("kept", "first"), responses=("is_correct", "size"),
        sessions=("session_id", "nunique"), share_correct=("is_correct", "mean"))
    side = st.groupby("kept").agg(students=("responses", "size"),
                                  responses_mean=("responses", "mean"),
                                  sessions_mean=("sessions", "mean"),
                                  sessions_median=("sessions", "median"),
                                  share_correct_mean=("share_correct", "mean"))
    dropped = st[~st.kept]
    side.loc[False, "all_correct"] = int((dropped.share_correct == 1).sum())
    side.loc[False, "all_wrong"] = int((dropped.share_correct == 0).sum())
    side.to_csv(d / "kept_dropped.csv")
    codes_gap = (Z[varies].mean() - Z[~varies].mean()).rename("kept_minus_dropped_sd")
    codes_gap.to_csv(d / "kept_dropped_codes.csv")

    wide = {des: g.set_index("column") for des, g in lpm.groupby("design")}
    a_, kp, kq = wide["lpm all"], wide["lpm kept"], wide["lpm kept, difficulty covariate"]
    cl_ = cl.set_index("column")
    surv = {des: sorted(g.index[g.bh_sig]) for des, g in
            {"lpm all": a_, "lpm kept": kp, "lpm kept, difficulty covariate": kq,
              "clogit": cl_}.items()}
    same_sign = lambda s, t: int((np.sign(s) == np.sign(t)).sum())  # noqa: E731
    out = {
        "run": a.run, "m_q": M_Q, "n_rows": int(len(tr)), "n_rows_kept": int(varies.sum()),
        "n_students": int(tr.student_id.nunique()),
        "n_students_kept": int(tr.student_id[varies].nunique()),
        "survivors": surv, "n_survivors": {k: len(v) for k, v in surv.items()},
        "raw_p05": {"lpm all": int((a_.p < 0.05).sum()), "lpm kept": int((kp.p < 0.05).sum()),
                    "lpm kept, difficulty covariate": int((kq.p < 0.05).sum()),
                    "clogit": int((cl_.p < 0.05).sum())},
        "n_survivors_clogit_model_se": int(cl.bh_sig_model_se.sum()),
        "same_sign_clogit_vs_lpm_cov": same_sign(cl_.logodds_per_sd, kq.pp_per_sd[cl_.index]),
        "spearman_clogit_vs_lpm_cov": float(stats.spearmanr(cl_.logodds_per_sd,
                                                            kq.pp_per_sd[cl_.index])[0]),
        "same_sign_clogit_vs_lpm_all": same_sign(cl_.logodds_per_sd, a_.pp_per_sd[cl_.index]),
        "spearman_clogit_vs_lpm_all": float(stats.spearmanr(cl_.logodds_per_sd,
                                                            a_.pp_per_sd[cl_.index])[0]),
        "spearman_lpm_kept_vs_all": float(stats.spearmanr(kp.pp_per_sd,
                                                          a_.pp_per_sd[kp.index])[0]),
        "lpm_all_survivors_in_clogit": {c: {"pp_per_sd": float(a_.pp_per_sd[c]),
                                            "lpm_kept_pp": float(kp.pp_per_sd[c]),
                                            "lpm_kept_bh": bool(kp.bh_sig[c]),
                                            "clogit_logodds": float(cl_.logodds_per_sd[c]),
                                            "clogit_ci": [float(cl_.lo[c]), float(cl_.hi[c])],
                                            "clogit_p": float(cl_.p[c]),
                                            "clogit_bh": bool(cl_.bh_sig[c])}
                                        for c in surv["lpm all"]},
        "kept_dropped": side.reset_index().to_dict("records"),
        "all_converged": bool(cl.converged.all()),
        "max_score_at_optimum": float(cl.grad_norm.max()),
    }
    runlog.finish(d, out)
    print(out["n_survivors"], out["raw_p05"])
    for c, v in out["lpm_all_survivors_in_clogit"].items():
        print(f"  {c:40s} lpm {v['pp_per_sd']:+.2f} kept {v['lpm_kept_pp']:+.2f} "
              f"clogit {v['clogit_logodds']:+.3f} [{v['clogit_ci'][0]:+.3f},{v['clogit_ci'][1]:+.3f}] "
              f"bh {v['clogit_bh']}")


if __name__ == "__main__":
    main()
