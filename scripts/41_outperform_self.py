#!/usr/bin/env python
"""Stage 41 — which codes go with a student beating their own usual, among strong and among weak
students, once the codes that only mark a good student are taken out?

Three numbers per code, from the same students and the same splits:

  **baseline**  How much more of the code strong students' sessions carry than weak students'
                (each student's mean share, averaged over the students of a third). This is "who
                gets which codes": a code can be high here only because good students use or
                receive it.
  **strong**    Among strong students: compared with that student's own average, how much more of
                the code is in the sessions where they did better than their own usual, against
                the sessions where they did worse.
  **weak**      The same among weak students.

Session-level, objective-aware. A session's performance is the mean over its answers of (correct
minus the question's usual correctness rate; `attainment.expected_correct`). "Better than their own
usual" means that number is above the student's own mean over the sessions analysed; a session's
code use is likewise its share minus the student's own mean share. Subtracting the student's own
averages on both sides removes everything stable about the student, so a code that only marks
good students contributes nothing to `strong` or `weak`. Contrasts are expressed in % of the
code's mean share in that third.

Split halves, as in stages 35 and 40. Training students with four or more sessions; each
student's sessions are split at random in two. One half ranks the student (question-adjusted
attainment, top and bottom third). The other half, at least two sessions per student, gives all three
numbers. Ten random splits, each used both ways, averaged. Intervals: 1,000 bootstrap draws over
students (thirds held fixed); p-values from the bootstrap standard error; Benjamini-Hochberg across
the 57 codes per panel.

Shares are raw shares of a role's labelled turns (122B labels). The contrasts are associations
with a better or worse session than usual, not effects: a tutor praises an answer because it was
right.

    python scripts/41_outperform_self.py --run outperform-self
"""
from __future__ import annotations

import argparse
import importlib

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import norm

from ttapub import attainment, agg, config, data, runlog, within
from ttapub import taxonomy_v3 as T

PANELS = ("baseline", "strong", "weak", "strong_minus_weak")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outperform-self")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-sessions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    shards = sorted(config.LABELS_V3.glob("v3_full_train_*_shard*.csv"))
    d = runlog.start(a.run, vars(a), [config.TRAIN_LABELS, config.METADATA, *shards])
    rng = np.random.default_rng(a.seed)
    s40 = importlib.import_module("40_four_cells")

    meta = data.load_metadata()
    tr = (data.load_train().merge(meta[["response_id", "student_id", "question_id"]], on="response_id")
          .reset_index(drop=True))
    tr["res"] = tr.is_correct - attainment.expected_correct(tr)
    sess = tr.groupby("session_id").agg(student=("student_id", "first"), res_sum=("res", "sum"),
                                        n=("res", "size"))
    S = s40.shares("train", sess.index)
    cols = list(S.columns)
    sess = sess[S.notna().all(axis=1)]
    sess = sess[sess.groupby("student").n.transform("size") >= a.min_sessions]
    S = S.loc[sess.index].to_numpy()
    st_codes, students = pd.factorize(sess.student)
    n_st = len(students)
    perf = (sess.res_sum / sess.n).to_numpy()
    print(f"{n_st:,} students with >= {a.min_sessions} sessions, {len(sess):,} sessions", flush=True)

    # the splits: per repetition, analysed sessions, their student's third, within-student deviations
    reps = []
    for _ in range(a.splits):
        order = rng.permutation(len(sess))
        rank_in_student = pd.Series(np.arange(len(order)), index=order).groupby(st_codes[order]).cumcount()
        half = (rank_in_student.reindex(np.arange(len(sess))).to_numpy()) % 2
        for h in (0, 1):
            A, B = half == h, half != h
            res_A = np.bincount(st_codes[A], sess.res_sum.to_numpy()[A], n_st)
            n_A = np.bincount(st_codes[A], sess.n.to_numpy()[A], n_st)
            attain = res_A / (n_A + attainment.M_ATTAINMENT)
            r = pd.Series(attain + rng.random(n_st) * 1e-12).rank(method="first").to_numpy()
            k = n_st // 3
            third = np.where(r > n_st - k, 1, np.where(r <= k, -1, 0))
            rows = np.flatnonzero(B & (third[st_codes] != 0))
            g = st_codes[rows]
            cnt = np.bincount(g, minlength=n_st)
            mean_S = np.stack([np.bincount(g, S[rows, j], n_st) for j in range(S.shape[1])], 1) / np.maximum(cnt, 1)[:, None]
            mean_p = np.bincount(g, perf[rows], n_st) / np.maximum(cnt, 1)
            dS = S[rows] - mean_S[g]
            dp = perf[rows] - mean_p[g]
            reps.append({"rows": rows, "student": g, "strong": third[g] == 1, "above": dp > 0,
                         "below": dp < 0, "dS": dS, "S": S[rows], "third": third,
                         "mean_S": mean_S, "has": cnt > 0})

    def estimate(w: np.ndarray) -> np.ndarray:
        """(4, n_codes) averaged over repetitions; w = bootstrap multiplicity per student."""
        acc = 0
        for rp in reps:
            third, has = rp["third"], rp["has"]
            ws = w * has
            out = []
            # baseline: each student's mean share, averaged over the students of a third
            m_strong = (ws * (third == 1)) @ rp["mean_S"] / (ws * (third == 1)).sum()
            m_weak = (ws * (third == -1)) @ rp["mean_S"] / (ws * (third == -1)).sum()
            out.append((m_strong / m_weak - 1) * 100)
            wr = w[rp["student"]]
            for grp, base in ((rp["strong"], m_strong), (~rp["strong"], m_weak)):
                up, dn = wr * (grp & rp["above"]), wr * (grp & rp["below"])
                diff = up @ rp["dS"] / up.sum() - dn @ rp["dS"] / dn.sum()
                out.append(diff / base * 100)
            out.append(out[1] - out[2])
            acc = acc + np.stack(out)
        return acc / len(reps)

    point = estimate(np.ones(n_st))
    draws = []
    for b in range(a.boot):
        draws.append(estimate(np.bincount(rng.integers(0, n_st, n_st), minlength=n_st).astype(float)))
        if (b + 1) % 250 == 0:
            print(f"  bootstrap {b + 1}/{a.boot}", flush=True)
    D = np.stack(draws)

    name = {agg.column(k, r): (k.lower(), r) for r in ("tutor", "student") for k in T.codes(r)}
    rows = []
    for i, panel in enumerate(PANELS):
        sd = D[:, i, :].std(axis=0, ddof=1)
        lo, hi = np.percentile(D[:, i, :], [2.5, 97.5], axis=0)
        p = 2 * norm.sf(np.abs(point[i]) / sd)
        bh = within.benjamini_hochberg(p, 0.05)
        for j, c in enumerate(cols):
            rows.append({"code": name[c][0], "role": name[c][1], "panel": panel, "estimate": point[i, j],
                         "lo": lo[j], "hi": hi[j], "p": p[j], "bh": bool(bh[j])})
    tab = pd.DataFrame(rows)
    tab.to_csv(d / "codes.csv", index=False)

    w = tab.pivot(index="code", columns="panel", values="estimate")
    sig = tab.pivot(index="code", columns="panel", values="bh")
    summary = {p: int(sig[p].sum()) for p in PANELS}
    corr = {"baseline_vs_strong": float(w.baseline.corr(w.strong)),
            "baseline_vs_weak": float(w.baseline.corr(w.weak)),
            "strong_vs_weak": float(w.strong.corr(w.weak))}
    print(f"BH per panel: {summary}; correlations across codes {corr}", flush=True)
    for p in ("strong", "weak", "strong_minus_weak"):
        s = tab[(tab.panel == p) & tab.bh].sort_values("estimate")
        print(f"\n{p}: {len(s)} codes clear BH", flush=True)
        for _, r in s.iterrows():
            b = w.loc[r.code, "baseline"]
            print(f"  {r.code:28s} {r.role:7s} {r.estimate:+6.1f}% [{r.lo:+.1f}, {r.hi:+.1f}]   baseline "
                  f"{b:+6.1f}%", flush=True)
    runlog.finish(d, {"run": a.run, "students": int(n_st), "sessions": int(len(sess)),
                      "splits": a.splits, "boot": a.boot, "bh": summary, "correlations": corr})


if __name__ == "__main__":
    main()
