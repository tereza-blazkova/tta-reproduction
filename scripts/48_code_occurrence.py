#!/usr/bin/env python
"""Stage 48 — how often each code occurs in the sessions of weak and of strong students.

The occurrence of a code is its share of a role's labelled turns in a session (tutor moves: share
of tutor turns; student signals: share of student turns; 122B labels). Per student, the mean share
over their analysed sessions; per group of students, the mean over its students.

Students are grouped by question-adjusted attainment (`attainment.expected_correct`; correct minus the
question's usual rate, summed and shrunk with `attainment.M_ATTAINMENT`), as in stage 41: training
students with four or more sessions, each student's sessions split at random in two, one half
ranks the student, the other half gives the occurrences. Ranking and measuring on different
sessions keeps a code that goes with correct answers in a session (feedback correct) from
manufacturing its own link to attainment. Ten random splits, each used both ways, averaged.

Groups: thirds (weak, middle, strong; the strong-vs-weak contrast is stage 41's baseline, and this
stage reproduces it exactly from the same seed: same splits, same bootstrap draws, contrast
computed within each split and then averaged) and fifths of attainment, for the gradient.
Contrasts in % of the weak group's occurrence: strong vs weak third, top vs bottom fifth.
Intervals: 1,000 bootstrap draws over students (groups held fixed); p-values from the bootstrap
standard error; Benjamini-Hochberg across the 57 codes.

    python scripts/48_code_occurrence.py --run code-occurrence
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

GROUPS = ["weak", "middle", "strong"] + [f"q{i}" for i in range(1, 6)] + ["all"]
CONTRASTS = {"strong_vs_weak": ("strong", "weak"), "q5_vs_q1": ("q5", "q1")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="code-occurrence")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-sessions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    shards = sorted(config.LABELS_V3.glob("v3_full_train_*_shard*.csv"))
    d = runlog.start(a.run, vars(a), [config.TRAIN_LABELS, config.METADATA, *shards])
    rng = np.random.default_rng(a.seed)
    s40 = importlib.import_module("40_four_cells")

    # same rows, same order and same random draws as stage 41
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
    print(f"{n_st:,} students with >= {a.min_sessions} sessions, {len(sess):,} sessions", flush=True)

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
            fifth = np.minimum((r - 1) * 5 // n_st, 4).astype(int) + 1
            rows = np.flatnonzero(B)
            g = st_codes[rows]
            cnt = np.bincount(g, minlength=n_st)
            mean_S = np.stack([np.bincount(g, S[rows, j], n_st) for j in range(S.shape[1])], 1) / np.maximum(cnt, 1)[:, None]
            has = cnt > 0
            member = {"weak": third == -1, "middle": third == 0, "strong": third == 1, "all": np.ones(n_st, bool)}
            member |= {f"q{i}": fifth == i for i in range(1, 6)}
            reps.append({"mean_S": mean_S, "member": {k_: m & has for k_, m in member.items()}})

    ROWS = GROUPS + list(CONTRASTS)

    def estimate(w: np.ndarray) -> np.ndarray:
        """(len(ROWS), n_codes): mean occurrence per group, then each contrast in %; every row is
        computed within a repetition and averaged over repetitions (stage 41 averages its baseline
        contrast the same way)."""
        acc = 0
        for rp in reps:
            E = np.stack([(w * rp["member"][gname]) @ rp["mean_S"] / (w * rp["member"][gname]).sum()
                          for gname in GROUPS])
            C = [(E[GROUPS.index(hi)] / E[GROUPS.index(lo)] - 1) * 100 for hi, lo in CONTRASTS.values()]
            acc = acc + np.vstack([E, *C])
        return acc / len(reps)

    def contrast(E: np.ndarray, cname: str) -> np.ndarray:
        return E[..., ROWS.index(cname), :]

    point = estimate(np.ones(n_st))
    draws = []
    for b in range(a.boot):
        draws.append(estimate(np.bincount(rng.integers(0, n_st, n_st), minlength=n_st).astype(float)))
        if (b + 1) % 250 == 0:
            print(f"  bootstrap {b + 1}/{a.boot}", flush=True)
    D = np.stack(draws)

    name = {agg.column(k, r): (k.lower(), r) for r in ("tutor", "student") for k in T.codes(r)}
    out = pd.DataFrame({"code": [name[c][0] for c in cols], "role": [name[c][1] for c in cols]})
    for i, gname in enumerate(GROUPS):
        out[f"share_{gname}"] = point[i]
    for cname in CONTRASTS:
        est, dist = contrast(point, cname), contrast(D, cname)
        sd = dist.std(axis=0, ddof=1)
        p = 2 * norm.sf(np.abs(est) / sd)
        out[cname] = est
        out[f"{cname}_lo"], out[f"{cname}_hi"] = np.percentile(dist, [2.5, 97.5], axis=0)
        out[f"{cname}_p"] = p
        out[f"{cname}_bh"] = within.benjamini_hochberg(p, 0.05)
    # gradient: Spearman of occurrence on the fifths' rank, per code (5 points; descriptive)
    q = out[[f"share_q{i}" for i in range(1, 6)]].to_numpy()
    out["monotone"] = [int(np.all(np.diff(row) > 0)) - int(np.all(np.diff(row) < 0)) for row in q]
    out.to_csv(d / "codes.csv", index=False)

    # parity with stage 41's baseline (same seed, same splits, same bootstrap draws)
    s41 = config.RESULTS / "outperform-self" / "codes.csv"
    parity = None
    if s41.exists():
        b41 = pd.read_csv(s41).query("panel == 'baseline'").set_index("code")
        diff = (out.set_index("code").strong_vs_weak - b41.estimate.reindex(out.code).to_numpy()).abs().max()
        parity = float(diff)
        print(f"max |strong_vs_weak - stage 41 baseline| = {diff:.2e}", flush=True)
        if diff > 1e-8:
            raise SystemExit("parity with stage 41's baseline failed")

    summary = {c: int(out[f"{c}_bh"].sum()) for c in CONTRASTS}
    both = out.strong_vs_weak_bh & out.q5_vs_q1_bh & (np.sign(out.strong_vs_weak) == np.sign(out.q5_vs_q1))
    print(f"BH: {summary}; both, same sign: {int(both.sum())}; monotone up {int((out.monotone == 1).sum())}, "
          f"down {int((out.monotone == -1).sum())}", flush=True)
    for _, r in out.sort_values("strong_vs_weak").iterrows():
        print(f"  {r.code:32s} {r.role:7s} all {100 * r.share_all:6.2f}%  weak {100 * r.share_weak:6.2f}%  "
              f"strong {100 * r.share_strong:6.2f}%  {r.strong_vs_weak:+6.1f}% [{r.strong_vs_weak_lo:+.1f}, "
              f"{r.strong_vs_weak_hi:+.1f}]{' *' if r.strong_vs_weak_bh else '  '}  q5/q1 {r.q5_vs_q1:+6.1f}%"
              f"{' *' if r.q5_vs_q1_bh else '  '} mono {r.monotone:+d}", flush=True)
    runlog.finish(d, {"run": a.run, "students": int(n_st), "sessions": int(len(sess)), "splits": a.splits,
                      "boot": a.boot, "bh": summary, "both_same_sign": int(both.sum()),
                      "monotone_up": int((out.monotone == 1).sum()), "monotone_down": int((out.monotone == -1).sum()),
                      "parity_stage41_max_abs": parity})


if __name__ == "__main__":
    main()
