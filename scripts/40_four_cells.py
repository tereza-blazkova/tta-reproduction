#!/usr/bin/env python
"""Stage 40 — within strong and within weak students, do sessions that go well look different
from sessions that go badly?

Students are split into the top and bottom third by question-adjusted attainment. Each of their
answers is then correct or incorrect, giving four cells:

                    correct               incorrect
    strong          expected              unexpected (strong student fails)
    weak            unexpected            expected
                    (weak student succeeds)

Per code, three contrasts of the code's share in the session behind the answer:
  * strong: correct against incorrect answers of strong students;
  * weak:   correct against incorrect answers of weak students;
  * difference: strong minus weak. A code whose link with success depends on the kind of student
    (a hint that helps weak students and not strong ones) shows up here.
Each is a regression of the code's share on the cell indicators with question fixed effects, so a
cell answered on harder questions does not differ for that reason; contrasts are expressed in % of
the code's mean share among these answers, so rare and common codes read on one scale.

**Main analysis, training, split halves.** Training students with four or more sessions. Each
student's sessions are split at random into halves: one half ranks the student (attainment on that
half's answers), the other half supplies the answers and codes of the four cells, so no session
both places a student in a third and appears in a cell. Ten random splits, each used both ways.
Intervals: 400 bootstrap draws over students, every split re-estimated per draw; Benjamini-
Hochberg across the 57 codes per contrast. p-values come from the bootstrap standard error (normal
approximation): a percentile p-value from 400 draws cannot go below 1/400, which is above the
first Benjamini-Hochberg threshold for 57 codes (0.05/57), so it would hide every contrast.

**Check, released test.** TSL test students with three or more training sessions, ranked by
attainment on all their training answers; the cells are their test answers. 2,000 student draws.

Shares are raw shares of a role's labelled turns (the 122B's labels), as in stage 35. The answer
is the unit and a session's share repeats over its answers; whole students are resampled. The
contrasts are associations: a tutor confirms an answer because the student just gave it.

    python scripts/40_four_cells.py --run four-cells
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import norm

from ttapub import attainment, agg, config, data, runlog, within
from ttapub import taxonomy_v3 as T

ROLES = ("tutor", "student")
CONTRASTS = ("strong", "weak", "difference")


def shares(split: str, sessions: pd.Index) -> pd.DataFrame:
    """Raw share of each code among a role's labelled turns, per session (NaN: no such turns)."""
    out = []
    for role in ROLES:
        files = sorted(config.LABELS_V3.glob(f"v3_full_{split}_{role}_shard*.csv"))
        lab = (pd.concat([pd.read_csv(f, dtype={"session_id": str}, usecols=["session_id", "turn", "code"])
                          for f in files], ignore_index=True).drop_duplicates(["session_id", "turn"]))
        cnt = pd.crosstab(lab.session_id, lab.code).reindex(index=sessions, columns=T.codes(role),
                                                            fill_value=0)
        n = cnt.sum(axis=1)
        sh = cnt.div(n.where(n > 0), axis=0)
        sh.columns = [agg.column(c, role) for c in sh.columns]
        out.append(sh)
    return pd.concat(out, axis=1)


def demean(M: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Subtract question means (one fixed effect: a single pass is exact)."""
    g = pd.factorize(q)[0]
    cnt = np.bincount(g)
    means = np.stack([np.bincount(g, M[:, j]) for j in range(M.shape[1])], axis=1) / cnt[:, None]
    return M - means[g]


def contrasts(S: np.ndarray, strong: np.ndarray, correct: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(3, n_codes): strong, weak and difference contrasts in % of each code's mean share."""
    X = np.c_[strong, correct * strong, correct * (1 - strong)].astype(float)
    Z = demean(np.c_[S, X], q)
    Sd, Xd = Z[:, :S.shape[1]], Z[:, S.shape[1]:]
    B = np.linalg.lstsq(Xd, Sd, rcond=None)[0]          # rows: strong, correct|strong, correct|weak
    base = S.mean(0)
    out = np.stack([B[1], B[2], B[1] - B[2]]) / base * 100
    return out


def thirds(a: pd.Series, rng) -> pd.Series:
    """+1 top third, -1 bottom third, 0 middle; ties at random."""
    r = (a + rng.random(len(a)) * 1e-9).rank(method="first")
    k = len(a) // 3
    return pd.Series(np.where(r > len(a) - k, 1, np.where(r <= k, -1, 0)), index=a.index)


def summarise(point: np.ndarray, draws: np.ndarray, cols: list[str], tag: str) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(CONTRASTS):
        d = draws[:, i, :]
        lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        p = 2 * norm.sf(np.abs(point[i]) / d.std(axis=0, ddof=1))
        bh = within.benjamini_hochberg(p, 0.05)
        for j, col in enumerate(cols):
            rows.append({"column": col, "contrast": c, f"{tag}": point[i, j], f"{tag}_lo": lo[j],
                         f"{tag}_hi": hi[j], f"{tag}_p": p[j], f"{tag}_bh": bool(bh[j])})
    return pd.DataFrame(rows).set_index(["column", "contrast"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="four-cells")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--boot-test", type=int, default=2000)
    ap.add_argument("--min-sessions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    shards = sorted(config.LABELS_V3.glob("v3_full_*_shard*.csv"))
    d = runlog.start(a.run, vars(a), [config.TRAIN_LABELS, config.TEST_LABELS, config.METADATA, *shards])
    rng = np.random.default_rng(a.seed)
    meta = data.load_metadata()
    tr = (data.load_train().merge(meta[["response_id", "student_id", "question_id"]], on="response_id")
          .reset_index(drop=True))
    tr["res"] = tr.is_correct - attainment.expected_correct(tr)
    ns = tr.groupby("student_id").session_id.transform("nunique")
    t4 = tr[ns >= a.min_sessions].reset_index(drop=True)
    S_tr = shares("train", pd.Index(t4.session_id.unique()))
    cols = list(S_tr.columns)
    ok = S_tr.reindex(t4.session_id).notna().all(axis=1).to_numpy()
    t4 = t4[ok].reset_index(drop=True)
    S4 = S_tr.reindex(t4.session_id).to_numpy()
    print(f"training: {t4.student_id.nunique():,} students with >= {a.min_sessions} sessions, "
          f"{len(t4):,} answers", flush=True)

    # the splits: per repetition, which rows are in the cells and each row's third
    reps = []
    sess = t4.groupby("session_id").student_id.first()
    for _ in range(a.splits):
        order = sess.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        half = order.groupby(order).cumcount() % 2      # alternate a student's shuffled sessions
        for h in (0, 1):
            rank_sess = set(half.index[half == h])
            in_rank = t4.session_id.isin(rank_sess).to_numpy()
            r = t4[in_rank].groupby("student_id").res.agg(["sum", "size"])
            third = thirds(r["sum"] / (r["size"] + attainment.M_ATTAINMENT), rng)
            g = t4.student_id.map(third).fillna(0).to_numpy()
            rows = np.flatnonzero(~in_rank & (g != 0))
            reps.append((rows, (g[rows] == 1).astype(int)))
    y = t4.is_correct.to_numpy()
    q = t4.question_id.to_numpy()

    def estimate(idx_map=None) -> np.ndarray:
        acc = 0
        for rows, strong in reps:
            if idx_map is not None:
                sel = idx_map(rows)
                rows_b, strong_b = rows[sel], strong[sel]
            else:
                rows_b, strong_b = rows, strong
            acc = acc + contrasts(S4[rows_b], strong_b, y[rows_b], q[rows_b])
        return acc / len(reps)

    point = estimate()
    students = t4.student_id.to_numpy()
    uniq, inv = np.unique(students, return_inverse=True)
    draws = []
    for b in range(a.boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        mult = w[inv]

        def idx_map(rows, mult=mult):
            return np.repeat(np.arange(len(rows)), mult[rows])
        draws.append(estimate(idx_map))
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{a.boot}", flush=True)
    train_tab = summarise(point, np.stack(draws), cols, "train")

    # check on the released test set
    te = data.load_test().reset_index(drop=True)
    _, a_te = attainment.train_and_test(tr, te.student_id)
    te["attainment"] = a_te
    te["k"] = te.student_id.map(tr.groupby("student_id").session_id.nunique()).fillna(0)
    h = te[(te.data_source == "tsl") & (te.k >= 3)].reset_index(drop=True)
    third = thirds(h.groupby("student_id").attainment.first(), rng)
    h["g"] = h.student_id.map(third)
    h = h[h.g != 0].reset_index(drop=True)
    S_te = shares("test", pd.Index(h.session_id.unique())).reindex(h.session_id)
    keep = S_te.notna().all(axis=1).to_numpy()
    h, St = h[keep].reset_index(drop=True), S_te[keep].to_numpy()
    st_strong, yt, qt = (h.g == 1).astype(int).to_numpy(), h.is_correct.to_numpy(), h.question_id.to_numpy()
    tpoint = contrasts(St, st_strong, yt, qt)
    u, iv = np.unique(h.student_id.to_numpy(), return_inverse=True)
    tdraws = []
    for _ in range(a.boot_test):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        ix = np.repeat(np.arange(len(h)), w[iv])
        tdraws.append(contrasts(St[ix], st_strong[ix], yt[ix], qt[ix]))
    test_tab = summarise(tpoint, np.stack(tdraws), cols, "test")
    cells_test = h.groupby([h.g.map({1: "strong", -1: "weak"}), "is_correct"]).size().to_dict()
    print(f"test: {h.student_id.nunique():,} students, {len(h):,} answers; cells "
          f"{ {f'{k[0]} {int(k[1])}': v for k, v in cells_test.items()} }", flush=True)

    name = {agg.column(k, r): (k.lower(), r) for r in ROLES for k in T.codes(r)}
    tab = train_tab.join(test_tab).reset_index()
    tab.insert(1, "code", tab.column.map(lambda c: name[c][0]))
    tab.insert(2, "role", tab.column.map(lambda c: name[c][1]))
    tab["mean_share_pct"] = tab.column.map(dict(zip(cols, 100 * S4.mean(0))))
    tab["same_sign_test"] = np.sign(tab.train) == np.sign(tab.test)
    tab.to_csv(d / "contrasts.csv", index=False)

    summary = {}
    for c in CONTRASTS:
        s = tab[tab.contrast == c]
        sig = s[s.train_bh]
        summary[c] = {"bh_train": int(s.train_bh.sum()), "bh_test": int(s.test_bh.sum()),
                      "r_train_test": float(np.corrcoef(s.train, s.test)[0, 1]),
                      "bh_train_same_sign_test": int(sig.same_sign_test.sum())}
        print(f"\n{c}: {len(sig)} codes clear BH on training (same sign on test: "
              f"{int(sig.same_sign_test.sum())}); r(train, test) across codes "
              f"{summary[c]['r_train_test']:.2f}", flush=True)
        for _, r in sig.sort_values("train").iterrows():
            print(f"  {r.code:28s} {r.role:7s} train {r.train:+6.1f}% [{r.train_lo:+.1f}, {r.train_hi:+.1f}]"
                  f"   test {r.test:+6.1f}% [{r.test_lo:+.1f}, {r.test_hi:+.1f}]", flush=True)
    runlog.finish(d, {"run": a.run, "train_students": int(t4.student_id.nunique()),
                      "train_answers": int(len(t4)), "splits": a.splits, "boot": a.boot,
                      "test_students": int(h.student_id.nunique()), "test_answers": int(len(h)),
                      "test_cells": {f"{k[0]} {int(k[1])}": int(v) for k, v in cells_test.items()},
                      "by_contrast": summary})


if __name__ == "__main__":
    main()
