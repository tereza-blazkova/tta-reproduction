#!/usr/bin/env python
"""Stage 46 — the attainment model, and what each code adds to it, out of fold and on the test set.

Models (stage 06, five seeds, out-of-fold over the frozen objective-disjoint folds and refitted
on every training row for the released test set):
  objective model    objective text only (`arm-lo-oof`);
  attainment model      objective text + the student's attainment (`arm-lo-attainment`), the baseline;
  transcript model   objective text + the 57 code rates (`arm-both-oof`), the paper's model;
  both               objective text + 57 codes + attainment (`arm-both-attainment`);
  attainment + one code (`add1oof-<column>`, 57 models), attainment + all tutor moves / all student
  signals (`arm-tutor-attainment`, `arm-student-attainment`);
  placebos: attainment + one code shuffled across sessions (`add1oofshuf-<column>`, five codes).

Attainment for a training row comes from the student's other training sessions, for a test row from
all their training answers (:mod:`ttapub.attainment`); no answer enters its own attainment.

Views. Out of fold: all rows; clean rows (sessions that never straddle folds, so no transcript is
shared between a training and an evaluation row); clean rows of students with at least 1, 3, 5
*other* training sessions. Test: TSL students, all and with at least 1, 3, 5 training sessions.
Intervals: 1,000 bootstrap draws (200 for AUROC), clustered by objective out of fold (the fold
unit) and by student on test.

Outputs:
  * `models.csv`: log loss and AUROC of the four main models per view (figure panel a), and each
    model minus the attainment model, paired on the same draws;
  * `add_one_in.csv`: each code's log-loss reduction and AUROC gain over the attainment model, its
    gain minus the placebos' mean gain (paired per draw) with interval and p (bootstrap SE), and
    Benjamini-Hochberg across the 57 codes; how far the code moves the predictions (mean absolute
    change in probability against the attainment model); the references and placebos in the same table;
  * `order.csv`: the codes ranked by their out-of-fold clean-row gain over the placebo, the order
    stage 47 adds them in. The test set plays no part in it.

    python scripts/46_attainment_incremental.py --run attainment-incremental
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from ttapub import agg, config, data, folds, runlog, within
from ttapub import taxonomy_v3 as T

MAIN = {"objective model": "arm-lo-oof", "attainment model": "arm-lo-attainment",
        "transcript model": "arm-both-oof", "both": "arm-both-attainment"}
REFS = {"all tutor moves": "arm-tutor-attainment", "all student signals": "arm-student-attainment",
        "all 57 codes": "arm-both-attainment"}
PLACEBO = ["v3s_explains_why_rate", "v3t_prompting_next_step_rate", "v3t_guiding_session_rate",
           "v3t_feedback_correct_rate", "v3t_praising_outcome_rate"]
BASE = "arm-lo-attainment"


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="attainment-incremental")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--boot-auroc", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cols = {agg.column(k, r): (k.lower(), r) for r in ("tutor", "student") for k in T.codes(r)}
    runs = {**{v: ("main", k) for k, v in MAIN.items()},
            **{f"add1oof-{c}": ("code", c) for c in cols},
            **{v: ("reference", k) for k, v in REFS.items()},
            **{f"add1oofshuf-{c}": ("placebo", c) for c in PLACEBO}}
    d = runlog.start(a.run, vars(a), [config.TEST_LABELS, config.METADATA,
                                      *(config.RESULTS / r / f for r in runs for f in ("oof.csv", "test.csv"))])
    rng = np.random.default_rng(a.seed)
    meta = data.load_metadata()
    tr = data.load_train().merge(meta[["response_id", "student_id"]], on="response_id").reset_index(drop=True)
    te = data.load_test().reset_index(drop=True)
    n_sess = tr.groupby("student_id").session_id.nunique()
    other = (tr.student_id.map(n_sess) - 1).to_numpy()
    te_k = te.student_id.map(n_sess).fillna(0).to_numpy()
    clean = folds.clean_mask(len(tr))

    def load(run, split, ids, yref):
        p = pd.read_csv(config.RESULTS / run / f"{split}.csv", dtype={"response_id": str}).set_index("response_id")
        if not np.array_equal(p.y.reindex(ids).to_numpy(), yref):
            raise SystemExit(f"{run}/{split}: labels do not match")
        return p.p.reindex(ids).to_numpy()

    yo, yt = tr.is_correct.to_numpy(float), te.is_correct.to_numpy(float)
    O = {r: load(r, "oof", tr.response_id, yo) for r in runs}
    Pt = {r: load(r, "test", te.response_id, yt) for r in runs}
    views = {"out of fold, all rows": ("oof", np.ones(len(tr), bool)),
             "out of fold, clean rows": ("oof", clean)}
    views |= {f"out of fold, clean, >= {x} other sessions": ("oof", clean & (other >= x)) for x in (1, 3, 5)}
    tsl = (te.data_source == "tsl").to_numpy()
    views |= {"test, all TSL": ("test", tsl)}
    views |= {f"test, >= {x} training sessions": ("test", tsl & (te_k >= x)) for x in (1, 3, 5)}
    views |= {"out of fold, clean, no other sessions": ("oof", clean & (other == 0)),   # attainment unknown
              "test, no training sessions": ("test", tsl & (te_k == 0))}

    mrows, arows = [], []
    for view, (split, m) in views.items():
        y = (yo if split == "oof" else yt)[m]
        P = {r: (O if split == "oof" else Pt)[r][m] for r in runs}
        groups = (tr.learning_objective_id.to_numpy() if split == "oof" else te.student_id.to_numpy())[m]
        ug, inv = np.unique(groups, return_inverse=True)
        W = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv] for _ in range(a.boot)])
        idx = [np.repeat(np.arange(len(y)), w) for w in W[: a.boot_auroc]]
        L = {r: ll(y, p) for r, p in P.items()}
        Lb = {r: (W @ v) / W.sum(1) for r, v in L.items()}                       # bootstrap mean LL
        Ab = {r: np.array([roc_auc_score(y[ix], p[ix]) for ix in idx]) for r, p in P.items()}
        for name, r in MAIN.items():
            mrows.append({"view": view, "model": name, "n": int(m.sum()), "clusters": len(ug),
                          "ll": float(L[r].mean()), "ll_lo": float(np.percentile(Lb[r], 2.5)),
                          "ll_hi": float(np.percentile(Lb[r], 97.5)), "auroc": 100 * float(roc_auc_score(y, P[r])),
                          "auroc_lo": 100 * float(np.percentile(Ab[r], 2.5)),
                          "auroc_hi": 100 * float(np.percentile(Ab[r], 97.5)),
                          "ll_vs_attainment": float(L[BASE].mean() - L[r].mean()),
                          "ll_vs_attainment_lo": float(np.percentile(Lb[BASE] - Lb[r], 2.5)),
                          "ll_vs_attainment_hi": float(np.percentile(Lb[BASE] - Lb[r], 97.5)),
                          "auroc_vs_attainment": 100 * float(roc_auc_score(y, P[r]) - roc_auc_score(y, P[BASE])),
                          "auroc_vs_attainment_lo": 100 * float(np.percentile(Ab[r] - Ab[BASE], 2.5)),
                          "auroc_vs_attainment_hi": 100 * float(np.percentile(Ab[r] - Ab[BASE], 97.5))})
        plac = [f"add1oofshuf-{c}" for c in PLACEBO]
        plac_ll_draw = np.mean([Lb[BASE] - Lb[p] for p in plac], axis=0)
        plac_ll = float(np.mean([L[BASE].mean() - L[p].mean() for p in plac]))
        for r, (kind, key) in runs.items():
            if kind == "main" and r != "arm-both-attainment":
                continue
            g_ll = Lb[BASE] - Lb[r]
            g_au = 100 * (Ab[r] - Ab[BASE])
            est_ll = float(L[BASE].mean() - L[r].mean())
            est_au = 100 * float(roc_auc_score(y, P[r]) - roc_auc_score(y, P[BASE]))
            vs = g_ll - plac_ll_draw
            code, role = (cols[key] if kind == "code" else
                          (f"placebo: {cols[key][0]}", "placebo") if kind == "placebo" else (key, "reference"))
            arows.append({"view": view, "code": code, "role": role, "ll_reduction": est_ll,
                          "ll_lo": float(np.percentile(g_ll, 2.5)), "ll_hi": float(np.percentile(g_ll, 97.5)),
                          "ll_p": float(2 * norm.sf(abs(est_ll) / g_ll.std(ddof=1))),
                          "vs_placebo": est_ll - plac_ll, "vs_placebo_lo": float(np.percentile(vs, 2.5)),
                          "vs_placebo_hi": float(np.percentile(vs, 97.5)),
                          "vs_placebo_p": float(2 * norm.sf(abs(est_ll - plac_ll) / vs.std(ddof=1))),
                          "d_auroc": est_au, "auroc_lo": float(np.percentile(g_au, 2.5)),
                          "auroc_hi": float(np.percentile(g_au, 97.5)),
                          "mean_abs_dp": float(np.abs(P[r] - P[BASE]).mean())})
        print(f"{view}: {int(m.sum()):,} rows, {len(ug):,} clusters; placebo mean {1e3 * plac_ll:+.2f}", flush=True)
    models, add = pd.DataFrame(mrows), pd.DataFrame(arows)
    for view in views:
        k = (add.view == view) & add.role.isin(["tutor", "student"])
        add.loc[k, "vs_placebo_bh"] = within.benjamini_hochberg(add.loc[k, "vs_placebo_p"].to_numpy(), 0.05)
        add.loc[k, "ll_bh"] = within.benjamini_hochberg(add.loc[k, "ll_p"].to_numpy(), 0.05)
    models.to_csv(d / "models.csv", index=False)
    add.to_csv(d / "add_one_in.csv", index=False)

    rank = add[(add.view == "out of fold, clean rows") & add.role.isin(["tutor", "student"])].sort_values(
        "vs_placebo", ascending=False).reset_index(drop=True)
    rank["column"] = rank.apply(lambda r: agg.column(r.code.upper(), r.role), axis=1)
    rank[["code", "role", "column", "vs_placebo", "vs_placebo_lo", "vs_placebo_hi", "vs_placebo_bh"]].to_csv(
        d / "order.csv", index=False)
    summary = {}
    for view in views:
        s = add[(add.view == view) & add.role.isin(["tutor", "student"])]
        top3 = s.set_index("code").loc[["explains_why", "prompting_next_step", "guiding_session"]]
        summary[view] = {"beat_placebo_bh": sorted(s[s.vs_placebo_bh.astype(bool) & (s.vs_placebo > 0)].code),
                         "top3_shap_vs_placebo": {c: [float(r.vs_placebo), float(r.vs_placebo_lo), float(r.vs_placebo_hi)]
                                                  for c, r in top3.iterrows()}}
        print(f"\n{view}: beat the placebo after BH: {summary[view]['beat_placebo_bh']}", flush=True)
        for c, r in top3.iterrows():
            print(f"  paper top-3 SHAP {c:22s} gain {1e3 * r.ll_reduction:+.2f}, minus placebo "
                  f"{1e3 * r.vs_placebo:+.2f} [{1e3 * r.vs_placebo_lo:+.2f}, {1e3 * r.vs_placebo_hi:+.2f}]", flush=True)
    print("\nout-of-fold order:", ", ".join(rank.code.head(10)), flush=True)
    runlog.finish(d, {"run": a.run, "boot": a.boot, "views": {v: int(m.sum()) for v, (_, m) in views.items()},
                      "summary": summary, "order_top10": rank.code.head(10).tolist()})


if __name__ == "__main__":
    main()
