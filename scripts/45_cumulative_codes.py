#!/usr/bin/env python
"""Stage 45 — how much of the transcript's gain over attainment do a few struggle codes recover?

Nested XGBoost models (stage 06 `--attainment --only ...`, five seeds), each the attainment model plus
the codes so far, in this order: prompting next step, prompting self-correction, feedback
incorrect, student unintelligible, giving hint, hedged answer, frustration (`cum-1` ... `cum-7`).
References: the attainment model (`arm-lo-attainment`, step 0), the attainment model plus all 57 codes
(`arm-both-attainment`, the gap to close), and the attainment model plus the same seven codes shuffled
across sessions (`cum-shuffled7`): adding columns alone buys a little, and this is its level.

The order was chosen from the add-one-in test results (stage 44), so the test curve is optimistic.
The out-of-fold view on training (objective-disjoint folds, the paper's primary partition) is not
touched by that choice and is reported beside it.

Per model, paired against the attainment model on the same rows: log-loss reduction and AUROC gain,
with 1,000 bootstrap draws (over students on test: all TSL and students with 1, 3, 5+ training
sessions; over objectives out of fold: all rows and clean rows).

    python scripts/45_cumulative_codes.py --run cumulative-codes
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ttapub import config, data, folds, runlog

STEPS = ["arm-lo-attainment"] + [f"cum-{i}" for i in range(1, 8)]
LABELS = ["attainment model", "+ prompting next step", "+ prompting self-correction", "+ feedback incorrect",
          "+ student unintelligible", "+ giving hint", "+ hedged answer", "+ frustration"]
REFS = {"all 57 codes": "arm-both-attainment", "7 codes shuffled": "cum-shuffled7"}


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(y, P: dict, groups, boot: int, rng) -> list[dict]:
    ug, inv = np.unique(groups, return_inverse=True)
    W = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv] for _ in range(boot)])
    idx = [np.repeat(np.arange(len(y)), w) for w in W[:200]]          # AUROC draws (200 suffice)
    base = P["arm-lo-attainment"]
    b_ll = ll(y, base)
    b_au = np.array([roc_auc_score(y[ix], base[ix]) for ix in idx])
    out = []
    for run, p in P.items():
        d = b_ll - ll(y, p)
        bd = (W @ d) / W.sum(1)
        au = 100 * (np.array([roc_auc_score(y[ix], p[ix]) for ix in idx]) - b_au)
        out.append({"run": run, "ll": float(ll(y, p).mean()), "auroc": 100 * float(roc_auc_score(y, p)),
                    "ll_reduction": float(d.mean()), "ll_lo": float(np.percentile(bd, 2.5)),
                    "ll_hi": float(np.percentile(bd, 97.5)),
                    "d_auroc": 100 * float(roc_auc_score(y, p) - roc_auc_score(y, base)),
                    "auroc_lo": float(np.percentile(au, 2.5)), "auroc_hi": float(np.percentile(au, 97.5))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cumulative-codes")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    runs = STEPS + list(REFS.values())
    d = runlog.start(a.run, vars(a), [config.TEST_LABELS, config.METADATA,
                                      *(config.RESULTS / r / f for r in runs for f in ("test.csv", "oof.csv"))])
    rng = np.random.default_rng(0)
    meta = data.load_metadata()
    tr = data.load_train().merge(meta[["response_id", "student_id"]], on="response_id").reset_index(drop=True)
    te = data.load_test().reset_index(drop=True)
    te["k"] = te.student_id.map(tr.groupby("student_id").session_id.nunique()).fillna(0)
    y = te.is_correct.to_numpy(float)
    label = dict(zip(STEPS, LABELS)) | {v: k for k, v in REFS.items()}
    rows = []

    def load(run, split, ref_ids, ref_y):
        p = pd.read_csv(config.RESULTS / run / f"{split}.csv", dtype={"response_id": str}).set_index("response_id")
        if not np.array_equal(p.y.reindex(ref_ids).to_numpy(), ref_y):
            raise SystemExit(f"{run}/{split}: labels do not match")
        return p.p.reindex(ref_ids).to_numpy()

    P = {r: load(r, "test", te.response_id, y) for r in runs}
    pops = {"test TSL, all": te.data_source == "tsl"}
    pops |= {f"test TSL, >= {x} training sessions": (te.data_source == "tsl") & (te.k >= x) for x in (1, 3, 5)}
    for pop, m in pops.items():
        m = m.to_numpy()
        for r in compare(y[m], {k: v[m] for k, v in P.items()}, te.student_id.to_numpy()[m], a.boot, rng):
            rows.append({"view": pop, "step": label[r["run"]], **r})

    yo = tr.is_correct.to_numpy(float)
    O = {r: load(r, "oof", tr.response_id, yo) for r in runs}
    lo = tr.learning_objective_id.to_numpy()
    clean = folds.clean_mask(len(tr))
    for view, m in (("out of fold, all rows", np.ones(len(tr), bool)), ("out of fold, clean rows", clean)):
        for r in compare(yo[m], {k: v[m] for k, v in O.items()}, lo[m], a.boot, rng):
            rows.append({"view": view, "step": label[r["run"]], **r})
    tab = pd.DataFrame(rows)
    tab.to_csv(d / "curve.csv", index=False)
    for view, s in tab.groupby("view", sort=False):
        s = s.set_index("step")
        gap = s.loc["all 57 codes", "ll_reduction"]
        print(f"\n{view}: all 57 codes {1e3 * gap:+.1f}, 7 shuffled {1e3 * s.loc['7 codes shuffled', 'll_reduction']:+.1f}"
              f" (log-loss reduction x 0.001 over the attainment model)", flush=True)
        for st in LABELS[1:]:
            r = s.loc[st]
            print(f"  {st:28s} {1e3 * r.ll_reduction:+5.1f} [{1e3 * r.ll_lo:+.1f}, {1e3 * r.ll_hi:+.1f}]  "
                  f"{100 * r.ll_reduction / gap:4.0f}% of the gap   AUROC {r.d_auroc:+.2f}", flush=True)
    runlog.finish(d, {"run": a.run, "boot": a.boot,
                      "curve": {f"{v} | {s}": {"ll_reduction": float(r.ll_reduction), "d_auroc": float(r.d_auroc)}
                                for (v, s), r in tab.set_index(["view", "step"]).iterrows()}})


if __name__ == "__main__":
    main()
