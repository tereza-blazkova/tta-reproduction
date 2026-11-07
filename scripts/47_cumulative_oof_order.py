#!/usr/bin/env python
"""Stage 47 — the attainment model plus codes one at a time, in an order chosen out of fold.

Stage 46 ranks the 57 codes by how much each adds to the attainment model beyond the placebo on
out-of-fold clean rows (`results/attainment-incremental/order.csv`); the test set plays no part in it.
Nested models add the top `--k` codes in that order (`cumoof-1` ... `cumoof-<k>`, stage 06
`--attainment --only ... --test`), with the same codes shuffled as the matched placebo
(`cumoof-shuffled`). Because the order was chosen out of fold, the test curve here is an honest
estimate; the out-of-fold curve is shown beside it and is optimistic for the same reason.

Paired against the attainment model, with the all-57-codes model as the gap to close (stage 45's
`compare`): out of fold (all and clean rows, objectives resampled) and test (all TSL and students
with 1, 3, 5+ training sessions, students resampled).

    python scripts/47_cumulative_oof_order.py --run cumulative-oof-order
"""
from __future__ import annotations

import argparse
import importlib

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import config, data, folds, runlog

ORDER = config.RESULTS / "attainment-incremental" / "order.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cumulative-oof-order")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    order = pd.read_csv(ORDER).head(a.k)
    steps = ["arm-lo-attainment"] + [f"cumoof-{i}" for i in range(1, a.k + 1)]
    labels = ["attainment model"] + [f"+ {c.replace('_', ' ')}" for c in order.code]
    refs = {"all 57 codes": "arm-both-attainment", f"{a.k} codes shuffled": "cumoof-shuffled"}
    runs = steps + list(refs.values())
    d = runlog.start(a.run, vars(a), [ORDER, config.TEST_LABELS, config.METADATA,
                                      *(config.RESULTS / r / f for r in runs for f in ("test.csv", "oof.csv"))])
    s45 = importlib.import_module("45_cumulative_codes")
    rng = np.random.default_rng(0)
    meta = data.load_metadata()
    tr = data.load_train().merge(meta[["response_id", "student_id"]], on="response_id").reset_index(drop=True)
    te = data.load_test().reset_index(drop=True)
    te["k"] = te.student_id.map(tr.groupby("student_id").session_id.nunique()).fillna(0)
    label = dict(zip(steps, labels)) | {v: k for k, v in refs.items()}

    def load(run, split, ids, yref):
        p = pd.read_csv(config.RESULTS / run / f"{split}.csv", dtype={"response_id": str}).set_index("response_id")
        if not np.array_equal(p.y.reindex(ids).to_numpy(), yref):
            raise SystemExit(f"{run}/{split}: labels do not match")
        return p.p.reindex(ids).to_numpy()

    rows = []
    yt = te.is_correct.to_numpy(float)
    Pt = {r: load(r, "test", te.response_id, yt) for r in runs}
    pops = {"test, all TSL": te.data_source == "tsl"}
    pops |= {f"test, >= {x} training sessions": (te.data_source == "tsl") & (te.k >= x) for x in (1, 3, 5)}
    for pop, m in pops.items():
        m = m.to_numpy()
        for r in s45.compare(yt[m], {k: v[m] for k, v in Pt.items()}, te.student_id.to_numpy()[m], a.boot, rng):
            rows.append({"view": pop, "step": label[r["run"]], **r})
    yo = tr.is_correct.to_numpy(float)
    Po = {r: load(r, "oof", tr.response_id, yo) for r in runs}
    lo = tr.learning_objective_id.to_numpy()
    clean = folds.clean_mask(len(tr))
    for view, m in (("out of fold, all rows", np.ones(len(tr), bool)), ("out of fold, clean rows", clean)):
        for r in s45.compare(yo[m], {k: v[m] for k, v in Po.items()}, lo[m], a.boot, rng):
            rows.append({"view": view, "step": label[r["run"]], **r})
    tab = pd.DataFrame(rows)
    tab["order"] = tab.step.map({s: i for i, s in enumerate(labels)})
    tab.to_csv(d / "curve.csv", index=False)
    for view, s in tab.groupby("view", sort=False):
        s = s.set_index("step")
        gap = s.loc["all 57 codes", "ll_reduction"]
        print(f"\n{view}: all 57 codes {1e3 * gap:+.1f}, shuffled {1e3 * s.loc[f'{a.k} codes shuffled', 'll_reduction']:+.1f}", flush=True)
        for st in labels[1:]:
            r = s.loc[st]
            print(f"  {st:30s} {1e3 * r.ll_reduction:+5.1f} [{1e3 * r.ll_lo:+.1f}, {1e3 * r.ll_hi:+.1f}]  "
                  f"{100 * r.ll_reduction / gap:4.0f}% of the gap   AUROC {r.d_auroc:+.2f}", flush=True)
    runlog.finish(d, {"run": a.run, "k": a.k, "order": order.code.tolist(),
                      "curve": {f"{v} | {s}": {"ll_reduction": float(r.ll_reduction), "d_auroc": float(r.d_auroc)}
                                for (v, s), r in tab.set_index(["view", "step"]).iterrows()}})


if __name__ == "__main__":
    main()
