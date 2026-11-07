#!/usr/bin/env python
"""Stage 53 — Lin-scheme baseline arms against the code-rate model, paired on identical rows.

Levels and paired differences on every view, with the repo's clustered bootstrap
(`metrics.boot_delta`): objective clusters out of fold, session clusters on test, as stage 06.
`d_ll` = LL(first) − LL(second), `d_auroc` = AUROC(first) − AUROC(second).

Refuses a pair unless response ids and outcomes match row for row in both prediction files.

    python scripts/53_lin_compare.py --run lin-compare \\
        arm-lin-rates:arm-both arm-lin-rates-tutor:arm-tutor arm-lin-rates-student:arm-student
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import config, data, folds, metrics, runlog


def views(a: str, b: str):
    """Yield (view, y, p_a, p_b, clusters), having checked the two runs share their rows."""
    train = data.load_train().reset_index(drop=True)
    pa, pb = (pd.read_csv(config.RESULTS / r / "oof.csv") for r in (a, b))
    if not (pa.response_id.equals(pb.response_id) and pa.y.equals(pb.y)):
        raise SystemExit(f"{a} and {b}: out-of-fold rows differ")
    if not pa.response_id.equals(train.response_id):
        raise SystemExit(f"{a}: out-of-fold rows are not in load_train() order")
    if not pa.fold.equals(pb.fold):
        raise SystemExit(f"{a} and {b}: folds differ")
    lo = train.learning_objective_id.to_numpy()
    clean = folds.clean_mask(len(train))
    y = pa.y.to_numpy(float)
    yield "oof_all", y, pa.p.to_numpy(), pb.p.to_numpy(), lo
    yield "oof_clean", y[clean], pa.p.to_numpy()[clean], pb.p.to_numpy()[clean], lo[clean]

    test = data.load_test().reset_index(drop=True)
    ta, tb = (pd.read_csv(config.RESULTS / r / "test.csv") for r in (a, b))
    if not (ta.response_id.equals(tb.response_id) and ta.y.equals(tb.y)):
        raise SystemExit(f"{a} and {b}: test rows differ")
    if not ta.response_id.equals(test.response_id):
        raise SystemExit(f"{a}: test rows are not in load_test() order")
    y = ta.y.to_numpy(float)
    sess = test.session_id.to_numpy()
    for src in ("tsl", "eedi", "all"):
        m = np.ones(len(y), bool) if src == "all" else (test.data_source == src).to_numpy()
        yield f"test_{src}", y[m], ta.p.to_numpy()[m], tb.p.to_numpy()[m], sess[m]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="first_run:second_run")
    ap.add_argument("--run", default="lin-compare")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [])
    res: dict = {"run": a.run, "pairs": {}}
    for pair in a.pairs:
        first, second = pair.split(":")
        res["pairs"][pair] = {}
        for view, y, p1, p0, cl in views(first, second):
            b1, b0 = metrics.basic(y, p1), metrics.basic(y, p0)
            r = metrics.boot_delta(y, p1, p0, cl, B=a.boot)
            r |= {"n": int(len(y)), "ll_first": b1["ll"], "ll_second": b0["ll"],
                  "auroc_first": b1["auroc"], "auroc_second": b0["auroc"],
                  "d_ll_point": float(b1["ll"] - b0["ll"]),
                  "d_auroc_point": float(b1["auroc"] - b0["auroc"]),
                  "first_ci": metrics.boot_ci(y, p1, cl, B=a.boot)}
            res["pairs"][pair][view] = r
            print(f"{pair:45s} {view:10s} n={len(y):6,}  LL {b1['ll']:.4f} vs {b0['ll']:.4f} "
                  f"d {r['d_ll_point']:+.4f} [{r['d_ll_ci'][0]:+.4f},{r['d_ll_ci'][1]:+.4f}]  "
                  f"AUROC {b1['auroc']:.2f} vs {b0['auroc']:.2f} d {r['d_auroc_point']:+.2f} "
                  f"[{r['d_auroc_ci'][0]:+.2f},{r['d_auroc_ci'][1]:+.2f}]", flush=True)
    runlog.finish(d, res)


if __name__ == "__main__":
    main()
