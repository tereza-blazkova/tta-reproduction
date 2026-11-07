#!/usr/bin/env python
"""Stage 42 — head to head on the same test answers: the transcript model against the attainment model.

Two XGBoost models from stage 06, fitted on every training row, five seeds:
  attainment model     objective text + the student's attainment (`arm-lo-attainment`), no transcript;
  transcript model  objective text + the 57 code rates (`arm-both-pred`), the paper's model;
and, for reference, both together (`arm-both-attainment`) and the objective alone (`arm-lo-pred`).

On TSL test students with at least X training sessions (X = 1, 3, 4, 5, 6, 7; attainment is measured
better as X grows) and on all TSL students:
  * each model's log loss and AUROC, and the paired difference transcript minus attainment model,
    with a 2,000-draw bootstrap over students;
  * the distribution of each model's predicted probability for answers that were right and for
    answers that were wrong (histograms, 2.5-point bins), and the mean prediction for each.

    python scripts/42_head_to_head.py --run head-to-head
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import config, data, metrics, runlog

MODELS = {"objective": "arm-lo-pred", "attainment": "arm-lo-attainment", "transcript": "arm-both-pred",
          "both": "arm-both-attainment"}
BINS = np.linspace(0, 1, 41)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="head-to-head")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min-sessions", default="1,3,4,5,6,7")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.TEST_LABELS, config.METADATA,
                                      *(config.RESULTS / r / "test.csv" for r in MODELS.values())])
    meta = data.load_metadata()
    tr = data.load_train().merge(meta[["response_id", "student_id"]], on="response_id")
    te = data.load_test().reset_index(drop=True)
    te["k"] = te.student_id.map(tr.groupby("student_id").session_id.nunique()).fillna(0)
    y = te.is_correct.to_numpy(float)
    P = {}
    for m, run in MODELS.items():
        p = pd.read_csv(config.RESULTS / run / "test.csv", dtype={"response_id": str}).set_index("response_id")
        if not np.array_equal(p.y.reindex(te.response_id).to_numpy(), y):
            raise SystemExit(f"{run}: labels do not match the test split")
        P[m] = p.p.reindex(te.response_id).to_numpy()

    pops = {"TSL, all": te.data_source == "tsl"}
    pops |= {f"TSL, >= {x} training sessions": (te.data_source == "tsl") & (te.k >= x)
             for x in (int(v) for v in a.min_sessions.split(","))}
    perf, pair, hist = [], [], []
    for pop, mask in pops.items():
        m = mask.to_numpy()
        g = te.student_id.to_numpy()[m]
        for name, p in P.items():
            r = metrics.basic(y[m], p[m]) | metrics.boot_ci(y[m], p[m], g, B=a.boot)
            perf.append({"population": pop, "model": name, "students": len(set(g)), "n": int(m.sum()),
                         "ll": r["ll"], "ll_lo": r["ll_ci"][0], "ll_hi": r["ll_ci"][1],
                         "auroc": r["auroc"], "auroc_lo": r["auroc_ci"][0], "auroc_hi": r["auroc_ci"][1],
                         "mean_p_right": float(p[m][y[m] == 1].mean()),
                         "mean_p_wrong": float(p[m][y[m] == 0].mean())})
            for outcome in (1, 0):
                h = np.histogram(p[m][y[m] == outcome], BINS)[0]
                hist.append(pd.DataFrame({"population": pop, "model": name,
                                          "outcome": "right" if outcome else "wrong",
                                          "bin_lo": BINS[:-1], "bin_hi": BINS[1:], "count": h}))
        for new, old in (("transcript", "attainment"), ("both", "attainment"), ("both", "transcript")):
            r = metrics.boot_delta(y[m], P[new][m], P[old][m], g, B=a.boot)
            pair.append({"population": pop, "comparison": f"{new} vs {old}",
                         "ll_reduction": -r["d_ll"], "ll_lo": -r["d_ll_ci"][1], "ll_hi": -r["d_ll_ci"][0],
                         "d_auroc": r["d_auroc"], "auroc_lo": r["d_auroc_ci"][0], "auroc_hi": r["d_auroc_ci"][1]})
        t = [q for q in pair if q["population"] == pop and q["comparison"] == "transcript vs attainment"][0]
        pr = {q["model"]: q for q in perf if q["population"] == pop}
        print(f"{pop}: {pr['attainment']['students']:,} students | attainment model LL {pr['attainment']['ll']:.4f} "
              f"AUROC {pr['attainment']['auroc']:.2f} | transcript model LL {pr['transcript']['ll']:.4f} "
              f"AUROC {pr['transcript']['auroc']:.2f} | transcript minus attainment: LL reduction "
              f"{t['ll_reduction']:+.4f} [{t['ll_lo']:+.4f}, {t['ll_hi']:+.4f}], AUROC {t['d_auroc']:+.2f} "
              f"[{t['auroc_lo']:+.2f}, {t['auroc_hi']:+.2f}]", flush=True)
    pd.DataFrame(perf).to_csv(d / "performance.csv", index=False)
    pd.DataFrame(pair).to_csv(d / "paired.csv", index=False)
    pd.concat(hist).to_csv(d / "histograms.csv", index=False)
    runlog.finish(d, {"run": a.run, "boot": a.boot, "populations": {p: int(m.sum()) for p, m in pops.items()},
                      "transcript_vs_attainment": {q["population"]: {k: q[k] for k in ("ll_reduction", "ll_lo", "ll_hi",
                                                                                        "d_auroc", "auroc_lo", "auroc_hi")}
                                                for q in pair if q["comparison"] == "transcript vs attainment"}})


if __name__ == "__main__":
    main()
