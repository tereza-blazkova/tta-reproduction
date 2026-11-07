#!/usr/bin/env python
"""Stage 55 — stage 46's code gains after one intercept per model: how much is calibration?

The attainment model over-predicts correctness (68.8% predicted where 63.4% are correct, clean rows
out of fold), so raw log loss rewards any column that pulls its predictions down. This stage
refits one intercept per model, a shift on the logit scale that keeps the ranking, and recomputes
each code's log-loss gain over the attainment model and over the placebos' mean gain (stage 46's
`PLACEBO`), three ways:

  raw        the predictions as saved;
  same rows  the intercept fitted on the rows being scored (a diagnostic: it sees their labels);
  crossfit   out of fold, fold k's intercept fitted on the other folds' rows of the same view;
             on test, the intercept fitted on all out-of-fold rows, frozen. No scored label
             enters its own intercept.

The codes are grouped as in `fig_code_occurrence`: those whose occurrence differs by attainment
after BH (stage 48), by direction. Point estimates only; intervals need stage 46's bootstrap
rerun on the shifted predictions.

Outputs: `recalibration.csv` (per view and run: gains x 1000, raw / same rows / crossfit, and the
code's side) and `metrics.json` (per view: the attainment model's mean prediction, share correct,
intercept and log loss; per side the mean gain beyond the placebo).

    python scripts/55_attainment_recalibration.py --run attainment-recalibration
"""
from __future__ import annotations

import argparse
import importlib

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from ttapub import config, data, runlog

s46 = importlib.import_module("46_attainment_incremental")
OCC = config.RESULTS / "code-occurrence" / "codes.csv"


def ll(y, p) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_shift(y, p) -> float:
    z = logit(np.clip(p, 1e-12, 1 - 1e-12))
    return float(minimize_scalar(lambda b: ll(y, expit(z + b)), bounds=(-3, 3), method="bounded").x)


def shifted(p, b):
    return expit(logit(np.clip(p, 1e-12, 1 - 1e-12)) + b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="attainment-recalibration")
    a = ap.parse_args()
    base = s46.BASE
    runs = ([*s46.MAIN.values(), *s46.REFS.values()]
            + [f"add1oof-{c}" for c in sorted(pd.read_csv(OCC).pipe(
                lambda o: [("v3t_" if r == "tutor" else "v3s_") + k + "_rate" for k, r in zip(o.code, o.role)]))]
            + [f"add1oofshuf-{c}" for c in s46.PLACEBO])
    runs = list(dict.fromkeys(runs))
    d = runlog.start(a.run, vars(a), [config.METADATA, OCC,
                                      *(config.RESULTS / r / f for r in runs for f in ("oof.csv", "test.csv"))])
    read = lambda r, f: pd.read_csv(config.RESULTS / r / f, dtype={"response_id": str}).set_index("response_id")
    oof, test = {r: read(r, "oof.csv") for r in runs}, {r: read(r, "test.csv") for r in runs}
    ro, rt = oof[base], test[base]
    for r in runs:                                   # same rows, same labels, same order everywhere
        oof[r], test[r] = oof[r].reindex(ro.index), test[r].reindex(rt.index)
        if not (np.array_equal(oof[r].y, ro.y) and np.array_equal(test[r].y, rt.y)):
            raise SystemExit(f"{r}: rows or labels differ from {base}")

    meta = data.load_metadata()
    tr = data.load_train().merge(meta[["response_id", "student_id"]], on="response_id")
    n = tr.groupby("student_id").session_id.nunique()
    other = (tr.set_index("response_id").student_id.map(n) - 1).reindex(ro.index).to_numpy()
    k = data.load_test().set_index("response_id").reindex(rt.index).student_id.map(n).fillna(0).to_numpy()
    clean = ro.clean.astype(bool).to_numpy()
    tsl = (rt.data_source == "tsl").to_numpy()
    views = {"out of fold, clean rows": ("oof", clean),
             "out of fold, clean, >= 3 other sessions": ("oof", clean & (other >= 3)),
             "test, all TSL": ("test", tsl), "test, >= 3 training sessions": ("test", tsl & (k >= 3))}

    fold, yo = ro.fold.to_numpy(), ro.y.to_numpy(float)
    frozen = {r: fit_shift(yo, oof[r].p.to_numpy()) for r in runs}   # test: from out-of-fold rows only
    occ = pd.read_csv(OCC)
    occ = occ[occ.strong_vs_weak_bh.astype(bool)]
    side = {f"add1oof-{('v3t_' if ro_ == 'tutor' else 'v3s_') + c}_rate": ("higher" if s > 0 else "lower")
            for c, ro_, s in zip(occ.code, occ.role, occ.strong_vs_weak)}
    rows, summary = [], {}
    for view, (split, m) in views.items():
        src = oof if split == "oof" else test
        y = src[base].y.to_numpy(float)[m]
        L = {"raw": {}, "same rows": {}, "crossfit": {}}
        for r in runs:
            p_all = src[r].p.to_numpy()
            p = p_all[m]
            L["raw"][r] = ll(y, p)
            L["same rows"][r] = ll(y, shifted(p, fit_shift(y, p)))
            if split == "oof":
                q = np.empty(len(p_all))
                for f in np.unique(fold):
                    fit = m & (fold != f)
                    q[fold == f] = shifted(p_all[fold == f], fit_shift(yo[fit], p_all[fit]))
                L["crossfit"][r] = ll(y, q[m])
            else:
                L["crossfit"][r] = ll(y, shifted(p, frozen[r]))
        plac = {w: np.mean([v[base] - v[f"add1oofshuf-{c}"] for c in s46.PLACEBO]) for w, v in L.items()}
        for r in runs:
            if r != base:
                rows.append({"view": view, "run": r, "side": side.get(r, ""),
                             **{f"gain_{w.replace(' ', '_')}": 1e3 * (v[base] - v[r]) for w, v in L.items()},
                             **{f"beyond_placebo_{w.replace(' ', '_')}": 1e3 * (v[base] - v[r] - plac[w])
                                for w, v in L.items()}})
        pb = src[base].p.to_numpy()[m]
        summary[view] = {"rows": int(m.sum()), "attainment_mean_p": float(pb.mean()), "share_correct": float(y.mean()),
                         "attainment_shift_same_rows": fit_shift(y, pb),
                         "attainment_ll": {w: v[base] for w, v in L.items()}}
    out = pd.DataFrame(rows)
    out.to_csv(d / "recalibration.csv", index=False)
    cols = [c for c in out if c.startswith("beyond_placebo")]
    g = out[out.side != ""].groupby(["view", "side"])[cols].mean()
    for view in views:
        summary[view]["beyond_placebo_by_side"] = {s: g.loc[(view, s)].round(4).to_dict() for s in ("higher", "lower")}
        print(view, summary[view]["attainment_mean_p"], summary[view]["share_correct"], flush=True)
        print(g.loc[view].round(2).to_string(), "\n", flush=True)
    runlog.finish(d, {"run": a.run, "codes_by_side": {s: int((occ.strong_vs_weak > 0).sum() if s == "higher"
                                                             else (occ.strong_vs_weak <= 0).sum())
                                                      for s in ("higher", "lower")},
                      "views": summary})


if __name__ == "__main__":
    main()
