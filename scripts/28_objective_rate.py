#!/usr/bin/env python
"""Stage 28 — the objective's historical correct rate, as a topic-difficulty baseline.

The question it answers: is the signal in the conversation anything more than the topic being
discussed? The paper's own answer is the objective-only arm of the ladder (`arm-lo`, XGBoost on
the objective's TF-IDF), and every block is measured on top of it. This is the literal version
of the baseline: predict each response with its objective's correct rate in the training rows.

**Mostly undefined here, by design, and the output says how often.** The folds are
objective-disjoint, so out-of-fold no held-out objective has any history and the prediction is
the training folds' base rate. On test, 9.3% of TSL rows and no Eedi rows have an objective that
appears in training. Where an objective has no history the prediction falls back to the training
base rate -- which is why the paper's model reads the objective's *text*, carrying difficulty
over from objectives that are worded alike.

The rate is shrunk toward the base rate with a fixed prior of `--prior` pseudo-responses, so an
objective seen a handful of times does not predict 0 or 1. The prior is fixed, not tuned.

    python scripts/28_objective_rate.py --run baseline-objective-rate
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import data, folds, metrics, runlog


def rate_model(lo_tr: np.ndarray, y_tr: np.ndarray, prior: float):
    """objective id -> shrunk training correct rate, and the base rate for unseen objectives."""
    base = float(y_tr.mean())
    g = pd.DataFrame({"lo": lo_tr, "y": y_tr}).groupby("lo").y.agg(["sum", "count"])
    rates = ((g["sum"] + prior * base) / (g["count"] + prior)).to_dict()
    return rates, base


def predict(rates: dict, base: float, lo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    seen = np.array([k in rates for k in lo])
    return np.array([rates.get(k, base) for k in lo]), seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline-objective-rate")
    ap.add_argument("--prior", type=float, default=10.0,
                    help="pseudo-responses at the base rate added to every objective; fixed")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a))

    train = data.load_train().reset_index(drop=True)
    y = train.is_correct.to_numpy(float)
    lo = train.learning_objective_id.to_numpy()
    fold = folds.lo_folds(len(train))
    clean = folds.clean_mask(len(train))
    res: dict = {"run": a.run, "model": "objective historical rate", "prior": a.prior}

    oof, seen_oof = np.zeros(len(y)), np.zeros(len(y), bool)
    for k in sorted(set(fold)):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        oof[te], seen_oof[te] = predict(*rate_model(lo[tr], y[tr], a.prior), lo[te])
    pd.DataFrame({"response_id": train.response_id, "y": y, "p": oof, "fold": fold,
                  "clean": clean, "objective_seen": seen_oof}).to_csv(d / "oof.csv", index=False)
    for view, msk in (("oof_all", np.ones(len(y), bool)), ("oof_clean", clean)):
        r = (metrics.basic(y[msk], oof[msk]) | metrics.boot_ci(y[msk], oof[msk], lo[msk], B=a.boot)
             | {"share_objective_seen": float(seen_oof[msk].mean())})
        res[view] = r
        print(metrics.fmt(view, r), f"| objective seen {r['share_objective_seen']:.1%}", flush=True)

    test = data.load_test().reset_index(drop=True)
    yt = test.is_correct.to_numpy(float)
    pt, seen = predict(*rate_model(lo, y, a.prior), test.learning_objective_id.to_numpy())
    pd.DataFrame({"response_id": test.response_id, "data_source": test.data_source, "y": yt,
                  "p": pt, "objective_seen": seen}).to_csv(d / "test.csv", index=False)
    for src in ("tsl", "eedi", "all"):
        msk = np.ones(len(yt), bool) if src == "all" else (test.data_source == src).to_numpy()
        r = (metrics.basic(yt[msk], pt[msk])
             | {"ll_icpt": metrics.intercept_refit_ll(yt[msk], pt[msk]),
                "share_objective_seen": float(seen[msk].mean())}
             | metrics.boot_ci(yt[msk], pt[msk], test.session_id.to_numpy()[msk], B=a.boot))
        res[f"test_{src}"] = r
        print(metrics.fmt(f"test {src}", r), f"| objective seen {r['share_objective_seen']:.1%}",
              flush=True)
    runlog.finish(d, res)


if __name__ == "__main__":
    main()
