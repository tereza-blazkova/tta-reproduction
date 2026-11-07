#!/usr/bin/env python
"""Stage 27 — what dropping a block costs, per learner, on the released test set.

Pairs two runs of the same learner that differ only in their columns (e.g. both blocks vs tutor
moves only) and reports the change on the same test rows, with a session-clustered bootstrap
95% CI (a session's responses share a transcript, so they are resampled together):

* dLL = LL(without) - LL(with): positive means the dropped block lowered log loss.
* dAUROC = AUROC(with) - AUROC(without): positive means the dropped block improved ranking.

    python scripts/27_paired_drop.py --run drop-student \\
        XGBoost=arm-both-pred:arm-tutor-pred "TabPFN 3.5=tabpfn35-tfidf-both:tabpfn35-tfidf-tutor"
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ttapub import config, data, runlog


def ll(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="label=with_run:without_run")
    ap.add_argument("--run", default="drop-student")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-spread", action="store_true",
                    help="resample each run's seeds too (needs the p_seed* columns a foundation "
                         "model writes), so the interval carries the model's run-to-run "
                         "randomness and not only which test sessions were drawn")
    a = ap.parse_args()

    test = data.load_test().reset_index(drop=True)[["response_id", "session_id", "data_source"]]
    d = runlog.start(a.run, vars(a), [])
    res, rows = {}, []
    for pair in a.pairs:
        label, runs = pair.split("=")
        w_run, wo_run = runs.split(":")
        w = pd.read_csv(config.RESULTS / w_run / "test.csv")
        wo = pd.read_csv(config.RESULTS / wo_run / "test.csv")
        sw = [c for c in w.columns if c.startswith("p_seed")]
        swo = [c for c in wo.columns if c.startswith("p_seed")]
        if a.seed_spread and not (sw and swo):
            raise SystemExit(f"{label}: --seed-spread needs p_seed* columns in both runs")
        w = w[["response_id", "y", "p"] + sw]
        wo = wo[["response_id", "p"] + swo].rename(columns={c: c + "_wo" for c in ["p"] + swo})
        m = test.merge(w, on="response_id").merge(wo, on="response_id")
        assert len(m) == len(test), f"{label}: {len(m)} rows matched of {len(test)}"
        for src in ("tsl", "eedi", "all"):
            s = m if src == "all" else m[m.data_source == src]
            y, p1, p0 = s.y.to_numpy(float), s.p.to_numpy(float), s.p_wo.to_numpy(float)
            P1 = s[sw].to_numpy(float) if a.seed_spread else None        # rows x seeds
            P0 = s[[c + "_wo" for c in swo]].to_numpy(float) if a.seed_spread else None
            dl = ll(y, p0) - ll(y, p1)
            point = {"dLL": float(dl.mean()),
                     "dAUROC": 100 * (roc_auc_score(y, p1) - roc_auc_score(y, p0)),
                     "LL_with": float(ll(y, p1).mean()), "LL_without": float(ll(y, p0).mean()),
                     "AUROC_with": 100 * roc_auc_score(y, p1),
                     "AUROC_without": 100 * roc_auc_score(y, p0)}
            sess = s.session_id.to_numpy()
            uniq, inv = np.unique(sess, return_inverse=True)
            members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
            rng = np.random.default_rng(a.seed)
            bl, ba = [], []
            for _ in range(a.boot):
                ix = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
                if len(np.unique(y[ix])) < 2:
                    continue
                q1, q0 = p1, p0
                if a.seed_spread:      # a fresh draw of each run's seeds, averaged as the run is
                    q1 = P1[:, rng.integers(0, P1.shape[1], P1.shape[1])].mean(1)
                    q0 = P0[:, rng.integers(0, P0.shape[1], P0.shape[1])].mean(1)
                bl.append((ll(y[ix], q0[ix]) - ll(y[ix], q1[ix])).mean())
                ba.append(100 * (roc_auc_score(y[ix], q1[ix]) - roc_auc_score(y[ix], q0[ix])))
            point |= {"dLL_ci": [float(q) for q in np.percentile(bl, [2.5, 97.5])],
                      "dAUROC_ci": [float(q) for q in np.percentile(ba, [2.5, 97.5])],
                      "n": int(len(s)), "with": w_run, "without": wo_run}
            res[f"{label}|{src}"] = point
            rows.append({"learner": label, "view": src} | {k: v for k, v in point.items()
                                                           if not k.endswith("_ci")}
                        | {"dLL_lo": point["dLL_ci"][0], "dLL_hi": point["dLL_ci"][1],
                           "dAUROC_lo": point["dAUROC_ci"][0], "dAUROC_hi": point["dAUROC_ci"][1]})
            print(f"{label:12s} {src:5s} n={len(s):6,}  dLL {point['dLL']:+.4f} "
                  f"[{point['dLL_ci'][0]:+.4f},{point['dLL_ci'][1]:+.4f}]  dAUROC "
                  f"{point['dAUROC']:+.2f} [{point['dAUROC_ci'][0]:+.2f},"
                  f"{point['dAUROC_ci'][1]:+.2f}]", flush=True)
    pd.DataFrame(rows).to_csv(d / "paired_drop.csv", index=False)
    runlog.finish(d, res)


if __name__ == "__main__":
    main()
