#!/usr/bin/env python
"""Stage 57 — what each block adds once the model knows the student's attainment.

Figure 2a's contrast (both blocks against the model without the block) with attainment
(:mod:`ttapub.attainment`) in every arm, for three learners on the same columns:

  XGBoost              `arm-{both,tutor,student}-attainment` (stage 06 --attainment); test predictions are
                       rebuilt from the committed models.joblib and must reproduce metrics.json;
  logistic regression  `attainment-logreg-{both,tutor,student}` (stage 22 --attainment);
  TabPFN 3.5           `attainment-tabpfn35-{both,tutor,student}` (stage 22 --attainment, GPU).

Bootstrap as in Figure 2a: 2,000 session clusters, `metrics.boot_delta`'s seed. Eedi students have
no attainment, so an Eedi contrast says nothing about attainment. XGBoost's arms ran on xgboost
3.3.0 and the paper's arms without attainment on 2.1.4: the version alone moves these contrasts,
so compare XGBoost with Figure 2a only against same-version arms.

Outputs: `blocks.csv` (learner, view, block, n, gains: d_auroc in points, d_ll negative = better)
and `metrics.json` (log loss and AUROC per arm and view, and the reproduction check).

    python scripts/57_attainment_blocks.py --run attainment-blocks
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd

from ttapub import attainment, config, data, metrics, runlog

ARMS = ("both", "tutor", "student")
VIEWS = {"TSL (test)": "tsl", "Eedi (test)": "eedi"}
REPRO_TOL = 1e-6


def xgb_predictions(arm: str, te: pd.DataFrame, ab: np.ndarray, tf: pd.DataFrame) -> np.ndarray:
    """A committed arm's test predictions, from its models; stops if they miss its metrics.json."""
    d = config.RESULTS / f"arm-{arm}-attainment"
    b = joblib.load(d / "models.joblib")
    rates = [c for c in b["num"] if c != "student_attainment"]
    X = te[["session_id"]].merge(tf[["session_id"] + rates], on="session_id", how="left")[rates]
    X = X.reset_index(drop=True)
    X["learning_objective"] = te.learning_objective.to_numpy()
    X["learning_objective_id"] = te.learning_objective_id.to_numpy()
    X["student_attainment"] = ab
    p = np.mean([m.predict_proba(X)[:, 1] for m in b["models"]], axis=0)
    m = json.loads((d / "metrics.json").read_text())
    y = te.is_correct.to_numpy(float)
    for src in VIEWS.values():
        k = (te.data_source == src).to_numpy()
        got = metrics.basic(y[k], p[k])["ll"]
        if abs(got - m[f"test_{src}"]["ll"]) > REPRO_TOL:
            raise SystemExit(f"arm-{arm}-attainment {src}: log loss {got} from the saved models, "
                             f"{m[f'test_{src}']['ll']} in metrics.json")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="attainment-blocks")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    stage22 = {"logistic regression": "attainment-logreg", "TabPFN 3.5": "attainment-tabpfn35"}
    inputs = [config.METADATA, config.features_path("test", "labels")]
    inputs += [config.RESULTS / f"arm-{x}-attainment" / "models.joblib" for x in ARMS]
    inputs += [config.RESULTS / f"{r}-{x}" / "test.csv" for r in stage22.values() for x in ARMS]
    d = runlog.start(a.run, vars(a), inputs)

    meta = data.load_metadata().set_index("response_id")
    tr = data.load_train().reset_index(drop=True)
    tr = tr.assign(student_id=tr.response_id.map(meta.student_id),
                   question_id=tr.response_id.map(meta.question_id))
    te = data.load_test().reset_index(drop=True)
    _, ab = attainment.train_and_test(tr, te.student_id)
    tf = pd.read_csv(config.features_path("test", "labels"), dtype={"session_id": str})
    y, g = te.is_correct.to_numpy(float), te.session_id.to_numpy()

    def stored(run: str) -> np.ndarray:
        o = pd.read_csv(config.RESULTS / run / "test.csv").set_index("response_id").reindex(te.response_id)
        if o.p.isna().any() or not np.array_equal(o.y.to_numpy(float), y):
            raise SystemExit(f"{run}: rows or labels differ from the test set")
        return o.p.to_numpy()

    pred = {"XGBoost": {x: xgb_predictions(x, te, ab, tf) for x in ARMS}}
    for learner, r in stage22.items():
        pred[learner] = {x: stored(f"{r}-{x}") for x in ARMS}

    rows, levels = [], {}
    for learner, P in pred.items():
        for view, src in VIEWS.items():
            k = (te.data_source == src).to_numpy()
            for x in ARMS:
                levels.setdefault(learner, {}).setdefault(src, {})[x] = metrics.basic(y[k], P[x][k])
            for block, without in (("tutor moves", "student"), ("student signals", "tutor")):
                r = metrics.boot_delta(y[k], P["both"][k], P[without][k], g[k], B=a.boot)
                rows.append(dict(learner=learner, view=view, block=block, n=int(k.sum()),
                                 d_auroc=r["d_auroc"], lo=r["d_auroc_ci"][0], hi=r["d_auroc_ci"][1],
                                 d_ll=r["d_ll"], ll_lo=r["d_ll_ci"][0], ll_hi=r["d_ll_ci"][1]))
    out = pd.DataFrame(rows)
    out.to_csv(d / "blocks.csv", index=False)
    print(out.round(4).to_string(index=False), flush=True)
    runlog.finish(d, {"run": a.run, "levels": levels, "test_rows_with_attainment": float(np.isfinite(ab).mean()),
                      "xgb_models_reproduce_metrics": True, "repro_tol_ll": REPRO_TOL})


if __name__ == "__main__":
    main()
