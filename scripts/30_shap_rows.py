#!/usr/bin/env python
"""Stage 30 — per-response SHAP values for the 57 code-rate columns, for the SI beeswarm.

Stage 07 keeps only each column's share and direction. The beeswarm needs the values row by row:
how large a column's contribution is for each response, and whether high rates push up or down.
This recomputes them exactly as stage 07 does (`shap_block`: TreeSHAP on the log-odds scale,
averaged over the five seeds' models of `arm-both`) on the same seeded draw of 8,000 training
responses, so the beeswarm and the main attribution figure describe the same numbers.

Writes `shap_rows.npz` (SHAP and the rates themselves, gitignored, reproducible from this script)
and a check in `metrics.json` that the shares recomputed from it equal stage 07's.

    python scripts/30_shap_rows.py --run shap-rows
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd

from ttapub import config, data, runlog

_spec = importlib.util.spec_from_file_location(
    "shap07", pathlib.Path(__file__).with_name("07_shap_clean57.py"))
shap07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shap07)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="shap-rows")
    ap.add_argument("--arm", default="arm-both")
    ap.add_argument("--n-train", type=int, default=8000, help="as stage 07")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.FEATURES_TRAIN])

    bundle = joblib.load(config.RESULTS / a.arm / "models.joblib")
    models, num = bundle["models"], bundle["num"]
    train = data.load_train().reset_index(drop=True)
    feats = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    X = train[["session_id"]].merge(feats[["session_id"] + num], on="session_id",
                                    how="left")[num]
    X["learning_objective"] = train.learning_objective.to_numpy()
    X["learning_objective_id"] = train.learning_objective_id.to_numpy()
    idx = np.random.default_rng(0).choice(len(X), min(a.n_train, len(X)), replace=False)
    Xs = X.iloc[idx]
    S, obj = shap07.shap_block(models, Xs, num)
    np.savez_compressed(d / "shap_rows.npz", shap=S, objective=obj,
                        rates=Xs[num].to_numpy(float), columns=np.array(num),
                        response_id=train.response_id.to_numpy()[idx])

    # the same shares stage 07 reports, recomputed from these rows: they must agree
    mine = shap07.table(S, obj, Xs, num, "train").set_index("column").share
    ref = pd.read_csv(config.RESULTS / a.arm / "shap_by_column.csv")
    ref = ref[ref.view == "train"].set_index("column").share
    gap = float((mine - ref.reindex(mine.index)).abs().max())
    print(f"{len(idx):,} rows x {len(num)} columns; max share difference vs stage 07: {gap:.2e}")
    runlog.finish(d, {"run": a.run, "arm": a.arm, "n_rows": int(len(idx)), "n_columns": len(num),
                      "max_share_diff_vs_stage07": gap})


if __name__ == "__main__":
    main()
