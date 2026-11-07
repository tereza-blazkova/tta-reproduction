#!/usr/bin/env python
"""Stage 07 — per-code attribution for clean57 (TreeSHAP).

Because each tutor move and each student signal is exactly **one** column, a code's attribution
is that column's SHAP: nothing to sum over correlated instruments, no ratio to apportion. This
is the whole reason the model has one column per code: a move's attribution spread over
several correlated columns could not be read from any one of them.

Reported per code: mean |SHAP| as a share of the model's total attribution, and the direction
(Spearman correlation between the column's value and its SHAP: + means more of this code raises
the predicted P(correct)). The objective's TF-IDF terms are summed into one `objective` row.

⚠️ **This describes what the model prices, not what helps students.** A move used with students
who were already succeeding is attributed as if it worked. The causal-direction question is the
within-student analysis, not this table — and a code that failed a distillation gate
(`results/<children-run>/gates.csv`) must not appear in a findings table at all.

    python scripts/07_shap_clean57.py --run clean57 [--gates results/clean57-children/gates.csv]
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
import xgboost as xgbl
from scipy.stats import spearmanr

from ttapub import agg, config, data, runlog


def shap_block(models, X, num):
    """Seed-averaged SHAP: (n, len(num)) for the code columns, (n,) for the objective block."""
    S = np.zeros((len(X), len(num)))
    obj = np.zeros(len(X))
    for m in models:
        names = list(m.named_steps["pre"].get_feature_names_out())
        Z = m[:-1].transform(X)
        c = m[-1].get_booster().predict(xgbl.DMatrix(Z, feature_names=names),
                                        pred_contribs=True)[:, :-1]
        S += c[:, [names.index(f"dlg__{k}") for k in num]]
        obj += c[:, [i for i, n in enumerate(names) if not n.startswith("dlg__")]].sum(1)
    return S / len(models), obj / len(models)


def table(S, obj, X, num, label: str) -> pd.DataFrame:
    tot = np.abs(S).mean(0).sum() + np.abs(obj).mean()
    rows = []
    for j, c in enumerate(num):
        v = X[c].to_numpy(float)
        ok = ~np.isnan(v)
        rho = spearmanr(v[ok], S[ok, j]).statistic if ok.sum() > 100 else np.nan
        rows.append({"column": c, "share": np.abs(S[:, j]).mean() / tot,
                     "direction": "up" if rho > 0.05 else ("down" if rho < -0.05 else "~"),
                     "spearman": rho, "view": label})
    rows.append({"column": "objective", "share": np.abs(obj).mean() / tot,
                 "direction": "", "spearman": np.nan, "view": label})
    return pd.DataFrame(rows).sort_values("share", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="clean57")
    ap.add_argument("--n-train", type=int, default=8000, help="training rows sampled for SHAP")
    ap.add_argument("--gates", default=None, help="gates.csv from stage 04; failing codes are "
                                                  "flagged in the output")
    a = ap.parse_args()

    d = config.RESULTS / a.run
    bundle = joblib.load(d / "models.joblib")
    models, num = bundle["models"], bundle["num"]
    train = data.load_train().reset_index(drop=True)
    feats = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    X = train[["session_id"]].merge(feats[["session_id"] + num], on="session_id",
                                    how="left")[num]
    X["learning_objective"] = train.learning_objective.to_numpy()
    X["learning_objective_id"] = train.learning_objective_id.to_numpy()
    idx = np.random.default_rng(0).choice(len(X), min(a.n_train, len(X)), replace=False)
    parts = [table(*shap_block(models, X.iloc[idx], num), X.iloc[idx], num, "train")]

    test = data.load_test().reset_index(drop=True)
    tfeats = pd.read_csv(config.FEATURES_TEST, dtype={"session_id": str})
    Xt = test[["session_id"]].merge(tfeats[["session_id"] + num], on="session_id",
                                    how="left")[num]
    Xt["learning_objective"] = test.learning_objective.to_numpy()
    Xt["learning_objective_id"] = test.learning_objective_id.to_numpy()
    for src in ("tsl", "eedi"):
        m = (test.data_source == src).to_numpy()
        parts.append(table(*shap_block(models, Xt[m], num), Xt[m], num, f"test_{src}"))

    out = pd.concat(parts, ignore_index=True)
    if a.gates:
        g = pd.read_csv(a.gates)
        g["column"] = [agg.column(c, r) for c, r in zip(g.code, g.role)]
        out = out.merge(g[["column", "status"]], on="column", how="left")
        out["readable"] = out.status.isin(["pass", np.nan]) | out.status.isna()
    out.to_csv(d / "shap_by_column.csv", index=False)
    print(out[out.view == "train"].head(25).to_string(index=False,
                                                      float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {d / 'shap_by_column.csv'}")
    if a.gates and (~out.readable).any():
        bad = sorted(set(out.loc[~out.readable, "column"]))
        print(f"⚠️  {len(bad)} columns failed a distillation gate and must not be read as "
              f"evidence about their concept: {', '.join(bad[:8])}"
              + (" …" if len(bad) > 8 else ""))


if __name__ == "__main__":
    main()
