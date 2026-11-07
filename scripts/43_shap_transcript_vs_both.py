#!/usr/bin/env python
"""Stage 43 — TreeSHAP of the transcript model against the same model with the student's attainment.

Two XGBoost models from stage 06, fitted on every training row, five seeds each:
  transcript model  objective text + the 57 code rates (`arm-both-pred`, the paper's model);
  both              objective text + the 57 code rates + attainment (`arm-both-attainment`), one model.

Seed-averaged TreeSHAP on the log-odds scale (stage 07's `shap_block`), computed once on every
released-test TSL answer and then summarised for all TSL students and for those with at least 1,
3, 4, 5, 6, 7 training sessions (`--min-sessions`), where attainment is measured better. Per
population and column: mean |SHAP| in log-odds, its share of the model's total, and its direction
(Spearman of the column's value against its SHAP). The objective's TF-IDF terms are summed into
one `objective` column. The attainment column's per-answer SHAP is saved for a dependence plot.
Per seed (each arm has five seed models, paired by position), every column's mean |SHAP| per
population is also saved, so a code's change between the two models has a seed range.

    python scripts/43_shap_transcript_vs_both.py --run shap-transcript-vs-both
"""
from __future__ import annotations

import argparse
import importlib

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ttapub import attainment, agg, config, data, runlog
from ttapub import taxonomy_v3 as T

RUNS = {"transcript": "arm-both-pred", "both": "arm-both-attainment"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="shap-transcript-vs-both")
    ap.add_argument("--min-sessions", default="1,3,4,5,6,7")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.TEST_LABELS, config.METADATA, config.FEATURES_TEST,
                                      *(config.RESULTS / r / "models.joblib" for r in RUNS.values())])
    s06, s07 = importlib.import_module("06_train_clean57"), importlib.import_module("07_shap_clean57")
    meta = data.load_metadata()
    tr = data.load_train().merge(meta[["response_id", "student_id", "question_id"]], on="response_id").reset_index(drop=True)
    te = data.load_test().reset_index(drop=True)
    _, ab = attainment.train_and_test(tr, te.student_id)
    k = te.student_id.map(tr.groupby("student_id").session_id.nunique()).fillna(0)
    keep = (te.data_source == "tsl").to_numpy()
    kk = k.to_numpy()[keep]
    pops = {"TSL, all": np.ones(keep.sum(), bool)}
    pops |= {f"TSL, >= {x} training sessions": kk >= x for x in (int(v) for v in a.min_sessions.split(","))}
    ft = pd.read_csv(config.FEATURES_TEST, dtype={"session_id": str})
    name = {agg.column(c, r): (c.lower(), r) for r in ("tutor", "student") for c in T.codes(r)}
    name["student_attainment"] = ("attainment", "attainment")
    rows, per_row, per_seed = [], [], []
    for model, run in RUNS.items():
        b = joblib.load(config.RESULTS / run / "models.joblib")
        num = b["num"]
        codes = [c for c in num if c != "student_attainment"]
        X = s06.design(te, ft, codes)
        if "student_attainment" in num:
            X["student_attainment"] = ab
        X = X[keep].reset_index(drop=True)
        S_all, obj_all = s07.shap_block(b["models"], X, num)
        for si, mdl in enumerate(b["models"]):
            Ss, objs = s07.shap_block([mdl], X, num)
            for pop, m in pops.items():
                vals = np.abs(Ss[m]).mean(0)
                per_seed.append(pd.DataFrame({"population": pop, "model": model, "seed": si,
                                              "code": [name[c][0] for c in num], "mean_abs_shap": vals}))
                per_seed.append(pd.DataFrame({"population": [pop], "model": [model], "seed": [si],
                                              "code": ["objective"], "mean_abs_shap": [np.abs(objs[m]).mean()]}))
        if "student_attainment" in num:
            j = num.index("student_attainment")
            per_row.append(pd.DataFrame({"model": model, "code": "attainment", "train_sessions": kk,
                                         "value": X["student_attainment"].to_numpy(float), "shap": S_all[:, j]}))
        for pop, m in pops.items():
            S, obj, Xp = S_all[m], obj_all[m], X[m]
            total = np.abs(S).mean(0).sum() + np.abs(obj).mean()
            for j, c in enumerate(num):
                v = Xp[c].to_numpy(float)
                ok = ~np.isnan(v)
                rho = spearmanr(v[ok], S[ok, j]).statistic if ok.sum() > 100 else np.nan
                rows.append({"population": pop, "answers": int(m.sum()), "model": model, "column": c,
                             "code": name[c][0], "kind": name[c][1],
                             "mean_abs_shap": float(np.abs(S[:, j]).mean()),
                             "share": float(np.abs(S[:, j]).mean() / total), "direction_rho": float(rho)})
            rows.append({"population": pop, "answers": int(m.sum()), "model": model, "column": "objective",
                         "code": "objective", "kind": "objective", "mean_abs_shap": float(np.abs(obj).mean()),
                         "share": float(np.abs(obj).mean() / total), "direction_rho": np.nan})
        print(f"{model}: SHAP on {keep.sum():,} TSL answers", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(d / "shap_columns.csv", index=False)
    pd.concat(per_row).to_csv(d / "shap_rows.csv", index=False)
    pd.concat(per_seed).to_csv(d / "shap_seeds.csv", index=False)
    comp = tab.groupby(["population", "model", "kind"]).share.sum().unstack().fillna(0)
    print(comp.round(3).to_string(), flush=True)
    w = tab[tab.population == "TSL, >= 3 training sessions"].pivot(index="code", columns="model",
                                                                    values="mean_abs_shap").dropna()
    w["change_pct"] = 100 * (w.both / w.transcript - 1)
    print(w.sort_values("transcript", ascending=False).head(15).round(4).to_string(), flush=True)
    runlog.finish(d, {"run": a.run, "answers": {p: int(m.sum()) for p, m in pops.items()},
                      "composition": {f"{p} | {m}": r for (p, m), r in comp.to_dict(orient="index").items()}})


if __name__ == "__main__":
    main()
