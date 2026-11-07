#!/usr/bin/env python
"""Stage 23 — the attribution analyses for logistic regression, for the supplementary material.

The paper's attribution is XGBoost's (stages 07 and 19). This repeats both for logistic
regression on the identical columns, so the SI can say which attribution findings belong to
the annotation and which to the booster.

**Linear SHAP is exact, not approximated.** For a linear model with the training mean as the
background, a column's SHAP value is `coef_j * (z_j - mean_j)` on the log-odds scale -- the same
scale as XGBoost's TreeSHAP margin contributions, so shares are comparable. The objective's
TF-IDF terms are summed into one `objective` row, exactly as stage 07 does.

**Why the by-state grid is a useful baseline.** A linear model has no interactions: a move's
SHAP depends on that move's own rate and nothing else. Any shift in its attribution across
student states therefore comes only from the move's rate being correlated with the state's
rate. XGBoost's grid (Table 3) contains that correlation *plus* whatever interaction the booster
learned; the difference between the two grids is the part that is genuinely about the student
state changing what the move is worth to the model.

Reuses `table()` from stage 07 and `state_grid()` from stage 19, and the same samples (seeded
draws of 8,000 and 15,000 training rows), so no number here can differ from XGBoost's by
construction rather than by learner.

    python scripts/23_attribution_logreg.py --run attribution-logreg
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, data, runlog

_HERE = pathlib.Path(__file__).parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shap07 = _load("shap07", "07_shap_clean57.py")
state19 = _load("state19", "19_shap_by_state.py")
base22 = _load("base22", "22_baseline.py")


def linear_shap(pipe, X_bg: pd.DataFrame, X: pd.DataFrame, num: list[str]):
    """(n, len(num)) SHAP for the code columns and (n,) for the summed objective block."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    names = list(pre.get_feature_names_out())
    coef = clf.coef_[0]
    mu = np.asarray(pre.transform(X_bg).mean(axis=0)).ravel()
    Z = pre.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    C = (Z - mu) * coef
    S = C[:, [names.index(f"rates__{k}") for k in num]]
    obj = C[:, [i for i, n in enumerate(names) if n.startswith("obj_text__")]].sum(axis=1)
    return S, obj


def design(frame: pd.DataFrame, feats: pd.DataFrame, num: list[str]) -> pd.DataFrame:
    X = frame[["session_id"]].merge(feats[["session_id"] + num], on="session_id",
                                    how="left")[num]
    X["learning_objective"] = frame.learning_objective.to_numpy()
    X["learning_objective_id"] = frame.learning_objective_id.to_numpy()
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="attribution-logreg")
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--n-train", type=int, default=8000, help="as stage 07")
    ap.add_argument("--n-state", type=int, default=15000, help="as stage 19")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.FEATURES_TRAIN])

    train = data.load_train().reset_index(drop=True)
    y = train.is_correct.to_numpy(float)
    feats = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    num = agg.code_columns(feats.columns, "tutor") + agg.code_columns(feats.columns, "student")
    X = design(train, feats, num)
    pipe = base22.make_model("logreg", num, a.C, 32, 42).fit(X, y)

    # ── shares and directions, as stage 07: training sample, then each test provider ──
    idx = np.random.default_rng(0).choice(len(X), min(a.n_train, len(X)), replace=False)
    parts = [shap07.table(*linear_shap(pipe, X, X.iloc[idx], num), X.iloc[idx], num, "train")]
    test = data.load_test().reset_index(drop=True)
    Xt = design(test, pd.read_csv(config.FEATURES_TEST, dtype={"session_id": str}), num)
    for src in ("tsl", "eedi"):
        m = (test.data_source == src).to_numpy()
        parts.append(shap07.table(*linear_shap(pipe, X, Xt[m], num), Xt[m], num, f"test_{src}"))
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(d / "shap_by_column.csv", index=False)

    # ── the by-state grid, as stage 19, on the same twelve moves as XGBoost's Table 3 ──
    xgb_moves = pd.read_csv(config.RESULTS / "shap-by-state" / "shap_by_state.csv",
                            index_col=0).index
    moves = [agg.column(m.upper(), "tutor") for m in xgb_moves]
    idx = np.random.default_rng(0).choice(len(X), min(a.n_state, len(X)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    S, _ = linear_shap(pipe, X, Xs, moves)
    grid = state19.state_grid(S, Xs, moves)
    grid.to_csv(d / "shap_by_state.csv")

    # how far the XGBoost grid is from the one correlation alone produces
    xg = pd.read_csv(config.RESULTS / "shap-by-state" / "shap_by_state.csv", index_col=0)
    states = list(state19.STATES)
    lr = grid.loc[xg.index, states].to_numpy(float).ravel()
    xb = xg[states].to_numpy(float).ravel()
    tr = out[out.view == "train"].set_index("column")
    xs = pd.read_csv(config.RESULTS / "arm-both" / "shap_by_column.csv")
    xs = xs[xs.view == "train"].set_index("column")
    both = tr.join(xs, lsuffix="_lr", rsuffix="_xgb", how="inner")
    codes = both.drop(index="objective")
    res = {"run": a.run, "C": a.C,
           "share_tutor": float(tr.loc[[c for c in tr.index if c.startswith("v3t_")], "share"].sum()),
           "share_student": float(tr.loc[[c for c in tr.index if c.startswith("v3s_")], "share"].sum()),
           "share_objective": float(tr.loc["objective", "share"]),
           "spearman_share_vs_xgb": float(codes.share_lr.rank().corr(codes.share_xgb.rank())),
           "direction_agreement_vs_xgb": float((codes.direction_lr == codes.direction_xgb).mean()),
           "bystate_corr_vs_xgb": float(np.corrcoef(lr, xb)[0, 1]),
           "bystate_max_range_lr": float(grid["range"].max()),
           "bystate_max_range_xgb": float(xg["range"].max())}
    print(f"shares (train): tutor {res['share_tutor']:.3f}, student {res['share_student']:.3f}, "
          f"objective {res['share_objective']:.3f}")
    print(f"vs XGBoost: share rank correlation {res['spearman_share_vs_xgb']:.2f}, "
          f"direction agreement {res['direction_agreement_vs_xgb']:.0%}")
    print(f"by-state grid: correlation with XGBoost's {res['bystate_corr_vs_xgb']:.2f}; "
          f"largest swing {res['bystate_max_range_lr']:.1f}% (XGBoost {res['bystate_max_range_xgb']:.1f}%)")
    print(grid.round(1).to_string())
    runlog.finish(d, res)


if __name__ == "__main__":
    main()
