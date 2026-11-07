#!/usr/bin/env python
"""Stage 19 — how the model's valuation of a tutor move shifts with the student's state.

The regression test of moderation (stage 17) is bounded by measurement: an interaction's
reliability is the product of its components', and at 0.12--0.22 nothing can be bounded below a
point per standard deviation. This asks a weaker but answerable question instead --- not *does
a move's effect depend on the student's state*, but *does the model price the move differently
when the student is doing something different* --- and it is answerable because attribution is
a property of the fitted model, measured without sampling error, rather than a parameter being
estimated through a noisy regressor.

Per-row SHAP for each tutor-move column, averaged
within tertiles of each student signal's session rate, **doubly centred** (row and column means
removed) so that neither a generally important move nor a generally extreme signal can drive a
cell, and expressed as a percentage of that column's own mean |SHAP| so moves of very different
weight are comparable.

⚠️ This describes the model, not the student. A cell says the model reads this move differently
in sessions where the student does more of that; it does not say the move works differently.

    python scripts/19_shap_by_state.py --run shap-by-state
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
import xgboost as xgbl

from ttapub import agg, config, data, runlog

STATES = {"reasoning shown": ["EXPLAINS_WHY", "SHOWS_METHOD"],
          "not knowing": ["EXPRESSES_NOT_KNOWING", "FRUSTRATION"],
          "confident answers": ["CONFIDENT_ANSWER"],
          "hedging": ["HEDGED_ANSWER", "ASKS_VERIFICATION"]}


def remove_own_rate(S: np.ndarray, Xs: pd.DataFrame, moves: list[str],
                    bins: int = 20) -> np.ndarray:
    """SHAP with each move's dependence on its *own* rate removed.

    Subtracts, within each of `bins` quantile bins of the move's rate, that bin's mean SHAP. What
    remains is how the move's attribution varies *at a fixed rate of that move* -- the part that
    can depend on anything else, including the student's state. For a model without
    interactions it is exactly zero, so it isolates what the plain grid cannot: the plain grid
    also moves whenever a move's rate is merely correlated with a state's.
    """
    R = S.copy()
    for j, k in enumerate(moves):
        q = pd.qcut(pd.Series(Xs[k].to_numpy(float)).rank(method="first"), bins,
                    labels=False).to_numpy()
        for b in np.unique(q):
            R[q == b, j] -= S[q == b, j].mean()
    return R


def state_grid(S: np.ndarray, Xs: pd.DataFrame, moves: list[str],
               scale: np.ndarray | None = None) -> pd.DataFrame:
    """The doubly centred grid: one row per move (columns of S), one column per student state.

    A function so that every learner's grid is computed by the same code -- stage 23 calls it
    for logistic regression, and a comparison across learners is only as good as that.
    """
    state_vals = {name: Xs[[agg.column(c, "student") for c in codes]].sum(axis=1).to_numpy()
                  for name, codes in STATES.items()}
    grid = pd.DataFrame(index=[m.replace("v3t_", "").replace("_rate", "") for m in moves],
                        columns=list(STATES), dtype=float)
    for name, v in state_vals.items():
        hi = v >= np.quantile(v, 2 / 3)
        grid[name] = S[hi].mean(axis=0)
    # double centring: remove what is generally true of a move and of a state
    g = grid.to_numpy(float)
    g = g - g.mean(axis=1, keepdims=True) - g.mean(axis=0, keepdims=True) + g.mean()
    # `scale` lets a residual grid be expressed on the full attribution's scale
    scale = np.abs(S).mean(axis=0) if scale is None else scale
    denom = scale[:, None]
    pct = pd.DataFrame(100 * g / np.where(denom == 0, np.nan, denom),
                       index=grid.index, columns=grid.columns)
    pct["mean_abs_shap"] = scale.round(4)
    pct["range"] = pct[list(STATES)].max(axis=1) - pct[list(STATES)].min(axis=1)
    return pct.sort_values("range", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="arm-both")
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--run", default="shap-by-state")
    ap.add_argument("--top", type=int, default=12,
                    help="keep only the moves the model actually uses. The grid is a "
                         "percentage of a column's own mean |SHAP|, so a column the model "
                         "barely reads has a near-zero denominator and posts a huge, "
                         "meaningless shift -- an unrestricted ranking is sorted by noise.")
    ap.add_argument("--own-rate-removed", action="store_true",
                    help="grid of SHAP with each move's own-rate dependence removed, on the "
                         "same moves as the plain run: what is left is interaction")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a))

    bundle = joblib.load(config.RESULTS / a.arm / "models.joblib")
    models, num = bundle["models"], bundle["num"]
    tr = data.load_train().reset_index(drop=True)
    feats = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    X = tr[["session_id"]].merge(feats[["session_id"] + num], on="session_id", how="left")[num]
    X["learning_objective"] = tr.learning_objective.to_numpy()
    X["learning_objective_id"] = tr.learning_objective_id.to_numpy()
    idx = np.random.default_rng(0).choice(len(X), min(a.n, len(X)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)

    moves_all = [c for c in num if c.startswith("v3t_")]
    moves = moves_all
    S = np.zeros((len(Xs), len(moves)))
    for m in models:
        names = list(m.named_steps["pre"].get_feature_names_out())
        Z = m[:-1].transform(Xs)
        c = m[-1].get_booster().predict(xgbl.DMatrix(Z, feature_names=names),
                                        pred_contribs=True)[:, :-1]
        S += c[:, [names.index(f"dlg__{k}") for k in moves]]
    S /= len(models)
    print(f"SHAP for {len(moves)} tutor moves over {len(Xs):,} rows, {len(models)} seeds")

    # keep the most-read moves, then recompute on that subset
    order = np.argsort(-np.abs(S).mean(axis=0))[: a.top]
    moves = [moves[i] for i in order]
    S = S[:, order]
    print(f"keeping the {len(moves)} moves with the largest mean |SHAP|")

    explained = None
    if a.own_rate_removed:
        # same twelve moves as the plain grid, so the two are read cell against cell
        plain = pd.read_csv(config.RESULTS / "shap-by-state" / "shap_by_state.csv", index_col=0)
        keep = [moves.index(agg.column(m.upper(), "tutor")) for m in plain.index]
        moves, S = [moves[i] for i in keep], S[:, keep]
        R = remove_own_rate(S, Xs, moves)
        explained = float(np.median(1 - R.var(0) / S.var(0)))
        print(f"own rate explains a median {explained:.0%} of each move's SHAP variance")
        pct = state_grid(R, Xs, moves, scale=np.abs(S).mean(axis=0))
    else:
        pct = state_grid(S, Xs, moves)
    pct.to_csv(d / "shap_by_state.csv")
    print("\n=== how the model's valuation of a move shifts across student states ===")
    print("    (doubly centred, % of that move's own mean |SHAP|; top third of each state)")
    print(pct.head(12).round(1).to_string())
    runlog.finish(d, {"run": a.run, "n_rows": int(len(Xs)), "n_moves": len(moves),
                      "max_range_pct": float(pct.range.max()),
                      "own_rate_removed": a.own_rate_removed,
                      "own_rate_share_of_shap_variance": explained})


if __name__ == "__main__":
    main()
