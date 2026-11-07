#!/usr/bin/env python
"""Stage 24 — how much of a tutor move's attribution is interaction with the student's state.

The by-state grid (stage 19, Table 3) cannot separate interaction from co-occurrence, and its
own-rate-removed version (`--own-rate-removed`) leaves at most 4.2%. This measures interaction
directly. TreeSHAP interaction values split each feature's SHAP exactly into a main effect
(the diagonal) plus one term per pairing with every other feature, summing back to the SHAP:

    SHAP_j = I[j, j] + sum_{k != j} I[j, k]

so `sum_{k in student signals} I[j, k]` is the part of move j's attribution that exists only
because of how the model combines it with the student's signals -- the learned interaction,
with no co-occurrence in it.

Per move: main-effect share, and the shares of the move's mean |SHAP| that are interaction with
the student signals, with the other tutor moves and with the objective text. Per move and
state: the interaction with that state's signals, averaged within the top third of sessions on
the state and centred on its overall mean, as a percentage of the move's mean |SHAP| -- Table 3's
cell with only interaction in it.

Cost: the pairwise array is (rows, 1515, 1515), about 61 s per 50 rows per model on a laptop, so
this uses one seed's model and a seeded sample of rows rather than all five seeds.

    python scripts/24_shap_interactions.py --run shap-interactions --n 1000
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import time

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
import xgboost as xgbl

from ttapub import agg, config, data, runlog

_spec = importlib.util.spec_from_file_location(
    "state19", pathlib.Path(__file__).with_name("19_shap_by_state.py"))
state19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(state19)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="shap-interactions")
    ap.add_argument("--arm", default="arm-both")
    ap.add_argument("--model", type=int, default=0, help="which seed's model to explain")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=50)
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a))

    bundle = joblib.load(config.RESULTS / a.arm / "models.joblib")
    m, num = bundle["models"][a.model], bundle["num"]
    tr = data.load_train().reset_index(drop=True)
    feats = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    X = tr[["session_id"]].merge(feats[["session_id"] + num], on="session_id", how="left")[num]
    X["learning_objective"] = tr.learning_objective.to_numpy()
    X["learning_objective_id"] = tr.learning_objective_id.to_numpy()
    Xs = X.iloc[np.random.default_rng(0).choice(len(X), min(a.n, len(X)),
                                                replace=False)].reset_index(drop=True)

    # the twelve moves of Table 3, in its order
    plain = pd.read_csv(config.RESULTS / "shap-by-state" / "shap_by_state.csv", index_col=0)
    moves = [agg.column(c.upper(), "tutor") for c in plain.index]
    names = list(m.named_steps["pre"].get_feature_names_out())
    J = [names.index(f"dlg__{c}") for c in moves]
    STU = [i for i, n in enumerate(names) if n.startswith("dlg__v3s_")]
    TUT = [i for i, n in enumerate(names) if n.startswith("dlg__v3t_")]
    OBJ = [i for i, n in enumerate(names) if n.startswith("obj_text__")]
    states = {s: [names.index(f"dlg__{agg.column(c, 'student')}") for c in codes]
              for s, codes in state19.STATES.items()}

    n, k = len(Xs), len(moves)
    total = np.zeros((n, k)); main_ = np.zeros((n, k))
    w_stu = np.zeros((n, k)); w_tut = np.zeros((n, k)); w_obj = np.zeros((n, k))
    w_state = {s: np.zeros((n, k)) for s in states}
    booster = m[-1].get_booster()
    t0 = time.time()
    for lo in range(0, n, a.chunk):
        hi = min(lo + a.chunk, n)
        Z = m[:-1].transform(Xs.iloc[lo:hi])
        I = booster.predict(xgbl.DMatrix(Z, feature_names=names), pred_interactions=True)
        for jj, j in enumerate(J):
            row = I[:, j, :-1]                     # move j against every feature (bias dropped)
            total[lo:hi, jj] = row.sum(axis=1)
            main_[lo:hi, jj] = row[:, j]
            w_stu[lo:hi, jj] = row[:, STU].sum(axis=1)
            w_tut[lo:hi, jj] = row[:, [t for t in TUT if t != j]].sum(axis=1)
            w_obj[lo:hi, jj] = row[:, OBJ].sum(axis=1)
            for s, idx in states.items():
                w_state[s][lo:hi, jj] = row[:, idx].sum(axis=1)
        print(f"  {hi}/{n} rows, {time.time() - t0:.0f}s", flush=True)

    scale = np.abs(total).mean(axis=0)
    labels = [c.replace("v3t_", "").replace("_rate", "") for c in moves]
    shares = pd.DataFrame({
        "mean_abs_shap": scale,
        "main_effect": np.abs(main_).mean(0) / scale,
        "x_student": np.abs(w_stu).mean(0) / scale,
        "x_other_tutor": np.abs(w_tut).mean(0) / scale,
        "x_objective": np.abs(w_obj).mean(0) / scale,
    }, index=labels)
    shares.to_csv(d / "interaction_shares.csv")

    # Table 3's cell, with only the interaction with that state's own signals in it
    state_vals = {s: Xs[[agg.column(c, "student") for c in codes]].sum(axis=1).to_numpy()
                  for s, codes in state19.STATES.items()}
    grid = pd.DataFrame(index=labels, columns=list(states), dtype=float)
    for s, v in state_vals.items():
        top = v >= np.quantile(v, 2 / 3)
        grid[s] = 100 * (w_state[s][top].mean(0) - w_state[s].mean(0)) / scale
    grid["mean_abs_shap"] = scale
    grid.to_csv(d / "interaction_by_state.csv")

    cells = grid[list(states)].abs().to_numpy()
    res = {"run": a.run, "model_seed_index": a.model, "n_rows": n, "moves": labels,
           "median_main_effect_share": float(shares.main_effect.median()),
           "median_x_student_share": float(shares.x_student.median()),
           "max_x_student_share": float(shares.x_student.max()),
           "max_x_student_move": shares.x_student.idxmax(),
           "median_x_other_tutor_share": float(shares.x_other_tutor.median()),
           "median_x_objective_share": float(shares.x_objective.median()),
           "grid_max_abs_cell": float(cells.max()), "grid_median_abs_cell": float(np.median(cells))}
    print(shares.round(3).to_string())
    print("\ninteraction-only by-state grid (% of the move's mean |SHAP|):")
    print(grid[list(states)].round(1).to_string())
    runlog.finish(d, res)


if __name__ == "__main__":
    main()
