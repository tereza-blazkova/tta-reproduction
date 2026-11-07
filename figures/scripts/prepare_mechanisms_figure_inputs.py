#!/usr/bin/env python
"""Distil the SHAP tables into the three CSVs the mechanisms figure reads.

Attribution comes from `07_shap_clean57.py`, which reports per-column mean |SHAP| as a share of
the model's total and the direction (Spearman of the column's value against its own SHAP).
Three views exist per run — train, test TSL, test Eedi — and both instruments were scored, so
"does the explanation transport?" and "does it depend on the instrument?" are answerable from
the same table.

    python figures/scripts/prepare_mechanisms_figure_inputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
IN = ROOT / "figures" / "input"
RES = ROOT / "results"

from ttapub import taxonomy_v3 as T  # noqa: E402


def load(run: str) -> pd.DataFrame:
    d = pd.read_csv(RES / run / "shap_by_column.csv")
    d = d[d.column != "objective"].copy()
    d["role"] = np.where(d.column.str.startswith("v3t_"), "tutor", "student")
    d["code"] = (d.column.str.replace(r"^v3[ts]_", "", regex=True)
                   .str.replace(r"_rate$", "", regex=True).str.upper())
    return d


def main() -> None:
    IN.mkdir(parents=True, exist_ok=True)
    L = load("arm-both")
    D = load("arm-both-dst")

    # a — what the model reads, on the training rows, plus the construct totals
    a = L[L.view == "train"][["code", "role", "share", "direction", "spearman"]].copy()
    a = a.sort_values("share", ascending=False)
    tot = pd.read_csv(RES / "arm-both" / "shap_by_column.csv")
    tot = tot[tot.view == "train"]
    obj = float(tot[tot.column == "objective"].share.iloc[0])
    totals = pd.DataFrame([
        {"construct": "tutor moves", "share": float(a[a.role == "tutor"].share.sum())},
        {"construct": "student signals", "share": float(a[a.role == "student"].share.sum())},
        {"construct": "learning objective", "share": obj}])
    a.to_csv(IN / "fig_mech_a_attribution.csv", index=False)
    totals.to_csv(IN / "fig_mech_a_totals.csv", index=False)
    # the same split for logistic regression (stage 23's linear SHAP, the same training rows), for
    # Fig. 2's second shares bar
    lt = pd.read_csv(RES / "attribution-logreg" / "shap_by_column.csv")
    lt = lt[lt.view == "train"]
    side = lt.column.str[:4].map({"v3t_": "tutor moves", "v3s_": "student signals"}).fillna("learning objective")
    pd.DataFrame({"construct": ["tutor moves", "student signals", "learning objective"],
                  "share": [float(lt[side == c].share.sum()) for c in
                            ("tutor moves", "student signals", "learning objective")]}).to_csv(
        IN / "fig_mech_a_totals_logreg.csv", index=False)

    # b — does the explanation transport? attribution on TSL against on Eedi
    piv = L.pivot_table(index=["code", "role"], columns="view",
                        values="share").reset_index()
    dirs = L.pivot_table(index=["code", "role"], columns="view", values="direction",
                         aggfunc="first").reset_index()
    b = piv.merge(dirs, on=["code", "role"], suffixes=("", "_dir"))
    b = b.rename(columns={"test_tsl": "share_tsl", "test_eedi": "share_eedi",
                          "train": "share_train", "test_tsl_dir": "dir_tsl",
                          "test_eedi_dir": "dir_eedi", "train_dir": "dir_train"})
    b["dir_flip"] = (b.dir_tsl != b.dir_eedi) & (b.dir_tsl != "~") & (b.dir_eedi != "~")
    b.to_csv(IN / "fig_mech_b_transport.csv", index=False)

    # c — does the explanation depend on the instrument? 122B labels against distilled children
    c = (L[L.view == "train"][["code", "role", "share", "direction"]]
         .merge(D[D.view == "train"][["code", "role", "share", "direction"]],
                on=["code", "role"], suffixes=("_labels", "_dst")))
    c["dir_flip"] = ((c.direction_labels != c.direction_dst)
                     & (c.direction_labels != "~") & (c.direction_dst != "~"))
    c.to_csv(IN / "fig_mech_c_instrument.csv", index=False)

    # d -- how the model's valuation of a move shifts with what the student is doing
    g = pd.read_csv(RES / "shap-by-state" / "shap_by_state.csv", index_col=0)
    g.index.name = "move"
    # where the code came from, so the table can mark the published moves
    g["origin"] = [T.origin(m.upper(), "tutor") for m in g.index]
    g.reset_index().to_csv(IN / "fig_mech_d_by_state.csv", index=False)

    # supplementary: the same grid for logistic regression, and for XGBoost with each move's
    # own-rate dependence removed -- both on Table 3's twelve moves, in Table 3's row order
    for run, name, f in (("attribution-logreg", "fig_si_bystate_logreg.csv", "shap_by_state.csv"),
                         ("shap-by-state-ownrate", "fig_si_bystate_ownrate.csv", "shap_by_state.csv"),
                         ("shap-interactions", "fig_si_bystate_interaction.csv",
                          "interaction_by_state.csv")):
        h = pd.read_csv(RES / run / f, index_col=0)
        h = h.loc[g.sort_values("mean_abs_shap", ascending=False).index]   # Table 3's rows
        h["mean_abs_shap"] = g["mean_abs_shap"] if run == "shap-by-state-ownrate" else h["mean_abs_shap"]
        h.index.name = "move"
        h["origin"] = g["origin"]
        h.reset_index().to_csv(IN / name, index=False)

    # supplementary: per-code attribution share, XGBoost against logistic regression
    xs = pd.read_csv(RES / "arm-both" / "shap_by_column.csv")
    ls = pd.read_csv(RES / "attribution-logreg" / "shap_by_column.csv")
    sc = (xs[xs.view == "train"][["column", "share", "direction"]]
          .merge(ls[ls.view == "train"][["column", "share", "direction"]], on="column",
                 suffixes=("_xgb", "_lr")))
    sc = sc[sc.column != "objective"]
    sc["role"] = np.where(sc.column.str.startswith("v3s_"), "student", "tutor")
    sc["code"] = sc.column.str.replace(r"^v3[ts]_", "", regex=True).str.replace("_rate$", "", regex=True)
    sc.to_csv(IN / "fig_si_attr_xgb_vs_lr.csv", index=False)

    pool = set(T.codes("tutor")) | set(T.codes("student"))
    for name, df in (("a", a), ("b", b), ("c", c)):
        bad = set(df.code) - pool
        if bad:
            raise SystemExit(f"panel {name}: codes outside the v3 pools: {sorted(bad)[:5]}")
    print(f"wrote 4 input CSVs | a: {len(a)} codes, b: {len(b)}, c: {len(c)} "
          f"({int(c.dir_flip.sum())} direction flips between instruments, "
          f"{int(b.dir_flip.sum())} between providers)")
    print("share range:", f"{a.share.min():.5f} .. {a.share.max():.4f}")


if __name__ == "__main__":
    main()
