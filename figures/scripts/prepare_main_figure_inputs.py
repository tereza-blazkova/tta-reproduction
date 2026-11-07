#!/usr/bin/env python
"""Distil `results/` into the four small CSVs the main figure reads.

Kept separate from the figure because a figure
script that reaches into `results/` cannot be re-rendered from a committed tree, and every
assertion about the data belongs on this side of the line. The figure reads only
`figures/input/`.

    python figures/scripts/prepare_main_figure_inputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
IN = ROOT / "figures" / "input"
RES = ROOT / "results"

from ttapub import taxonomy_v3 as T  # noqa: E402


#: One ladder per learner, laddered by identical contrasts (08_block_ladder.py --prefix).
#: A learner whose ladder does not exist yet is left out, and fig_ablation draws its slot as
#: "not yet run" -- a missing ladder must never become a row of zeros.
#: `ladder-tabpfn35` exists but is built from the SVD32 arms, which reduce the objective text
#: and so do not see the same columns; it is deliberately not read here.
LADDERS = {"XGBoost": "ladder-v3", "logistic regression": "ladder-logreg"}

#: Foundation models: each block dropped from the full model on test (27_paired_drop.py, labels
#: "<learner>|<block>"). The same contrast as the ladder's "block over objective + the other
#: block", with the opposite sign convention: dLL = LL(without) - LL(with).
#: Every learner on the same 1,514 columns, as XGBoost and logistic regression are: the
#: foundation models' 5-seed runs at full width, session-clustered intervals like the ladders'.
#: (`drop-blocks-svd32`, the 89-column check, and `drop-blocks-tfidf5-seeds`, the seed-aware
#: intervals, are reported in the appendix, not drawn here.)
DROPS = ["drop-blocks-tfidf5"]
SIZE: dict[str, str] = {}


def panel_a() -> pd.DataFrame:
    """What each block adds, by provider and by learner, with intervals."""
    want = {"student signals over objective + tutor moves": "student signals",
            "tutor moves over objective + student signals": "tutor moves"}
    rows = []
    for learner, run in LADDERS.items():
        path = RES / run / "metrics.json"
        if not path.exists():
            print(f"  {learner}: no {run}, drawn as not yet run")
            continue
        m = json.loads(path.read_text())
        for view, label in (("oof_all", "TSL (out-of-fold)"), ("test_tsl", "TSL (test)"),
                            ("test_eedi", "Eedi (test)")):
            pairs = m["views"][view]["pairs"]
            for key, block in want.items():
                if key not in pairs:
                    raise SystemExit(f"{run} is missing the pair {key!r} in {view}")
                p = pairs[key]
                rows.append({"learner": learner, "view": label, "block": block, "n": p["n"],
                             "d_auroc": p["d_auroc"], "lo": p["d_auroc_ci"][0],
                             "hi": p["d_auroc_ci"][1], "d_ll": p["d_ll"],
                             "ll_lo": p["d_ll_ci"][0], "ll_hi": p["d_ll_ci"][1]})
    d = pd.concat([pd.read_csv(RES / run / "paired_drop.csv") for run in DROPS])
    d = d[d.view.isin(["tsl", "eedi"])]
    for _, r in d.iterrows():
        learner, block = r.learner.split("|")
        learner = SIZE.get(learner, learner)
        label = {"tsl": "TSL (test)", "eedi": "Eedi (test)"}[r["view"]]
        rows.append({"learner": learner, "view": label, "block": block, "n": r.n,
                     "d_auroc": r.dAUROC, "lo": r.dAUROC_lo, "hi": r.dAUROC_hi,
                     "d_ll": -r.dLL, "ll_lo": -r.dLL_hi, "ll_hi": -r.dLL_lo})
    return pd.DataFrame(rows)


def panel_b() -> pd.DataFrame:
    """Per-code cross-annotator kappa, averaged over the three annotator pairings."""
    b = pd.read_csv(RES / "panel-v3" / "panel_by_code.csv")
    k = (b.groupby(["role", "code"]).kappa.mean().reset_index()
          .rename(columns={"kappa": "kappa_mean"}))
    k["n_pairings"] = b.groupby(["role", "code"]).kappa.size().values
    return k.sort_values(["role", "kappa_mean"])


def panel_c() -> pd.DataFrame:
    """The two instruments' performance, by provider."""
    rows = []
    for run, inst in (("arm-both", "122B labels"), ("arm-both-dst", "distilled children")):
        m = json.loads((RES / run / "metrics.json").read_text())
        for view, label in (("test_tsl", "TSL (test)"), ("test_eedi", "Eedi (test)")):
            rows.append({"instrument": inst, "view": label, "auroc": m[view]["auroc"],
                         "ll": m[view]["ll"], "n": m[view]["n"]})
    return pd.DataFrame(rows)


def panel_d() -> pd.DataFrame:
    """Per code: cross-annotator kappa against how well a CPU child reproduces its rate
    off-provider. Two independent measurement axes, one row per code."""
    k = panel_b()
    f = pd.read_csv(RES / "clean57-children" / "feature_fidelity_r.csv")
    f = f[(f.split == "test") & (f.group == "eedi")].copy()
    # v3t_<code>_rate / v3s_<code>_rate -> CODE
    f["code"] = (f.col.str.replace(r"^v3[ts]_", "", regex=True)
                      .str.replace(r"_rate$", "", regex=True).str.upper())
    m = k.merge(f[["code", "role", "r"]], on=["code", "role"], how="inner")
    missing = set(k.code) - set(m.code)
    if missing:
        print(f"  note: {len(missing)} codes have kappa but no Eedi fidelity "
              f"(too rare to correlate): {sorted(missing)[:6]}")
    return m.rename(columns={"r": "fidelity_r_eedi"})


def main() -> None:
    IN.mkdir(parents=True, exist_ok=True)
    for name, fn in (("fig_main_a_transfer", panel_a), ("fig_main_b_kappa", panel_b),
                     ("fig_main_c_instrument", panel_c), ("fig_main_d_convergence", panel_d)):
        df = fn()
        # sanity: the pools must be the ones the codebook defines, or the figure is mislabelled
        if "code" in df.columns:
            pool = set(T.codes("tutor")) | set(T.codes("student"))
            bad = set(df.code) - pool
            if bad:
                raise SystemExit(f"{name}: codes outside the v3 pools: {sorted(bad)[:5]}")
        df.to_csv(IN / f"{name}.csv", index=False)
        print(f"wrote {IN / f'{name}.csv'}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
