#!/usr/bin/env python
"""Inputs for fig_attainment_vs_transcript, fig_add_one_in and fig_cumulative_oof (stages 46 and 47):
aggregates per view, model and code, no identifiers.

    python figures/scripts/prepare_attainment_incremental_inputs.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "figures" / "input"
RES = ROOT / "results"


def main() -> None:
    pd.read_csv(RES / "attainment-incremental" / "models.csv").to_csv(IN / "fig_attinc_models.csv", index=False)
    pd.read_csv(RES / "attainment-incremental" / "add_one_in.csv").to_csv(IN / "fig_attinc_add.csv", index=False)
    cur = RES / "cumulative-oof-order" / "curve.csv"
    if cur.exists():
        pd.read_csv(cur).to_csv(IN / "fig_attinc_curve.csv", index=False)
    print("ok")


if __name__ == "__main__":
    main()
