#!/usr/bin/env python
"""Inputs for fig_codes_add_attainment (stage 59, `results/attainment-recalibration-intervals`): per view
and code, the gain over the attainment model after recalibration, beyond the placebo, with its interval
and BH; and the same for the all-tutor, all-student and all-57 models. Aggregates only, no identifiers.

    python figures/scripts/prepare_recalibration_intervals_inputs.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "figures" / "input"
R = ROOT / "results" / "attainment-recalibration-intervals"


def main() -> None:
    pd.read_csv(R / "codes.csv").to_csv(IN / "fig_recal_codes.csv", index=False)
    pd.read_csv(R / "references.csv").to_csv(IN / "fig_recal_references.csv", index=False)
    print("ok")


if __name__ == "__main__":
    main()
