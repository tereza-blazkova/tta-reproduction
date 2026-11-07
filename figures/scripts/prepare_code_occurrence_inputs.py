#!/usr/bin/env python
"""Inputs for fig_code_occurrence (stage 48, `results/code-occurrence`): per code, its mean share of
the role's turns among weak, middle and strong students (thirds) and attainment fifths, and the
strong-vs-weak and top-vs-bottom-fifth contrasts. One row per code, no identifiers.

    python figures/scripts/prepare_code_occurrence_inputs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "figures" / "input"
R = ROOT / "results" / "code-occurrence"


def main() -> None:
    m = json.loads((R / "metrics.json").read_text())
    pd.read_csv(R / "codes.csv").to_csv(IN / "fig_occur_codes.csv", index=False)
    pd.Series({"students": m["students"], "sessions": m["sessions"], **{f"bh_{k}": v for k, v in m["bh"].items()},
               "both_same_sign": m["both_same_sign"]}, name="value").rename_axis("key").to_csv(IN / "fig_occur_summary.csv")
    print(m["bh"], m["both_same_sign"])


if __name__ == "__main__":
    main()
