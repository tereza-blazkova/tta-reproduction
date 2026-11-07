#!/usr/bin/env python
"""Input for fig_attainment_blocks: stage 57's contrasts, aggregates with no identifiers.

    python figures/scripts/prepare_attainment_blocks_inputs.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    pd.read_csv(ROOT / "results" / "attainment-blocks" / "blocks.csv").to_csv(
        ROOT / "figures" / "input" / "fig_attainment_blocks.csv", index=False)
