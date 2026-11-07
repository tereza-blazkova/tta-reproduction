#!/usr/bin/env python
"""Inputs for the block-level exchanges panel: the pairs that clear BH pooled (stage 33).

Reads `results/block-sequences/block_sequences.csv` (hard labels, the primary arm) and writes
`figures/input/fig_block_sequences.csv` in the schema `fig_rq1.panel_b` draws: one row per
pair, pooled and within-student estimates with their intervals, and who opens the exchange.
"""
from __future__ import annotations

import pandas as pd

import style as st

#: short names for the panel; the caption spells the blocks and states out
SHORT = {"eliciting": "eliciting", "telling": "telling", "evaluating": "evaluating",
         "affective": "affective", "reasoning_shown": "reasoning", "not_knowing": "not knowing",
         "answering_confidently": "confident answer", "hedging": "hedging"}


def main() -> None:
    o = pd.read_csv(st.ROOT / "results" / "block-sequences" / "block_sequences.csv")
    o = o[(o.labels == "hard") & o.bh_sig_pooled]
    o = o.reindex(o.pooled_pp_per_sd.abs().sort_values(ascending=False).index)
    a, b = o.pair.str.split(" > ").str[0], o.pair.str.split(" > ").str[1]
    out = pd.DataFrame({
        "code": [f"{SHORT[x]} > {SHORT[y]}" for x, y in zip(a, b)],
        "role": ["tutor" if k == "uptake" else "student" for k in o.kind],
        "pp_per_sd_pooled": o.pooled_pp_per_sd, "lo_pooled": o.pooled_lo,
        "hi_pooled": o.pooled_hi, "pp_per_sd_within_student": o.pp_per_sd,
        "lo_within_student": o.lo, "hi_within_student": o.hi,
        "bh_sig_within_student": o.bh_sig.astype(int)})
    p = st.INPUT / "fig_block_sequences.csv"
    out.to_csv(p, index=False)
    print(f"wrote {p} ({len(out)} pairs)")


if __name__ == "__main__":
    main()
