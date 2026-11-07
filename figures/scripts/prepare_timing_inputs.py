#!/usr/bin/env python
"""Inputs for fig_timing_{tutor,student}_si: where in a session each code occurs.

Descriptive only: the model never sees position (its 57 conversation columns are session rates),
and the paper has no localisation analysis. This shows where the
codes the rates count come from.

Reads the training turn-level labels (`data/derived/labels_v3/v3_full_train_<role>_shard*.csv`,
the primary `code`, which is what the rates use) and `data/derived/turns_train.parquet` for each
turn's position. A turn's relative position is pos / (n_turns - 1), 0 the first turn and 1 the
last, split into 20 equal bins. Sessions shorter than 20 turns are left out, so a bin is never
narrower than a turn.

Writes `figures/input/fig_timing.csv`: one row per (role, code, bin) with the code's turns in the
bin, and `lift` -- the code's share of its turns in that bin over the share of all that role's
labelled turns in the bin. Lift 1 is a code spread like the role's turns as a whole; 2 is twice
as concentrated there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
IN = ROOT / "figures" / "input"
BINS, MIN_TURNS = 20, 20

from ttapub import config  # noqa: E402


def main() -> None:
    turns = pd.read_parquet(ROOT / "data" / "derived" / "turns_train.parquet",
                            columns=["session_id", "turn", "pos", "n_turns"])
    turns = turns[turns.n_turns >= MIN_TURNS]
    turns["bin"] = np.minimum((turns.pos / (turns.n_turns - 1) * BINS).astype(int), BINS - 1)
    out = []
    for role in ("tutor", "student"):
        lab = pd.concat([pd.read_csv(f, dtype={"session_id": str}, usecols=["session_id", "turn", "code"])
                         for f in sorted(config.LABELS_V3.glob(f"v3_full_train_{role}_shard*.csv"))])
        m = lab.merge(turns[["session_id", "turn", "bin"]], on=["session_id", "turn"])
        base = m.bin.value_counts(normalize=True).sort_index()
        n = m.groupby(["code", "bin"]).size().unstack(fill_value=0).reindex(columns=range(BINS),
                                                                            fill_value=0)
        lift = n.div(n.sum(1), axis=0) / base.to_numpy()
        for code in n.index:
            out += [dict(role=role, code=code, bin=b, n=int(n.loc[code, b]),
                         lift=round(float(lift.loc[code, b]), 4)) for b in range(BINS)]
        print(f"{role}: {len(m):,} labelled turns in {m.session_id.nunique():,} sessions "
              f"of {MIN_TURNS}+ turns, {len(n)} codes")
    pd.DataFrame(out).to_csv(IN / "fig_timing.csv", index=False)
    print(f"wrote {IN / 'fig_timing.csv'}")


if __name__ == "__main__":
    main()
