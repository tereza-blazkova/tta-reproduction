#!/usr/bin/env python
"""Inputs for Fig. 1d: what one session looks like (and how long sessions are, no longer drawn).

Reads the training turn labels (`data/derived/labels_v3/v3_full_train_<role>_shard*.csv`, the
primary `code`). A turn is one tutor-student exchange: 99.4% carry both a tutor and a student
label, and every session's turns are numbered without gaps, so a session's length is its number
of labelled turns.

The example session is chosen by a rule, not by eye: among sessions within 8 turns of the median
length, the one whose share of turns in each code group (below) is closest to the corpus average
(least squares over the ten groups). Code groups are stage 33's (the tutor blocks, after Vanacore
et al. 2026, and stage 17's student states), plus one group per role for the codes that organise
the session (tutor: guiding, technical, tool, context, choice) or step outside it (student:
social, off task, technical). Codes in no group (residuals, no move, and a few others) leave the
turn blank in the example.

Writes to `figures/input/`:
* `fig_session_length.csv` -- training sessions by length in turns (`turns`, `sessions`).
* `fig_example_session.csv` -- the example: one row per (turn, role) with its code and group (empty
  for a code in no group), and `n_turns`. No identifier: the session is printed when this runs and
  is recovered by rerunning the rule.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
IN = ROOT / "figures" / "input"
WINDOW = 8                                    # turns either side of the median length

from ttapub import config  # noqa: E402

s33 = importlib.import_module("33_block_sequences")
TUTOR = dict(s33.BLOCKS) | {"organising": ["GUIDING_SESSION", "TECHNICAL_ISSUES", "EXPLAINING_TOOL",
                                           "ASKING_TO_CLARIFY_CONTEXT", "OFFERING_CHOICE"]}
STUDENT = dict(s33.STATES) | {"social": ["STUDENT_SOCIAL", "OFF_TASK", "STUDENT_TECHNICAL"]}
GROUPS = {"tutor": TUTOR, "student": STUDENT}


def main() -> None:
    lab = []
    for role in ("tutor", "student"):
        d = pd.concat([pd.read_csv(f, dtype={"session_id": str}, usecols=["session_id", "turn", "code"])
                       for f in sorted(config.LABELS_V3.glob(f"v3_full_train_{role}_shard*.csv"))])
        code2group = {c: g for g, cs in GROUPS[role].items() for c in cs}
        d["role"], d["group"] = role, d.code.map(code2group)
        lab.append(d)
    lab = pd.concat(lab, ignore_index=True)
    span = lab.groupby("session_id").turn.agg(["min", "max", "nunique"])
    if not (span["nunique"] == span["max"] - span["min"] + 1).all():
        raise SystemExit("a session's turns are not numbered without gaps")
    n = span["nunique"]
    pd.DataFrame({"turns": n.value_counts().sort_index().index,
                  "sessions": n.value_counts().sort_index().to_numpy()}).to_csv(
        IN / "fig_session_length.csv", index=False)
    # the example: typical length, then typical mix of groups
    keys = [(r, g) for r in ("tutor", "student") for g in GROUPS[r]]
    share = (lab.dropna(subset=["group"]).groupby(["session_id", "role", "group"]).size()
             .unstack(["role", "group"], fill_value=0).reindex(columns=keys, fill_value=0)
             .div(n, axis=0).fillna(0))
    near = n[(n - n.median()).abs() <= WINDOW].index
    dist = ((share.loc[near] - share.mean()) ** 2).sum(axis=1)
    sid = dist.idxmin()
    ex = lab[lab.session_id == sid].copy()
    ex["turn"] = ex.turn - ex.turn.min()                  # 0 = the first turn
    ex = ex.sort_values(["turn", "role"])[["turn", "role", "code", "group"]]
    ex["n_turns"] = int(n[sid])
    ex.to_csv(IN / "fig_example_session.csv", index=False)
    print(f"{len(n):,} sessions, median {n.median():.0f} turns; example {sid}: {n[sid]} turns, "
          f"distance {dist.min():.4f} (median of the {len(near):,} candidates {dist.median():.4f})")


if __name__ == "__main__":
    main()
