#!/usr/bin/env python
"""Stage 01 — transcripts -> the per-turn table every later stage is keyed on.

One row per paired turn, for both splits: ids, position, the two child input texts, and the
`has_teacher` / `has_student` flags that define each role's turn axis. This is the join key for
the LLM labels (stage 02), the distilled probabilities (stage 04) and the session rates
(stage 05), so it is built once and never rebuilt selectively — a turn numbering that differs
between stages silently mislabels turns.

    python scripts/01_build_turn_table.py --split train
    python scripts/01_build_turn_table.py --split test
"""
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import pandas as pd

from ttapub import config, data, dialogue

OUT = {"train": config.TURNS_TRAIN, "test": config.TURNS_TEST}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N sessions only (smoke test)")
    a = ap.parse_args()

    frame = data.load_train() if a.split == "train" else data.load_test()
    sids = list(dict.fromkeys(frame.session_id))
    if a.limit:
        sids = sids[: a.limit]
    store = data.store(a.split)

    t0 = time.time()
    parts, empty = [], []
    for i, (sid, tx) in enumerate(store.iter(sids), 1):
        turns = dialogue.build_turns(tx)
        if not turns:
            empty.append(sid)
            continue
        parts.append(dialogue.turn_frame(sid, turns))
        if i % 2000 == 0:
            print(f"  {i:,}/{len(sids):,} sessions  {time.time() - t0:.0f}s", flush=True)

    df = pd.concat(parts, ignore_index=True)
    config.ensure(config.DERIVED)
    out = OUT[a.split] if not a.limit else OUT[a.split].with_suffix(".smoke.parquet")
    df.to_parquet(out, index=False)
    print(f"[{a.split}] {len(df):,} turns over {df.session_id.nunique():,} sessions "
          f"({len(empty)} sessions with no usable turn)\n"
          f"  tutor turns {df.has_teacher.sum():,}  student turns {df.has_student.sum():,}  "
          f"median turns/session {df.groupby('session_id').size().median():.0f}\n"
          f"wrote {out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
