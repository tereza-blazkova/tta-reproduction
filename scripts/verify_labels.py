#!/usr/bin/env python
"""Verify an annotation run before anything is built from it.

Checks, per split and role: turn-level coverage against the stage-01 turn table, session
coverage, duplicate `(session_id, turn)` keys, null codes, and codes outside the pool (which
would mean the annotation and `docs/taxonomy_v3.md` disagree).

**A session with no turns of a role is not a gap.** Five training sessions are single-turn
transcripts in which the student never speaks, so the student pass has nothing to classify.
An earlier version of this check counted those as missing labels and cried wolf; coverage is
therefore asserted at the level that actually matters — **turns**, not sessions.

    python scripts/verify_labels.py --tag full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from ttapub import config, data
from ttapub import taxonomy_v3 as T

FLAG = {"tutor": "has_teacher", "student": "has_student"}
TURNS = {"train": config.TURNS_TRAIN, "test": config.TURNS_TEST}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--max-unlabelled", type=float, default=1e-4,
                    help="tolerated share of turns with no label (default 0.01%%). The runtime "
                         "guard in stage 02 counts CHUNK parse failures; it cannot see a turn "
                         "the model silently omitted from otherwise valid JSON, which is what "
                         "this catches. A handful of omitted backchannels is not a broken run, "
                         "so the verdict is a rate with the count always printed.")
    a = ap.parse_args()

    ok = True
    for split, loader in (("train", data.load_train), ("test", data.load_test)):
        # only the key and flag columns: the text columns are ~95% of the file and this check
        # never looks at them (reading them all took minutes per split)
        turns = pd.read_parquet(TURNS[split],
                                columns=["session_id", "turn", "has_teacher", "has_student"])
        in_split = set(loader().session_id)
        print(f"=== {split}: {len(in_split):,} sessions, {len(turns):,} paired turns ===")
        for role in ("tutor", "student"):
            files = sorted(config.LABELS_V3.glob(f"v3_{a.tag}_{split}_{role}_shard*.csv"))
            if not files:
                print(f"  {role:8s} NO LABEL FILES for tag {a.tag!r}")
                ok = False
                continue
            df = pd.concat([pd.read_csv(f, dtype={"session_id": str}) for f in files],
                           ignore_index=True)
            axis = turns[turns[FLAG[role]]]
            # The real coverage question: did every turn of this role get a label? Done as an
            # outer merge rather than python set arithmetic — the set version took >10 min on
            # 2.4M turns, which made the check something you skip, which defeats its purpose.
            j = axis[["session_id", "turn"]].merge(
                df[["session_id", "turn"]].assign(_lab=1),
                on=["session_id", "turn"], how="outer", indicator=True)
            unlabelled = int((j._merge == "left_only").sum())   # turn with no label
            orphan = int((j._merge == "right_only").sum())      # label with no turn
            unlab_rate = unlabelled / max(len(axis), 1)
            over = unlab_rate > a.max_unlabelled
            dup = int(df.duplicated(["session_id", "turn"]).sum())
            nulls = int(df.code.isna().sum())
            off = sorted(set(df.code.dropna()) - set(T.codes(role)))
            alsos = df.get("also", pd.Series(dtype=str)).fillna("")
            off_also = sorted(set(alsos[alsos != ""]) - set(T.codes(role)))
            # build the set ONCE: the obvious comprehension rebuilt it per session, which is
            # O(n^2) over a 2.4M-row column and turned a 10-second check into an 8-minute one
            with_turns = set(axis.session_id.unique())
            sess_no_turns = len(in_split - with_turns)
            print(f"  {role:8s} {len(df):>9,} labels / {len(axis):>9,} {role} turns"
                  f" | unlabelled {unlabelled} ({unlab_rate:.5%}{'  OVER LIMIT' if over else ''})"
                  f" | orphan {orphan} | dup {dup}"
                  f" | null {nulls} | off-pool {off or 'none'}"
                  f" | also off-pool {off_also or 'none'}"
                  + (f" | {sess_no_turns} sessions have no {role} turn (expected)"
                     if sess_no_turns else ""))
            if unlabelled and not over:
                gaps = j[j._merge == "left_only"][["session_id", "turn"]]
                shown = ", ".join(f"{r.session_id}:{r.turn}" for r in
                                  gaps.head(5).itertuples())
                print(f"           within tolerance; omitted turns: {shown}"
                      + (" …" if len(gaps) > 5 else ""))
            if over or orphan or dup or nulls or off or off_also:
                ok = False
    print("\n" + ("✅ verified: every turn of both roles carries one in-pool code, within "
                   "tolerance" if ok else
                   "❌ PROBLEM — do not build features from this annotation"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
