#!/usr/bin/env python
"""Stage 14 — code rates computed over the objective's own stretch of the session.

One row per RESPONSE (not per session), so two objectives taught in the same session get
different features and a session fixed effect becomes estimable. Each rate is also computed
over the **placebo** span — a different segment of the same session — which is what tells us
whether localisation carries objective-specific signal or merely re-cuts noise.

Rates use the same shrinkage as the session-level ones, toward the same corpus prior, so a
short segment is not allowed to produce an extreme rate. The denominator is the number of turns
of the coded role inside the span.

    python scripts/14_localised_features.py --split train --tag full
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, runlog
from ttapub import taxonomy_v3 as T

TURNS = {"train": config.TURNS_TRAIN, "test": config.TURNS_TEST}
FLAG = {"tutor": "has_teacher", "student": "has_student"}


def load_labels(tag: str, split: str, role: str) -> pd.DataFrame:
    files = sorted(config.LABELS_V3.glob(f"v3_{tag}_{split}_{role}_shard*.csv"))
    d = pd.concat([pd.read_csv(f, dtype={"session_id": str}) for f in files],
                  ignore_index=True).drop_duplicates(["session_id", "turn"])
    return d[["session_id", "turn", "code"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--tag", default="full")
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    run = a.run or f"localised-features-{a.split}"
    d = runlog.start(run, vars(a))

    turns = pd.read_parquet(TURNS[a.split],
                            columns=["session_id", "turn", "pos", "has_teacher", "has_student"])
    turns = turns.sort_values(["session_id", "pos"]).reset_index(drop=True)
    # row index of (session, pos): pos is contiguous from 0 within a session, so one offset
    # per session is enough, and a span never crosses a session boundary — which is what makes
    # a single global cumsum below correct.
    first = turns.groupby("session_id", sort=False).apply(lambda g: g.index[0], include_groups=False)
    offset = first.to_dict()

    loc = pd.read_csv(config.DERIVED / f"localisation_{a.split}.csv",
                      dtype={"session_id": str, "response_id": str})
    prior = agg.load_prior()

    out = {"response_id": loc.response_id.to_numpy()}
    for which, (c0, c1) in (("", ("pos_start", "pos_end")),
                            ("pl_", ("placebo_start", "placebo_end"))):
        base = np.array([offset[s] for s in loc.session_id])
        start = base + loc[c0].to_numpy()
        end = base + loc[c1].to_numpy()
        for role in ("tutor", "student"):
            mask = turns[FLAG[role]].to_numpy()
            csm = np.concatenate([[0], np.cumsum(mask)])
            n_role = csm[end] - csm[start]
            lab = load_labels(a.tag, a.split, role)
            code_of = (turns[["session_id", "turn"]].merge(lab, on=["session_id", "turn"],
                                                           how="left").code.to_numpy())
            for code in T.codes(role):
                ind = (code_of == code)
                cs = np.concatenate([[0], np.cumsum(ind)])
                hits = cs[end] - cs[start]
                r = (hits + agg.SHRINK_KAPPA * prior[code]) / (n_role + agg.SHRINK_KAPPA)
                out[which + agg.column(code, role)] = r.astype(np.float32)
            if which == "":
                out[f"n_{role}_turns_span"] = n_role.astype(np.int32)

    df = pd.DataFrame(out)
    path = config.DERIVED / f"features_localised_{a.split}.parquet"
    df.to_parquet(path, index=False)

    cols = agg.code_columns(df.columns)
    pl = [c for c in df.columns if c.startswith("pl_")]
    print(f"[{a.split}] {len(df):,} responses x {len(cols)} localised rates "
          f"(+{len(pl)} placebo)")
    print(f"  median turns in span: tutor {np.median(df.n_tutor_turns_span):.0f}, "
          f"student {np.median(df.n_student_turns_span):.0f}")
    # does localisation actually move the features? if matched and placebo agree, it does not
    same = np.mean([np.corrcoef(df[c], df["pl_" + c])[0, 1] for c in cols[:20]])
    print(f"  mean correlation between matched and placebo rates (first 20 codes): {same:.3f}")
    print(f"wrote {path}")
    runlog.finish(d, {"run": run, "split": a.split, "n": len(df), "n_cols": len(cols),
                      "matched_vs_placebo_r": float(same)})


if __name__ == "__main__":
    main()
