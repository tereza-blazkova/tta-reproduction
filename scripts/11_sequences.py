#!/usr/bin/env python
"""Stage 11 — adjacent-turn sequences of tutor moves and student signals.

A turn here is one exchange: the tutor speaks, the student replies. That gives two directions,
and they answer different questions:

* **uptake** — the tutor's move at turn *t* followed by the student's signal in the *same* turn,
  i.e. what the student does in reply to that move. "Does an eliciting move actually elicit
  reasoning?"
* **response** — the student's signal at turn *t* followed by the tutor's move at turn *t+1*,
  i.e. what the tutor does about what the student just did. "What does a tutor do after a bare
  answer, or after the student says they do not know?"

Both are computed per session as a rate over the eligible transitions, shrunk toward the corpus
prior exactly as the single-code rates are (`ttapub.agg`), because a pair is rarer than either
of its components and a short session's pair rate is correspondingly noisier.

Writes the session x pair table for pairs that clear `--min-count`, plus the pair inventory
with lift over chance (the product of the two components' marginals), which is what makes an
exchange matrix readable.

    python scripts/11_sequences.py --tag full --split train
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, runlog
from ttapub import taxonomy_v3 as T

KAPPA = agg.SHRINK_KAPPA


def load_labels(tag: str, split: str) -> pd.DataFrame:
    """One row per (session, turn) with the tutor code and the student code side by side."""
    out = {}
    for role in ("tutor", "student"):
        files = sorted(config.LABELS_V3.glob(f"v3_{tag}_{split}_{role}_shard*.csv"))
        if not files:
            raise SystemExit(f"no {role} labels for {tag}_{split}")
        d = pd.concat([pd.read_csv(f, dtype={"session_id": str}) for f in files],
                      ignore_index=True).drop_duplicates(["session_id", "turn"])
        out[role] = d.rename(columns={"code": role})[["session_id", "turn", role]]
    return out["tutor"].merge(out["student"], on=["session_id", "turn"], how="outer")


def build_pairs(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """(session_id, pair) rows for one direction."""
    df = df.sort_values(["session_id", "turn"])
    if kind == "uptake":                       # tutor move -> the student's reply, same turn
        a, b = df.tutor, df.student
        ok = a.notna() & b.notna()
        pair = a.where(ok) + " > " + b.where(ok)
    else:                                      # student signal -> the tutor's next move
        nxt = df.groupby("session_id", sort=False).tutor.shift(-1)
        same = df.groupby("session_id", sort=False).turn.shift(-1) == df.turn + 1
        ok = df.student.notna() & nxt.notna() & same.fillna(False)
        pair = df.student.where(ok) + " > " + nxt.where(ok)
    out = pd.DataFrame({"session_id": df.session_id, "pair": pair})
    return out[out.pair.notna()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--min-count", type=int, default=2000,
                    help="a pair must occur this often across the corpus to get a column. "
                         "Rarer pairs give session rates that are mostly zero and whose "
                         "within-student variance is dominated by sampling noise.")
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    run = a.run or f"sequences-{a.split}"
    d = runlog.start(run, vars(a))

    lab = load_labels(a.tag, a.split)
    print(f"[{a.split}] {len(lab):,} turns over {lab.session_id.nunique():,} sessions")

    inventories, tables = [], []
    for kind in ("uptake", "response"):
        p = build_pairs(lab, kind)
        n_elig = p.groupby("session_id").size().rename("n_transitions")
        counts = p.pair.value_counts()
        keep = counts[counts >= a.min_count]
        print(f"  {kind}: {len(p):,} transitions, {len(counts):,} distinct pairs, "
              f"{len(keep):,} clear min-count {a.min_count:,}")

        # lift over chance: how much more often the pair occurs than its components' marginals
        left = p.pair.str.split(" > ").str[0].value_counts(normalize=True)
        right = p.pair.str.split(" > ").str[1].value_counts(normalize=True)
        inv = keep.rename("count").to_frame().reset_index(names="pair")
        inv["kind"] = kind
        inv["share"] = inv["count"] / len(p)
        inv["from"] = inv.pair.str.split(" > ").str[0]
        inv["to"] = inv.pair.str.split(" > ").str[1]
        inv["expected"] = inv["from"].map(left).to_numpy() * inv["to"].map(right).to_numpy()
        inv["lift"] = inv.share / inv.expected
        inventories.append(inv)

        sub = p[p.pair.isin(keep.index)]
        wide = (sub.groupby(["session_id", "pair"]).size().unstack(fill_value=0)
                .reindex(n_elig.index, fill_value=0))
        prior = (wide.sum() / n_elig.sum()).to_numpy()
        rates = (wide.to_numpy() + KAPPA * prior) / (n_elig.to_numpy()[:, None] + KAPPA)
        tbl = pd.DataFrame(rates, index=wide.index,
                           columns=[f"seq_{kind[:3]}_{c}" for c in wide.columns],
                           dtype=np.float32)
        tables.append(tbl)

    inv = pd.concat(inventories, ignore_index=True)
    feats = tables[0].join(tables[1], how="outer").reset_index()
    out_f = config.DERIVED / f"sequences_v3_{a.split}.parquet"
    out_i = config.DERIVED / f"sequence_inventory_{a.split}.csv"
    feats.to_parquet(out_f, index=False)
    inv.to_csv(out_i, index=False)
    print(f"wrote {out_f} ({feats.shape[0]:,} sessions x {feats.shape[1] - 1} pair columns)")
    print(f"wrote {out_i} ({len(inv)} pairs with lift)")
    runlog.finish(d, {"run": run, "split": a.split, "min_count": a.min_count,
                      "n_pair_columns": int(feats.shape[1] - 1),
                      "pairs_by_kind": inv.kind.value_counts().to_dict()})


if __name__ == "__main__":
    main()
