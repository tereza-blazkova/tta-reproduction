#!/usr/bin/env python
"""Inputs for Fig. 1a-c: in how many sessions each code occurs, and how much of a session it
takes up when it does.

Reads the training turn labels (`data/derived/labels_v3/v3_full_train_<role>_shard*.csv`, the
primary `code`, which is what the session rates count). For each code and each session on the
role's axis (a session with at least one labelled turn of that role):

* the session counts once if the code labels at least one of the role's turns there;
* in those sessions, the code's share of the role's labelled turns (the raw rate, unshrunk).

Before writing, the counts are checked against the model's own columns: shrinking them toward
the prior (`data/derived/code_prior_v3.json`, kappa 25) must reproduce
`data/derived/features_v3_train.csv`, so the bars count the same turns the model's rates do.

Writes `figures/input/fig_presence.csv`: one row per code, with role, code, label, residual, the
role's sessions, the sessions with the code and their share, and the share of turns in those
sessions (25th, 50th and 75th percentiles and the mean).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
IN = ROOT / "figures" / "input"
GATE = 1e-9                                   # largest allowed gap to the feature table

from ttapub import agg, config  # noqa: E402


def main() -> None:
    meta = pd.read_csv(ROOT / "tables" / "table_columns.csv")
    feats = pd.read_csv(ROOT / "data" / "derived" / "features_v3_train.csv",
                        dtype={"session_id": str}).set_index("session_id")
    prior = json.loads((ROOT / "data" / "derived" / "code_prior_v3.json").read_text())["prior"]
    out = []
    for role in ("tutor", "student"):
        lab = pd.concat([pd.read_csv(f, dtype={"session_id": str}, usecols=["session_id", "turn", "code"])
                         for f in sorted(config.LABELS_V3.glob(f"v3_full_train_{role}_shard*.csv"))])
        lab = lab[lab.code.notna()]
        if lab.duplicated(["session_id", "turn"]).any():
            raise SystemExit(f"{role}: a turn carries two primary codes")
        codes = list(meta[meta.role == role].code)
        cnt = lab.groupby(["session_id", "code"]).size().unstack(fill_value=0).reindex(columns=codes,
                                                                                    fill_value=0)
        n = lab.groupby("session_id").size().reindex(cnt.index)
        # gate: the shrunk rates of these counts are the model's columns
        pri = np.array([prior[c] for c in codes])
        rate = (cnt.to_numpy() + agg.SHRINK_KAPPA * pri) / (n.to_numpy()[:, None] + agg.SHRINK_KAPPA)
        ref = feats.loc[cnt.index, [agg.column(c, role) for c in codes]].to_numpy()
        gap = np.nanmax(np.abs(rate - ref))
        if not gap < GATE:
            raise SystemExit(f"{role}: counts do not reproduce the feature table (max gap {gap:.2e})")
        share = cnt.div(n, axis=0)
        for m in meta[meta.role == role].itertuples():
            s = share.loc[cnt[m.code] > 0, m.code].to_numpy()
            out.append(dict(role=role, code=m.code, label=m.label, residual=m.residual,
                            sessions=len(cnt), sessions_with=len(s), share_sessions=len(s) / len(cnt),
                            turns_q25=np.quantile(s, 0.25), turns_q50=np.quantile(s, 0.5),
                            turns_q75=np.quantile(s, 0.75), turns_mean=s.mean()))
        print(f"{role}: {len(cnt):,} sessions, {len(lab):,} labelled turns, {len(codes)} codes; "
              f"counts reproduce the feature table (max gap {gap:.1e})")
    pd.DataFrame(out).to_csv(IN / "fig_presence.csv", index=False)
    print(f"wrote {IN / 'fig_presence.csv'}")


if __name__ == "__main__":
    main()
