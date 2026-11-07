#!/usr/bin/env python
"""Stage 09 — LLM-panel agreement on the validation sample.

Several models code the **same turns with the identical prompt**, so the spread between them
measures how much of a code is the codebook and how much is the particular annotator. This is
the LLM--LLM half of the validation plan; the human round is the other half, and neither
substitutes for the other: two models can agree and both be wrong.

Reports, per role and per code, pairwise Cohen's kappa for every pair of annotators, plus each
annotator's turn **coverage** — which belongs in the same table, because an annotator that
labels only the turns it finds easy can post a flattering kappa on its own selection. Agreement
is computed on turns both members of a pair labelled, and the coverage column is what says
whether that set is the whole sample.

    python scripts/09_panel_agreement.py --run panel \
        --annotator "122B=full_train" --annotator "Qwen3-32B=panel32" \
        --annotator "Mistral-24B=panelmistral_cov"
"""
from __future__ import annotations

import argparse
from itertools import combinations

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import config, runlog
from ttapub import taxonomy_v3 as T

ROLES = ("tutor", "student")
FLAG = {"tutor": "has_teacher", "student": "has_student"}


def load(tag: str, role: str) -> pd.DataFrame:
    files = sorted(config.LABELS_V3.glob(f"v3_{tag}_{role}_shard*.csv"))
    if not files:
        raise SystemExit(f"no labels for tag {tag!r} role {role!r}")
    df = pd.concat([pd.read_csv(f, dtype={"session_id": str}) for f in files],
                   ignore_index=True)
    return df.drop_duplicates(["session_id", "turn"])[["session_id", "turn", "code"]]


def kappa(a: pd.Series, b: pd.Series, labels: list[str]) -> float:
    idx = {c: i for i, c in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        if x in idx and y in idx:
            M[idx[x], idx[y]] += 1
    tot = M.sum()
    if tot == 0:
        return float("nan")
    po = np.trace(M) / tot
    pe = float((M.sum(0) / tot) @ (M.sum(1) / tot))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", action="append", required=True,
                    help="NAME=TAG, repeatable; the first is the reference")
    ap.add_argument("--run", default="panel")
    a = ap.parse_args()
    ann = dict(x.split("=", 1) for x in a.annotator)
    d = runlog.start(a.run, vars(a))

    turns = pd.read_parquet(config.TURNS_TRAIN,
                            columns=["session_id", "turn", "has_teacher", "has_student"])
    rows, per_code, cov_rows = [], [], []
    for role in ROLES:
        codes = T.codes(role)
        labs = {n: load(t, role) for n, t in ann.items()}
        # the sample is the sessions every annotator was run on
        sample = set.intersection(*[set(df.session_id) for df in labs.values()])
        axis = turns[turns[FLAG[role]] & turns.session_id.isin(sample)]
        for n, df in labs.items():
            k = df[df.session_id.isin(sample)]
            cov_rows.append({"annotator": n, "role": role, "labels": len(k),
                             "turns_in_sample": len(axis),
                             "coverage": round(len(k) / max(len(axis), 1), 4)})
        for x, y in combinations(ann, 2):
            m = labs[x].merge(labs[y], on=["session_id", "turn"], suffixes=("_x", "_y"))
            m = m[m.session_id.isin(sample)]
            if m.empty:
                continue
            rows.append({"role": role, "a": x, "b": y, "n": len(m),
                         "share_of_turns": round(len(m) / max(len(axis), 1), 4),
                         "raw_agreement": round(float((m.code_x == m.code_y).mean()), 4),
                         "kappa": round(kappa(m.code_x, m.code_y, codes), 4)})
            for c in codes:
                ia, ib = m.code_x == c, m.code_y == c
                if ia.sum() < 10 and ib.sum() < 10:
                    continue
                per_code.append({"role": role, "code": c, "a": x, "b": y,
                                 "n_a": int(ia.sum()), "n_b": int(ib.sum()),
                                 "kappa": round(kappa(np.where(ia, c, "_o"),
                                                      np.where(ib, c, "_o"),
                                                      [c, "_o"]), 4)})
    overall = pd.DataFrame(rows)
    coverage = pd.DataFrame(cov_rows)
    bycode = pd.DataFrame(per_code)
    overall.to_csv(d / "panel_overall.csv", index=False)
    coverage.to_csv(d / "panel_coverage.csv", index=False)
    bycode.to_csv(d / "panel_by_code.csv", index=False)

    print("=== turn coverage (an annotator that skips turns can flatter itself) ===")
    print(coverage.to_string(index=False))
    print("\n=== overall agreement ===")
    print(overall.to_string(index=False))
    for role in ROLES:
        b = bycode[bycode.role == role]
        if b.empty:
            continue
        piv = b.pivot_table(index="code", columns=["a", "b"], values="kappa")
        piv["mean"] = piv.mean(axis=1)
        print(f"\n=== {role}: per-code kappa, worst first ===")
        print(piv.sort_values("mean").to_string(float_format=lambda v: f"{v:.3f}"))
    runlog.finish(d, {"run": a.run, "annotators": ann,
                      "overall": overall.to_dict("records"),
                      "coverage": coverage.to_dict("records")})


if __name__ == "__main__":
    main()
