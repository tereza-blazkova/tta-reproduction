#!/usr/bin/env python
"""Plumbing test: fake v3 labels -> stages 04-08, on a few hundred sessions.

The labels are drawn at random from the real pools, so **no number this produces means
anything** — it exists to prove the stage contract holds end to end (join keys, column names,
gates, priors, folds, views) before a GPU hour is spent. It writes to `--tag smoke` and
`results/smoke-*`, so it can never be mistaken for a real run.

    python scripts/smoke_e2e.py --sessions 300
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import config
from ttapub import taxonomy_v3 as T

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def fake_labels(sessions: int, seed: int = 0) -> None:
    turns = pd.read_parquet(config.TURNS_TRAIN)
    sids = list(dict.fromkeys(turns.session_id))[:sessions]
    turns = turns[turns.session_id.isin(set(sids))]
    rng = np.random.default_rng(seed)
    config.LABELS_V3.mkdir(parents=True, exist_ok=True)
    for role, flag in (("tutor", "has_teacher"), ("student", "has_student")):
        sub = turns[turns[flag]]
        codes = T.codes(role)
        # a skewed draw, so some codes land under the positives gate like real ones do
        w = rng.dirichlet(np.linspace(3, 0.2, len(codes)))
        out = pd.DataFrame({"session_id": sub.session_id.to_numpy(),
                            "turn": sub.turn.to_numpy(),
                            "code": rng.choice(codes, len(sub), p=w),
                            "also": ""})
        p = config.LABELS_V3 / f"v3_smoke_{role}_shard0.csv"
        out.to_csv(p, index=False)
        print(f"fake {role} labels: {len(out):,} turns -> {p}")


def run(*cmd: str) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([PY, *cmd], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--test-sessions", type=int, default=400)
    a = ap.parse_args()

    run("scripts/01_build_turn_table.py", "--split", "train", "--limit", str(a.sessions))
    smoke_tr = config.TURNS_TRAIN.with_suffix(".smoke.parquet")
    smoke_tr.replace(config.TURNS_TRAIN)
    run("scripts/01_build_turn_table.py", "--split", "test", "--limit", str(a.test_sessions))
    config.TURNS_TEST.with_suffix(".smoke.parquet").replace(config.TURNS_TEST)

    fake_labels(a.sessions)
    run("scripts/04_distil_v3.py", "--tag", "smoke", "--run", "smoke-children",
        "--fit-cap", "40000")
    run("scripts/05_build_features.py", "--run", "smoke-features")
    for blocks, arm in (("", "smoke-arm-lo"), ("student", "smoke-arm-student"),
                        ("tutor", "smoke-arm-tutor"), ("tutor,student", "smoke-arm-both")):
        run("scripts/06_train_clean57.py", "--blocks", blocks, "--lo", "text", "--seeds", "1",
            "--test", "--run", arm, "--boot", "200", "--n-jobs", "4")
    run("scripts/07_shap_clean57.py", "--run", "smoke-arm-both", "--n-train", "2000",
        "--gates", "results/smoke-children/gates.csv")
    run("scripts/08_block_ladder.py", "--run", "smoke-ladder", "--boot", "200",
        "--arms", "smoke-arm-lo", "smoke-arm-student", "smoke-arm-tutor", "smoke-arm-both")
    print("\n✅ plumbing ok. Every number above is from RANDOM labels — delete "
          "results/smoke-* and data/derived/* before a real run.")


if __name__ == "__main__":
    main()
