#!/usr/bin/env python
"""Inputs for fig_head_to_head (stage 42, `results/head-to-head`): the transcript model against
the attainment model on the same test answers.

* `fig_h2h_a_perf.csv`: per TSL population, each model's log loss and AUROC with intervals;
* `fig_h2h_points.csv`: one row per TSL test answer whose student has 3+ training sessions: the
  attainment model's and the transcript model's predicted probability and whether the answer was
  right. No identifier of any answer, session or student. Individual-level: keep this repository
  private, as for `fig_beeswarm_points.csv`.

    python figures/scripts/prepare_head_to_head_inputs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ttapub import data  # noqa: E402

IN = ROOT / "figures" / "input"
RES = ROOT / "results"
MIN_SESSIONS = 3


def main() -> None:
    perf = pd.read_csv(RES / "head-to-head" / "performance.csv")
    perf[perf.model.isin(["attainment", "transcript", "both"])].to_csv(IN / "fig_h2h_a_perf.csv", index=False)
    meta = data.load_metadata()
    tr = data.load_train().merge(meta[["response_id", "student_id"]], on="response_id")
    te = data.load_test().reset_index(drop=True)
    k = te.student_id.map(tr.groupby("student_id").session_id.nunique()).fillna(0)
    keep = ((te.data_source == "tsl") & (k >= MIN_SESSIONS)).to_numpy()
    out = pd.DataFrame({"right": te.is_correct.to_numpy()[keep]})
    for name, run in (("p_attainment", "arm-lo-attainment"), ("p_transcript", "arm-both-pred")):
        p = pd.read_csv(RES / run / "test.csv", dtype={"response_id": str}).set_index("response_id")
        if not np.array_equal(p.y.reindex(te.response_id).to_numpy(), te.is_correct.to_numpy(float)):
            raise SystemExit(f"{run}: labels do not match the test split")
        out[name] = p.p.reindex(te.response_id).to_numpy()[keep]
    out = out.sample(frac=1, random_state=0).reset_index(drop=True)   # row order carries nothing
    out.to_csv(IN / "fig_h2h_points.csv", index=False)
    print(f"{len(out):,} answers, {out.right.mean():.3f} right")


if __name__ == "__main__":
    main()
