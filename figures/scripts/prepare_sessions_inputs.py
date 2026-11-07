#!/usr/bin/env python
"""Inputs for fig_sessions_per_student: how many sessions each training student contributes.

A within-student design identifies an association only from students with two or more
sessions, and estimates it better the more sessions they have. Whether to report student fixed
effects at all depends on this distribution, so it is drawn before that decision is taken.

Writes `figures/input/fig_sessions_per_student.csv`: one row per session count, with the number
of students at that count and the responses they contribute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
IN = ROOT / "figures" / "input"

from ttapub import data  # noqa: E402


def main() -> None:
    meta = data.load_metadata()[["response_id", "student_id"]]
    tr = data.load_train()[["response_id", "session_id"]].merge(meta, on="response_id")
    if tr.student_id.isna().any():
        raise SystemExit(f"{tr.student_id.isna().sum()} training responses have no student_id")
    per = tr.groupby("student_id").agg(n_sessions=("session_id", "nunique"),
                                       n_responses=("response_id", "size"))
    out = (per.groupby("n_sessions")
              .agg(n_students=("n_responses", "size"), n_responses=("n_responses", "sum"))
              .reset_index())
    out.to_csv(IN / "fig_sessions_per_student.csv", index=False)
    two = per.n_sessions >= 2
    print(f"{len(per):,} students, {per.n_sessions.median():.0f} median sessions, max "
          f"{per.n_sessions.max()}; {two.mean():.1%} with 2+ sessions, covering "
          f"{per.n_responses[two].sum() / per.n_responses.sum():.1%} of responses")


if __name__ == "__main__":
    main()
