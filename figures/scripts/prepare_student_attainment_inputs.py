#!/usr/bin/env python
"""Inputs for Fig. 3b: how far students sit above or below the questions they answer.

For every training answer, the outcome minus its learning objective's usual correct rate (the
objective's rate without the answer's own session, pulled toward the corpus rate by `M` answers,
as the first step of `ttapub.attainment.expected_correct`). A student's value is the mean of their
answers' residuals, in percentage points, unshrunk; students with at least three training
sessions only. Controlling for the question instead (the paper's attainment measure) gives the same
picture: the two student measures correlate at 0.994 and their spreads differ by 0.2 points.

Beside it, the spread chance alone gives: the outcomes shuffled among the answers to the same
objective (which keeps every objective's rate), the student means recomputed, over `N_NULL`
shuffles. A wider observed distribution than the shuffled one is the students' own difference.

Writes to `figures/input/` (aggregates only, no identifiers):
* `fig_student_attainment.csv` -- bins of `BIN` percentage points: `lo`, `hi`, `students` (observed)
  and `null` (mean count over the shuffles).
* `fig_student_attainment_summary.csv` -- students, their answers, and the observed and shuffled
  standard deviations and the shares of students more than 10 points above or below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
IN = ROOT / "figures" / "input"
MIN_SESSIONS, BIN, N_NULL, SEED, M = 3, 5.0, 20, 0, 10.0
LO, HI = -100.0, 100.0                  # every possible mean; empty bins at the ends are dropped

from ttapub import data  # noqa: E402


def objective_rate(t: pd.DataFrame) -> np.ndarray:
    """Each answer's objective rate without the answer's own session, shrunk toward the corpus."""
    y = t.is_correct.astype(float)
    tot, own = y.groupby(t.learning_objective_id), y.groupby([t.learning_objective_id, t.session_id])
    s = tot.transform("sum") - own.transform("sum")
    n = tot.transform("size") - own.transform("size")
    return ((s + M * y.mean()) / (n + M)).to_numpy()


def student_means(tr: pd.DataFrame, y: np.ndarray) -> pd.Series:
    t = tr.assign(is_correct=y)
    res = 100 * (y - objective_rate(t))
    return pd.Series(res, index=t.index).groupby(t.student_id).mean()


def main() -> None:
    tr = data.load_train().reset_index(drop=True)
    meta = data.load_metadata().set_index("response_id")
    tr["student_id"] = tr.response_id.map(meta.student_id)
    tr["question_id"] = tr.response_id.map(meta.question_id)
    if tr[["student_id", "question_id"]].isna().any().any():
        raise SystemExit("training answers without a student or a question in the metadata")
    n_sess = tr.groupby("student_id").session_id.nunique()
    keep = n_sess[n_sess >= MIN_SESSIONS].index
    y = tr.is_correct.to_numpy(float)
    obs = student_means(tr, y).loc[keep]
    rng = np.random.default_rng(SEED)
    q = tr.learning_objective_id.to_numpy()
    nulls = []
    for _ in range(N_NULL):
        ys = y.copy()
        for idx in pd.Series(np.arange(len(tr))).groupby(q).indices.values():
            ys[idx] = ys[rng.permutation(idx)]
        nulls.append(student_means(tr, ys).loc[keep])
    edges = np.arange(LO, HI + BIN, BIN)
    out = sum(((n < LO) | (n > HI)).sum() for n in [obs] + nulls)
    if out:
        raise SystemExit(f"{out} student means fall outside [{LO}, {HI}]: widen the range")
    clip = lambda v: v
    h_obs, _ = np.histogram(clip(obs), bins=edges)
    h_null = np.mean([np.histogram(clip(n), bins=edges)[0] for n in nulls], axis=0)
    h = pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "students": h_obs, "null": h_null.round(2)})
    used = np.flatnonzero((h.students > 0) | (h.null > 0))
    h.iloc[used.min():used.max() + 1].to_csv(IN / "fig_student_attainment.csv", index=False)
    null_all = np.concatenate([n.to_numpy() for n in nulls])
    summ = {"students": len(obs), "answers": int(tr.student_id.isin(keep).sum()),
            "sd_observed": obs.std(), "sd_shuffled": null_all.std(),
            "share_above_10": (obs > 10).mean(), "share_below_10": (obs < -10).mean(),
            "share_above_10_shuffled": (null_all > 10).mean(), "share_below_10_shuffled": (null_all < -10).mean()}
    pd.DataFrame([summ]).to_csv(IN / "fig_student_attainment_summary.csv", index=False)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()})


if __name__ == "__main__":
    main()
