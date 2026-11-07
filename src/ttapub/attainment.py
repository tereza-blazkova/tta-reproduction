"""A student's attainment, measured on data other than the answer being predicted.

**Definition (stages 35 and 36: `35_student_heterogeneity.py`, `36_beyond_the_student.py`).**
Each answer is scored against its question: the outcome minus the question's correctness rate, computed without the answer's own session and shrunk toward its
objective's rate (and an objective's toward the corpus rate) when the question is rare. A student's
attainment is the sum of those residuals over the answers it may use, divided by (n + `M_ATTAINMENT`),
so a student with few answers is pulled toward 0. Easy questions do not make a student look able.

**Which answers it may use.**
  * a training row: the student's *other* training sessions. An answer never enters its own
    feature, so a model fitted with this column cannot read the label through it;
  * a test row: all of the student's training answers. No test outcome enters.
A student with nothing to use gets NaN, which XGBoost routes as missing: "unknown" is not
"average".

`M_ATTAINMENT` = 10 is the pseudo-count stage 36 chose by predicting a held-out session's residual
from the student's other sessions (`results/beyond-the-student`, `attainment_shrinkage`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Pseudo-count, in responses, shrinking a rare question's rate toward its objective's (stage 35).
M_SHRINK = 10.0
#: Pseudo-count, in responses, shrinking a student's mean residual toward 0 (stage 36).
M_ATTAINMENT = 10.0


def expected_correct(tr: pd.DataFrame, m: float = M_SHRINK) -> np.ndarray:
    """Each response's question correctness rate, leaving out the response's own session.

    Identical to `expected_correct` in `35_student_heterogeneity.py`; stage 39 checks the two agree."""
    y = tr.is_correct.astype(float)
    g = y.mean()

    def loso(key: str) -> tuple[np.ndarray, np.ndarray]:
        tot = y.groupby(tr[key])
        own = y.groupby([tr[key], tr.session_id])
        s = tot.transform("sum") - own.transform("sum")
        n = tot.transform("size") - own.transform("size")
        return s.to_numpy(), n.to_numpy()

    s_lo, n_lo = loso("learning_objective_id")
    p_lo = (s_lo + m * g) / (n_lo + m)
    s_q, n_q = loso("question_id")
    return (s_q + m * p_lo) / (n_q + m)


def train_and_test(train: pd.DataFrame, test_students: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(attainment for every training row, attainment for every test row), in their row order.

    `train` needs session_id, student_id, question_id, learning_objective_id, is_correct;
    `test_students` is the test rows' student_id."""
    res = train.is_correct.to_numpy(float) - expected_correct(train)
    r = pd.Series(res, index=train.index)
    st_sum = r.groupby(train.student_id).transform("sum")
    st_n = r.groupby(train.student_id).transform("size")
    se_sum = r.groupby(train.session_id).transform("sum")
    se_n = r.groupby(train.session_id).transform("size")
    other_n = (st_n - se_n).to_numpy(float)
    a_train = np.where(other_n > 0, (st_sum - se_sum).to_numpy() / (other_n + M_ATTAINMENT), np.nan)
    by_student = r.groupby(train.student_id).agg(["sum", "size"])
    full = by_student["sum"] / (by_student["size"] + M_ATTAINMENT)
    a_test = test_students.map(full).to_numpy(float)
    return a_train, a_test
