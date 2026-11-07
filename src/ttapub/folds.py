"""The frozen LO-disjoint partition and the leakage-free row mask.

Both are **loaded, never recomputed**. `data/folds/folds_gkf5_lo.npy` is the exact 5-way
`GroupKFold(learning_objective_id)` partition every number in the paper was measured on, and
`clean_rows_gkf5.npy` the mask derived from it. Recomputing them is not reproducible: sklearn
orders groups by size with an unstable quicksort, 256 of the 398 objectives are tied on
response count, and numpy's sort dispatches on CPU features — so the same code produced two
different (both valid) partitions on two cluster nodes, 0.0046 log loss apart, which silently
turned a booster A/B into a comparison of two different splits. The shas below pin the files
that every committed number belongs to.

**What the clean mask is.** `GroupKFold(learning_objective_id)` controls objective
memorisation but gives up session-disjointness: 36.7% of sessions carry more than one
objective, so one transcript is split across folds and 52.4% of responses are trained on their
own conversation. A row is *clean* iff every response of its session lands in one fold. No
5-way split can be both session- and LO-disjoint (93.6% of the data sits in one connected
component of the session-objective graph), so one axis must leak; the mask makes the cost
measurable instead of silent.

⚠️ Clean rows are a **different population**, not a random subsample (base rate 0.634 vs the
leaked half's 0.765). Compare models *within* the clean rows; never compare a clean level
against a pooled level and call the difference an effect.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import config

FOLD_SHA = "59f0c807dad3ac08"
CLEAN_SHA = "cc365b58eaea4bdb"
N_ROWS = 35_072


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:16]


def lo_folds(n_rows: int | None = None) -> np.ndarray:
    """The frozen fold vector, one entry per training row (row order = `data.load_train()`)."""
    folds = np.load(config.FOLD_VECTOR)
    got = _sha(folds.astype(np.int64))
    if got != FOLD_SHA:
        raise SystemExit(f"fold vector sha {got} != {FOLD_SHA}: {config.FOLD_VECTOR} is not "
                         "the frozen partition, so its numbers are not comparable.")
    if n_rows is not None and len(folds) != n_rows:
        raise SystemExit(f"fold vector has {len(folds)} rows, the frame has {n_rows}.")
    return folds


def clean_mask(n_rows: int | None = None) -> np.ndarray:
    """The frozen leakage-free row mask, aligned with :func:`lo_folds`."""
    mask = np.load(config.CLEAN_MASK)
    got = _sha(mask.astype(np.bool_))
    if got != CLEAN_SHA:
        raise SystemExit(f"clean mask sha {got} != {CLEAN_SHA}: the mask is derived from the "
                         "fold vector, so the partition changed.")
    if n_rows is not None and len(mask) != n_rows:
        raise SystemExit(f"clean mask has {len(mask)} rows, the frame has {n_rows}.")
    return mask
