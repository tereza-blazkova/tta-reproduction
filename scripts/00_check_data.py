#!/usr/bin/env python
"""Stage 00 — verify the copied data is the data every committed number was measured on.

Checks the row counts, the base rates, the split composition, the transcript inventories and
the frozen fold/clean shas. Run it once after cloning or re-copying: every later stage assumes
these hold, and a silently different input file is the one error that invalidates everything
downstream without failing anything.

    python scripts/00_check_data.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import numpy as np

from ttapub import config, data, folds

EXPECT = {
    "train_rows": 35_072, "train_sessions": 22_821, "train_los": 398, "train_base": 0.7025,
    "test_rows": 10_508, "test_tsl": 9_640, "test_eedi": 868,
    "test_tsl_base": 0.7034, "test_eedi_base": 0.4931,
}


def close(got: float, want: float, tol: float = 5e-4) -> str:
    return "ok" if abs(got - want) <= tol else f"MISMATCH (expected {want})"


def main() -> None:
    bad = 0
    tr = data.load_train()
    print(f"train rows      {len(tr):,}  {close(len(tr), EXPECT['train_rows'], 0)}")
    print(f"train sessions  {tr.session_id.nunique():,}  "
          f"{close(tr.session_id.nunique(), EXPECT['train_sessions'], 0)}")
    print(f"train objectives {tr.learning_objective_id.nunique():,}  "
          f"{close(tr.learning_objective_id.nunique(), EXPECT['train_los'], 0)}")
    print(f"train base rate {tr.is_correct.mean():.4f}  "
          f"{close(tr.is_correct.mean(), EXPECT['train_base'])}")

    te = data.load_test()
    print(f"test rows       {len(te):,}  {close(len(te), EXPECT['test_rows'], 0)}")
    for src in ("tsl", "eedi"):
        m = te.data_source == src
        print(f"test {src:4s}       {m.sum():,} rows, {te[m].session_id.nunique():,} sessions, "
              f"base {te[m].is_correct.mean():.4f}  "
              f"{close(m.sum(), EXPECT[f'test_{src}'], 0)} / "
              f"{close(te[m].is_correct.mean(), EXPECT[f'test_{src}_base'])}")

    # The transcripts are needed only by stages 01-02 (turn table, annotation). Fitting a model
    # on the finished feature tables does not touch them, so a missing zip is reported and
    # skipped rather than treated as a broken install -- but never silently: it says so.
    for name, path, getter, frame in (
            ("train", config.TRAIN_TRANSCRIPTS, data.train_store, tr),
            ("test", config.TEST_TRANSCRIPTS, data.test_store, te)):
        if not path.exists():
            print(f"{name} transcripts NOT PRESENT ({path.name}) -- skipped; only stages "
                  f"01-02 need them")
            continue
        store = getter()
        have = store.sessions()
        want = set(frame.session_id)
        missing = want - have
        print(f"{name} transcripts {len(have):,} members, "
              f"{len(missing)} sessions missing" + (" ok" if not missing else " MISMATCH"))
        bad += bool(missing)

    f = folds.lo_folds(len(tr))
    c = folds.clean_mask(len(tr))
    print(f"folds           sha ok, {len(set(f))} folds, sizes "
          f"{np.bincount(f).tolist()}")
    print(f"clean rows      {c.sum():,} / {len(c):,} ({c.mean():.1%}), "
          f"base {tr.is_correct[c].mean():.3f} vs leaked {tr.is_correct[~c].mean():.3f}")

    bad += sum("MISMATCH" in line for line in [])  # counts above print their own verdicts
    print("\nIf any line says MISMATCH, stop: the inputs are not the ones the committed "
          "numbers belong to.")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
