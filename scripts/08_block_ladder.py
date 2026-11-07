#!/usr/bin/env python
"""Stage 08 — the nested block ladder: what each feature block adds, on identical rows.

Reads the arms stage 06 wrote and reports, per view, each arm's log loss / AUROC and the paired
deltas with clustered bootstrap intervals. The pairs are the questions the paper asks:

    student signals over the objective alone
    tutor moves over the objective alone
    student signals over objective + tutor moves      <- what student talk adds
    tutor moves over objective + student signals
    both over the objective alone

Pairs are computed on the rows both arms scored, so a delta is never a population difference.
Out-of-fold deltas cluster on the objective (the unit the folds are disjoint on); test deltas
cluster on the session.

    # fit the arms first
    for b in "" student tutor tutor,student; do
      python scripts/06_train_clean57.py --blocks "$b" --lo text --seeds 5 --test \
        --run "arm-${b:-lo}"
    done
    python scripts/08_block_ladder.py --arms arm-lo arm-student arm-tutor arm-tutor,student
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import config, data, metrics, runlog

#: The arm holding both blocks. Accepts either name so a run made under the older
#: auto-generated name still ladders: a silently-skipped pair is worse than a loud error, and
#: the two incremental contrasts below are the paper's central numbers.
def pairs(prefix: str = "arm-") -> list:
    """The ladder's contrasts for one learner's arms, named `<prefix>lo` ... `<prefix>both`.

    A prefix rather than a second hard-coded list, so that every learner is laddered by
    exactly the same contrasts -- the comparison across learners is only meaningful if the
    pairs are identical.
    """
    both = (f"{prefix}both", f"{prefix}tutor,student")
    return [
        (f"{prefix}student", f"{prefix}lo", "student signals over objective"),
        (f"{prefix}tutor", f"{prefix}lo", "tutor moves over objective"),
        (both, f"{prefix}tutor", "student signals over objective + tutor moves"),
        (both, f"{prefix}student", "tutor moves over objective + student signals"),
        (both, f"{prefix}lo", "both over objective"),
    ]


def _resolve(name, frames):
    """An arm name, or the first of several aliases that is present."""
    if isinstance(name, str):
        return name if name in frames else None
    return next((n for n in name if n in frames), None)


def load(arm: str, which: str) -> pd.DataFrame | None:
    p = config.RESULTS / arm / f"{which}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="arm-",
                    help="arm-name prefix: arm- for XGBoost, logreg- for logistic regression")
    ap.add_argument("--arms", nargs="+", default=None,
                    help="defaults to <prefix>lo, student, tutor, both")
    ap.add_argument("--run", default="ladder")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    arms = list(dict.fromkeys(a.arms or [f"{a.prefix}{x}"
                                         for x in ("lo", "student", "tutor", "both")]))

    train = data.load_train().reset_index(drop=True)
    lo_of = dict(zip(train.response_id, train.learning_objective_id))
    test = data.load_test().reset_index(drop=True)
    sess_of = dict(zip(test.response_id, test.session_id))

    d = runlog.start(a.run, vars(a))
    res: dict = {"arms": arms, "views": {}}

    for which, cluster_of in (("oof", lo_of), ("test", sess_of)):
        frames = {arm: load(arm, which) for arm in arms}
        frames = {k: v for k, v in frames.items() if v is not None}
        if not frames:
            continue
        sources = ([("all", None)] if which == "oof"
                   else [("tsl", "tsl"), ("eedi", "eedi"), ("all", None)])
        for label, src in sources:
            view = f"{which}_{label}"
            print(f"\n══ {view} ══")
            rows = {}
            for arm, f in frames.items():
                m = np.ones(len(f), bool) if src is None else (f.data_source == src).to_numpy()
                r = metrics.basic(f.y.to_numpy()[m], f.p.to_numpy()[m])
                if which == "test":
                    r["ll_icpt"] = metrics.intercept_refit_ll(f.y.to_numpy()[m],
                                                              f.p.to_numpy()[m])
                rows[arm] = r
                print(metrics.fmt(arm, r))
            res["views"][view] = {"arms": rows, "pairs": {}}
            for new_spec, old_spec, what in pairs(a.prefix):
                new, old = _resolve(new_spec, frames), _resolve(old_spec, frames)
                if new is None or old is None:
                    print(f"  (skipped: {what} — missing arm)")
                    continue
                A, B = frames[new], frames[old]
                j = A[["response_id", "y", "p"]].merge(B[["response_id", "p"]],
                                                       on="response_id",
                                                       suffixes=("_new", "_old"))
                if src is not None:
                    keep = A[["response_id", "data_source"]]
                    j = j.merge(keep, on="response_id")
                    j = j[j.data_source == src]
                g = np.array([cluster_of.get(r, r) for r in j.response_id])
                dd = metrics.boot_delta(j.y.to_numpy(float), j.p_new.to_numpy(),
                                        j.p_old.to_numpy(), g, B=a.boot)
                res["views"][view]["pairs"][what] = dd | {"n": int(len(j))}
                print(f"  {what:46s} n={len(j):,}  "
                      f"dLL {dd['d_ll']:+.4f} [{dd['d_ll_ci'][0]:+.4f},{dd['d_ll_ci'][1]:+.4f}]"
                      f"  dAUROC {dd['d_auroc']:+.2f} "
                      f"[{dd['d_auroc_ci'][0]:+.2f},{dd['d_auroc_ci'][1]:+.2f}]")

    runlog.finish(d, res)
    (d / "ladder.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
