#!/usr/bin/env python
"""Stage 54 — does adding student columns help more with our codes than with the Lin scheme?

For each scheme, the student gain is the metric of tutor + student columns minus the metric of
tutor columns alone. The contrast is `gain(ours) − gain(Lin)`, resampled with the same clusters
in all four models (`arm-both`, `arm-tutor`, `arm-lin-rates`, `arm-lin-rates-tutor`), so it tests
the difference between the gains directly rather than comparing two separate intervals.

Signs: a **negative** log-loss contrast or a **positive** AUROC contrast means student columns add
more with our codes. Clusters match stage 53: objectives out of fold, sessions on test.

Fixed before this script was run: `test_tsl` is the primary view, because it is the same provider
on unseen sessions. The other views are reported beside it, and nothing is corrected for
multiplicity. Stage 53 had already shown the separate gains, so this is a follow-up test, not a
preregistered one.

    python scripts/54_lin_student_gain.py --run lin-student-gain
    python scripts/54_lin_student_gain.py --codes-only --run lin-student-gain-codes-only

`--codes-only` makes the same contrast on the arms fit without the learning objective
(`--lo none`: `arm-nolo-both`, `arm-nolo-tutor`, `arm-lin-nolo-both`, `arm-lin-nolo-tutor`),
the comparison Lin et al. made, who predicted from dialogue acts alone.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from ttapub import metrics, runlog

_spec = importlib.util.spec_from_file_location(
    "lin_compare", Path(__file__).with_name("53_lin_compare.py"))
_lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lc)

ARMS = {False: (("arm-both", "arm-tutor"), ("arm-lin-rates", "arm-lin-rates-tutor")),
        True: (("arm-nolo-both", "arm-nolo-tutor"), ("arm-lin-nolo-both", "arm-lin-nolo-tutor"))}


def score(y, p) -> tuple[float, float]:
    return log_loss(y, p), 100 * roc_auc_score(y, p)


def contrast(y, ours_full, ours_tutor, lin_full, lin_tutor) -> dict[str, float]:
    ll = {}
    au = {}
    for k, p in (("of", ours_full), ("ot", ours_tutor), ("lf", lin_full), ("lt", lin_tutor)):
        ll[k], au[k] = score(y, p)
    return {"gain_ours_ll": ll["of"] - ll["ot"], "gain_lin_ll": ll["lf"] - ll["lt"],
            "contrast_ll": (ll["of"] - ll["ot"]) - (ll["lf"] - ll["lt"]),
            "gain_ours_auroc": au["of"] - au["ot"], "gain_lin_auroc": au["lf"] - au["lt"],
            "contrast_auroc": (au["of"] - au["ot"]) - (au["lf"] - au["lt"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lin-student-gain")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--codes-only", action="store_true",
                    help="compare the arms fit without the learning objective (--lo none)")
    a = ap.parse_args()
    OURS, LIN = ARMS[a.codes_only]
    d = runlog.start(a.run, vars(a), [])
    res: dict = {"run": a.run, "primary_view": "test_tsl", "ours": list(OURS), "lin": list(LIN),
                 "views": {}}
    # views() checks each pair's rows, outcomes and folds; the third pair ties the two schemes
    for vo, vl, vx in zip(_lc.views(*OURS), _lc.views(*LIN), _lc.views(OURS[0], LIN[0])):
        view, y, of, ot, cl = vo
        _, y2, lf, lt, cl2 = vl
        _, y3, _, _, cl3 = vx
        if not (np.array_equal(y, y2) and np.array_equal(y, y3)
                and np.array_equal(cl, cl2) and np.array_equal(cl, cl3)):
            raise SystemExit(f"{view}: rows differ between the schemes")
        point = contrast(y, of, ot, lf, lt)
        rng = np.random.default_rng(0)
        idx = metrics._cluster_index(cl)
        draws: dict[str, list[float]] = {k: [] for k in point}
        for _ in range(a.boot):
            ix = np.concatenate([idx[g] for g in rng.integers(0, len(idx), len(idx))])
            if y[ix].min() == y[ix].max():
                continue
            for k, v in contrast(y[ix], of[ix], ot[ix], lf[ix], lt[ix]).items():
                draws[k].append(v)
        r: dict = {"n": int(len(y)), "clusters": len(idx), "draws": len(draws["contrast_ll"])}
        for k, v in point.items():
            r[k] = float(v)
            r[f"{k}_ci"] = [float(np.percentile(draws[k], q)) for q in (2.5, 97.5)]
        res["views"][view] = r
        print(f"{view:10s} n={len(y):6,}  "
              f"LL gain ours {r['gain_ours_ll']:+.4f} Lin {r['gain_lin_ll']:+.4f} "
              f"contrast {r['contrast_ll']:+.4f} "
              f"[{r['contrast_ll_ci'][0]:+.4f},{r['contrast_ll_ci'][1]:+.4f}]  "
              f"AUROC gain ours {r['gain_ours_auroc']:+.2f} Lin {r['gain_lin_auroc']:+.2f} "
              f"contrast {r['contrast_auroc']:+.2f} "
              f"[{r['contrast_auroc_ci'][0]:+.2f},{r['contrast_auroc_ci'][1]:+.2f}]", flush=True)
    runlog.finish(d, res)


if __name__ == "__main__":
    main()
