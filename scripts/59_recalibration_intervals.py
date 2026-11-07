#!/usr/bin/env python
"""Stage 59 — intervals and a corrected test for stage 55's cross-fitted code gains.

Stage 55 gives point estimates. This stage runs stage 46's bootstrap on the same shifted predictions:
1,000 draws, clustered by objective out of fold and by student on test; each code's gain over the
attainment model minus the placebos' mean gain, paired per draw; p from the bootstrap SE;
Benjamini-Hochberg across the 57 codes per view. The intercepts are stage 55's crossfit: out of fold,
fold k's fitted on the other folds' rows of the view; on test, fitted on all out-of-fold rows, frozen.

The draws hold those intercepts fixed. On out-of-fold clean rows a sensitivity refits them inside every
draw (`refit_*`), so that their own uncertainty enters. An intercept keeps the ranking, so AUROC is
stage 46's and is not repeated. The views are stage 55's: Eedi is not among them.

Also the mean gain of the codes whose occurrence differs by attainment (stage 48), per side, and the
difference between the sides, with intervals; and the same gains for the attainment model plus all tutor
moves, all student signals and all 57 codes (stage 46's references).

Reads `results/<run>/{oof,test}.csv` of `arm-lo-attainment`, `add1oof-*`, `add1oofshuf-*` and stage 46's
references, and
`results/code-occurrence/codes.csv`.

Outputs: `codes.csv` (view, code, gain beyond the placebo, interval, p, BH, sensitivity), `sides.csv`,
`references.csv`, `metrics.json`.

    python scripts/59_recalibration_intervals.py --run attainment-recalibration-intervals --jobs 8
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import importlib

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from scipy.special import logit
from scipy.stats import norm

from ttapub import config, data, runlog, within

s46 = importlib.import_module("46_attainment_incremental")
OCC = config.RESULTS / "code-occurrence" / "codes.csv"
SENSITIVITY_VIEW = "out of fold, clean rows"


def row_ll(y, z):
    """Per-row log loss of the logit `z`."""
    return np.logaddexp(0, z) - y * z


def fit_shift(y, z, w=None) -> float:
    f = (lambda b: row_ll(y, z + b).mean()) if w is None else (lambda b: (w * row_ll(y, z + b)).sum() / w.sum())
    return minimize_scalar(f, bounds=(-3, 3), method="bounded").x


def refit_draw(w, y, zs, fold) -> dict:
    """One bootstrap draw's mean log loss per run, the intercepts refitted on the draw's weights."""
    out = {}
    for r, z in zs.items():
        total = 0.0
        for f in np.unique(fold):
            i = fold == f
            total += (w[i] * row_ll(y[i], z[i] + fit_shift(y[~i], z[~i], w[~i]))).sum()
        out[r] = total / w.sum()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="attainment-recalibration-intervals")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    base = s46.BASE
    occ = pd.read_csv(OCC)
    column = lambda code, role: ("v3t_" if role == "tutor" else "v3s_") + code + "_rate"
    codes = {f"add1oof-{column(c, r)}": (c, r) for c, r in zip(occ.code, occ.role)}
    if len(codes) != 57:
        raise SystemExit(f"{len(codes)} codes, expected 57")
    placebos = [f"add1oofshuf-{c}" for c in s46.PLACEBO]
    refs = {r: name for name, r in s46.REFS.items()}
    runs = [base, *codes, *placebos, *refs]
    d = runlog.start(a.run, vars(a), [config.METADATA, OCC,
                                      *(config.RESULTS / r / f for r in runs for f in ("oof.csv", "test.csv"))])
    read = lambda r, f: pd.read_csv(config.RESULTS / r / f, dtype={"response_id": str}).set_index("response_id")
    oof, test = {r: read(r, "oof.csv") for r in runs}, {r: read(r, "test.csv") for r in runs}
    ro, rt = oof[base], test[base]
    for r in runs:                                   # same rows, same labels, same order everywhere
        oof[r], test[r] = oof[r].reindex(ro.index), test[r].reindex(rt.index)
        if not (np.array_equal(oof[r].y, ro.y) and np.array_equal(test[r].y, rt.y)):
            raise SystemExit(f"{r}: rows or labels differ from {base}")

    meta = data.load_metadata()
    tr = data.load_train().merge(meta[["response_id", "student_id"]], on="response_id")
    te = data.load_test().set_index("response_id").reindex(rt.index)
    n_sess = tr.groupby("student_id").session_id.nunique()
    tr = tr.set_index("response_id").reindex(ro.index)
    other = (tr.student_id.map(n_sess) - 1).to_numpy()
    k = te.student_id.map(n_sess).fillna(0).to_numpy()
    clean = ro.clean.astype(bool).to_numpy()
    tsl = (rt.data_source == "tsl").to_numpy()
    views = {"out of fold, clean rows": ("oof", clean),
             "out of fold, clean, >= 3 other sessions": ("oof", clean & (other >= 3)),
             "test, all TSL": ("test", tsl), "test, >= 3 training sessions": ("test", tsl & (k >= 3))}

    fold, yo = ro.fold.to_numpy(), ro.y.to_numpy(float)
    to_logit = lambda p: logit(np.clip(p, 1e-12, 1 - 1e-12))
    z_oof = {r: to_logit(oof[r].p.to_numpy()) for r in runs}
    z_test = {r: to_logit(test[r].p.to_numpy()) for r in runs}
    frozen = {r: fit_shift(yo, z_oof[r]) for r in runs}   # test: from out-of-fold rows only
    sig = occ[occ.strong_vs_weak_bh.astype(bool)]
    side = {f"add1oof-{column(c, r)}": ("higher" if s > 0 else "lower")
            for c, r, s in zip(sig.code, sig.role, sig.strong_vs_weak)}

    rng = np.random.default_rng(a.seed)
    rows, side_rows, ref_rows = [], [], []
    for view, (split, m) in views.items():
        y = (yo if split == "oof" else rt.y.to_numpy(float))[m]
        L = {}
        for r in runs:
            if split == "oof":
                q = np.empty(len(yo))
                for f in np.unique(fold):
                    fit = m & (fold != f)
                    q[fold == f] = z_oof[r][fold == f] + fit_shift(yo[fit], z_oof[r][fit])
                L[r] = row_ll(y, q[m])
            else:
                L[r] = row_ll(y, z_test[r][m] + frozen[r])
        groups = (tr.learning_objective_id.to_numpy() if split == "oof" else te.student_id.to_numpy())[m]
        ug, inv = np.unique(groups, return_inverse=True)
        W = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug))[inv]
                      for _ in range(a.boot)]).astype(float)
        Lb = {r: (W @ v) / W.sum(1) for r, v in L.items()}
        plac_draw = np.mean([Lb[base] - Lb[p] for p in placebos], axis=0)
        plac = float(np.mean([L[base].mean() - L[p].mean() for p in placebos]))
        draws, est = {}, {}
        first = len(rows)
        for r, (code, role) in codes.items():
            est[r] = float(L[base].mean() - L[r].mean() - plac)
            draws[r] = Lb[base] - Lb[r] - plac_draw
            rows.append({"view": view, "code": code, "role": role, "side": side.get(r, ""), "n": int(m.sum()),
                         "clusters": len(ug), "vs_placebo": est[r],
                         "vs_placebo_lo": float(np.percentile(draws[r], 2.5)),
                         "vs_placebo_hi": float(np.percentile(draws[r], 97.5)),
                         "vs_placebo_p": float(2 * norm.sf(abs(est[r]) / draws[r].std(ddof=1)))})
        for r, name in refs.items():
            v = Lb[base] - Lb[r]
            e = float(L[base].mean() - L[r].mean())
            ref_rows.append({"view": view, "model": name, "gain": e, "gain_lo": float(np.percentile(v, 2.5)),
                             "gain_hi": float(np.percentile(v, 97.5)), "vs_placebo": e - plac,
                             "vs_placebo_lo": float(np.percentile(v - plac_draw, 2.5)),
                             "vs_placebo_hi": float(np.percentile(v - plac_draw, 97.5))})
        higher = [r for r in side if side[r] == "higher"]
        lower = [r for r in side if side[r] == "lower"]
        for name, e, v, n in (
                ("higher", np.mean([est[r] for r in higher]), np.mean([draws[r] for r in higher], axis=0), len(higher)),
                ("lower", np.mean([est[r] for r in lower]), np.mean([draws[r] for r in lower], axis=0), len(lower)),
                ("higher minus lower", np.mean([est[r] for r in higher]) - np.mean([est[r] for r in lower]),
                 np.mean([draws[r] for r in higher], axis=0) - np.mean([draws[r] for r in lower], axis=0),
                 len(higher) + len(lower))):
            side_rows.append({"view": view, "side": name, "n_codes": n, "mean": float(e),
                              "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))})
        if view == SENSITIVITY_VIEW:
            res = Parallel(n_jobs=a.jobs)(delayed(refit_draw)(w, yo[m], {r: z_oof[r][m] for r in runs}, fold[m]) for w in W)
            A = {r: np.array([x[r] for x in res]) for r in runs}
            plac_refit = np.mean([A[base] - A[p] for p in placebos], axis=0)
            for row in ref_rows[-len(refs):]:
                v = A[base] - A[{n: r for r, n in refs.items()}[row["model"]]] - plac_refit
                row.update(refit_lo=float(np.percentile(v, 2.5)), refit_hi=float(np.percentile(v, 97.5)))
            for i, r in enumerate(codes):
                v = A[base] - A[r] - plac_refit
                rows[first + i].update(refit_lo=float(np.percentile(v, 2.5)), refit_hi=float(np.percentile(v, 97.5)),
                                       refit_p=float(2 * norm.sf(abs(est[r]) / v.std(ddof=1))))
        print(f"{view}: {int(m.sum()):,} rows, {len(ug):,} clusters", flush=True)

    out = pd.DataFrame(rows)
    out["vs_placebo_bh"] = False
    for view in views:
        i = out.view == view
        out.loc[i, "vs_placebo_bh"] = within.benjamini_hochberg(out.loc[i, "vs_placebo_p"].to_numpy(), 0.05)
    i = out.view == SENSITIVITY_VIEW
    out.loc[i, "refit_bh"] = within.benjamini_hochberg(out.loc[i, "refit_p"].to_numpy(), 0.05)
    out.to_csv(d / "codes.csv", index=False)
    sides = pd.DataFrame(side_rows)
    sides.to_csv(d / "sides.csv", index=False)
    references = pd.DataFrame(ref_rows)
    references.to_csv(d / "references.csv", index=False)
    summary = {}
    for view in views:
        s = out[out.view == view]
        summary[view] = {"beat_placebo_bh": sorted(s[s.vs_placebo_bh.astype(bool) & (s.vs_placebo > 0)].code),
                         "sides_x1000": {r.side: [1e3 * r["mean"], 1e3 * r.lo, 1e3 * r.hi]
                                         for _, r in sides[sides.view == view].iterrows()},
                         "references_vs_placebo_x1000": {r.model: [1e3 * r.vs_placebo, 1e3 * r.vs_placebo_lo,
                                                                   1e3 * r.vs_placebo_hi]
                                                         for _, r in references[references.view == view].iterrows()}}
    summary[SENSITIVITY_VIEW]["beat_placebo_bh_refit"] = sorted(
        out[i & out.refit_bh.eq(True) & (out.vs_placebo > 0)].code)
    runlog.finish(d, {"run": a.run, "boot": a.boot, "views": summary})


if __name__ == "__main__":
    main()
