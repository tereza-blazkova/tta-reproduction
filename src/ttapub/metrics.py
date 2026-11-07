"""Scoring: log loss and AUROC together, always, with clustered bootstrap intervals.

Two rules this module exists to enforce.

1. **Both metrics, every view.** Log loss is sensitive to level; AUROC is scale-free and sensitive to ranking. They disagree — on the released test
   set a model can carry the TSL ranking and still lose on Eedi purely through the base rate —
   and the disagreement is the finding, so neither may be dropped.
2. **Rows are not independent.** A session contributes several responses sharing one
   transcript, and an objective contributes many sessions. Intervals therefore resample
   *clusters*: objectives for the out-of-fold views (the unit the folds are disjoint on),
   sessions for the test views.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def basic(y, p) -> dict:
    return {"n": int(len(y)), "base": float(np.mean(y)), "mean_p": float(np.mean(p)),
            "ll": float(log_loss(y, p)), "auroc": 100 * float(roc_auc_score(y, p))}


def _cluster_index(groups):
    ug, inv = np.unique(groups, return_inverse=True)
    return [np.where(inv == g)[0] for g in range(len(ug))]


def boot_ci(y, p, groups, B: int = 2000, seed: int = 0) -> dict:
    """Cluster-bootstrap 95% intervals for log loss and AUROC."""
    rng = np.random.default_rng(seed)
    idx_by = _cluster_index(groups)
    ll, au = [], []
    for _ in range(B):
        ix = np.concatenate([idx_by[g] for g in rng.integers(0, len(idx_by), len(idx_by))])
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        ll.append(log_loss(yy, p[ix]))
        au.append(100 * roc_auc_score(yy, p[ix]))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"ll_ci": q(ll), "auroc_ci": q(au)}


def boot_delta(y, p_new, p_old, groups, B: int = 2000, seed: int = 0) -> dict:
    """Paired cluster bootstrap of (new - old) on identical rows."""
    rng = np.random.default_rng(seed)
    idx_by = _cluster_index(groups)
    dl, da = [], []
    for _ in range(B):
        ix = np.concatenate([idx_by[g] for g in rng.integers(0, len(idx_by), len(idx_by))])
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        dl.append(log_loss(yy, p_new[ix]) - log_loss(yy, p_old[ix]))
        da.append(100 * (roc_auc_score(yy, p_new[ix]) - roc_auc_score(yy, p_old[ix])))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"d_ll": float(np.mean(dl)), "d_ll_ci": q(dl),
            "d_auroc": float(np.mean(da)), "d_auroc_ci": q(da)}


def intercept_refit_ll(y, p) -> float:
    """Diagnostic only: log loss after a rank-preserving intercept shift fitted on these rows.

    Separates "wrong level" (a base rate that did not transport, as on Eedi) from "wrong
    ranking". It is fitted on the evaluation rows, so it is **never** used to predict — it is
    reported beside the honest log loss to say which of the two failures we are looking at.
    """
    from scipy.optimize import minimize_scalar
    z = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    f = lambda b: log_loss(y, 1 / (1 + np.exp(-(z + b))))  # noqa: E731
    return float(minimize_scalar(f, bounds=(-5, 5), method="bounded").fun)


def fmt(name: str, r: dict) -> str:
    s = (f"  {name:16s} n={r['n']:,}  LL {r['ll']:.4f}")
    if "ll_icpt" in r:
        s += f" (icpt {r['ll_icpt']:.4f})"
    s += f"  AUROC {r['auroc']:.2f}"
    if "auroc_ci" in r:
        s += f" [{r['auroc_ci'][0]:.2f},{r['auroc_ci'][1]:.2f}]"
    if "base" in r:
        s += f"  base {r['base']:.4f} mean_p {r['mean_p']:.4f}"
    return s
