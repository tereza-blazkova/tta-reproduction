"""Within-student estimation: two-way fixed effects with cluster-robust standard errors.

The question RQ1 and RQ2 both ask is *within* a student: does this student do better after
sessions in which they showed more of a signal than they usually do? Absorbing a student fixed
effect removes every stable difference between students — attainment, talkativeness, how their
tutor was assigned — and leaves only variation across that student's own sessions. A question
fixed effect does the same for item difficulty.

**Linear probability model, not fixed-effects logit.** The outcome is binary, so an LPM is
misspecified in the textbook sense; it is still the right primary here:

* a conditional logit drops every student whose outcome never varies — 2,398 of 5,933 students
  and 6,003 responses — and those students are not missing at random: they are the always-right
  and always-wrong. The LPM keeps all 35,072 rows, so the estimate is for the whole corpus;
* coefficients read directly as percentage points, which is what the paper reports;
* two-way absorption is tractable, and with fixed effects in the thousands a logit's incidental
  parameters problem is real.

The FE logit is the robustness check, not the headline, and the two are reported together.

**Absorption by alternating projections.** Demeaning by student and by question in turn,
iterated to convergence, is equivalent to including both dummy sets (Frisch–Waugh–Lovell) but
never forms the 3,535 x 1,015 design. `check_against_dummies()` proves that equivalence on a
subsample rather than asserting it.

**Standard errors cluster on the student.** A student contributes several sessions and a
session several responses, all sharing one transcript; independent-row SEs would be far too
small. The CR1 finite-sample correction is applied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FEResult:
    """One regression's coefficients, with everything needed to report them."""
    params: pd.Series
    se: pd.Series
    n: int
    n_clusters: int
    absorbed: dict[str, int]
    dof: int

    @property
    def t(self) -> pd.Series:
        return self.params / self.se

    def ci(self, level: float = 0.95) -> pd.DataFrame:
        from scipy import stats
        q = stats.t.ppf(0.5 + level / 2, max(self.dof, 1))
        return pd.DataFrame({"estimate": self.params,
                             "lo": self.params - q * self.se,
                             "hi": self.params + q * self.se})

    def pvalues(self) -> pd.Series:
        from scipy import stats
        return pd.Series(2 * stats.t.sf(np.abs(self.t), max(self.dof, 1)),
                         index=self.params.index)

    def table(self, level: float = 0.95) -> pd.DataFrame:
        out = self.ci(level)
        out["se"] = self.se
        out["t"] = self.t
        out["p"] = self.pvalues()
        return out


def _demean(M: np.ndarray, codes: list[np.ndarray], tol: float = 1e-10,
            max_iter: int = 200) -> np.ndarray:
    """Project out several sets of fixed effects by alternating within-transformations."""
    M = M.astype(float, copy=True)
    if not codes:
        return M - M.mean(axis=0)
    for _ in range(max_iter):
        before = M.copy()
        for c in codes:
            n_lev = c.max() + 1
            counts = np.bincount(c, minlength=n_lev).astype(float)
            for j in range(M.shape[1]):
                means = np.bincount(c, weights=M[:, j], minlength=n_lev) / counts
                M[:, j] -= means[c]
        if np.max(np.abs(M - before)) < tol:
            break
    return M


def _factorize(s) -> np.ndarray:
    return pd.factorize(pd.Series(s).astype(str), sort=True)[0]


def fe_ols(y, X: pd.DataFrame, absorb: dict, cluster) -> FEResult:
    """Two-way (or k-way) absorbed OLS with cluster-robust SEs.

    `absorb` maps a name to the grouping array to absorb; `cluster` is the clustering key.
    Columns of `X` that become collinear after absorption (no within variation) are dropped and
    reported as absent rather than silently returning a singular fit.
    """
    y = np.asarray(y, dtype=float)
    cols = list(X.columns)
    codes = [_factorize(v) for v in absorb.values()]
    Z = _demean(np.column_stack([y] + [X[c].to_numpy(float) for c in cols]), codes)
    yd, Xd = Z[:, 0], Z[:, 1:]

    keep = [j for j in range(Xd.shape[1]) if np.std(Xd[:, j]) > 1e-12]
    dropped = [cols[j] for j in range(len(cols)) if j not in keep]
    if dropped:
        print(f"  [fe_ols] no within variation, dropped: {dropped}")
    Xd, cols = Xd[:, keep], [cols[j] for j in keep]

    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    beta = XtX_inv @ (Xd.T @ yd)
    resid = yd - Xd @ beta

    g = _factorize(cluster)
    n_g = g.max() + 1
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    order = np.argsort(g, kind="stable")
    gs, Xs, rs = g[order], Xd[order], resid[order]
    starts = np.searchsorted(gs, np.arange(n_g))
    ends = np.searchsorted(gs, np.arange(n_g), side="right")
    for a, b in zip(starts, ends):
        if b > a:
            u = Xs[a:b].T @ rs[a:b]
            meat += np.outer(u, u)
    n, k = Xd.shape
    n_absorbed = sum(c.max() + 1 for c in codes)
    dof_resid = max(n - k - n_absorbed, 1)
    # CR1: the usual finite-sample scaling, with the absorbed effects counted as parameters.
    scale = (n_g / max(n_g - 1, 1)) * ((n - 1) / dof_resid)
    V = scale * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return FEResult(params=pd.Series(beta, index=cols), se=pd.Series(se, index=cols),
                    n=n, n_clusters=n_g,
                    absorbed={k_: int(c.max() + 1) for k_, c in zip(absorb, codes)},
                    dof=max(n_g - 1, 1))


def absorb_matrix(M: pd.DataFrame, absorb: dict) -> tuple[pd.DataFrame, int]:
    """Demean a whole matrix once, for many small regressions that share the same fixed effects.

    Frisch-Waugh-Lovell holds per regression, so a column demeaned once can be reused in every
    model that absorbs the same effects. Demeaning 480 columns together instead of 4 columns
    421 times is the difference between seconds and an hour, and the estimates are identical.
    """
    codes = [_factorize(v) for v in absorb.values()]
    out = _demean(M.to_numpy(float), codes)
    return pd.DataFrame(out, columns=M.columns, index=M.index), sum(c.max() + 1 for c in codes)


def ols_demeaned(yd: np.ndarray, Xd: pd.DataFrame, cluster, n_absorbed: int) -> FEResult:
    """OLS with cluster-robust SEs on already-demeaned data (see :func:`absorb_matrix`)."""
    cols = list(Xd.columns)
    X = Xd.to_numpy(float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ yd)
    resid = yd - X @ beta
    g = _factorize(cluster)
    n_g = g.max() + 1
    order = np.argsort(g, kind="stable")
    gs, Xs, rs = g[order], X[order], resid[order]
    starts = np.searchsorted(gs, np.arange(n_g))
    ends = np.searchsorted(gs, np.arange(n_g), side="right")
    meat = np.zeros((X.shape[1], X.shape[1]))
    for a, b in zip(starts, ends):
        if b > a:
            u = Xs[a:b].T @ rs[a:b]
            meat += np.outer(u, u)
    n, k = X.shape
    dof_resid = max(n - k - n_absorbed, 1)
    scale = (n_g / max(n_g - 1, 1)) * ((n - 1) / dof_resid)
    V = scale * (XtX_inv @ meat @ XtX_inv)
    return FEResult(params=pd.Series(beta, index=cols),
                    se=pd.Series(np.sqrt(np.clip(np.diag(V), 0, None)), index=cols),
                    n=n, n_clusters=n_g, absorbed={"(pre-absorbed)": n_absorbed},
                    dof=max(n_g - 1, 1))


def check_against_dummies(y, X: pd.DataFrame, absorb: dict, cluster,
                          tol_se: float = 0.02) -> None:
    """Prove the absorption equals explicit dummies, on whatever sample it is handed.

    Frisch-Waugh-Lovell says it must; this asserts it, because a demeaning loop that has not
    converged produces plausible-looking coefficients that are simply wrong.

    **The tolerance is a fraction of the standard error, not an absolute number.** Alternating
    projections converge to a tolerance, so the two fits will never be bit-identical; what
    matters is whether they could differ in anything reported. A coefficient gap of 2.6e-06
    against an SE of 0.027 is a ten-thousandth of a standard error and cannot move a published
    digit, while the same gap on a tightly estimated coefficient would matter — which is
    exactly what scaling by the SE captures.

    ⚠️ Hand this whole STUDENTS, not random rows. A random row subsample of a corpus with ~6
    responses per student leaves most students with one observation, so the within-student
    design is near-unidentified and the comparison tests arithmetic nobody will run.
    """
    import statsmodels.api as sm

    D = [pd.get_dummies(pd.Series(v).astype(str), prefix=k, drop_first=True, dtype=float)
         for k, v in absorb.items()]
    full = pd.concat([X.reset_index(drop=True)] + [d.reset_index(drop=True) for d in D], axis=1)
    full = sm.add_constant(full, has_constant="add")
    ref = sm.OLS(np.asarray(y, float), full).fit(
        cov_type="cluster", cov_kwds={"groups": _factorize(cluster)})
    got = fe_ols(y, X, absorb, cluster)
    worst = 0.0
    for c in got.params.index:
        se = max(ref.bse[c], 1e-12)
        d_b, d_se = abs(got.params[c] - ref.params[c]) / se, abs(got.se[c] - ref.bse[c]) / se
        worst = max(worst, d_b, d_se)
        if d_b > tol_se or d_se > tol_se:
            raise AssertionError(
                f"absorbed fit disagrees with dummies for {c!r}: "
                f"beta {got.params[c]:.6g} vs {ref.params[c]:.6g} ({d_b:.3g} SE), "
                f"se {got.se[c]:.6g} vs {ref.bse[c]:.6g} ({d_se:.3g} SE)")
    print(f"  [check] absorbed == dummies for {len(got.params)} coefficients "
          f"(n={got.n:,}, clusters={got.n_clusters:,}; worst gap {worst:.2g} SE)")


def benjamini_hochberg(p, alpha: float = 0.05) -> np.ndarray:
    """BH step-up: which hypotheses are rejected at FDR `alpha`."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out
