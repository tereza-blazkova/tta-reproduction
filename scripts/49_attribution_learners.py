#!/usr/bin/env python
"""Stage 49 — what each learner attributes to the codes, on the same held-out responses.

Fig. 2b shows XGBoost's TreeSHAP. This computes the same kind of attribution for logistic
regression and TabPFN 3.5, each fitted as in the paper's model table, so the codes' attributions
can be compared across learners response by response.

**Rows.** 2,000 TSL test responses (a seeded draw), the same for every learner. Test rows, not
training rows: an in-context learner holds its training rows in its context, and a row explained
while it sits in the context can be looked up rather than predicted.

**Players.** The 57 code rates, one player each, and the objective's text as one player (its
TF-IDF terms move together). A learner's attributions for a response therefore sum to its
log-odds for that response minus the background's.

**One estimator for every learner.** Shapley values by permutation sampling, with the
interventional value function. For each response, `--perms` orderings of the 58 players come
in antithetic pairs (an ordering and its reverse), each ordering with its own background
response drawn from the training rows. Along an ordering the players switch, one at a time, from the
background's values to the response's, and each switch's change in log-odds is that player's
contribution. The orderings and the backgrounds depend only on the response's position in the
draw, so they are identical across learners and shards: the learners' values are paired.

Exact references beside it: XGBoost's TreeSHAP (as Fig. 2b) and logistic regression's closed
form. For a linear model every switch adds the same amount whatever the order, so the sampled
estimator must reproduce the closed form exactly given its own backgrounds; that equality is the
check on the bookkeeping, and it stops the run if it fails.

Writes `results/attribution-learners/<learner>[.<i>of<n>].npz` (gitignored) and, after
`--merge`, `attribution_by_column.csv` and `metrics.json`.

    python scripts/49_attribution_learners.py --learner logreg
    python scripts/49_attribution_learners.py --learner xgb
    python scripts/49_attribution_learners.py --learner tabpfn35 --shard 0 --n-shards 4   # GPU
    python scripts/49_attribution_learners.py --merge
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import config, data, foundation, metrics, runlog, xgb

_HERE = pathlib.Path(__file__).parent
RUN = "attribution-learners"
LEARNERS = ("logreg", "xgb", "tabpfn35")
SEED = 49


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s06 = _load("s06", "06_train_clean57.py")
s07 = _load("s07", "07_shap_clean57.py")
s22 = _load("s22", "22_baseline.py")
s23 = _load("s23", "23_attribution_logreg.py")


def frames(n_rows: int):
    train = data.load_train().reset_index(drop=True)
    feats = pd.read_csv(config.features_path("train", "labels"), dtype={"session_id": str})
    num = s06.feature_columns(feats.columns, {"tutor", "student"})
    X = s06.design(train, feats, num)
    y = train.is_correct.to_numpy(float)
    test = data.load_test().reset_index(drop=True)
    tfeats = pd.read_csv(config.features_path("test", "labels"), dtype={"session_id": str})
    Xt = s06.design(test, tfeats, num)
    tsl = np.flatnonzero((test.data_source == "tsl").to_numpy())
    rows = np.random.default_rng(0).choice(tsl, n_rows, replace=False)
    return X, y, test, Xt, num, tsl, rows


def groups(names: list[str], num: list[str], prefix: str) -> list[np.ndarray]:
    """The 58 players as column indices of the learner's own design matrix: 57 codes, then the
    objective's TF-IDF block."""
    g = [np.array([names.index(f"{prefix}__{c}")]) for c in num]
    g.append(np.array([i for i, n in enumerate(names) if n.startswith("obj_text__")]))
    if sum(len(x) for x in g) != len(names):
        raise SystemExit("the players do not cover the design matrix exactly")
    return g


def draws(pos: int, perms: int, n_train: int, n_players: int):
    """Orderings (antithetic pairs) and one background per pair, from the row's position only."""
    rng = np.random.default_rng([SEED, pos])
    half = [rng.permutation(n_players) for _ in range(perms // 2)]
    order = [o for h in half for o in (h, h[::-1])]
    # one background per ordering: the backgrounds, not the orderings, carry most of the
    # variance (for an additive model the ordering does not matter at all)
    return np.array(order), rng.choice(n_train, perms, replace=False)


def permutation_shap(f, Zx, Zbg, pos, g, perms, batch, log=None):
    """(phi, se, fx, fb): sampled Shapley values (r, players), their standard errors over the
    orderings, f at the rows, and the mean of f over each row's backgrounds."""
    r, P, k = len(Zx), perms, len(g)
    phi_p = np.zeros((r, P, k))
    fx, fb, ends = np.zeros(r), np.zeros(r), np.zeros((r, P))
    per = max(1, batch // (P * (k + 1)))                     # rows per evaluation batch
    t0 = time.time()
    for a in range(0, r, per):
        idx = range(a, min(a + per, r))
        paths, meta = [], []
        for i in idx:
            order, bg = draws(pos[i], P, len(Zbg), k)
            for p in range(P):
                z = Zbg[bg[p]].copy()
                path = [z.copy()]
                for j in order[p]:
                    z[g[j]] = Zx[i, g[j]]
                    path.append(z.copy())
                paths.append(np.stack(path))
                meta.append((i, p, order[p]))
        F = f(np.concatenate(paths)).reshape(len(paths), k + 1)
        for (i, p, o), Fi in zip(meta, F):
            phi_p[i, p, o] = np.diff(Fi)
            # every path ends at the row itself; a learner whose output moves with the batch it
            # shares (TabPFN at reduced precision) ends at slightly different values, so f(x) is
            # their mean and their spread is recorded as the evaluation noise
            fx[i] += Fi[-1] / P
            fb[i] += Fi[0] / P
            ends[i, p] = Fi[-1]
        if log:
            log(f"  rows {a + len(idx):,}/{r:,}  {time.time() - t0:.0f}s")
    se = phi_p.std(1, ddof=1) / np.sqrt(P)                    # over the orderings' backgrounds
    permutation_shap.eval_noise = float((ends.max(1) - ends.min(1)).max())
    return phi_p.mean(1), se, fx, fb


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def fit_logreg(X, y, num):
    pipe = s22.make_model("logreg", num, 0.1, 32, 42).fit(X, y)
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    names = list(pre.get_feature_names_out())

    def Z(frame):
        z = pre.transform(frame)
        return np.asarray(z.toarray() if hasattr(z, "toarray") else z, dtype=np.float64)

    w, b = clf.coef_[0], clf.intercept_[0]
    return pipe, Z, (lambda Zq: Zq @ w + b), groups(names, num, "rates"), w


def fit_xgb(X, y, Xt, num, n_jobs):
    p, models = xgb.fit_predict(X, y, Xt, num, lo="text", seeds=xgb.SEEDS[:5], n_jobs=n_jobs,
                                aug_seed=100, augment=True)
    import xgboost as xgbl
    names = list(models[0].named_steps["pre"].get_feature_names_out())
    if any(list(m.named_steps["pre"].get_feature_names_out()) != names for m in models):
        raise SystemExit("the seeds' pipelines disagree on the design columns")

    def Z(frame):
        return xgb.densify(models[0].named_steps["pre"].transform(frame))

    def f(Zq):                     # the seed-averaged margin, the scale TreeSHAP splits
        dm = xgbl.DMatrix(Zq, feature_names=names, missing=np.nan)
        return np.mean([m[-1].get_booster().predict(dm, output_margin=True) for m in models], 0)

    return p, models, Z, f, groups(names, num, "dlg")


def fit_tabpfn(X, y, num, seed: int, n_est: int):
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    pre = s22.make_model("tabpfn35", num, 0.1, 32, 42, learner=lambda: None,
                         objective="tfidf").named_steps["pre"].fit(X)
    names = list(pre.get_feature_names_out())

    def Z(frame):
        return np.asarray(pre.transform(frame), dtype=np.float32)

    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3_5, model_path=foundation.TabPFN35._weights(), n_estimators=n_est,
        device="cuda", ignore_pretraining_limits=True, random_state=seed, inference_precision="auto")
    with torch.inference_mode():
        clf.fit(Z(X), y.astype(int))

    def f(Zq):
        with torch.inference_mode():
            return logit(foundation._p1(clf.predict_proba(Zq.astype(np.float32)), clf.classes_))

    return Z, f, groups(names, num, "rates")


def run_learner(a) -> None:
    X, y, test, Xt, num, tsl, rows = frames(a.rows)
    pos = np.arange(a.rows)[a.shard::a.n_shards]
    if a.limit:
        pos = pos[: a.limit]
    sel = rows[pos]
    tag = a.learner + (f".{a.shard}of{a.n_shards}" if a.n_shards > 1 else "") + (
        f".limit{a.limit}" if a.limit else "")
    d = runlog.start(f"{RUN}/{tag}", vars(a), [config.features_path("train", "labels")])
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    out, checks = {}, {}
    if a.learner == "logreg":
        pipe, Z, f, g, w = fit_logreg(X, y, num)
        pt = pipe.predict_proba(Xt)[:, 1]
        checks["tsl_ll"] = metrics.basic(test.is_correct.to_numpy(float)[tsl], pt[tsl])["ll"]
        S, obj = s23.linear_shap(pipe, X, Xt.iloc[sel], num)
        out["exact"] = np.column_stack([S, obj])
        # the weights themselves: log-odds per standard deviation of each code's rate (the rates
        # are standardised inside the pipeline)
        names = list(pipe.named_steps["pre"].get_feature_names_out())
        out["coef"] = np.array([w[names.index(f"rates__{c}")] for c in num])
    elif a.learner == "xgb":
        pt, models, Z, f, g = fit_xgb(X, y, Xt, num, a.n_jobs)
        checks["tsl_ll"] = metrics.basic(test.is_correct.to_numpy(float)[tsl], pt[tsl])["ll"]
        S, obj = s07.shap_block(models, Xt.iloc[sel], num)
        out["exact"] = np.column_stack([S, obj])
    else:
        Z, f, g = fit_tabpfn(X, y, num, 42, a.n_est)
    Zx, Zbg = Z(Xt.iloc[sel]), Z(X)
    log(f"[{tag}] fitted in {time.time() - t0:.0f}s; {len(sel):,} rows x {a.perms} orderings x "
        f"{len(g) + 1} evaluations")
    if a.learner == "tabpfn35" and a.shard == 0:
        # the paper's TabPFN is the mean of three seeds; rebuild it on these rows and compare
        ps = [1 / (1 + np.exp(-f(Zx)))]
        for s in xgb.SEEDS[1:3]:
            _, fs, _ = fit_tabpfn(X, y, num, s, a.n_est)
            ps.append(1 / (1 + np.exp(-fs(Zx))))
        ref = pd.read_csv(config.RESULTS / "tabpfn35-tfidf-both" / "test.csv").set_index("response_id")
        ref = ref.loc[test.response_id.iloc[sel], "p"].to_numpy()
        dif = np.abs(np.mean(ps, 0) - ref)
        checks |= {"p_vs_paper_mean_abs": float(dif.mean()), "p_vs_paper_max_abs": float(dif.max()),
                   "p_vs_paper_corr": float(np.corrcoef(np.mean(ps, 0), ref)[0, 1])}
        log(f"  3-seed TabPFN vs the paper's test.csv: mean |dp| {dif.mean():.4f}, "
            f"max {dif.max():.4f}, r {checks['p_vs_paper_corr']:.5f}")
        if checks["p_vs_paper_corr"] < 0.99:
            raise SystemExit("this TabPFN does not reproduce the paper's predictions")
    phi, se, fx, fb = permutation_shap(f, Zx, Zbg, pos, g, a.perms, a.batch, log)
    eff = np.abs(phi.sum(1) - (fx - fb)).max()
    checks["efficiency_max_abs"] = float(eff)
    checks["eval_noise_max"] = permutation_shap.eval_noise    # spread of f(x) across its batches
    log(f"  efficiency gap {eff:.1e}; f(x) spread across batches up to {checks['eval_noise_max']:.1e}")
    if eff > 1e-6 * max(1.0, np.abs(fx).max()):
        raise SystemExit(f"attributions do not sum to f(x) - f(background): {eff:.2e}")
    if a.learner == "logreg":
        # the closed form with each row's own sampled backgrounds must equal the estimate
        lin = np.zeros_like(phi)
        for r_, p_ in enumerate(pos):
            _, bg = draws(p_, a.perms, len(Zbg), len(g))
            dz = (Zx[r_] - Zbg[bg].mean(0)) * w
            lin[r_] = [dz[gi].sum() for gi in g]
        checks["closed_form_max_abs"] = float(np.abs(lin - phi).max())
        if checks["closed_form_max_abs"] > 1e-9:
            raise SystemExit(f"the estimator misses the closed form by {checks['closed_form_max_abs']:.2e}")
    np.savez_compressed(d / f"{tag}.npz", pos=pos, response_id=test.response_id.iloc[sel].to_numpy().astype(str),
                        phi=phi, se=se, fx=fx, fb=fb, rates=Xt.iloc[sel][num].to_numpy(float),
                        players=np.array(num + ["objective"]), **out)
    checks["seconds"] = time.time() - t0
    runlog.finish(d, {"learner": a.learner, "rows": len(sel), "perms": a.perms, **checks})


def merge(a) -> None:
    """One file per learner, every row present once, and the per-column summary."""
    base = config.RESULTS / RUN
    got, summary, m = {}, [], {}
    for learner in LEARNERS:
        parts = sorted(base.glob(f"{learner}*/{learner}*.npz"))
        parts = [p for p in parts if "limit" not in p.name]
        if not parts:
            raise SystemExit(f"no results for {learner}")
        z = [dict(np.load(p, allow_pickle=True)) for p in parts]   # our own files (older ones hold ids as objects)
        pos = np.concatenate([q["pos"] for q in z])
        if sorted(pos) != list(range(a.rows)):
            raise SystemExit(f"{learner}: rows missing or repeated across {len(parts)} shards")
        o = np.argsort(pos)
        # per-row arrays are stacked over the shards; the rest (players, the weights) is the learner's
        per_row = [k for k in z[0] if k != "players" and z[0][k].ndim and len(z[0][k]) == len(z[0]["pos"])]
        got[learner] = {k: (np.concatenate([q[k] for q in z])[o] if k in per_row else z[0][k])
                        for k in z[0] if k != "players"}
        got[learner]["players"] = z[0]["players"]
    players = [str(p) for p in got["xgb"]["players"]]
    rid = got["xgb"]["response_id"]
    for learner in LEARNERS:
        if not (got[learner]["response_id"] == rid).all():
            raise SystemExit(f"{learner} explains different rows")
    rates = got["xgb"]["rates"]
    for j, col in enumerate(players):
        row = {"column": col}
        for learner in LEARNERS:
            v = got[learner]["phi"][:, j]
            row[f"{learner}_mean_abs"] = np.abs(v).mean()
            row[f"{learner}_se_mean"] = np.nanmean(got[learner]["se"][:, j])
            if j < rates.shape[1]:
                ok = ~np.isnan(rates[:, j])
                row[f"{learner}_spearman"] = pd.Series(rates[ok, j]).corr(pd.Series(v[ok]),
                                                                          method="spearman")
            if "exact" in got[learner]:
                row[f"{learner}_exact_mean_abs"] = np.abs(got[learner]["exact"][:, j]).mean()
        for p_, q_ in (("xgb", "logreg"), ("xgb", "tabpfn35"), ("logreg", "tabpfn35")):
            row[f"r_{p_}_{q_}"] = np.corrcoef(got[p_]["phi"][:, j], got[q_]["phi"][:, j])[0, 1]
        summary.append(row)
    s = pd.DataFrame(summary)
    s.to_csv(base / "attribution_by_column.csv", index=False)
    codes = s[s.column != "objective"]
    for p_, q_ in (("xgb", "logreg"), ("xgb", "tabpfn35"), ("logreg", "tabpfn35")):
        m[f"spearman_mean_abs_{p_}_{q_}"] = float(codes[f"{p_}_mean_abs"].corr(
            codes[f"{q_}_mean_abs"], method="spearman"))
    m["xgb_tree_vs_permutation_r"] = float(np.corrcoef(got["xgb"]["exact"][:, :57].ravel(),
                                                       got["xgb"]["phi"][:, :57].ravel())[0, 1])
    np.savez_compressed(base / "merged.npz", response_id=rid, rates=rates, players=np.array(players),
                        **{f"{k}_{l}": got[l][k] for l in LEARNERS for k in ("phi", "se", "fx", "fb")},
                        **{f"exact_{l}": got[l]["exact"] for l in ("xgb", "logreg")})
    runlog.finish(base, {"rows": a.rows, "perms": a.perms, **m})
    print(s.sort_values("xgb_mean_abs", ascending=False).head(15).round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", choices=LEARNERS)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--perms", type=int, default=8, help="orderings per row (even: antithetic pairs)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N rows of the shard (smoke test)")
    ap.add_argument("--batch", type=int, default=16384, help="model evaluations per call")
    ap.add_argument("--n-est", type=int, default=8, help="TabPFN ensemble members, as the paper")
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    if a.perms % 2:
        raise SystemExit("--perms must be even (antithetic pairs)")
    if a.merge:
        merge(a)
    elif a.learner:
        run_learner(a)
    else:
        raise SystemExit("--learner or --merge")


if __name__ == "__main__":
    main()
