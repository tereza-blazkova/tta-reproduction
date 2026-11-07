#!/usr/bin/env python
"""Stage 22 — a different learner on the identical columns.

The paper's model is XGBoost on 57 code rates plus the objective's TF-IDF text. This script
holds those columns, the frozen objective-disjoint folds, the clean-row mask, the views and the
bootstrap exactly fixed, and changes only the learner. That is the point of the comparison:
given the same 1,514 columns, how much of the performance belongs to the features and how much
to gradient boosting? A baseline that quietly used other folds or other columns could not answer
it.

    python scripts/22_baseline.py --model logreg --run baseline-logreg --test
    python scripts/22_baseline.py --model tabpfn35 --run tabpfn35-both --test   # CUDA GPU

`--attainment` adds the student's attainment as one more rate column, as in `06_train_clean57.py`
(:mod:`ttapub.attainment`); a student without one is median-imputed, since these learners need a value.

**What is the same as `06_train_clean57.py`:** `design()`, `folds.lo_folds`, `folds.clean_mask`,
the four views (`oof_all`, `oof_clean`, `test_tsl`, `test_eedi`) and the clustered bootstrap, so
the numbers go straight into Table 1 beside the XGBoost row.

**What differs, deliberately:**

* **No training-time augmentation.** The XGBoost recipe doubles its training set with a
  degraded copy. A baseline gets none of that: it is the plain learner on the plain columns.
  Logistic regression is deterministic and fits once; the foundation models average `--seeds`
  runs of their own ensembles (:mod:`ttapub.foundation`), as XGBoost averages its seeds.
* **Rates are imputed and standardised.** A linear model needs it; XGBoost routes NaN as
  missing. Imputation uses the training fold's median only.
* **By default the foundation models get the objective text reduced to `--svd` components**
  (fitted on the training fold, like the TF-IDF): 57 rates + 32 components = 89 columns. That
  was a choice, not a model limit, and it costs: logreg on the same compressed text loses ~1.9
  test-TSL AUROC (`logreg-svd32-both`). `--objective tfidf` gives any learner the XGBoost columns
  exactly (densified; TabPFN 3.5 reads up to 20,000), and the output records which was used
  (`objective: tfidf` or `tfidf-svd<k>`).
* **Nothing is fitted in the foundation models.** They learn in context: a fold's "model" is
  its training rows. So they need no tuning, and none is done.

**Rules that carry over from the repo (do not relax them for a baseline):**

* Folds are loaded, never recomputed. Eedi is evaluation-only: nothing, including `--C`, is
  chosen on it. If you tune a hyper-parameter, do it on the out-of-fold views only.
* No objective-ID one-hots; the objective enters as text.
* Report log loss *and* AUROC on every view.
* Test rows are predicted independently. For a foundation model that is measured, not assumed:
  :func:`ttapub.foundation.independence_gate` runs on fold 0 before anything is scored and the
  run stops if fp32 predictions of the same rows differ by 1e-4 with different neighbours.
"""
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import attainment, config, data, folds, foundation, metrics, runlog, xgb

# the design builder is shared, not copied, so a change to the columns changes both models
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "train_clean57", pathlib.Path(__file__).with_name("06_train_clean57.py"))
_t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t)
design, feature_columns = _t.design, _t.feature_columns

#: fp32 disagreement above this between the same rows in different company stops the run
GATE_TOL = 1e-4


def make_model(kind: str, num: list[str], C: float, svd: int, seed: int, learner=None,
               objective: str = "auto"):
    """``objective`` is how the text enters: ``tfidf`` (the XGBoost columns, 1,457 on the full
    training set) or ``svd`` (TF-IDF reduced to ``svd`` components). ``auto`` is tfidf for
    logreg and svd for the foundation models."""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler

    objective = resolve_objective(kind, objective)
    rates = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    # the objective-only arm of the ladder has no rate columns at all
    with_rates = [("rates", rates, num)] if num else []
    obj = TfidfVectorizer(max_features=5_000, ngram_range=(1, 2))
    if objective == "svd":
        obj = make_pipeline(obj, TruncatedSVD(n_components=svd, random_state=seed))
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        pre = ColumnTransformer([("obj_text", obj, "learning_objective")] + with_rates,
                                sparse_threshold=1.0)
        clf = LogisticRegression(C=C, max_iter=2_000, solver="lbfgs")
        return Pipeline([("pre", pre), ("clf", clf)])
    if kind in foundation.LEARNERS:
        # the foundation models read dense arrays, so the TF-IDF arm is densified here
        pre = ColumnTransformer([("obj_text", obj, "learning_objective")] + with_rates,
                                sparse_threshold=0.0)
        return Pipeline([("pre", pre), ("clf", learner())])
    raise SystemExit(f"unknown --model {kind!r}")


def resolve_objective(kind: str, objective: str) -> str:
    if objective == "auto":
        return "tfidf" if kind == "logreg" else "svd"
    return objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg", *foundation.LEARNERS], required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--C", type=float, default=0.1,
                    help="inverse L2 strength for logreg. Tune on the out-of-fold views only.")
    ap.add_argument("--svd", type=int, default=32,
                    help="objective-text components for the foundation models, which cannot "
                         "take TF-IDF")
    ap.add_argument("--objective", choices=["auto", "tfidf", "svd"], default="auto",
                    help="objective text as the XGBoost TF-IDF columns or reduced by --svd; auto = "
                         "tfidf for logreg, svd for the foundation models")
    ap.add_argument("--seed", type=int, default=42, help="the SVD's seed")
    ap.add_argument("--seeds", type=int, default=3,
                    help="foundation models: runs averaged per prediction (xgb.SEEDS[:n])")
    ap.add_argument("--n-est", type=int, default=8,
                    help="foundation models: ensemble members per run (each library's own)")
    ap.add_argument("--blocks", default="tutor,student",
                    help="as in 06_train_clean57.py; '' is the objective-only arm of the ladder")
    ap.add_argument("--source", choices=["labels", "probs"], default="labels")
    ap.add_argument("--attainment", action="store_true",
                    help="add the student's attainment from their other training sessions")
    ap.add_argument("--test", action="store_true", help="also fit on all rows and score test")
    ap.add_argument("--no-oof", action="store_true",
                    help="skip the out-of-fold views: one fit on all rows, test only (needs --test); "
                         "the independence gate then runs on the test rows")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    size = "small"
    if a.no_oof and not a.test:
        raise SystemExit("--no-oof leaves nothing to score without --test")

    name = a.run or f"baseline-{a.model}"
    fm = a.model in foundation.LEARNERS
    seeds = xgb.SEEDS[: a.seeds] if fm else [a.seed]
    learner = ((lambda fp32=None: foundation.make(a.model, a.n_est, seeds, size, fp32))
               if fm else None)
    train = data.load_train().reset_index(drop=True)
    y = train.is_correct.to_numpy(float)
    lo = train.learning_objective_id.to_numpy()
    fold = folds.lo_folds(len(train))
    clean = folds.clean_mask(len(train))

    feats = pd.read_csv(config.features_path("train", a.source), dtype={"session_id": str})
    num = feature_columns(feats.columns, {b for b in a.blocks.split(",") if b})
    X = design(train, feats, num)
    n_codes = len(num)
    if a.attainment:
        meta = data.load_metadata().set_index("response_id")
        tr_meta = train.assign(student_id=train.response_id.map(meta.student_id),
                               question_id=train.response_id.map(meta.question_id))
        ab_train, ab_test = attainment.train_and_test(
            tr_meta, data.load_test().reset_index(drop=True).student_id)
        X["student_attainment"] = ab_train
        num = num + ["student_attainment"]

    d = runlog.start(name, vars(a), [config.features_path("train", a.source)])
    print(f"[{name}] {a.model}: {n_codes} code columns + objective text"
          + (" + attainment" if a.attainment else ""), flush=True)
    t0 = time.time()
    res: dict = {"run": name, "model": a.model, "blocks": a.blocks, "source": a.source,
                 "n_cols": n_codes, "attainment": a.attainment,
                 "C": a.C if a.model == "logreg" else None,
                 "objective": ("tfidf" if resolve_objective(a.model, a.objective) == "tfidf"
                               else f"tfidf-svd{a.svd}"),
                 "augment": False, "seeds": seeds, "columns": num}
    if fm:
        res["learner"] = learner().describe()

    def gate(m, Xc, yc, Xq) -> None:
        """Stop before anything is scored if query rows influence each other's predictions."""
        pre = m.named_steps["pre"]
        g = foundation.independence_gate(
            lambda fp32: foundation.make(a.model, a.n_est, seeds[:1], size, fp32),
            pre.transform(Xc), yc, pre.transform(Xq))
        res["independence_gate"] = g | {"tol_fp32": GATE_TOL}
        print(f"  independence gate: max |dp| fp32 {g['fp32']:.2e}, "
              f"production {g['production']:.2e}", flush=True)
        if g["fp32"] > GATE_TOL:
            raise SystemExit("query rows influence each other's predictions in fp32: "
                             "this learner pools across test rows, nothing is scored")

    res["oof"] = not a.no_oof
    oof = np.zeros(len(y))
    for k in ([] if a.no_oof else sorted(set(fold))):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        m = make_model(a.model, num, a.C, a.svd, a.seed, learner, a.objective).fit(
            X.iloc[tr].reset_index(drop=True), y[tr])
        if fm and k == min(fold):
            gate(m, X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te].reset_index(drop=True))
        oof[te] = m.predict_proba(X.iloc[te].reset_index(drop=True))[:, 1]
        print(f"  fold {k}: {time.time() - t0:.0f}s", flush=True)
    if not a.no_oof:
        pd.DataFrame({"response_id": train.response_id, "y": y, "p": oof,
                      "fold": fold, "clean": clean}).to_csv(d / "oof.csv", index=False)
        for view, msk in (("oof_all", np.ones(len(y), bool)), ("oof_clean", clean)):
            r = metrics.basic(y[msk], oof[msk]) | metrics.boot_ci(y[msk], oof[msk], lo[msk],
                                                                  B=a.boot)
            res[view] = r
            print(metrics.fmt(view, r), flush=True)

    if a.test:
        test = data.load_test().reset_index(drop=True)
        tfeats = pd.read_csv(config.features_path("test", a.source), dtype={"session_id": str})
        Xt = design(test, tfeats, num[:n_codes])
        if a.attainment:
            Xt["student_attainment"] = ab_test
        yt = test.is_correct.to_numpy(float)
        m = make_model(a.model, num, a.C, a.svd, a.seed, learner, a.objective).fit(X, y)
        if fm and a.no_oof:
            gate(m, X, y, Xt)
        pt = m.predict_proba(Xt)[:, 1]
        print(f"  test predicted: {time.time() - t0:.0f}s", flush=True)
        out = pd.DataFrame({"response_id": test.response_id, "data_source": test.data_source,
                            "y": yt, "p": pt})
        for s_, ps in getattr(m.named_steps["clf"], "seed_p_", {}).items():
            out[f"p_seed{s_}"] = ps                  # foundation models: one column per seed
        out.to_csv(d / "test.csv", index=False)
        for src in ("tsl", "eedi", "all"):
            msk = (np.ones(len(yt), bool) if src == "all"
                   else (test.data_source == src).to_numpy())
            r = (metrics.basic(yt[msk], pt[msk])
                 | {"ll_icpt": metrics.intercept_refit_ll(yt[msk], pt[msk])}
                 | metrics.boot_ci(yt[msk], pt[msk], test.session_id.to_numpy()[msk],
                                   B=a.boot))
            res[f"test_{src}"] = r
            print(metrics.fmt(f"test {src}", r), flush=True)

    runlog.finish(d, res | {"seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
