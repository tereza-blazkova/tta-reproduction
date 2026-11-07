#!/usr/bin/env python
"""Stage 06 — fit **clean57** and score it on every view.

One column per code, the objective as TF-IDF text, the fixed booster
(:mod:`ttapub.xgb`). Arms are selected with `--blocks` and `--lo`, which is how the nested
ladder of the paper's performance table is produced — the same recipe, the same frozen folds,
only the feature blocks changing:

    --blocks ""              objective only
    --blocks student         + the 16 student signals
    --blocks tutor           + the 41 tutor moves
    --blocks tutor,student   + both  (= clean57)

`--attainment` adds one column, the student's question-adjusted attainment measured on their other
training sessions (test rows: on all their training answers; see :mod:`ttapub.attainment`). It is
passed through untouched and is not attenuated by the augmentation, which models transcripts.

Views, all reported, none optional:

* **oof_all** — out-of-fold over the frozen LO-disjoint partition, every training row.
* **oof_clean** — the same predictions on leakage-free rows only (a different *population*,
  not a subsample: compare models within it, never a clean level against a pooled level).
* **test_tsl / test_eedi / test_all** — the released test split by provider, fitted on all
  training rows. Eedi is a provider absent from training, so it is zero-shot transfer and its
  base rate differs by 21 points; the intercept-refit log loss is printed beside the honest one
  to separate "wrong level" from "wrong ranking".


    python scripts/06_train_clean57.py --blocks tutor,student --lo text --seeds 5 --test \
        --run clean57
"""
from __future__ import annotations

import argparse
import json
import time

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd

from ttapub import attainment, agg, config, data, folds, metrics, runlog, xgb


def feature_columns(cols, blocks: set[str]) -> list[str]:
    keep = []
    if "tutor" in blocks:
        keep += agg.code_columns(cols, "tutor")
    if "student" in blocks:
        keep += agg.code_columns(cols, "student")
    return keep


def design(frame: pd.DataFrame, feats: pd.DataFrame, num: list[str]) -> pd.DataFrame:
    """Response rows x (code rates + objective). A session with no features gets NaN, which
    XGBoost routes as missing — it is not imputed, because "could not measure" and "rate 0"
    are different statements."""
    X = frame[["session_id"]].merge(feats[["session_id"] + num], on="session_id",
                                    how="left")[num].reset_index(drop=True)
    X["learning_objective"] = frame["learning_objective"].to_numpy()
    X["learning_objective_id"] = frame["learning_objective_id"].to_numpy()
    return X


def reference_oof(response_id: pd.Series, y: np.ndarray) -> dict[str, np.ndarray]:
    """Out-of-fold predictions of the models this one is compared against."""
    out = {}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--blocks", default="tutor,student")
    ap.add_argument("--lo", default="text", choices=list(xgb.LO_ARMS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--drop", default="", help="comma list of code columns to drop")
    ap.add_argument("--only", default="", help="comma list of code columns to keep (all others in "
                                               "--blocks are dropped); stage 44's add-one-in arms")
    ap.add_argument("--shuffle", action="store_true",
                    help="negative control: permute the code columns across sessions (train and test "
                         "separately, seed 0), so they keep their distribution and carry no information")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--xgb", default="{}", help="JSON overrides of XGB_PARAMS")
    ap.add_argument("--test", action="store_true", help="also fit on all rows and score test")
    ap.add_argument("--test-only", action="store_true", help="skip the out-of-fold views")
    ap.add_argument("--attainment", action="store_true",
                    help="add the student's attainment from their other training sessions")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--source", choices=["labels", "probs"], default="labels",
                    help="which measurement instrument's feature table to fit on: the 122B's "
                         "own labels (the paper's main path) or the distilled classifiers' "
                         "probabilities (the appendix arm). Recorded in metrics.json, because "
                         "a number is only interpretable with its instrument named.")
    a = ap.parse_args()

    blocks = {b for b in a.blocks.split(",") if b}
    name = a.run or ("clean57-" + ("+".join(sorted(blocks)) or "lo") + f"-{a.lo}")
    over = json.loads(a.xgb)
    seeds = xgb.SEEDS[: a.seeds]
    a.test = a.test or a.test_only

    train = data.load_train().reset_index(drop=True)
    y = train.is_correct.to_numpy(float)
    lo = train.learning_objective_id.to_numpy()
    fold = folds.lo_folds(len(train))
    clean = folds.clean_mask(len(train))

    feats = pd.read_csv(config.features_path("train", a.source), dtype={"session_id": str})
    num = [c for c in feature_columns(feats.columns, blocks) if c not in a.drop.split(",")]
    if a.only:
        keep = set(a.only.split(","))
        missing = keep - set(num)
        if missing:
            raise SystemExit(f"--only names columns not in the blocks: {sorted(missing)}")
        num = [c for c in num if c in keep]
    if a.shuffle:
        feats = feats.copy()
        feats[num] = feats[num].to_numpy()[np.random.default_rng(0).permutation(len(feats))]
    X = design(train, feats, num)
    cols, aug = num, None
    if a.attainment:
        meta = data.load_metadata().set_index("response_id")
        tr_meta = train.assign(student_id=train.response_id.map(meta.student_id),
                               question_id=train.response_id.map(meta.question_id))
        test_students = data.load_test().reset_index(drop=True).student_id
        ab_train, ab_test = attainment.train_and_test(tr_meta, test_students)
        X["student_attainment"] = ab_train
        cols, aug = num + ["student_attainment"], num

    d = runlog.start(name, vars(a), [config.features_path("train", a.source),
                                     config.prior_path(a.source)])
    head = f"[{name}] {len(num)} code columns, lo={a.lo}, seeds={seeds}"
    if num:
        head += f", rows with no features {X[num[0]].isna().mean():.2%}"
    print(head, flush=True)
    t0 = time.time()

    res: dict = {"run": name, "blocks": sorted(blocks), "lo": a.lo, "source": a.source,
                 "n_cols": len(num), "attainment": a.attainment, "shuffled": a.shuffle,
                 "seeds": seeds, "augment": not a.no_augment, "xgb_overrides": over,
                 "columns": num}

    # ── out-of-fold views ──
    if not a.test_only:
        oof = np.zeros(len(y))
        for k in sorted(set(fold)):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            oof[te], _ = xgb.fit_predict(
                X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te].reset_index(drop=True),
                cols, lo=a.lo, seeds=seeds, n_jobs=a.n_jobs, aug_seed=100 + k,
                augment=not a.no_augment, aug_cols=aug, **over)
            print(f"  fold {k}: {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame({"response_id": train.response_id, "y": y, "p": oof,
                      "fold": fold, "clean": clean}).to_csv(d / "oof.csv", index=False)

        refs = reference_oof(train.response_id, y)
        for view, m in (("oof_all", np.ones(len(y), bool)), ("oof_clean", clean)):
            r = metrics.basic(y[m], oof[m]) | metrics.boot_ci(y[m], oof[m], lo[m], B=a.boot)
            for ref_name, p_ref in refs.items():
                r[f"vs_{ref_name}"] = (
                    {"ll": float(np.round(metrics.basic(y[m], p_ref[m])["ll"], 6)),
                     "auroc": metrics.basic(y[m], p_ref[m])["auroc"]}
                    | metrics.boot_delta(y[m], oof[m], p_ref[m], lo[m], B=a.boot))
            res[view] = r
            print(metrics.fmt(view, r), flush=True)
            for ref_name in refs:
                v = r[f"vs_{ref_name}"]
                print(f"      vs {ref_name:12s} LL {v['ll']:.4f} -> d {v['d_ll']:+.4f} "
                      f"[{v['d_ll_ci'][0]:+.4f},{v['d_ll_ci'][1]:+.4f}]   "
                      f"AUROC {v['auroc']:.2f} -> d {v['d_auroc']:+.2f} "
                      f"[{v['d_auroc_ci'][0]:+.2f},{v['d_auroc_ci'][1]:+.2f}]", flush=True)

    # ── released test set, fitted on every training row ──
    if a.test:
        test = data.load_test().reset_index(drop=True)
        tfeats = pd.read_csv(config.features_path("test", a.source), dtype={"session_id": str})
        if a.shuffle:
            tfeats[num] = tfeats[num].to_numpy()[np.random.default_rng(0).permutation(len(tfeats))]
        Xt = design(test, tfeats, num)
        if a.attainment:
            Xt["student_attainment"] = ab_test
        yt = test.is_correct.to_numpy(float)
        pt, models = xgb.fit_predict(X, y, Xt, cols, lo=a.lo, seeds=seeds, n_jobs=a.n_jobs,
                                     aug_seed=100, augment=not a.no_augment, aug_cols=aug, **over)
        pd.DataFrame({"response_id": test.response_id, "data_source": test.data_source,
                      "y": yt, "p": pt}).to_csv(d / "test.csv", index=False)
        joblib.dump({"models": models, "num": cols, "lo": a.lo}, d / "models.joblib")
        if num:
            res["test_rows_without_features"] = float(Xt[num[0]].isna().mean())
        for src in ("tsl", "eedi", "all"):
            m = (np.ones(len(yt), bool) if src == "all"
                 else (test.data_source == src).to_numpy())
            r = (metrics.basic(yt[m], pt[m])
                 | {"ll_icpt": metrics.intercept_refit_ll(yt[m], pt[m])}
                 | metrics.boot_ci(yt[m], pt[m], test.session_id.to_numpy()[m], B=a.boot))
            res[f"test_{src}"] = r
            print(metrics.fmt(f"test {src}", r), flush=True)

    runlog.finish(d, res | {"seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
