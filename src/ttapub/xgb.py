"""The clean57 booster: XGBoost on one column per code, plus the objective's text.

**The design, in one paragraph.** Every conversation column is one named code's session rate —
no second annotator, no regex lexicon, no derived ratio, no interaction column. The only other
input is the learning objective as TF-IDF of its text. Objective-ID one-hots are deliberately
absent: the hidden test set contains objectives that never appear in training, so a per-objective
statistic degrades to a fallback exactly where it is needed, and the generalising signal is the
objective's *text* plus the transcript, not which objective it is.

Hyper-parameters are fixed in advance and were not tuned on these columns.

**Ablation-augmented training.** Each fit sees one extra copy of every training row with its
rate columns attenuated by U(0.2, 1.0). This simulates a shorter, differently-registered corpus
(the Eedi regime) without using any Eedi data. It is part of the recipe, not a tuning knob: switch it off
only as a reported arm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: The booster: `lossguide` at 31 leaves.
XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=0, grow_policy="lossguide",
    max_leaves=31, min_child_weight=126.0, reg_lambda=80.0, reg_alpha=10.0,
    colsample_bytree=0.35, subsample=0.6, gamma=0.02, tree_method="hist",
)

SEEDS = [42, 143, 244, 345, 446]

#: Objective arms. `text` is the publication default; `none` and `id` exist as reported arms.
LO_ARMS = ("none", "text", "id")


def densify(Z):
    """Sparse -> dense float32, so an unstored TF-IDF entry stays 0 instead of becoming
    XGBoost's `missing`. A genuine NaN (a session with no transcript) survives as NaN and is
    still routed as missing, which is the distinction the rate columns rely on."""
    if hasattr(Z, "toarray"):
        return Z.astype(np.float32).toarray()
    return np.asarray(Z, dtype=np.float32)


def build_pipeline(code_cols, lo: str = "text", seed: int = 42, n_jobs: int = 8,
                   max_features: int = 5_000, **overrides):
    """The clean57 pipeline: [objective TF-IDF] + passthrough code rates -> dense -> XGBoost.

    `X` must carry `learning_objective` (text) and every column in `code_cols`;
    `learning_objective_id` is needed only for `lo="id"`.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
    from xgboost import XGBClassifier

    if lo not in LO_ARMS:
        raise ValueError(f"lo must be one of {LO_ARMS}, got {lo!r}")
    transformers = []
    if lo in ("text", "id"):
        transformers.append(("obj_text", TfidfVectorizer(max_features=max_features,
                                                         ngram_range=(1, 2)),
                             "learning_objective"))
    if lo == "id":
        transformers.append(("obj_id", OneHotEncoder(handle_unknown="ignore"),
                             ["learning_objective_id"]))
    transformers.append(("dlg", "passthrough", list(code_cols)))
    pre = ColumnTransformer(transformers, sparse_threshold=1.0)
    clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss", missing=np.nan,
                        n_jobs=n_jobs, random_state=seed, verbosity=0,
                        **dict(XGB_PARAMS, **overrides))
    return Pipeline([("pre", pre),
                     ("dense", FunctionTransformer(densify, accept_sparse=True)),
                     ("clf", clf)])


def ablate(X: pd.DataFrame, code_cols, seed: int) -> pd.DataFrame:
    """A degraded copy of X: rate columns attenuated by U(0.2, 1.0) per row.

    The objective columns are never perturbed — only the transcript-derived rates, which is the
    whole point of the augmentation.
    """
    rng = np.random.default_rng(seed)
    Z = X.copy()
    cols = [c for c in code_cols if c in Z.columns]
    if cols:
        Z[cols] = Z[cols].to_numpy() * rng.uniform(0.2, 1.0, size=(len(Z), len(cols)))
    return Z


def fit_predict(Xtr, ytr, Xte, code_cols, lo: str = "text", seeds=None, n_jobs: int = 8,
                aug_seed: int = 100, augment: bool = True, aug_cols=None, **overrides):
    """Fit a seed ensemble and average its probabilities. Returns (p, models).

    `aug_cols` are the columns the augmentation attenuates, by default all of `code_cols`. A
    passthrough column that is not a transcript rate (stage 06's `--attainment`) is left out of it.
    """
    seeds = list(seeds or SEEDS[:3])
    if augment:
        cols = list(code_cols) if aug_cols is None else list(aug_cols)
        Xa = ablate(Xtr, cols, seed=aug_seed) if len(cols) else Xtr.copy()
        Xtr = pd.concat([Xtr, Xa], ignore_index=True)
        ytr = np.concatenate([ytr, ytr])
    p = np.zeros(len(Xte))
    models = []
    for s in seeds:
        m = build_pipeline(code_cols, lo=lo, seed=s, n_jobs=n_jobs, **overrides).fit(Xtr, ytr)
        p += m.predict_proba(Xte)[:, 1]
        models.append(m)
    return p / len(seeds), models
