"""Distilling the LLM annotation into CPU children, one binary head per code.

**Why distil at all.** The 122B annotator cannot be run on every session of both splits, and a
model that needed it at inference could not be scored on the released test set in the same way
it was fitted. So the LLM labels ~3,000 sessions, and a small text classifier per code
reproduces those labels everywhere. The feature model then runs with no language model in the
loop, and a code's column means the same thing on train and on test.

**The recipe (frozen).**
TF-IDF over words (1-2 grams, min_df 4) union char_wb 4-grams (min_df 10), both sublinear, into
one logistic head per code, C=1.0. Folds are **session-disjoint**: a session the head was fitted
on would otherwise get an in-sample probability, and its session rate would be sharper than any
test session's. Labelled sessions therefore receive out-of-fold probabilities, and unlabelled
sessions (and the whole test split) are scored by heads fitted on all labelled sessions.

**Per-code gates (spec §6), applied identically to every code whatever its source.** Without a
gate a code can fail silently in distillation: its per-turn probability is near-constant, so
the session column carries no information about the concept while still receiving attribution
— which is the shrinkage formula leaking session length. A code that
fails a gate is *reported*, then merged into its parent or left out. It is never shipped as a
constant column, and its attribution is never quoted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_union

N_FOLDS = 5
FIT_CAP = 250_000          # turns per fold fit; LR on TF-IDF saturates far below this
PREDICT_CHUNK = 200_000

# ── gates (spec §6) ──
MIN_POSITIVES = 300        # positive LLM labels needed before a code is distilled at all
GATE_AUROC = 0.75          # held-out per-turn AUROC
GATE_SD = 0.01             # per-turn SD of the predicted probability
GATE_P50 = 1               # at least this many turns with P > 0.5
RARE_PREVALENCE = 0.05     # below this, the head is class-weighted


def vectorizer() -> object:
    """Words for phrasing ("does that make sense"), char n-grams for the surface the word
    tokeniser destroys (ASR spellings, numbers, elongations)."""
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=4, max_features=300_000,
                        sublinear_tf=True, dtype=np.float32),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=10,
                        max_features=150_000, sublinear_tf=True, dtype=np.float32),
    )


def head(prevalence: float) -> LogisticRegression:
    """One binary head. Codes under `RARE_PREVALENCE` get class weights, per spec §6.2."""
    return LogisticRegression(C=1.0, max_iter=400,
                              class_weight="balanced" if prevalence < RARE_PREVALENCE else None)


@dataclass
class CodeReport:
    """One code's distillation record — the row its gate decision is reported from."""
    code: str
    role: str
    n_pos: int
    prevalence: float
    auroc: float = float("nan")
    p_sd: float = float("nan")
    n_over_half: int = 0
    status: str = "pending"
    notes: list[str] = field(default_factory=list)

    def gate(self) -> str:
        if self.n_pos < MIN_POSITIVES:
            self.status = "too_rare"
            self.notes.append(f"{self.n_pos} positives < {MIN_POSITIVES}: needs the "
                              "enrichment pass before it can be distilled")
        elif not (self.auroc >= GATE_AUROC):
            self.status = "failed_auroc"
            self.notes.append(f"held-out AUROC {self.auroc:.3f} < {GATE_AUROC}")
        elif not (self.p_sd >= GATE_SD):
            self.status = "near_constant"
            self.notes.append(f"per-turn SD {self.p_sd:.4f} < {GATE_SD}: the column would "
                              "carry no information about the concept")
        elif self.n_over_half < GATE_P50:
            self.status = "never_fires"
            self.notes.append("no turn reaches P > 0.5")
        else:
            self.status = "pass"
        return self.status

    def as_row(self) -> dict:
        return {"code": self.code, "role": self.role, "n_pos": self.n_pos,
                "prevalence": round(self.prevalence, 6), "auroc": round(self.auroc, 4),
                "p_sd": round(self.p_sd, 5), "n_over_half": self.n_over_half,
                "status": self.status, "notes": "; ".join(self.notes)}


def fit_children(text: np.ndarray, Y: pd.DataFrame, groups: np.ndarray, role: str,
                 fit_cap: int = FIT_CAP, seed: int = 221, verbose: bool = True):
    """Session-disjoint OOF probabilities for every code, plus heads fitted on all rows.

    Returns `(oof, children, reports)`:
      * `oof` — DataFrame of out-of-fold per-turn probabilities, one column per code;
      * `children` — {"vectorizer": fitted, "heads": {code: head}, "role": role,
                      "prevalence": {...}} for scoring unlabelled turns and the test split;
      * `reports` — one :class:`CodeReport` per code, gates applied.
    """
    codes = list(Y.columns)
    prevalence = {c: float(Y[c].mean()) for c in codes}
    reports = {c: CodeReport(code=c, role=role, n_pos=int(Y[c].sum()),
                             prevalence=prevalence[c]) for c in codes}
    oof = pd.DataFrame(index=Y.index, columns=codes, dtype=float)
    rng = np.random.default_rng(seed)

    distil = [c for c in codes if reports[c].n_pos >= MIN_POSITIVES]
    skipped = [c for c in codes if c not in distil]
    if verbose and skipped:
        print(f"  [{role}] {len(skipped)} codes under {MIN_POSITIVES} positives, not distilled: "
              + ", ".join(skipped), flush=True)

    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(text, Y[codes[0]], groups)):
        fit_idx = rng.choice(tr, fit_cap, replace=False) if fit_cap and len(tr) > fit_cap else tr
        vec = vectorizer()
        Xtr = vec.fit_transform(text[fit_idx])
        Xte = vec.transform(text[te])
        if verbose:
            print(f"  [{role}] fold {k + 1}/{N_FOLDS}: {Xtr.shape[0]:,} x {Xtr.shape[1]:,}",
                  flush=True)
        for c in codes:
            if c not in distil:
                oof.loc[oof.index[te], c] = float(Y[c].to_numpy()[tr].mean())
                continue
            ytr = Y[c].to_numpy()[fit_idx]
            if ytr.sum() < 1:
                oof.loc[oof.index[te], c] = float(Y[c].to_numpy()[tr].mean())
                continue
            m = head(prevalence[c]).fit(Xtr, ytr)
            oof.loc[oof.index[te], c] = _predict(m, Xte)
        del vec, Xtr, Xte

    for c in codes:
        r = reports[c]
        p = oof[c].to_numpy(float)
        if c in distil and Y[c].nunique() > 1:
            r.auroc = float(roc_auc_score(Y[c], p))
        r.p_sd = float(np.nanstd(p))
        r.n_over_half = int((p > 0.5).sum())
        r.gate()

    # The serving heads are fitted on a capped sample, like the folds. Fitting them on all
    # 2.37M turns instead blew the memory on a laptop and the process was SIGKILLed with no
    # traceback (measured 2026-10-14) — a TF-IDF vocabulary over 2.4M documents with word 1-2
    # grams plus char 4-grams is the peak, not the heads. Capping also keeps the serving heads
    # comparable with the out-of-fold ones that produced the reported gates: fitting them on
    # 9x more data than their own evaluation would make the gate optimistic for the wrong
    # reason.
    fit_all = (rng.choice(len(text), fit_cap, replace=False)
               if fit_cap and len(text) > fit_cap else np.arange(len(text)))
    vec = vectorizer()
    Xall = vec.fit_transform(text[fit_all])
    heads = {}
    for c in distil:
        ya = Y[c].to_numpy()[fit_all]
        if ya.sum() < 1:          # the cap can drop every positive of a very rare code
            continue
        heads[c] = head(prevalence[c]).fit(Xall, ya)
    children = {"vectorizer": vec, "heads": heads, "role": role, "prevalence": prevalence,
                "codes": codes}
    return oof, children, reports


def _predict(model, X) -> np.ndarray:
    return np.concatenate([model.predict_proba(X[i:i + PREDICT_CHUNK])[:, 1]
                           for i in range(0, X.shape[0], PREDICT_CHUNK)])


def score_turns(children: dict, text: np.ndarray) -> pd.DataFrame:
    """Per-turn probabilities for unlabelled turns (and the whole test split).

    A code with no head (it failed the positives gate) gets its training prevalence, so the
    column exists, is constant, and its gate status says not to read it.
    """
    X = children["vectorizer"].transform(text)
    out = {}
    for c in children["codes"]:
        m = children["heads"].get(c)
        out[c] = _predict(m, X) if m is not None else np.full(X.shape[0],
                                                              children["prevalence"][c])
    return pd.DataFrame(out)
