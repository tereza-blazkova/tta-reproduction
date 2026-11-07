#!/usr/bin/env python
"""Stage 31 — should a practitioner track the student's explaining, or the tutor's prompting?

Three analyses of one question, all on the code-rate scale the paper uses (session shares of
turns, shrunk toward the training-corpus share with kappa 25):

1. **Head to head, within student.** The outcome on the student's rate of explaining why
   (EXPLAINS_WHY) and the tutor's rate of prompting -- the five prompting codes summed, and
   prompting self-explanation alone -- singly and together, with student and question fixed
   effects absorbed and SEs clustered on the student (`ttapub.within`, as stages 12 and 15).
2. **Prompts that land.** Every turn is one of four kinds by the tutor's code and the student's
   reply in the same turn (the uptake direction of stage 11): a prompt answered by an
   explanation (*landed*), a prompt not answered by one (*not landed*), an explanation with no
   prompt (*unprompted*), or neither. The first three enter one within-student model as
   shrunk shares of the session's turns. Reported per SD, and per percentage point of the
   session's turns, where `landed - not_landed` is tested directly by reparametrising
   (y = (b_L - b_N) L + b_N (L + N)).
3. **One number, held out.** A logistic regression on a single standardised session rate, fitted
   on all training responses and scored on the released test set, TSL and Eedi separately, with
   a session-clustered bootstrap of the paired difference (as stage 27).

Associations within a student, not effects: a tutor prompts in response to what the student is
doing, and a student explains more on a good day.

    python scripts/31_explain_vs_prompt.py --run explain-vs-prompt
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from ttapub import agg, config, data, runlog, within

KAPPA = agg.SHRINK_KAPPA
PROMPTS = ["PROMPTING_NEXT_STEP", "PROMPTING_SELF_EXPLANATION", "PROMPTING_SELF_CORRECTION",
           "PROMPTING_RELATED_CONCEPTS", "PROMPTING_ALTERNATIVE_REPRESENTATION"]
EXPLAIN = "EXPLAINS_WHY"
EXPLAIN_COL = agg.column(EXPLAIN, "student")
PROMPT_COLS = [agg.column(c, "tutor") for c in PROMPTS]
PSE_COL = agg.column("PROMPTING_SELF_EXPLANATION", "tutor")


def labels(split: str) -> pd.DataFrame:
    """One row per (session, turn): the tutor's code and the student's code in that turn."""
    out = {}
    for role in ("tutor", "student"):
        f = sorted(config.LABELS_V3.glob(f"v3_full_{split}_{role}_shard*.csv"))
        d = pd.concat([pd.read_csv(x, dtype={"session_id": str}) for x in f], ignore_index=True)
        out[role] = d.drop_duplicates(["session_id", "turn"]).rename(columns={"code": role})[
            ["session_id", "turn", role]]
    return out["tutor"].merge(out["student"], on=["session_id", "turn"], how="outer")


def turn_kinds(lab: pd.DataFrame, prompts: list[str]) -> pd.DataFrame:
    """Shrunk session shares of landed / not-landed prompts and unprompted explanations."""
    p = lab.tutor.isin(prompts)
    e = lab.student.eq(EXPLAIN)
    kinds = pd.DataFrame({"session_id": lab.session_id, "landed": p & e, "not_landed": p & ~e,
                          "unprompted": ~p & e})
    k = kinds.groupby("session_id").sum()
    n = kinds.groupby("session_id").size()
    prior = k.sum() / n.sum()
    return (k + KAPPA * prior).div(n + KAPPA, axis=0), prior


def fit(y, Md: pd.DataFrame, cols: list[str], cluster, n_abs) -> pd.DataFrame:
    t = within.ols_demeaned(y, Md[cols], cluster, n_abs).table()
    return t[["estimate", "lo", "hi", "p"]] * [100, 100, 100, 1]


def standardise(d: pd.DataFrame) -> pd.DataFrame:
    d = d.astype(float).fillna(d.mean())
    return (d - d.mean()) / d.std(ddof=0)


def bootstrap(y, p1, p0, sess, boot: int, seed: int) -> dict:
    """dLL = LL(p0) - LL(p1) and dAUROC = AUROC(p1) - AUROC(p0), session-clustered CIs."""
    ll = lambda p: -(y * np.log(p) + (1 - y) * np.log(1 - p))  # noqa: E731
    dl = ll(np.clip(p0, 1e-12, 1)) - ll(np.clip(p1, 1e-12, 1))
    uniq, inv = np.unique(sess, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    bl, ba = [], []
    for _ in range(boot):
        ix = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        bl.append(dl[ix].mean())
        ba.append(100 * (roc_auc_score(y[ix], p1[ix]) - roc_auc_score(y[ix], p0[ix])))
    return {"dLL": float(dl.mean()), "dLL_lo": float(np.percentile(bl, 2.5)),
            "dLL_hi": float(np.percentile(bl, 97.5)),
            "dAUROC": 100 * (roc_auc_score(y, p1) - roc_auc_score(y, p0)),
            "dAUROC_lo": float(np.percentile(ba, 2.5)), "dAUROC_hi": float(np.percentile(ba, 97.5))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="explain-vs-prompt")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.FEATURES_TRAIN, config.FEATURES_TEST])
    metrics: dict = {"run": a.run}

    meta = data.load_metadata()
    tr = (data.load_train().merge(meta[["response_id", "student_id", "question_id"]],
                                  on="response_id").reset_index(drop=True))
    feats = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str})
    feats["prompting_all"] = feats[PROMPT_COLS].sum(1)
    lab = labels("train")
    kinds_all, prior_all = turn_kinds(lab, PROMPTS)
    kinds_pse, prior_pse = turn_kinds(lab, ["PROMPTING_SELF_EXPLANATION"])
    kinds = kinds_all.add_prefix("all_").join(kinds_pse.add_prefix("pse_"))
    X = (tr[["session_id"]].merge(feats, on="session_id", how="left")
                           .merge(kinds, left_on="session_id", right_index=True, how="left"))
    y = tr.is_correct.to_numpy(float)
    cluster = tr.student_id.to_numpy()
    print(f"rows {len(tr):,} | students {tr.student_id.nunique():,} | "
          f"turn-kind shares, all prompts: {prior_all.round(4).to_dict()} | "
          f"self-explanation prompts: {prior_pse.round(4).to_dict()}")
    metrics["corpus_share_of_turns"] = {"all_prompts": prior_all.to_dict(),
                                        "self_explanation_prompts": prior_pse.to_dict()}
    kind_cols = list(kinds.columns)

    # ---- 1 and 2, per SD and per percentage point of turns, pooled and within student ------
    use = ["prompting_all", PSE_COL, EXPLAIN_COL] + kind_cols
    Zs = standardise(X[use])
    Zr = X[use].astype(float).fillna(X[use].mean()) * 100        # percentage points of turns
    Zr["all_sum"] = Zr.all_landed + Zr.all_not_landed
    Zr["pse_sum"] = Zr.pse_landed + Zr.pse_not_landed
    designs = {"within student": {"student": tr.student_id, "question": tr.question_id},
               "pooled": {"question": tr.question_id}}
    rows = []
    for design, absorb in designs.items():
        Ms, n_abs = within.absorb_matrix(pd.concat([pd.Series(y, name="_y"), Zs], axis=1), absorb)
        Mr, _ = within.absorb_matrix(pd.concat([pd.Series(y, name="_y"), Zr], axis=1), absorb)
        models = {
            "1a explaining alone": (Ms, [EXPLAIN_COL], "sd"),
            "1b prompting (5 codes) alone": (Ms, ["prompting_all"], "sd"),
            "1c prompting self-explanation alone": (Ms, [PSE_COL], "sd"),
            "1d explaining + prompting (5 codes)": (Ms, [EXPLAIN_COL, "prompting_all"], "sd"),
            "1e explaining + prompting self-explanation": (Ms, [EXPLAIN_COL, PSE_COL], "sd"),
            "2a turn kinds, any prompt": (Ms, ["all_landed", "all_not_landed", "all_unprompted"], "sd"),
            "2b turn kinds, self-explanation prompt": (Ms, ["pse_landed", "pse_not_landed",
                                                            "pse_unprompted"], "sd"),
            "2c turn kinds, any prompt (per pp)": (Mr, ["all_landed", "all_not_landed",
                                                        "all_unprompted"], "pp"),
            "2d turn kinds, self-explanation prompt (per pp)": (Mr, ["pse_landed", "pse_not_landed",
                                                                     "pse_unprompted"], "pp"),
            "2e landed minus not landed, any prompt (per pp)": (Mr, ["all_landed", "all_sum",
                                                                     "all_unprompted"], "pp"),
            "2f landed minus not landed, self-explanation (per pp)": (Mr, ["pse_landed", "pse_sum",
                                                                          "pse_unprompted"], "pp"),
        }
        for model, (M, cols, unit) in models.items():
            t = fit(M["_y"].to_numpy(), M, cols, cluster, n_abs)
            for c, r in t.iterrows():
                term = c
                if model.startswith(("2e", "2f")):
                    if not c.endswith("landed") or "not" in c:
                        continue
                    term = c.replace("landed", "landed_minus_not_landed")
                rows.append({"design": design, "model": model, "term": term, "unit": unit,
                             "estimate": r.estimate, "lo": r.lo, "hi": r.hi, "p": r.p})
    assoc = pd.DataFrame(rows)
    assoc.to_csv(d / "associations.csv", index=False)
    show = assoc[assoc.design == "within student"]
    print("\n=== within student: outcome pp per SD (or per pp of the session's turns) ===")
    print(show[["model", "term", "estimate", "lo", "hi", "p"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    metrics["associations"] = {f"{r.design}|{r.model}|{r.term}":
                               {"estimate": r.estimate, "lo": r.lo, "hi": r.hi, "p": r.p,
                                "unit": r.unit} for r in assoc.itertuples()}

    # ---- 3, one number held out --------------------------------------------------------------
    te = data.load_test().reset_index(drop=True)
    ft = pd.read_csv(config.FEATURES_TEST, dtype={"session_id": str})
    ft["prompting_all"] = ft[PROMPT_COLS].sum(1)
    Xt = te[["session_id"]].merge(ft, on="session_id", how="left")
    single = {"explaining": EXPLAIN_COL, "prompting (5 codes)": "prompting_all",
              "prompting self-explanation": PSE_COL}
    mu, sd = X[list(single.values())].mean(), X[list(single.values())].std(ddof=0)
    preds = {"base rate": np.full(len(te), y.mean())}
    coefs = {}
    for name, c in single.items():
        z = ((X[[c]].fillna(mu[c]) - mu[c]) / sd[c]).to_numpy()
        zt = ((Xt[[c]].fillna(mu[c]) - mu[c]) / sd[c]).to_numpy()
        m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        preds[name] = m.predict_proba(zt)[:, 1]
        coefs[name] = float(m.coef_[0, 0])
    yt = te.is_correct.to_numpy(float)
    held = []
    for src in ("tsl", "eedi"):
        s = (te.data_source == src).to_numpy()
        for name, p in preds.items():
            held.append({"view": src, "model": name, "n": int(s.sum()),
                         "LL": log_loss(yt[s], p[s], labels=[0, 1]),
                         "AUROC": 100 * roc_auc_score(yt[s], p[s]) if name != "base rate" else 50.0})
        for one, other in (("explaining", "prompting (5 codes)"),
                           ("explaining", "prompting self-explanation"),
                           ("explaining", "base rate"), ("prompting (5 codes)", "base rate"),
                           ("prompting self-explanation", "base rate")):
            b = bootstrap(yt[s], preds[one][s], preds[other][s], te.session_id.to_numpy()[s],
                          a.boot, a.seed)
            held.append({"view": src, "model": f"{one} vs {other}"} | b)
    held = pd.DataFrame(held)
    held.to_csv(d / "held_out.csv", index=False)
    print("\n=== held out: one standardised rate, logistic regression ===")
    print(held.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    metrics["held_out"] = held.to_dict(orient="records")
    metrics["held_out_coef_per_sd_logodds"] = coefs
    runlog.finish(d, metrics)


if __name__ == "__main__":
    main()
