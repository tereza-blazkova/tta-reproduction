#!/usr/bin/env python
"""Stage 39 — what the transcript adds once the model already knows how good the student is.

The code-rate model can predict well by recognising a strong student, which says nothing about
the tutoring. So the model is refitted with one more column, the student's attainment measured on
their other training sessions (stage 06 `--attainment`, :mod:`ttapub.attainment`), and evaluated on the
released-test students whose attainment is measured well: TSL students with at least X training
sessions (`--min-sessions`, default 1, 3, 4, 5, 6, 7). A model that is told who the student is can only gain from the
transcript by reading something about the session.

Arms, all from stage 06 on the released test set, fitted on every training row, five seeds:
    without attainment   arm-{lo,tutor,student,both}-pred
    with attainment      arm-{lo,tutor,student,both}-attainment

Per population, paired on the same rows, with a 2,000-draw bootstrap over students (a student
contributes several sessions, so students are resampled whole):
  * the transcript's gain, both blocks against the objective only, without and with attainment;
  * each block's own gain, both against the other block alone, without and with attainment;
  * attainment's gain, with against without, for the objective-only and the both-blocks arms;
  * attribution: TreeSHAP on these rows for the both-blocks model without and with attainment. Code
    shares are of the 57 code columns' total, so the rankings compare which codes carry the
    transcript's signal; the objective's and attainment's shares of the whole are reported beside.

Eedi has no student with training sessions, so attainment is missing for every Eedi row; it is
scored only to show the attainment arms change nothing there.

    python scripts/39_attainment_adjusted.py --run attainment-adjusted
"""
from __future__ import annotations

import argparse
import importlib

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ttapub import attainment, agg, config, data, metrics, runlog
from ttapub import taxonomy_v3 as T

ARMS = ("lo", "tutor", "student", "both")


def run_dir(arm: str, with_attainment: bool):
    return config.RESULTS / f"arm-{arm}-{'attainment' if with_attainment else 'pred'}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="attainment-adjusted")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min-sessions", default="1,3,4,5,6,7")
    a = ap.parse_args()
    ladder = [int(x) for x in a.min_sessions.split(",")]
    inputs = [run_dir(k, w) / "test.csv" for k in ARMS for w in (False, True)]
    d = runlog.start(a.run, vars(a), [config.TRAIN_LABELS, config.TEST_LABELS, config.METADATA,
                                      config.FEATURES_TEST, *inputs])

    meta = data.load_metadata()
    tr = data.load_train().merge(meta[["response_id", "student_id", "question_id"]], on="response_id")
    # gate: the attainment module's question rate is stage 35's
    s35 = importlib.import_module("35_student_heterogeneity")
    gap = np.abs(attainment.expected_correct(tr) - s35.expected_correct(tr)).max()
    if gap > 0:
        raise SystemExit(f"attainment.expected_correct differs from stage 35's by {gap}")
    n_sess = tr.groupby("student_id").session_id.nunique()

    te = data.load_test().reset_index(drop=True)
    te["train_sessions"] = te.student_id.map(n_sess).fillna(0).astype(int)
    y = te.is_correct.to_numpy(float)
    P = {}
    for k in ARMS:
        for w in (False, True):
            p = pd.read_csv(run_dir(k, w) / "test.csv", dtype={"response_id": str}).set_index("response_id")
            if not np.array_equal(p.y.reindex(te.response_id).to_numpy(), y):
                raise SystemExit(f"{run_dir(k, w)}: labels do not match the test split")
            P[(k, w)] = p.p.reindex(te.response_id).to_numpy()

    pops = {f"TSL, >= {x} training sessions": (te.data_source == "tsl") & (te.train_sessions >= x)
            for x in ladder}
    pops["TSL, all"] = te.data_source == "tsl"
    pops["Eedi"] = te.data_source == "eedi"
    pairs = {"transcript (both vs objective)": (("both", None), ("lo", None)),
             "student signals (both vs tutor)": (("both", None), ("tutor", None)),
             "tutor moves (both vs student)": (("both", None), ("student", None))}

    rows, drows = [], []
    for pop, m in pops.items():
        m = m.to_numpy()
        g = te.student_id.to_numpy()[m]
        print(f"{pop}: {m.sum():,} responses, {len(set(g)):,} students", flush=True)
        for (k, w), p in P.items():
            rows.append({"population": pop, "arm": k, "attainment": w, "n": int(m.sum()),
                         "students": len(set(g)), **metrics.basic(y[m], p[m]),
                         **metrics.boot_ci(y[m], p[m], g, B=a.boot)})
        for label, ((k1, _), (k0, _)) in pairs.items():
            for w in (False, True):
                r = metrics.boot_delta(y[m], P[(k1, w)][m], P[(k0, w)][m], g, B=a.boot)
                drows.append({"population": pop, "contrast": label, "attainment": w, **r})
                print(f"  {label:34s} {'with' if w else 'without':7s} attainment: dLL {-r['d_ll']:+.4f} "
                      f"[{-r['d_ll_ci'][1]:+.4f}, {-r['d_ll_ci'][0]:+.4f}]  dAUROC {r['d_auroc']:+.2f} "
                      f"[{r['d_auroc_ci'][0]:+.2f}, {r['d_auroc_ci'][1]:+.2f}]", flush=True)
        for k in ("lo", "both"):
            r = metrics.boot_delta(y[m], P[(k, True)][m], P[(k, False)][m], g, B=a.boot)
            drows.append({"population": pop, "contrast": f"attainment ({k})", "attainment": True, **r})
            print(f"  attainment on top of {k:4s}                     : dLL {-r['d_ll']:+.4f} "
                  f"[{-r['d_ll_ci'][1]:+.4f}, {-r['d_ll_ci'][0]:+.4f}]  dAUROC {r['d_auroc']:+.2f} "
                  f"[{r['d_auroc_ci'][0]:+.2f}, {r['d_auroc_ci'][1]:+.2f}]", flush=True)
    perf, deltas = pd.DataFrame(rows), pd.DataFrame(drows)
    # report a reduction as positive: LL(without the block) - LL(with it)
    deltas["ll_reduction"] = -deltas.d_ll
    deltas["ll_reduction_lo"] = -deltas.d_ll_ci.str[1]
    deltas["ll_reduction_hi"] = -deltas.d_ll_ci.str[0]
    deltas["auroc_lo"], deltas["auroc_hi"] = deltas.d_auroc_ci.str[0], deltas.d_auroc_ci.str[1]
    perf.drop(columns=["ll_ci", "auroc_ci"]).assign(
        ll_lo=perf.ll_ci.str[0], ll_hi=perf.ll_ci.str[1],
        auroc_lo=perf.auroc_ci.str[0], auroc_hi=perf.auroc_ci.str[1]).to_csv(d / "performance.csv", index=False)
    deltas.drop(columns=["d_ll_ci", "d_auroc_ci"]).to_csv(d / "deltas.csv", index=False)

    # attribution: which codes carry the transcript's signal, without and with attainment
    s07 = importlib.import_module("07_shap_clean57")
    s06 = importlib.import_module("06_train_clean57")
    ft = pd.read_csv(config.FEATURES_TEST, dtype={"session_id": str})
    _, ab_test = attainment.train_and_test(tr.reset_index(drop=True), te.student_id)
    srows = []
    for w in (False, True):
        bundle = joblib.load(run_dir("both", w) / "models.joblib")
        num = bundle["num"]
        codes = [c for c in num if c != "student_attainment"]
        Xt = s06.design(te, ft, codes)
        if w:
            Xt["student_attainment"] = ab_test
        for pop in (f"TSL, >= {x} training sessions" for x in ladder):
            m = pops[pop].to_numpy()
            S, obj = s07.shap_block(bundle["models"], Xt[m].reset_index(drop=True), num)
            absS = np.abs(S).mean(0)
            total = absS.sum() + np.abs(obj).mean()
            code_tot = absS[[num.index(c) for c in codes]].sum()
            for j, c in enumerate(num):
                srows.append({"population": pop, "attainment": w, "column": c,
                              "share_of_codes": absS[j] / code_tot if c in codes else np.nan,
                              "share_of_all": absS[j] / total})
            srows.append({"population": pop, "attainment": w, "column": "objective",
                          "share_of_codes": np.nan, "share_of_all": np.abs(obj).mean() / total})
    shap = pd.DataFrame(srows)
    name = {agg.column(k, r): (k.lower(), r) for r in ("tutor", "student") for k in T.codes(r)}
    shap["code"] = shap.column.map(lambda c: name[c][0] if c in name else c)
    shap["role"] = shap.column.map(lambda c: name[c][1] if c in name else "")
    shap.to_csv(d / "shap.csv", index=False)

    rank = {}
    for pop in (f"TSL, >= {x} training sessions" for x in ladder):
        s = shap[(shap.population == pop) & shap.share_of_codes.notna()].pivot(
            index="column", columns="attainment", values="share_of_codes")
        rank[pop] = float(spearmanr(s[False], s[True]).statistic)
        top = s.assign(change=s[True] - s[False]).sort_values("change")
        print(f"{pop}: code-share rankings without vs with attainment, Spearman {rank[pop]:.2f}; "
              f"largest falls {', '.join(name[c][0] for c in top.index[:4])}; largest rises "
              f"{', '.join(name[c][0] for c in top.index[-4:])}", flush=True)
    key = deltas.set_index(["population", "contrast", "attainment"])
    runlog.finish(d, {
        "run": a.run, "boot": a.boot, "populations": {p: int(m.sum()) for p, m in pops.items()},
        "students": {p: int(te.student_id[m].nunique()) for p, m in pops.items()},
        "transcript_gain": {f"{p} | {'with' if w else 'without'} attainment":
                            {"ll_reduction": float(key.loc[(p, "transcript (both vs objective)", w)].ll_reduction),
                             "d_auroc": float(key.loc[(p, "transcript (both vs objective)", w)].d_auroc)}
                            for p in pops for w in (False, True)},
        "spearman_code_shares_without_vs_with_attainment": rank,
    })


if __name__ == "__main__":
    main()
