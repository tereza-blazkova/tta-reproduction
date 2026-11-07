#!/usr/bin/env python
"""Stage 34 — which findings survive a different annotator.

The 122B's labels are one instrument. Two other LLMs coded every turn of the corpus with the
same codebook (Qwen3-32B with the identical prompt; Mistral-Small-4 with a coverage reminder),
and every stage that turns labels into a finding was re-run on each (`TTAPUB_ANNOTATOR`). This
stage lines the three up, code by code, beside how well each code's labels agree with the
122B's (per-code Cohen's kappa over all training turns, `panel-full`):

* **attribution** (stage 07, TreeSHAP of the 1,514-column XGBoost): each code's share of total
  attribution and its direction, per view;
* **within-student, one code at a time** (stage 15, design 2) and **all codes at once**
  (stage 32): the estimate in pp per SD, its 95% interval and its BH verdict.

Summaries per annotator against the 122B: rank correlation of attribution shares and of
within-student estimates, direction agreement (unweighted and weighted by the 122B's share), and
for the codes BH-significant under the 122B, how many keep their sign and their significance.
The pre-registered expectation (the run manifest (reproduce/runs.tsv) §1): performance similar, attribution diverging
most on the low-kappa tutor codes.

    python scripts/34_annotator_robustness.py --run annotator-robustness
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ttapub import agg, config, runlog

ANNOTATORS = {"122B": "", "Qwen3-32B": "-q32", "Mistral-Small-4": "-m4"}
SECOND = [a for a in ANNOTATORS if a != "122B"]


def _kappa() -> pd.DataFrame:
    k = pd.read_csv(config.RESULTS / "panel-full" / "panel_by_code.csv")
    k = k[k.a == "122B"]
    k["column"] = [agg.column(c, r) for c, r in zip(k.code, k.role)]
    return k.pivot_table(index="column", columns="b", values="kappa").add_prefix("kappa_")


def _shap(view: str) -> pd.DataFrame:
    out = []
    for a, sfx in ANNOTATORS.items():
        s = pd.read_csv(config.RESULTS / f"arm-both{sfx}" / "shap_by_column.csv")
        s = s[(s.view == view) & (s.column != "objective")].set_index("column")
        out.append(s[["share", "direction"]].add_suffix(f"_{a}"))
    return pd.concat(out, axis=1)


def _within(kind: str) -> pd.DataFrame:
    out = []
    for a, sfx in ANNOTATORS.items():
        if kind == "one":
            w = pd.read_csv(config.RESULTS / f"rq1-within{sfx or '-session'}" / "rq1_designs.csv")
            w = w[w.design == "2 within student"].rename(columns={"lo": "lo_sd", "hi": "hi_sd"})
        else:
            w = pd.read_csv(config.RESULTS / f"joint-within{sfx}" / "joint_within.csv")
            w = w[w.design == "within student"]
        w = w.set_index("column")[["pp_per_sd", "lo_sd", "hi_sd", "bh_sig"]]
        out.append(w.add_suffix(f"_{a}"))
    return pd.concat(out, axis=1)


def _summ_shap(t: pd.DataFrame, a: str) -> dict:
    ok = t[[f"share_122B", f"share_{a}"]].notna().all(axis=1)
    t = t[ok]
    same = t[f"direction_122B"] == t[f"direction_{a}"]
    w = t["share_122B"]
    top = t.sort_values("share_122B", ascending=False).head(12)
    return {"rho_share": round(float(spearmanr(t.share_122B, t[f"share_{a}"])[0]), 3),
            "direction_agree": round(float(same.mean()), 3),
            "direction_agree_weighted": round(float((same * w).sum() / w.sum()), 3),
            "top12_overlap": int(len(set(top.index) &
                                     set(t.sort_values(f"share_{a}", ascending=False)
                                         .head(12).index))),
            "top12_same_direction": int((top.direction_122B == top[f"direction_{a}"]).sum())}


def _summ_within(t: pd.DataFrame, a: str) -> dict:
    ok = t[[f"pp_per_sd_122B", f"pp_per_sd_{a}"]].notna().all(axis=1)
    t = t[ok]
    sig = t[t["bh_sig_122B"].astype(bool)]
    keep_sign = np.sign(sig.pp_per_sd_122B) == np.sign(sig[f"pp_per_sd_{a}"])
    return {"rho_estimate": round(float(spearmanr(t.pp_per_sd_122B, t[f"pp_per_sd_{a}"])[0]), 3),
            "sign_agree": round(float((np.sign(t.pp_per_sd_122B) ==
                                       np.sign(t[f"pp_per_sd_{a}"])).mean()), 3),
            "bh_sig_122B": int(len(sig)),
            "of_which_same_sign": int(keep_sign.sum()),
            "of_which_still_bh_sig_same_sign": int((keep_sign &
                                                    sig[f"bh_sig_{a}"].astype(bool)).sum()),
            "bh_sig_here": int(t[f"bh_sig_{a}"].astype(bool).sum()),
            "codes": int(len(t))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="annotator-robustness")
    a = ap.parse_args()
    inputs = [config.RESULTS / "panel-full" / "panel_by_code.csv"]
    for sfx in ANNOTATORS.values():
        inputs += [config.RESULTS / f"arm-both{sfx}" / "shap_by_column.csv"]
    d = runlog.start(a.run, vars(a), inputs)

    kappa = _kappa()
    summary, tables = [], {}
    for view in ("train", "test_tsl", "test_eedi"):
        t = _shap(view).join(kappa)
        tables[f"shap_{view}"] = t
        for b in SECOND:
            summary.append({"what": f"attribution ({view})", "annotator": b} | _summ_shap(t, b))
    for kind, label in (("one", "within student, one code"), ("all", "within student, all codes")):
        t = _within(kind).join(kappa)
        tables[f"within_{kind}"] = t
        for b in SECOND:
            summary.append({"what": label, "annotator": b} | _summ_within(t, b))

    # does disagreement follow the measurement? rank correlation of kappa with the share gap
    gaps = []
    t = tables["shap_train"]
    for b in SECOND:
        gap = (t[f"share_{b}"] - t.share_122B).abs() / t[["share_122B", f"share_{b}"]].mean(axis=1)
        flip = (t.direction_122B != t[f"direction_{b}"]).astype(float)
        k = t[f"kappa_{b}"]
        ok = gap.notna() & k.notna()
        gaps.append({"annotator": b,
                     "rho_kappa_vs_relative_share_gap": round(float(spearmanr(k[ok], gap[ok])[0]), 3),
                     "mean_kappa_direction_flipped": round(float(k[ok & (flip == 1)].mean()), 3),
                     "mean_kappa_direction_kept": round(float(k[ok & (flip == 0)].mean()), 3),
                     "n_flipped": int((ok & (flip == 1)).sum())})

    summ = pd.DataFrame(summary)
    summ.to_csv(d / "summary.csv", index=False)
    pd.DataFrame(gaps).to_csv(d / "kappa_vs_gap.csv", index=False)
    for name, t in tables.items():
        t.reset_index().to_csv(d / f"by_code_{name}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print()
    print(pd.DataFrame(gaps).to_string(index=False))
    runlog.finish(d, {"run": a.run, "summary": summ.to_dict("records"), "kappa_vs_gap": gaps})


if __name__ == "__main__":
    main()
