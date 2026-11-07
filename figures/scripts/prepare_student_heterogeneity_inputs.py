#!/usr/bin/env python
"""Inputs for fig_student_heterogeneity: do students differ, and do different students get
different codes? (stage 35, `results/student-heterogeneity`).

Writes, all aggregates with no identifier of any student, session or response:

* `fig_studhet_a_spread.csv`: histogram of students' share of answers correct, observed and
  under sessions dealt to students at random (median and 95% band per bin);
* `fig_studhet_b_halves.csv`: students binned by one half of their sessions, against the same
  measure on the other half;
* `fig_studhet_c_jsd.csv` / `fig_studhet_c_null.csv`: divergence between the code distributions
  of the top and bottom thirds, observed and the permutation draws;
* `fig_studhet_d_codes.csv`: per code, how much more of it the top third's sessions carry, beside
  the code's pooled and within-student association with the outcome (`fig_rq1_a_scatter.csv`,
  which `prepare_rq1_figure_inputs.py` writes from `results/rq1-within-session`).

    python figures/scripts/prepare_student_heterogeneity_inputs.py [--run student-heterogeneity]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "figures" / "input"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="student-heterogeneity")
    a = ap.parse_args()
    R = ROOT / "results" / a.run
    m = json.loads((R / "metrics.json").read_text())
    if m["outcome"] != "correct":
        raise SystemExit(f"{a.run} ranks students by {m['outcome']}; the figure is drawn for "
                         f"the share of answers correct")

    pd.read_csv(R / "spread_hist.csv").to_csv(IN / "fig_studhet_a_spread.csv", index=False)
    pd.read_csv(R / "split_half_bins.csv").to_csv(IN / "fig_studhet_b_halves.csv", index=False)
    pd.read_csv(R / "jsd.csv").to_csv(IN / "fig_studhet_c_jsd.csv", index=False)
    pd.read_csv(R / "jsd_null.csv").to_csv(IN / "fig_studhet_c_null.csv", index=False)

    codes = pd.read_csv(R / "per_code.csv")
    codes["code"] = codes.code.str.lower()
    rq1 = pd.read_csv(IN / "fig_rq1_a_scatter.csv")[
        ["code", "role", "pp_per_sd_pooled", "pp_per_sd_within_student",
         "bh_sig_pooled", "bh_sig_within_student"]]
    d = codes.merge(rq1, on=["code", "role"], how="left", validate="1:1")
    if d.pp_per_sd_pooled.isna().any():
        raise SystemExit(f"codes without a pooled estimate: {d.code[d.pp_per_sd_pooled.isna()].tolist()}")
    d.to_csv(IN / "fig_studhet_d_codes.csv", index=False)

    link = {k: stats.spearmanr(d.log2_ratio, d[f"pp_per_sd_{k}"]) for k in ("pooled", "within_student")}
    A, sh = m["spread"], m["split_half"]
    pd.Series({"students": m["students_multi_session"], "mean_correct_pct": A["mean_pp"],
               "sd_observed_pp": A["sd_observed_pp"],
               "sd_null_pp": A["sd_null_pp"], "sd_stable_trait_pp": A["sd_stable_trait_pp"],
               "split_half_r": sh["r"], "spearman_brown": sh["spearman_brown"],
               "spearman_pooled": link["pooled"].statistic,
               "spearman_within_student": link["within_student"].statistic},
              name="value").rename_axis("key").to_csv(IN / "fig_studhet_summary.csv")
    print(f"{m['students_multi_session']:,} students: SD {A['sd_observed_pp']:.1f} pp vs "
          f"{A['sd_null_pp']:.1f} by chance, stable part {A['sd_stable_trait_pp']:.1f} pp; "
          f"split-half r {sh['r']:.3f}")
    print(f"BH-significant codes: {m['codes_bh_sig']}, within tutor {m['codes_bh_sig_within_tutor']}")
    for k, r in link.items():
        print(f"log2 ratio vs {k} association: Spearman {r.statistic:+.2f} (p {r.pvalue:.1e})")


if __name__ == "__main__":
    main()
