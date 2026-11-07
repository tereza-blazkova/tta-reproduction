#!/usr/bin/env python
"""Stage 10 — the paper's model comparison table, every model on the identical test rows.

One test set (10,508 responses: 9,640 TSL + 868 Eedi), so the columns are comparable by
construction.

`LL*` is log loss after a rank-preserving intercept-only refit: a diagnostic that separates a
wrong overall level from a wrong ranking. It is fitted on the evaluation rows, so it is never a
prediction, and AUROC is untouched by it.

    python scripts/10_model_table.py
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import config, data, metrics, runlog
from ttapub.latex import caption_below

#: Logistic regression on the identical columns (`22_baseline.py`). The one comparison that
#: changes only the learner: same 57 rates, same objective TF-IDF, same folds.
LOGREG = "logistic regression (same columns)"
#: The topic-difficulty baselines: what the objective alone predicts, as the ladder's
#: objective-only XGBoost arm and as the objective's historical correct rate (stage 28).
OBJ_TEXT = "objective text only (XGBoost)"
OBJ_RATE = "objective's historical correct rate"
BASELINES = [OBJ_RATE, OBJ_TEXT]
#: A baseline using the dialogue-act scheme of Lin et al. (2022), not a replication
#: (`docs/lin_scheme.md`). Same learner, objective terms and folds as our XGBoost row.
LIN = "Lin-scheme XGBoost"


#: The model that reads the transcript itself. Quoted from its own run (LLMKT fine-tuned on
#: all 35,062 TSL training rows, checkpoints 10-14), reported 2026-10-27 as LL / AUROC per split
#: with Eedi LL* in brackets. Its per-row predictions are not committed here, so the row carries
#: a `\\todo{}` until they are and it can be recomputed with `from_predictions`.
LLMKT = "LLMKT"
QUOTED = {LLMKT: {"tsl": (0.6385, None, 58.76), "eedi": (0.8061, 0.7003, 54.66),
                  "all": (0.6523, None, 60.27)}}

META = {
    "code-rate XGBoost (one col per code, v3)": ("v3", "122B labels", "57", "recomputed"),
    LOGREG: ("v3", "122B labels", "57", "recomputed"),
    LLMKT: ("—", "—", "—", "quoted"),
    OBJ_TEXT: ("—", "—", "1,457", "recomputed"),
    LIN: ("Lin scheme", "122B labels", "47", "recomputed"),
    OBJ_RATE: ("—", "—", "1", "recomputed"),
}


def from_predictions(path, y_col="y", p_col="p") -> dict:
    df = pd.read_csv(path)
    if "data_source" not in df.columns:
        df = df.merge(data.load_test()[["response_id", "data_source"]], on="response_id")
    out = {}
    for src in ("tsl", "eedi", "all"):
        m = np.ones(len(df), bool) if src == "all" else (df.data_source == src).to_numpy()
        y, p = df[y_col].to_numpy(float)[m], df[p_col].to_numpy(float)[m]
        b = metrics.basic(y, p)
        out[src] = (b["ll"], metrics.intercept_refit_ll(y, p), b["auroc"], int(m.sum()))
    return out


#: Tabular foundation models on the identical columns (`22_baseline.py`, full TF-IDF). Their
#: per-row predictions are not committed, so the row is read from the run's `metrics.json`. A
#: run whose provenance is dirty gets a red marker in the table, which disappears once it is
#: rerun from a committed tree. TabPFN is the five-seed run, matching Fig. 2a and XGBoost.
FOUNDATION = {"TabPFN 3.5": "tabpfn35-tfidf-s5-both"}


def foundation_rows() -> list[dict]:
    rows = []
    for name, run in FOUNDATION.items():
        d = config.RESULTS / run
        m = json.loads((d / "metrics.json").read_text())
        assert m["objective"] == "tfidf" and m["n_cols"] == 57, f"{run} is not the full columns"
        dirty = json.loads((d / "provenance.json").read_text())["git"]["dirty"]
        rows.append({"model": name, "run": run, "dirty": dirty,
                     "tsl_ll": m["test_tsl"]["ll"], "tsl_auroc": m["test_tsl"]["auroc"],
                     "eedi_ll": m["test_eedi"]["ll"], "eedi_ll_refit": m["test_eedi"]["ll_icpt"],
                     "eedi_auroc": m["test_eedi"]["auroc"],
                     "all_ll": m["test_all"]["ll"], "all_auroc": m["test_all"]["auroc"]})
    return rows

OURS = "code-rate XGBoost (one col per code, v3)"

#: What each model is given. The column exists because the paper's claim is about what a model
#: reads, and a table of log losses without it invites the reader to compare architectures.
READS = {
    "code-rate XGBoost (one col per code, v3)": "57 code rates + obj.\\ text",
    LOGREG: "57 code rates + obj.\\ text",
    LLMKT: "transcript text",
    OBJ_TEXT: "obj.\\ text",
    LIN: "\\begin{tabular}[t]{@{}l@{}}47 dialogue-act\\\\ rates + obj.\\ text\\end{tabular}",
    OBJ_RATE: "objective id",
    "best constant (base rate)": "nothing",
}


def _total_columns() -> str:
    """The fitted model's column count, from the column-table run rather than restated.

    Our row and the foundation-model row must show the same number: both are given the same
    1,514 columns, and the comparison is the architecture, not the feature set.
    """
    m = config.RESULTS / "column-table" / "metrics.json"
    if not m.exists():
        raise SystemExit(f"{m} missing -- run scripts/20_column_table.py first; the column "
                         f"count is read from it, not assumed")
    return f"{json.loads(m.read_text())['n_total']:,}"


def to_latex(recs: list[dict]) -> str:
    """The paper's model table, with the unrun comparators as visible placeholders."""
    def f(v, nd=4):
        return "---" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"
    total = _total_columns()
    out = [r"% generated by scripts/10_model_table.py -- do not edit by hand",
           r"\begin{table*}[t]",
           r"\centering\small\setlength{\tabcolsep}{4pt}",
           r"\caption{\textbf{Predictive performance on the held-out test splits.} The best value "
           r"in each column is in bold. LL* refits the intercept on the evaluation split, so the gap "
           r"between LL and LL* reflects the base rate and not the ranking. The objective-rate "
           r"baseline uses the objective's training correct rate where it exists (9.3\% of TSL and "
           r"no Eedi test responses) and the base rate otherwise.}",
           r"\label{tab:models}",
           r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llr rr rrr rr@{}}", r"\toprule",
           r"& & & \multicolumn{2}{c}{TSL} & \multicolumn{3}{c}{Eedi} "
           r"& \multicolumn{2}{c}{TSL + Eedi} \\",
           r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
           r"model & reads & cols & LL & AUROC & LL & LL* & AUROC & LL & AUROC \\"]
    # the floor (the objective alone), then the field's baseline (LLMKT, which reads the
    # transcript), then the learners on the code-rate columns (trained, then in-context)
    by_name = {x["model"]: x for x in recs}

    # every measured row first, so each column's best value can be found before formatting;
    # two groups: every baseline (objective alone, LLMKT, the Lin scheme), then the code-rate models
    label = {OBJ_RATE: "Objective id correct rate", OBJ_TEXT: "Objective text (XGBoost)"}
    groups = [
        ("Baselines",
         [(label[name], READS[name], "1,457" if name == OBJ_TEXT else "1", by_name[name])
          for name in BASELINES]
         # short, so the note does not widen the table; the reason is at QUOTED above
         + [(LLMKT + r" \todo{quoted}", READS[LLMKT], "---", by_name[LLMKT]),
            (LIN, READS[LIN], f"{int(total.replace(',', '')) - 57 + 47:,}", by_name[LIN])]),
        ("Code-rate tabular models",
         [("Logistic regression", READS[LOGREG], total, by_name[LOGREG]),
          (r"\textbf{XGBoost}", READS[OURS], total, by_name[OURS])]
         + [(r["model"] + (r" \todo{rerun clean}" if r["dirty"] else ""),
             r"57 code rates + obj.\ text", total, r) for r in foundation_rows()]),
    ]
    rows = [row for _, g in groups for row in g]
    # (column, decimals, lower is better); the best shown value is bold, ties included
    # AUROC on its 0-1 scale, like log loss and like the figures (the records keep points)
    metrics_ = [("tsl_ll", 4, True), ("tsl_auroc", 4, False), ("eedi_ll", 4, True),
                ("eedi_ll_refit", 4, True), ("eedi_auroc", 4, False),
                ("all_ll", 4, True), ("all_auroc", 4, False)]

    def val(r, k):
        v = r[k]
        if v is None or np.isnan(v):
            return np.nan
        return v / 100 if k.endswith("auroc") else v

    best = {}
    for k, nd, low in metrics_:
        vals = [round(val(r, k), nd) for *_, r in rows if not np.isnan(val(r, k))]
        best[k] = min(vals) if low else max(vals)

    def cell(r, k, nd):
        x = val(r, k)
        v = f(x, nd)
        ok = not np.isnan(x) and round(x, nd) == best[k]
        return rf"\textbf{{{v}}}" if ok else v

    for head, g in groups:
        out += [r"\midrule", rf"\multicolumn{{10}}{{@{{}}l}}{{\emph{{{head}}}}} \\"]
        for name, reads, cols, r in g:
            mark = ""
            out.append(rf"{mark}\quad {name} & {reads} & {cols} & "
                       + " & ".join(cell(r, k, nd) for k, nd, _ in metrics_) + r" \\")
    out += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
    return caption_below("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="model-table")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a))

    rows: dict = {}
    rows["code-rate XGBoost (one col per code, v3)"] = from_predictions(config.RESULTS / "arm-both"
                                                              / "test.csv")
    lr = config.RESULTS / "baseline-logreg" / "test.csv"
    if not lr.exists():
        raise SystemExit(f"{lr} missing -- run `22_baseline.py --model logreg --test` first; "
                         f"the table's third row is read from it, not assumed")
    rows[LOGREG] = from_predictions(lr)
    rows[OBJ_TEXT] = from_predictions(config.RESULTS / "arm-lo" / "test.csv")
    rows[LIN] = from_predictions(config.RESULTS / "arm-lin-rates" / "test.csv")
    rows[OBJ_RATE] = from_predictions(config.RESULTS / "baseline-objective-rate" / "test.csv")
    rows.update(QUOTED)
    # the floor: the best single constant
    te = data.load_test()
    base = {}
    for src in ("tsl", "eedi", "all"):
        m = np.ones(len(te), bool) if src == "all" else (te.data_source == src).to_numpy()
        y = te.is_correct.to_numpy(float)[m]
        pi = y.mean()
        base[src] = (float(-(y * np.log(pi) + (1 - y) * np.log(1 - pi)).mean()), None, 50.0,
                     int(m.sum()))
    rows["best constant (base rate)"] = base
    META["best constant (base rate)"] = ("—", "—", "0", "recomputed")

    recs = []
    for name, v in rows.items():
        ann, inst, ncol, prov = META[name]
        r = {"model": name, "annotation": ann, "instrument": inst, "cols": ncol,
             "provenance": prov}
        for src in ("tsl", "eedi", "all"):
            ll, llr, auc, n = v[src] if len(v[src]) == 4 else (*v[src], None)
            r[f"{src}_ll"] = ll
            r[f"{src}_ll_refit"] = llr
            r[f"{src}_auroc"] = auc
        recs.append(r)
    t = pd.DataFrame(recs)
    t.to_csv(d / "model_table.csv", index=False)

    def fmt(v, nd=4):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"
    lines = ["| model | ann. | instrument | cols | TSL LL | TSL LL* | TSL AUROC | "
             "Eedi LL | Eedi LL* | Eedi AUROC | All LL | All AUROC | source |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in recs:
        lines.append("| {model} | {ann} | {inst} | {cols} | {a} | {b} | {c} | {e} | {f} | "
                     "{g} | {h} | {i} | {prov} |".format(
                         model=r["model"], ann=r["annotation"], inst=r["instrument"],
                         cols=r["cols"], prov=r["provenance"],
                         a=fmt(r["tsl_ll"]), b=fmt(r["tsl_ll_refit"]),
                         c=fmt(r["tsl_auroc"], 2), e=fmt(r["eedi_ll"]),
                         f=fmt(r["eedi_ll_refit"]), g=fmt(r["eedi_auroc"], 2),
                         h=fmt(r["all_ll"]), i=fmt(r["all_auroc"], 2)))
    md = "\n".join(lines)
    (d / "model_table.md").write_text(md + "\n")
    tables = config.ROOT / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "table_models.tex").write_text(to_latex(recs), encoding="utf-8")
    print(f"wrote {tables / 'table_models.tex'}")
    print(md)
    # keyed off READS rather than a literal name, which the clean57 -> code-rate rename broke
    ours = next(k for k in rows if k.startswith("code-rate"))
    print(f"\nn: TSL {rows[ours]['tsl'][3]:,}, Eedi {rows[ours]['eedi'][3]:,}, "
          f"all {rows[ours]['all'][3]:,}")
    runlog.finish(d, {"run": a.run, "rows": recs})


if __name__ == "__main__":
    main()
