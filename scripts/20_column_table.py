#!/usr/bin/env python
"""Stage 20 — the model's column table, for the paper.

clean57 has one column per annotated code, so "what are the model's features" and "what is the
codebook" are the same question. The paper needs that answered in one place, and the thing a
reader actually wants to know about a column is **where the code came from**: a move taken
verbatim from the published NTO taxonomy carries different weight from one we added after
reading our own transcripts, and a reader cannot tell which is which from the name.

Three origins cover all 57 (`ttapub.taxonomy_v3.ORIGIN_LABEL`): a code is published, came from
asking tutors, or came from reading transcripts closely. Residual is marked separately, because
it says what a code *is* --- it absorbs turns rather than asserting a construct --- not where it
came from; the residuals were themselves introduced during human annotation, on finding that
without them turns with no pedagogical content were absorbed by substantive codes. Nothing is
attributed outside these three: see the provenance correction in `docs/taxonomy_v3.md` §4,
where four student codes had been credited to the NTO's published Sandpiper codebooks, which
carry no student scheme at all (`ttapub.taxonomy_v3.ORIGIN_LABEL`), all parsed
from `docs/taxonomy_v3.md` rather than restated here --- the codebook is the source of truth,
so a code that moves between sections moves in this table without anyone remembering to edit it.

Two further columns, because a bare list of names would invite exactly the misreadings the rest
of the paper spends its length correcting:

* **share of the role's turns** --- the corpus prevalence from the annotation's own prior
  artifact. A code at 0.3% of turns cannot support a session-level claim however interesting
  its definition.
* **cross-annotator kappa** --- mean over the three LLM pairings of the panel. The repo's rule
  is that every tutor-move claim carries its kappa; putting it in the column table is what makes
  that cheap for a reader to check.

    python scripts/20_column_table.py --run column-table
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

from ttapub import config, runlog
from ttapub.latex import caption_below
from ttapub import taxonomy_v3 as T

#: Order the blocks within a role. Residuals last: they are bookkeeping, not constructs.
#: Checked against ORIGIN_LABEL below rather than kept in step by hand -- when an origin was
#: once renamed this list was missed, pandas turned the four unlisted codes into NaN and they
#: vanished from the table without a word. 53 of 57 columns is not a thing to discover in a
#: proof.
ORIGIN_ORDER = ["NTO", "INT", "CLOSE"]

PANEL = config.RESULTS / "panel-v3" / "panel_by_code.csv"


def _nice(code: str) -> str:
    return code.lower().replace("_", " ")


def build() -> pd.DataFrame:
    prior = json.loads(config.prior_path("labels").read_text())["prior"]
    kap = (pd.read_csv(PANEL).groupby(["role", "code"]).kappa.mean()
           if PANEL.exists() else pd.Series(dtype=float))
    rows = []
    for role in ("tutor", "student"):
        for code, entry in T.pool(role).items():
            key = f"{role}:{code}"
            rows.append({
                "role": role, "code": code, "label": _nice(code),
                "origin": entry["origin"],
                "origin_label": T.ORIGIN_LABEL[entry["origin"]],
                "residual": T.is_residual(code, role),
                "share_of_turns": prior.get(key, prior.get(code)),
                "kappa": float(kap.get((role, code))) if (role, code) in kap.index else None,
            })
    d = pd.DataFrame(rows)
    missing = d[d.share_of_turns.isna()]
    if len(missing):
        raise SystemExit(f"no corpus prior for {list(missing.code)} — the prior artifact and "
                         f"the codebook disagree, which means the features were built from a "
                         f"different taxonomy than this table describes")
    unordered = sorted(set(d.origin) - set(ORIGIN_ORDER))
    if unordered or set(ORIGIN_ORDER) - set(T.ORIGIN_LABEL):
        raise SystemExit(f"ORIGIN_ORDER does not cover {unordered or ''} and/or names an "
                         f"origin ORIGIN_LABEL does not: an unordered origin becomes NaN and "
                         f"its codes drop out of the table silently")
    d["origin"] = pd.Categorical(d.origin, ORIGIN_ORDER, ordered=True)
    # residuals last within their origin block: they are listed for completeness, and leading
    # a block with `student no move` at 14.4% would put the largest non-construct at the top
    return d.sort_values(["role", "origin", "residual", "share_of_turns"],
                         ascending=[False, True, True, False]).reset_index(drop=True)


def _lo_columns() -> int:
    """How many TF-IDF columns the objective's text contributes, from the fitted model.

    Read off the fitted vectoriser rather than the `max_features` cap, which is 5,000 and is
    not reached: the training split's 1,015 objectives yield far fewer unigram+bigram terms.
    The vectoriser is refit inside each fold, so this is the full-training-split figure.
    """
    import joblib
    path = config.RESULTS / "arm-both" / "models.joblib"
    if not path.exists():
        raise SystemExit(f"{path} missing -- the LO column count is read from the fitted "
                         f"model, not assumed; run stage 06 first")
    d = joblib.load(path)
    for m in d["models"]:
        for _, step in getattr(m, "named_steps", {}).items():
            if "ColumnTransformer" in type(step).__name__:
                for _, t, _ in step.transformers_:
                    if "Tfidf" in type(t).__name__:
                        return len(t.vocabulary_)
    raise SystemExit("no fitted TfidfVectorizer found in the saved models")


def _block_lines(d: pd.DataFrame, role: str, origin: str) -> list[str]:
    """One origin block of one role as row *contents* (four cells, no row terminator).

    No `\\addlinespace` here: the two panels are zipped row by row, so a spacing command
    would have to appear in a cell, where TeX rejects it. Blocks are separated by an empty
    row instead, which pairs with the other panel like any other.
    """
    g = d[(d.role == role) & (d.origin == origin)]
    if g.empty:
        return []
    what = "Tutor moves" if role == "tutor" else "Student signals"
    out = [rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{{what} --- {T.ORIGIN_LABEL[origin]}}} "
           rf"({len(g)})}}"]
    for _, r in g.iterrows():
        k = "" if pd.isna(r.kappa) or r.residual else f"{r.kappa:.2f}"
        mark = r"$^\dagger$" if r.residual else ""
        out.append(rf"\quad {r.label}{mark} & {100 * r.share_of_turns:.1f}\% & {k}")
    return out


BLANK = "& &"


def _panel(d: pd.DataFrame, blocks: list[tuple[str, str]]) -> list[str]:
    """Several blocks stacked into one panel, separated by an empty row."""
    out: list[str] = []
    for role, origin in blocks:
        rows = _block_lines(d, role, origin)
        if not rows:
            continue
        if out:
            out.append(BLANK)
        out += rows
    return out


def to_latex(d: pd.DataFrame, n_lo: int) -> str:
    """Two panels side by side, with the objective's TF-IDF block as a full-width first row.

    A single 57-row list ran over two pages and left two thirds of the line width empty. The
    split is by block rather than by halving the rows, so no origin group is broken across the
    gutter: the 28 NTO codes fill the left panel, and the remaining 13 tutor codes and all 16
    student signals fill the right, which happens to balance to within a few lines.
    """
    left = _panel(d, [("tutor", "NTO")])
    right = _panel(d, [("tutor", "INT"), ("tutor", "CLOSE"),
                       ("student", "INT"), ("student", "CLOSE")])
    n = max(len(left), len(right))
    left += [BLANK] * (n - len(left))
    right += [BLANK] * (n - len(right))
    n_tutor = int((d.role == "tutor").sum())
    n_student = int((d.role == "student").sum())
    out = [r"% generated by scripts/20_column_table.py -- do not edit by hand",
           r"\begin{table*}[t]",
           r"\centering\footnotesize\setlength{\tabcolsep}{5pt}",
           rf"\caption{{\textbf{{The {n_tutor + n_student} code rates.}} \emph{{Turns}} is the "
           rf"code's share of its role's turns. $\kappa$ is Cohen's $\kappa$ averaged over three "
           rf"pairs of LLM annotators (Qwen3.5-122B-A10B, Qwen3-32B, Mistral-Small-24B) on the "
           rf"same 100 sessions (\cref{{sec:validation}}). $\dagger$ marks a residual code.}}",
           r"\label{tab:columns}",
           r"\begin{tabular}{@{}lrr@{\qquad}lrr@{}}", r"\toprule",
           r"code & turns & $\kappa$ & code & turns & $\kappa$ \\",
           r"\midrule"]
    out += [rf"{a} & {b} \\" for a, b in zip(left, right)]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return caption_below("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="column-table")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.prior_path("labels")])
    t = build()
    n = T.N_TUTOR + T.N_STUDENT
    if len(t) != n:
        raise SystemExit(f"built {len(t)} rows for a {n}-column model")
    t.to_csv(d / "columns.csv", index=False)
    tables = config.ROOT / "tables"
    tables.mkdir(exist_ok=True)
    n_lo = _lo_columns()
    (tables / "table_columns.tex").write_text(to_latex(t, n_lo), encoding="utf-8")
    t.to_csv(tables / "table_columns.csv", index=False)
    print(f"residuals: {int(t.residual.sum())} "
          f"({', '.join(t.code[t.residual].str.lower())})")
    counts = (t.groupby(["role", "origin"], observed=True).size()
              .unstack(fill_value=0).to_dict("index"))
    for role, c in counts.items():
        print(f"{role:8s} " + "  ".join(f"{k} {v}" for k, v in c.items() if v))
    print(f"\nwrote {tables / 'table_columns.tex'}")
    print(t.groupby("role", observed=True).kappa.describe()[["count", "mean", "min", "max"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    runlog.finish(d, {"run": a.run, "n_columns": len(t), "n_lo_columns": n_lo,
                      "n_total": n_lo + len(t), "counts": counts})


if __name__ == "__main__":
    main()
