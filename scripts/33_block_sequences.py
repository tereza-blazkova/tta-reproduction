#!/usr/bin/env python
"""Stage 33 — sequences at block level: pre-specified, pooled for reliability, hard and soft.

Stage 12 tested 421 code-to-code pairs and none survived within student; stage 16 then showed
that a pair rate's within-student reliability is a median 0.13, so that null is mostly silence.
This stage asks the same question with the three levers that can actually buy power, all fixed
here before the first run:

1. **Coarser constructs.** Tutor moves pooled into four blocks and student codes into four
   states, so a session's pair rate rests on many transitions instead of a handful.
2. **Fewer tests, fixed in advance.** Every block x state combination in both directions,
   4 x 4 x 2 = 32 tests, Benjamini-Hochberg across the 32 --- no selection among them.
3. **Soft labels** as a sensitivity arm: the distilled children's per-turn probabilities
   (all 57 codes pass the stage 04 gates) in place of the 122B's hard labels. The children were
   trained on those labels, so their errors are not independent of them; the arm tests whether
   a smoother measure changes the answer, not whether a second annotator agrees.

The tutor blocks follow the three tutoring states of Vanacore et al. (2026) --- inquiry
elicitation, direct instruction, affective support --- with evaluative feedback split out as a
fourth, because confirming an answer is neither eliciting nor telling. Logistical codes and the
residuals belong to none. The student states are stage 17's composites, unchanged.

Model, per pair, exactly as stage 12:

    outcome ~ pair rate + block rate + state rate + student FE + question FE

standardised, SEs clustered on the student, so the estimate is the pairing *beyond* its two
components (percentage points per SD). The pooled design (question FE only) is reported beside
it. Each pair's within-student split-half reliability (stage 16's block10 scheme, unshrunk)
gives a disattenuated bound, max(|lo|, |hi|) / sqrt(rel) (stage 21's factor).

Primary: hard labels. Sensitivity: soft labels, corrected as their own family of 32.

⚠️ Associations within a student, not effects: the tutor's next move answers what the student
just did, and absorbing the student does not randomise that.

    python scripts/33_block_sequences.py --run block-sequences
"""
from __future__ import annotations

import argparse
import importlib

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, data, runlog, within

KAPPA = agg.SHRINK_KAPPA
MIN_PER_HALF = 5

#: Tutor blocks, after Vanacore et al. (2026), with feedback as its own block (see docstring).
BLOCKS = {
    "eliciting": ["PROMPTING_RELATED_CONCEPTS", "PROMPTING_ALTERNATIVE_REPRESENTATION",
                  "PROMPTING_SELF_EXPLANATION", "PROMPTING_NEXT_STEP",
                  "PROMPTING_SELF_CORRECTION", "PROBING_PRIOR_KNOWLEDGE", "PROBING_UNDERSTAND",
                  "TRANSFER_PROBE", "RELEASE_HANDOFF"],
    "telling": ["GIVING_HINT", "GIVING_EXAMPLE", "EXPLAINING_CONCEPTUAL",
                "EXPLAINING_PROCEDURAL", "GIVING_ANSWER", "WORKED_DEMONSTRATION",
                "SIMPLIFYING_TO_SUBPROBLEM", "SUMMARIZING_PROGRESS", "STRATEGIZING"],
    "evaluating": ["FEEDBACK_CORRECT", "FEEDBACK_INCORRECT", "FEEDBACK_NEUTRAL", "REVOICING",
                   "RESTATING"],
    "affective": ["BUILDING_RAPPORT", "ENCOURAGING", "ASKING_FEELING", "VALIDATING_FEELING",
                  "PRAISING_PROCESS", "PRAISING_OUTCOME", "PRAISING_TRAITS",
                  "NORMALIZING_DIFFICULTY", "MOTIVATING_RELEVANCE"],
}
#: Student states: stage 17's composites.
STATES = {
    "reasoning_shown": ["EXPLAINS_WHY", "SHOWS_METHOD"],
    "not_knowing": ["EXPRESSES_NOT_KNOWING", "FRUSTRATION"],
    "answering_confidently": ["CONFIDENT_ANSWER"],
    "hedging": ["HEDGED_ANSWER", "ASKS_VERIFICATION"],
}


def turn_table(tag: str) -> pd.DataFrame:
    """One row per turn: hard and soft membership of every block and state."""
    tp = pd.read_parquet(config.TURN_PROBS_TRAIN)
    tp["session_id"] = tp.session_id.astype(str)
    lab = importlib.import_module("11_sequences").load_labels(tag, "train")
    df = tp.merge(lab, on=["session_id", "turn"], how="left").sort_values(["session_id", "pos"])
    for name, codes in BLOCKS.items():
        df[f"hard_t_{name}"] = df.tutor.isin(codes).astype(float).where(df.tutor.notna())
        # single-label coding, so a turn's block probability is the sum over its codes
        soft = df[[f"t_{c}" for c in codes]].sum(axis=1).clip(0, 1)
        df[f"soft_t_{name}"] = soft.where(df.has_teacher)
    for name, codes in STATES.items():
        df[f"hard_s_{name}"] = df.student.isin(codes).astype(float).where(df.student.notna())
        soft = df[[f"s_{c}" for c in codes]].sum(axis=1).clip(0, 1)
        df[f"soft_s_{name}"] = soft.where(df.has_student)
    keep = ["session_id", "turn"] + [c for c in df.columns if c.startswith(("hard_", "soft_"))]
    return df[keep].reset_index(drop=True)


def transitions(df: pd.DataFrame, src: str, kind: str) -> pd.DataFrame:
    """One row per eligible transition, with the value of every pair on it (0-1)."""
    if kind == "uptake":                       # tutor move -> the student's reply, same turn
        t = df[[f"{src}_t_{b}" for b in BLOCKS]].to_numpy()
        s = df[[f"{src}_s_{st}" for st in STATES]].to_numpy()
    else:                                      # student signal -> the tutor's next move
        nxt = df.groupby("session_id", sort=False)[[f"{src}_t_{b}" for b in BLOCKS]].shift(-1)
        adjacent = (df.groupby("session_id", sort=False).turn.shift(-1) == df.turn + 1)
        t = nxt.where(adjacent, np.nan).to_numpy()
        s = df[[f"{src}_s_{st}" for st in STATES]].to_numpy()
    ok = ~np.isnan(t).any(1) & ~np.isnan(s).any(1)
    vals = {}
    for i, b in enumerate(BLOCKS):
        for j, st in enumerate(STATES):
            name = f"{b} > {st}" if kind == "uptake" else f"{st} > {b}"
            vals[name] = t[ok, i] * s[ok, j]
    out = pd.DataFrame(vals)
    out.insert(0, "session_id", df.session_id.to_numpy()[ok])
    out["r"] = out.groupby("session_id", sort=False).cumcount()
    return out


def shrunk(values: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Session rates of `cols`, shrunk toward their corpus mean as every other rate is."""
    g = values.groupby("session_id", sort=False)[cols]
    tot, n = g.sum(), g.size()
    prior = values[cols].mean().to_numpy()
    r = (tot.to_numpy() + KAPPA * prior) / (n.to_numpy()[:, None] + KAPPA)
    return pd.DataFrame(r, index=tot.index, columns=cols)


def reliability(values: pd.DataFrame, cols: list[str], student_of: dict) -> pd.Series:
    """Within-student split-half reliability, block10 halves, unshrunk (stage 16's design)."""
    v = values.assign(half=(values.r // 10) % 2)
    n = v.groupby(["session_id", "half"]).size().unstack("half")
    ok = n.min(axis=1) >= MIN_PER_HALF
    m = v.groupby(["session_id", "half"])[cols].mean().unstack("half")
    m = m[ok.reindex(m.index, fill_value=False)]
    g = pd.Series([student_of.get(s) for s in m.index], index=m.index)
    rep = g.map(g.value_counts()) >= 2
    m, g = m[rep], g[rep]
    out = {}
    for c in cols:
        a = m[(c, 0)] - m[(c, 0)].groupby(g).transform("mean")
        b = m[(c, 1)] - m[(c, 1)].groupby(g).transform("mean")
        both = a.notna() & b.notna()
        a, b = a[both], b[both]
        r = np.corrcoef(a, b)[0, 1] if a.std() > 1e-12 and b.std() > 1e-12 else np.nan
        out[c] = 2 * r / (1 + r) if r > -1 else np.nan
    return pd.Series(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--run", default="block-sequences")
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a), [config.TURN_PROBS_TRAIN])

    meta = data.load_metadata()
    tr = (data.load_train()
          .merge(meta[["response_id", "student_id", "question_id"]], on="response_id")
          .reset_index(drop=True))
    tr["session_id"] = tr.session_id.astype(str)
    student_of = dict(tr[["session_id", "student_id"]].drop_duplicates().to_numpy())
    df = turn_table(a.tag)
    print(f"{len(df):,} turns over {df.session_id.nunique():,} sessions", flush=True)

    rows = []
    for src in ("hard", "soft"):
        # session rates: the 32 pairs and their 8 components
        feats, rel = [], []
        for kind in ("uptake", "response"):
            tv = transitions(df, src, kind)
            pairs = [c for c in tv.columns if " > " in c]
            feats.append(shrunk(tv, pairs).add_prefix(f"{kind}|"))
            rel.append(reliability(tv, pairs, student_of).add_prefix(f"{kind}|"))
            print(f"  [{src}] {kind}: {len(tv):,} transitions", flush=True)
        for role, groups in (("t", BLOCKS), ("s", STATES)):
            cols = [f"{src}_{role}_{k}" for k in groups]
            sub = df[["session_id"] + cols].dropna()
            feats.append(shrunk(sub, cols))
        F = pd.concat(feats, axis=1)
        rel = pd.concat(rel)
        X = tr[["session_id"]].merge(F, left_on="session_id", right_index=True, how="left")
        use = X.drop(columns="session_id").astype(float)
        use = use.fillna(use.mean())
        use = ((use - use.mean()) / use.std(ddof=0).replace(0, np.nan)).fillna(0.0)
        M = pd.concat([pd.Series(tr.is_correct.to_numpy(float), name="_y"), use], axis=1)
        Md, n_abs = within.absorb_matrix(M, {"student": tr.student_id,
                                             "question": tr.question_id})
        Mp, n_abs_p = within.absorb_matrix(M, {"question": tr.question_id})
        cluster = tr.student_id.to_numpy()
        for col in [c for c in F.columns if "|" in c]:
            kind, pair = col.split("|")
            left, right = pair.split(" > ")
            blk, st = (left, right) if kind == "uptake" else (right, left)
            comps = [f"{src}_t_{blk}", f"{src}_s_{st}"]
            r = within.ols_demeaned(Md["_y"].to_numpy(), Md[[col] + comps], cluster, n_abs)
            rp = within.ols_demeaned(Mp["_y"].to_numpy(), Mp[[col] + comps], cluster, n_abs_p)
            t, tp = r.table().loc[col], rp.table().loc[col]
            rows.append({"labels": src, "kind": kind, "pair": pair, "block": blk, "state": st,
                         "pp_per_sd": 100 * t.estimate, "lo": 100 * t.lo, "hi": 100 * t.hi,
                         "p": t.p, "pooled_pp_per_sd": 100 * tp.estimate,
                         "pooled_lo": 100 * tp.lo, "pooled_hi": 100 * tp.hi,
                         "pooled_p": tp.p, "rel_within": rel[col]})
        print(f"  [{src}] fitted 32 pairs", flush=True)

    out = pd.DataFrame(rows)
    for src in ("hard", "soft"):
        m = out.labels == src
        out.loc[m, "bh_sig"] = within.benjamini_hochberg(out.loc[m, "p"].to_numpy(), a.alpha)
        out.loc[m, "bh_sig_pooled"] = within.benjamini_hochberg(
            out.loc[m, "pooled_p"].to_numpy(), a.alpha)
    out["bh_sig"] = out.bh_sig.astype(bool)
    out["bh_sig_pooled"] = out.bh_sig_pooled.astype(bool)
    out["bound"] = (np.maximum(out.lo.abs(), out.hi.abs())
                    / np.sqrt(out.rel_within.clip(lower=1e-3)))
    out = out.sort_values(["labels", "p"])
    out.to_csv(d / "block_sequences.csv", index=False)

    summary = {"run": a.run, "n_tests_per_family": 32, "alpha": a.alpha}
    for src in ("hard", "soft"):
        o = out[out.labels == src]
        summary[src] = {"n_bh_sig": int(o.bh_sig.sum()),
                        "n_bh_sig_pooled": int(o.bh_sig_pooled.sum()),
                        "n_raw_p05": int((o.p < 0.05).sum()),
                        "median_rel_within": float(o.rel_within.median()),
                        "median_bound": float(o.bound.median())}
        print(f"\n=== {src} labels: {summary[src]} ===")
        print(o.head(8)[["kind", "pair", "pooled_pp_per_sd", "pp_per_sd", "lo", "hi", "p",
                         "bh_sig", "rel_within", "bound"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    runlog.finish(d, summary)


if __name__ == "__main__":
    main()
