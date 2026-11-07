#!/usr/bin/env python
"""Stage 52 — Lin-scheme labels -> session columns, gates, reliability. Lin-scheme baseline only.

Run with `TTAPUB_ANNOTATOR=lin`, the repo's existing switch, so the tables land in
`data/derived/features_v3_lin_{train,test}.csv` and `code_prior_v3_lin.json`. Stage 06 then fits
them unchanged, which is what makes this "the same model" rather than a copy of it. The main-path
files cannot be overwritten: the script exits unless the switch is set.

**Main arm (`--kind rates`, TTAPUB_ANNOTATOR=lin).** Rates only, shrunk with kappa = 25 toward a
prior rebuilt from these labels on the training split. One column per role-tag that clears the
gate, plus one per first-level tag with at least two second-level children for that role. A
first-level tag with one child would be the same column twice. Kappa stays 25, though the unit is
now usually the sentence, so the shrinkage is relatively weaker than on turns; this is recorded in
`docs/lin_scheme.md`.

`UNCODABLE` gets no column and leaves the denominator: not measured is not "did not occur".

**Gate.** At least 300 positives on the training split. A failing tag gets no column. Split-half
reliability comes from stage 16's estimator, unshrunk, and is reported beside every tag.

**Exploratory arm (`--kind faith`, TTAPUB_ANNOTATOR=linfaith).** Lin's counts and percentages per
role-tag at both levels, without the mined patterns. Counts break the rates-only rule and carry
session length, so this arm stays out of the main comparison. Its columns still end in `_rate`
because stage 06 selects on that suffix; `_count_` in the name says what they are.

    TTAPUB_ANNOTATOR=lin python scripts/52_build_lin_features.py --tag full --run lin-features
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import agg, config, data, dialogue, runlog
from ttapub import taxonomy_lin as L

LABELS = config.DERIVED / "labels_lin"
FIELD = {"tutor": "teacher", "student": "student"}
MIN_POSITIVES = 300
IMAGE = re.compile(r"\[\s*image|\bimage\s*link\b|https?://|\.(?:png|jpe?g|gif)\b", re.I)


def load_labels(tag: str, role: str) -> pd.DataFrame:
    files = sorted(LABELS.glob(f"lin_{tag}_{role}_shard*.csv"))
    if not files:
        raise SystemExit(f"no label files lin_{tag}_{role}_shard*.csv in {LABELS}")
    df = pd.concat([pd.read_csv(f, dtype={"session_id": str}) for f in files], ignore_index=True)
    off = sorted(set(df.code) - set(L.codes(role)))
    if off:
        raise SystemExit(f"{role} labels outside the pool: {off[:5]}")
    return df.drop_duplicates(["session_id", "turn", "sent"])


def expected_units(split: str, unit: str) -> tuple[pd.DataFrame, dict]:
    """Units per session and role from the transcripts, and a count of image-like utterances."""
    frame = data.load_train() if split == "train" else data.load_test()
    source = dict(frame[["session_id", "data_source"]].drop_duplicates().to_numpy()) \
        if "data_source" in frame.columns else {}
    rows, images, utts = [], {}, {}
    for sid, tx in data.store(split).iter(list(dict.fromkeys(frame.session_id))):
        src = source.get(sid, split)
        text = tx.content.dropna().astype(str)
        utts[src] = utts.get(src, 0) + len(text)
        images[src] = images.get(src, 0) + int(text.map(lambda c: bool(IMAGE.search(c))).sum())
        n = {"tutor": 0, "student": 0}
        for t in dialogue.build_turns(tx):
            for role, f in FIELD.items():
                if unit == "turn":
                    n[role] += int(bool(t[f].strip()))
                else:
                    n[role] += len(L.segment(t[f]))
        rows.append({"session_id": sid, **n})
    return pd.DataFrame(rows), {"utterances": utts, "image_or_link": images}


def coverage(lab: pd.DataFrame, exp: pd.DataFrame, role: str) -> dict:
    got = lab.groupby("session_id").size()
    e = exp.set_index("session_id")[role]
    e = e[e > 0]
    return {"sessions_with_units": int(len(e)),
            "sessions_labelled": int(e.index.isin(got.index).sum()),
            "units_expected": int(e.sum()),
            "units_labelled": int(got.reindex(e.index).fillna(0).sum()),
            "unit_coverage": round(float(got.reindex(e.index).fillna(0).sum() / max(e.sum(), 1)),
                                   6),
            "share_uncodable": round(float((lab.code == L.UNCODABLE).mean()), 5)}


def one_hot(lab: pd.DataFrame, role: str) -> pd.DataFrame:
    """Indicator columns over this role's coded units, `UNCODABLE` removed."""
    lab = lab[lab.code != L.UNCODABLE]
    out = pd.DataFrame({f"lin_{c.lower()}": (lab.code == c).astype(float).to_numpy()
                        for c in L.feature_codes(role)})
    for first, kids in L.first_level_groups(role).items():
        if len(kids) >= 2:
            out[f"lin1_{first.lower()}"] = out[[f"lin_{c.lower()}" for c in kids]].sum(axis=1)
    out.insert(0, "session_id", lab.session_id.to_numpy())
    return out


def reliability_table(lab: pd.DataFrame, role: str, student_of: dict) -> pd.DataFrame:
    """Stage 16's estimator, on this role's coded units, for both levels."""
    spec = importlib.util.spec_from_file_location(
        "stage16", Path(__file__).with_name("16_reliability.py"))
    s16 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s16)
    lab = lab[lab.code != L.UNCODABLE].sort_values(["session_id", "turn", "sent"]).copy()
    lab["r"] = lab.groupby("session_id", sort=False).cumcount()
    lab["n_role"] = lab.groupby("session_id", sort=False).r.transform("size")
    parts = []
    for scheme in ("alt", "block10", "half"):
        second = lab.assign(code="lin_" + lab.code.str.lower())
        parts.append(s16.reliability(second, sorted(second.code.unique()), scheme, student_of))
        parents = {c: k for k, v in L.first_level_groups(role).items() if len(v) >= 2 for c in v}
        # every coded unit stays in the frame: stage 16 takes each rate's denominator from the
        # rows it is given, so dropping the other tags would make every first-level rate 1
        firsts = lab.assign(code=("lin1_" + lab.code.map(parents).str.lower()).fillna("_rest"))
        wanted = sorted(set(firsts.code) - {"_rest"})
        parts.append(s16.reliability(firsts, wanted, scheme, student_of))
    r = pd.concat(parts, ignore_index=True)
    # pivot, not pivot_table: a reliability that is NaN throughout must stay a visible NaN column
    return r.pivot(index="code", columns="scheme", values="rel_within").reset_index()


def gate_table(frames: dict[str, pd.DataFrame], rel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for role, f in frames.items():
        for c in f.columns.drop("session_id"):
            pos = int(f[c].sum())
            rows.append({"role": role, "column": agg.column(c, role), "tag": c, "positives": pos,
                         "prevalence_pct": round(100 * pos / max(len(f), 1), 3),
                         "passes": pos >= MIN_POSITIVES})
    out = pd.DataFrame(rows)
    rr = pd.concat([v.assign(role=k) for k, v in rel.items()], ignore_index=True) \
        .rename(columns={"code": "tag", "alt": "rel_within_upper", "half": "rel_within_lower",
                         "block10": "rel_within_block10"})
    return out.merge(rr, on=["role", "tag"], how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--run", default="lin-features")
    ap.add_argument("--unit", default="sentence", choices=["sentence", "turn"])
    ap.add_argument("--kind", default="rates", choices=["rates", "faith"])
    a = ap.parse_args()
    want = {"rates": "lin", "faith": "linfaith"}[a.kind]
    if config.ANNOTATOR != want:
        raise SystemExit(f"--kind {a.kind} needs TTAPUB_ANNOTATOR={want}, so that the main-path "
                         f"feature tables cannot be overwritten")

    d = runlog.start(a.run, vars(a), [])
    labs = {s: {r: load_labels(f"{a.tag}_{s}", r) for r in FIELD} for s in ("train", "test")}

    meta = data.load_metadata()
    tr = data.load_train()
    student_of = dict(tr.merge(meta[["response_id", "student_id"]], on="response_id")
                      [["session_id", "student_id"]].drop_duplicates().to_numpy())

    res: dict = {"run": a.run, "kind": a.kind, "tag": a.tag, "unit": a.unit,
                 "kappa": agg.SHRINK_KAPPA, "min_positives": MIN_POSITIVES,
                 "codebook_digest": runlog.digest(L.SPEC)}
    exp = {}
    for s in ("train", "test"):
        exp[s], res[f"markers_{s}"] = expected_units(s, a.unit)
        res[f"coverage_{s}"] = {r: coverage(labs[s][r], exp[s], r) for r in FIELD}
        print(s, json.dumps(res[f"coverage_{s}"]), json.dumps(res[f"markers_{s}"]), flush=True)

    frames = {r: one_hot(labs["train"][r], r) for r in FIELD}
    rel = {r: reliability_table(labs["train"][r], r, student_of) for r in FIELD}
    gates = gate_table(frames, rel)
    gates.to_csv(d / "gates.csv", index=False)
    print(gates.to_string(float_format=lambda v: f"{v:.3f}"), flush=True)
    keep = {r: gates[(gates.role == r) & gates.passes].tag.tolist() for r in FIELD}
    res["failed_gate"] = gates[~gates.passes][["role", "tag", "positives"]].to_dict("records")
    res["n_columns"] = {r: len(keep[r]) for r in FIELD}
    res["median_rel_within_block10"] = {
        r: float(gates[(gates.role == r) & gates.tag.str.startswith("lin_")]
                 .rel_within_block10.median()) for r in FIELD}

    prior: dict[str, float] = {}
    for r in FIELD:
        for c in keep[r]:
            prior[f"{r}:{c}"] = float(frames[r][c].mean())
    agg.save_prior(prior, {"run": a.run, "tag": a.tag, "unit": a.unit, "kind": a.kind,
                           "git": json.loads((d / "provenance.json").read_text())["git"]},
                   path=config.prior_path("labels"))

    for s in ("train", "test"):
        blocks = []
        for r in FIELD:
            f = one_hot(labs[s][r], r)[["session_id"] + keep[r]]
            if a.kind == "rates":
                pri = {c: prior[f"{r}:{c}"] for c in keep[r]}
                blocks.append(agg.shrunk_rates(f, keep[r], r, pri))
            else:
                g = f.groupby("session_id")[keep[r]]
                pct = g.mean().rename(columns=lambda c: agg.column(c.replace(
                    "lin", "linfaith_pct", 1), r))
                cnt = g.sum().rename(columns=lambda c: agg.column(c.replace(
                    "lin", "linfaith_count", 1), r))
                blocks.append(pct.join(cnt))
        feats = blocks[0].join(blocks[1], how="outer")
        feats.index.name = "session_id"
        feats = feats.reset_index()
        feats.to_csv(config.features_path(s, "labels"), index=False)
        cols = agg.code_columns(feats.columns)
        res[f"table_{s}"] = {"sessions": len(feats), "columns": len(cols),
                             "na_share": float(np.mean(feats[cols].isna().to_numpy()))}
        print(f"[{s}] {len(feats):,} sessions x {len(cols)} columns -> "
              f"{config.features_path(s, 'labels')}", flush=True)
    res["prior"] = prior
    runlog.finish(d, res)


if __name__ == "__main__":
    main()
