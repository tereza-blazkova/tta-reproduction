#!/usr/bin/env python
"""Stage 05 — per-turn labels (or probabilities) -> the 57 session-rate columns, both splits.

One column per code: the shrunk session rate (see :mod:`ttapub.agg`). Tutor rates are computed
over turns with tutor text and student rates over turns with student text — the two axes have
different denominators, and crossing them would silently change what every rate means.

**Two sources, and `labels` is the main path.**

* ``--source labels`` (default) reads the LLM's own per-turn labels and one-hot encodes them, so
  a session rate is the share of that role's turns the 122B assigned the code. This is possible
  because the full corpus — both splits, every session — is annotated directly (≈5.3 h on three
  B200s), which was not affordable when the design assumed a 3,000-session
  sample. It removes the distillation step from every number in the paper, and distillation is
  where a code can fail silently: its per-turn probabilities go near-constant, so the column
  stops carrying information about the concept while still receiving attribution.
* ``--source probs`` reads stage 04's distilled per-turn probabilities. Kept for the appendix
  arm that asks how faithfully a CPU classifier can reproduce the 122B's features — a
  deployment question, not a measurement one.

A turn with no label (a parse failure left it out) is **excluded from its denominator**, not
counted as a zero: "not measured" and "the code did not occur" are different statements. The
share of such turns is reported.

The corpus prior each rate is shrunk toward is **computed here, from this annotation's own
turns over the training split**, and written with the run that produced it. A prior from any other
annotation would shrink a rate toward a different instrument's corpus mean.

    python scripts/05_build_features.py --run clean57-features --tag full
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

from ttapub import agg, config, runlog
from ttapub import taxonomy_v3 as T

PREFIX = {"tutor": "t_", "student": "s_"}
FLAG = {"tutor": "has_teacher", "student": "has_student"}


def _label_frame(turns: pd.DataFrame, labels: pd.DataFrame, role: str) -> tuple[pd.DataFrame, dict]:
    """One-hot the LLM's labels onto this role's turn axis.

    Returns (frame of indicator columns keyed by session_id, coverage dict).

    Two different kinds of gap, kept apart because they mean different things:

    * **session coverage** — sessions of the split that carry no labels at all. During a run in
      progress this is most of them; at the end it must be ~100%, and a shortfall means a shard
      died rather than that the annotator found nothing.
    * **unlabelled turns within annotated sessions** — a chunk whose generation failed to parse
      left its turns without a label. These are excluded from the rate's denominator, since
      "not measured" is not "the code did not occur". This is the number that bounds the
      measurement, and it should be near zero (the pilot ran at 0.00% parse failures).
    """
    axis = turns.loc[turns[FLAG[role]], ["session_id", "turn"]]
    lab = labels[labels.role == role][["session_id", "turn", "code"]]
    labelled_sessions = set(lab.session_id)
    n_sessions = axis.session_id.nunique()
    axis = axis[axis.session_id.isin(labelled_sessions)]
    m = axis.merge(lab, on=["session_id", "turn"], how="left")
    cov = {"sessions_in_split": int(n_sessions),
           "sessions_annotated": int(axis.session_id.nunique()),
           "session_coverage": round(axis.session_id.nunique() / max(n_sessions, 1), 5),
           "unlabelled_turns_in_annotated": round(float(m.code.isna().mean()), 5)}
    m = m[m.code.notna()]
    codes = T.codes(role)
    bad = sorted(set(m.code) - set(codes))
    if bad:
        raise SystemExit(f"labels contain codes outside the {role} pool: {bad[:5]} — the "
                         f"annotation and docs/taxonomy_v3.md disagree")
    out = pd.DataFrame({c: (m.code == c).astype(float).to_numpy() for c in codes})
    out.insert(0, "session_id", m.session_id.to_numpy())
    return out, cov


def _role_frame(turn_probs: pd.DataFrame, role: str) -> pd.DataFrame:
    """The turns of one role's axis, with columns renamed from `t_<CODE>` to `<CODE>`."""
    cols = [f"{PREFIX[role]}{c}" for c in T.codes(role)]
    missing = [c for c in cols if c not in turn_probs.columns]
    if missing:
        raise SystemExit(f"turn probabilities missing {len(missing)} {role} columns "
                         f"(first: {missing[:3]}) — re-run stage 04.")
    sub = turn_probs.loc[turn_probs[FLAG[role]], ["session_id"] + cols].copy()
    sub.columns = ["session_id"] + T.codes(role)
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="clean57-features")
    ap.add_argument("--source", choices=["labels", "probs"], default="labels",
                    help="labels = the LLM's own per-turn codes (the main path); "
                         "probs = stage 04's distilled probabilities (appendix arm)")
    ap.add_argument("--tag", default="full",
                    help="annotation tag for --source labels; the split's suffix is added, so "
                         "'full' reads v3_full_train_* and v3_full_test_*")
    ap.add_argument("--kappa", type=float, default=agg.SHRINK_KAPPA)
    ap.add_argument("--reuse-prior", action="store_true",
                    help="keep the existing prior instead of recomputing it (only when "
                         "rebuilding features from unchanged children)")
    a = ap.parse_args()

    if a.source == "labels":
        turns_tr = pd.read_parquet(config.TURNS_TRAIN)
        labels_tr = _load_labels(f"{a.tag}_train")
        d = runlog.start(a.run, vars(a), [config.TURNS_TRAIN])
    else:
        tr = pd.read_parquet(config.TURN_PROBS_TRAIN)
        d = runlog.start(a.run, vars(a), [config.TURN_PROBS_TRAIN])

    # ── the prior: each code's corpus mean over every scored turn of its own axis ──
    if a.reuse_prior and config.prior_path(a.source).exists():
        prior = agg.load_prior(config.prior_path(a.source))
        print(f"reusing {config.prior_path(a.source)}")
    else:
        prior, n_turns = {}, {}
        for role in ("tutor", "student"):
            if a.source == "labels":
                f, _ = _label_frame(turns_tr, labels_tr, role)
            else:
                f = _role_frame(tr, role)
            prior |= agg.compute_prior(f, T.codes(role))
            n_turns[role] = int(len(f))
        prov = json.loads((d / "provenance.json").read_text())
        agg.save_prior(prior, {"run": a.run, "git": prov["git"], "source": a.source,
                               "tag": a.tag if a.source == "labels" else None,
                               "n_turns_tutor": n_turns["tutor"],
                               "n_turns_student": n_turns["student"]},
                       path=config.prior_path(a.source))
        print(f"wrote {config.prior_path(a.source)} (kappa {agg.SHRINK_KAPPA}, "
              f"source {a.source}, "
              f"{n_turns['tutor']:,} tutor / {n_turns['student']:,} student turns)")

    out = {}
    turn_tables = {"train": config.TURNS_TRAIN, "test": config.TURNS_TEST}
    prob_tables = {"train": config.TURN_PROBS_TRAIN, "test": config.TURN_PROBS_TEST}
    for split in ("train", "test"):
        dest = config.features_path(split, a.source)
        src = turn_tables[split] if a.source == "labels" else prob_tables[split]
        if not src.exists():
            print(f"skipping {split}: {src} missing")
            continue
        blocks, unlab = [], {}
        if a.source == "labels":
            turns = pd.read_parquet(src)
            try:
                labels = _load_labels(f"{a.tag}_{split}")
            except SystemExit as e:
                print(f"skipping {split}: {e}")
                continue
        else:
            probs = pd.read_parquet(src)
        for role in ("tutor", "student"):
            if a.source == "labels":
                f, unlab[role] = _label_frame(turns, labels, role)
            else:
                f = _role_frame(probs, role)
            blocks.append(agg.shrunk_rates(f, T.codes(role), role, prior, kappa=a.kappa))
        feats = blocks[0].join(blocks[1], how="outer")
        feats.index.name = "session_id"
        feats = feats.reset_index()
        feats.to_csv(dest, index=False)
        cols = agg.code_columns(feats.columns)
        out[split] = {"sessions": len(feats), "columns": len(cols),
                      "na_share": float(feats[cols].isna().to_numpy().mean()),
                      "unlabelled_turn_share": unlab}
        extra = ""
        if unlab:
            c = unlab["tutor"]
            extra = (f"  | sessions annotated {c['sessions_annotated']:,}/"
                     f"{c['sessions_in_split']:,} ({c['session_coverage']:.1%}); "
                     f"unlabelled turns within them: " +
                     ", ".join(f"{r} {v['unlabelled_turns_in_annotated']:.3%}"
                               for r, v in unlab.items()))
        print(f"[{split}] {len(feats):,} sessions x {len(cols)} code columns -> {dest}{extra}")

    runlog.finish(d, {"run": a.run, "kappa": a.kappa, "source": a.source, "tag": a.tag,
                      "splits": out, "prior_path": str(config.prior_path(a.source)),
                      "prior": prior})


def _load_labels(tag: str) -> pd.DataFrame:
    """Every shard of one annotation run, as session_id, turn, code, also, role."""
    parts = []
    for role in ("tutor", "student"):
        files = sorted(config.LABELS_V3.glob(f"v3_{tag}_{role}_shard*.csv"))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f, dtype={"session_id": str}) for f in files],
                       ignore_index=True)
        df["role"] = role
        parts.append(df)
    if not parts:
        raise SystemExit(f"no label files v3_{tag}_*_shard*.csv in {config.LABELS_V3}")
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["role", "session_id", "turn"])


if __name__ == "__main__":
    main()
