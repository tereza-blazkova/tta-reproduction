#!/usr/bin/env python
"""Stage 16 — split-half reliability of the session-level rates.

Every feature the model uses is a session rate: the share of a role's turns carrying a code.
The question here is whether that number is stable --- if the same session were measured twice,
would it come out the same? A session cannot be measured twice, so each session's turns are
split in half, the rate is computed on each half, and the two halves are correlated across
sessions; the Spearman--Brown correction scales a half-length correlation up to full length.

**The reliability that matters is the within-student one.** A rate can separate students
perfectly --- Anna explains more than Ben --- while carrying no information about which of
Anna's own sessions was which, and the second is what RQ1 and RQ2 run on. So the half-rates are
centred within student before correlating, and only students with at least two sessions can
contribute. Both the raw (total) and the within-student figure are reported, because the gap
between them is itself the finding.

**Three splits, because no single one is right.**
  alt    odd and even turns. Neighbouring turns are correlated --- an episode of being stuck
         spans several turns --- so the halves share local context and this OVERSTATES
         reliability. Upper bound.
  half   first half against second half. Real change inside a session counts as disagreement,
         so this UNDERSTATES it. Lower bound.
  blockK alternating blocks of K turns: a point estimate between the two.

**Rates here are unshrunk.** The model consumes rates shrunk toward a corpus prior, but
shrinkage pulls both halves toward the same constant, so measuring reliability on shrunk rates
would credit the prior with the agreement. What is reported is the reliability of the
measurement itself; shrinkage then trades variance for bias on top of it.

    python scripts/16_reliability.py --split train
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from ttapub import config, data, runlog
from ttapub import taxonomy_v3 as T

FLAG = {"tutor": "has_teacher", "student": "has_student"}
MIN_PER_HALF = 5          # a half with fewer turns gives a rate that is mostly sampling noise


def load_pair_transitions(tag: str, split: str, kind: str) -> pd.DataFrame:
    """One row per eligible adjacent-turn transition, with the pair it realises.

    The same two directions stage 11 builds: `uptake` is a tutor move and the student's reply
    in the same turn, `response` is a student signal and the tutor's next move. A pair rate is
    rarer than either of its components, so this is where attenuation is expected to bite
    hardest and where the RQ2 bound needs it.
    """
    turns = pd.read_parquet(config.TURNS_TRAIN if split == "train" else config.TURNS_TEST,
                            columns=["session_id", "turn", "pos"])
    lab = {}
    for role in ("tutor", "student"):
        files = sorted(config.LABELS_V3.glob(f"v3_{tag}_{split}_{role}_shard*.csv"))
        d = pd.concat([pd.read_csv(f, dtype={"session_id": str}) for f in files],
                      ignore_index=True).drop_duplicates(["session_id", "turn"])
        lab[role] = d.rename(columns={"code": role})[["session_id", "turn", role]]
    df = turns.merge(lab["tutor"], on=["session_id", "turn"], how="left") \
              .merge(lab["student"], on=["session_id", "turn"], how="left")
    df = df.sort_values(["session_id", "pos"])
    if kind == "uptake":
        ok = df.tutor.notna() & df.student.notna()
        pair = df.tutor.where(ok) + " > " + df.student.where(ok)
    else:
        nxt = df.groupby("session_id", sort=False).tutor.shift(-1)
        adjacent = df.groupby("session_id", sort=False).turn.shift(-1) == df.turn + 1
        ok = df.student.notna() & nxt.notna() & adjacent.fillna(False)
        pair = df.student.where(ok) + " > " + nxt.where(ok)
    out = pd.DataFrame({"session_id": df.session_id, "code": pair})
    out = out[out.code.notna()].copy()
    out["r"] = out.groupby("session_id", sort=False).cumcount()
    out["n_role"] = out.groupby("session_id", sort=False).r.transform("size")
    return out


def load_role_turns(tag: str, split: str, role: str) -> pd.DataFrame:
    turns = pd.read_parquet(config.TURNS_TRAIN if split == "train" else config.TURNS_TEST,
                            columns=["session_id", "turn", "pos", FLAG[role]])
    turns = turns[turns[FLAG[role]]].copy()
    files = sorted(config.LABELS_V3.glob(f"v3_{tag}_{split}_{role}_shard*.csv"))
    lab = pd.concat([pd.read_csv(f, dtype={"session_id": str}) for f in files],
                    ignore_index=True).drop_duplicates(["session_id", "turn"])
    df = turns.merge(lab[["session_id", "turn", "code"]], on=["session_id", "turn"], how="inner")
    df = df.sort_values(["session_id", "pos"])
    # rank within the role's own axis, so "first half" means first half of that role's turns
    df["r"] = df.groupby("session_id", sort=False).cumcount()
    df["n_role"] = df.groupby("session_id", sort=False).r.transform("size")
    return df


def assign(df: pd.DataFrame, scheme: str) -> pd.Series:
    if scheme == "alt":
        return df.r % 2
    if scheme == "half":
        return (df.r >= df.n_role // 2).astype(int)
    k = int(scheme.replace("block", ""))
    return (df.r // k) % 2


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r) if r > -1 else np.nan


def reliability(df: pd.DataFrame, codes: list[str], scheme: str,
                student_of: dict) -> pd.DataFrame:
    df = df.assign(half=assign(df, scheme))
    n = df.groupby(["session_id", "half"]).size().rename("n")
    cnt = (df.groupby(["session_id", "half"]).code.value_counts()
             .unstack(fill_value=0).reindex(columns=codes, fill_value=0))
    rate = cnt.div(n, axis=0)
    wide = rate.unstack("half")                      # columns: (code, half)
    ok = n.unstack("half").min(axis=1) >= MIN_PER_HALF
    wide = wide[ok.reindex(wide.index, fill_value=False)]
    stu = pd.Series({s: student_of.get(s) for s in wide.index.get_level_values(0)},
                    index=wide.index)
    out = []
    for c in codes:
        if (c, 0) not in wide.columns or (c, 1) not in wide.columns:
            continue
        a, b = wide[(c, 0)], wide[(c, 1)]
        keep = a.notna() & b.notna()
        a, b, g = a[keep], b[keep], stu[keep]
        raw = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
        # within student: centre each half on that student's own mean, keep repeat students
        d = pd.DataFrame({"a": a.to_numpy(), "b": b.to_numpy(), "g": g.to_numpy()})
        rep = d.g.map(d.g.value_counts()) >= 2
        d = d[rep]
        if len(d) > 50 and d.g.nunique() > 20:
            d["a"] -= d.groupby("g").a.transform("mean")
            d["b"] -= d.groupby("g").b.transform("mean")
            win = (np.corrcoef(d.a, d.b)[0, 1]
                   if d.a.std() > 1e-12 and d.b.std() > 1e-12 else np.nan)
        else:
            win = np.nan
        out.append({"code": c, "scheme": scheme, "n_sessions": int(keep.sum()),
                    "n_sessions_within": int(len(d)), "n_students": int(d.g.nunique()),
                    "r_total": raw, "rel_total": spearman_brown(raw),
                    "r_within": win, "rel_within": spearman_brown(win)})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--split", default="train")
    ap.add_argument("--run", default="reliability")
    ap.add_argument("--unit", default="codes", choices=["codes", "sequences"],
                    help="codes = the 57 single-code rates; sequences = the 421 adjacent-turn "
                         "pair rates that RQ2 uses")
    a = ap.parse_args()
    d = runlog.start(a.run, vars(a))

    meta = data.load_metadata()
    frame = data.load_train() if a.split == "train" else data.load_test()
    student_of = dict(frame.merge(meta[["response_id", "student_id"]], on="response_id")
                      [["session_id", "student_id"]].drop_duplicates().to_numpy())

    parts = []
    if a.unit == "sequences":
        inv = pd.read_csv(config.DERIVED / f"sequence_inventory_{a.split}.csv")
        for kind in ("uptake", "response"):
            keep = inv[inv.kind == kind].pair.tolist()
            df = load_pair_transitions(a.tag, a.split, kind)
            df = df[df.code.isin(keep)]
            print(f"[{kind}] {len(df):,} transitions, {len(keep)} pairs")
            for scheme in ("alt", "block5", "block10", "half"):
                r = reliability(df, keep, scheme, student_of)
                r["role"] = kind
                parts.append(r)
                print(f"  {scheme:8s} median within-student reliability "
                      f"{r.rel_within.median():.3f}  (total {r.rel_total.median():.3f})")
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(d / "reliability_sequences.csv", index=False)
        b10 = out[out.scheme == "block10"]
        print(f"\nmedian within-student reliability of a PAIR rate: {b10.rel_within.median():.3f}")
        print(f"  quartiles: {b10.rel_within.quantile(.25):.3f} / {b10.rel_within.quantile(.75):.3f}")
        runlog.finish(d, {"run": a.run, "unit": "sequences",
                          "median_within_block10": float(b10.rel_within.median())})
        return
    for role in ("tutor", "student"):
        df = load_role_turns(a.tag, a.split, role)
        print(f"[{role}] {len(df):,} turns over {df.session_id.nunique():,} sessions")
        for scheme in ("alt", "block5", "block10", "half"):
            r = reliability(df, T.codes(role), scheme, student_of)
            r["role"] = role
            parts.append(r)
            print(f"  {scheme:8s} median within-student reliability "
                  f"{r.rel_within.median():.3f}  (total {r.rel_total.median():.3f})")
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(d / "reliability.csv", index=False)

    piv = out.pivot_table(index=["role", "code"], columns="scheme", values="rel_within")
    piv = piv[["half", "block10", "block5", "alt"]]
    piv.columns = ["lower (1st/2nd)", "block10", "block5", "upper (alt)"]
    print("\n=== within-student reliability, bounds per code (worst 12 by the block10 estimate) ===")
    print(piv.sort_values("block10").head(12).to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n=== best 8 ===")
    print(piv.sort_values("block10").tail(8).to_string(float_format=lambda v: f"{v:.3f}"))
    runlog.finish(d, {"run": a.run, "median_within_block10":
                      float(out[out.scheme == "block10"].rel_within.median()),
                      "median_total_block10":
                      float(out[out.scheme == "block10"].rel_total.median())})


if __name__ == "__main__":
    main()
