#!/usr/bin/env python
"""Stage 13 — assign each response the stretch of its session that handled ITS objective.

Writes one row per response: the matched segment's turn span, the placebo span (a different
segment of the same session), and how the match was reached. Validation is the point of the
stage, so it reports the segmenter against the tutor's own stated lesson count — the number of
objectives the session was graded on — which is how the original was tuned and is independent
of the outcome.

    python scripts/13_localise.py --split train
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ttapub import config, data, localise, runlog

TURNS = {"train": config.TURNS_TRAIN, "test": config.TURNS_TEST}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    run = a.run or f"localise-{a.split}"
    d = runlog.start(run, vars(a))

    frame = data.load_train() if a.split == "train" else data.load_test()
    turns = pd.read_parquet(TURNS[a.split],
                            columns=["session_id", "turn", "pos", "tutor_text", "student_text"])
    # the child-input texts carry the turn's own words after a fixed truncation; for boundary
    # detection that is the tutor's side, which is where announcements live
    turns["t_txt"] = turns.tutor_text.str.split(" [T] ", regex=False).str[-1]
    turns["s_txt"] = turns.student_text.str.split(" [S] ", regex=False).str[-1]
    by_session = {sid: g.sort_values("pos") for sid, g in turns.groupby("session_id")}

    rows, seg_counts = [], []
    for sid, g in by_session.items():
        t_txt, s_txt = g.t_txt.tolist(), g.s_txt.tolist()
        segs = localise.segment_session(t_txt)
        seg_counts.append({"session_id": sid, "n_segments": len(segs) - 1})
        sub = frame[frame.session_id == sid]
        for _, r in sub.iterrows():
            k = localise.match_segment(segs, t_txt, s_txt, r.learning_objective)
            kp = localise.wrong_segment(k, len(segs) - 1, str(r.response_id))
            whole = k < 0
            lo_i, hi_i = (0, len(g)) if whole else (segs[k], segs[k + 1])
            pl_i, ph_i = (0, len(g)) if whole else (segs[kp], segs[kp + 1])
            pos = g.pos.to_numpy()
            rows.append({"response_id": r.response_id, "session_id": sid,
                         "n_segments": len(segs) - 1, "matched": not whole,
                         "pos_start": int(pos[lo_i]), "pos_end": int(pos[hi_i - 1]) + 1,
                         "placebo_start": int(pos[pl_i]), "placebo_end": int(pos[ph_i - 1]) + 1,
                         "n_turns_session": len(g)})
    out = pd.DataFrame(rows)
    out["span_turns"] = out.pos_end - out.pos_start
    seg = pd.DataFrame(seg_counts)

    # ── validation against the tutor's stated lesson count (never against the outcome) ──
    n_obj = frame.groupby("session_id").learning_objective_id.nunique().rename("n_objectives")
    v = seg.set_index("session_id").join(n_obj).dropna()
    exact = (v.n_segments == v.n_objectives).mean()
    within1 = ((v.n_segments - v.n_objectives).abs() <= 1).mean()
    rho = spearmanr(v.n_segments, v.n_objectives).statistic
    print(f"[{a.split}] segmenter vs the session's objective count: "
          f"exact {exact:.0%}, within +/-1 {within1:.0%}, Spearman {rho:.2f}")
    print(f"[{a.split}] {len(out):,} responses | matched to a segment "
          f"{out.matched.mean():.1%} | median span {out.span_turns.median():.0f} turns "
          f"of {out.n_turns_session.median():.0f}")
    m = out[out.matched]
    if len(m):
        print(f"  among matched: median span {m.span_turns.median():.0f} turns "
              f"({(m.span_turns / m.n_turns_session).median():.0%} of the session)")

    path = config.DERIVED / f"localisation_{a.split}.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    runlog.finish(d, {"run": run, "split": a.split, "n": len(out),
                      "matched_share": float(out.matched.mean()),
                      "seg_vs_objectives": {"exact": float(exact), "within1": float(within1),
                                            "spearman": float(rho)}})


if __name__ == "__main__":
    main()
