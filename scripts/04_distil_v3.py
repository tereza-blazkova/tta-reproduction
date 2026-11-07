#!/usr/bin/env python
"""Stage 04 — distil the v3 LLM labels into fast CPU children ("the fast annotators").

**This is no longer on the main path.** Every session of both splits carries the 122B's own
labels, so stage 05 builds features from those directly (`--source labels`). What the children
are for now is the deployment question the paper should still answer: *can these features be
produced without a language model in the loop, and how faithfully?* A TF-IDF + logistic head per
code runs on a laptop in seconds per session, against ~2 s on a B200.

Because the test split is also annotated, that question is now answerable **on held-out data**:
`--fidelity-tag` scores the children against the 122B's labels on the test split and reports
per-code AUROC and argmax agreement **split by provider**, so the fast annotator's transfer to
an unseen provider (Eedi) is measured rather than assumed. That was impossible under the
original design, where the only LLM labels were the ones the children were fitted on.

Fits one binary head per code on session-disjoint folds (see :mod:`ttapub.distil` for the
recipe and why each part of it is frozen), then applies the spec §6 gates and writes:

    data/derived/turn_probs_v3_train.parquet   per-turn probabilities, every training turn
    data/derived/turn_probs_v3_test.parquet    the same for the released test split
    data/derived/children_v3.joblib            fitted vectorizer + heads, for re-scoring
    results/<run>/gates.csv                    one row per code: positives, AUROC, SD, status

Labelled sessions get **out-of-fold** probabilities; unlabelled sessions and the whole test
split are scored with heads fitted on all labelled turns. That is the serving path, and it is
why a labelled session's rate is not sharper than a test session's.

    python scripts/04_distil_v3.py --tag full --run clean57-children
"""
from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401
import joblib
import numpy as np
import pandas as pd

from ttapub import config, data, distil, runlog
from ttapub import taxonomy_v3 as T

ROLES = ("tutor", "student")
TEXT_COL = {"tutor": "tutor_text", "student": "student_text"}
FLAG_COL = {"tutor": "has_teacher", "student": "has_student"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full",
                    help="annotation tag from stage 02; '_train' is appended for fitting")
    ap.add_argument("--fidelity-tag", default=None,
                    help="tag whose TEST labels the children are scored against, e.g. 'full' "
                         "-> v3_full_test_*. Omit to skip the held-out fidelity report.")
    ap.add_argument("--run", default="clean57-children")
    ap.add_argument("--fit-cap", type=int, default=distil.FIT_CAP)
    ap.add_argument("--no-test", action="store_true", help="skip scoring the test split")
    ap.add_argument("--limit-sessions", type=int, default=0, help="smoke test")
    a = ap.parse_args()

    # the label loader is inlined (`03_validate` is not an importable module name)
    lab = _load_labels(f"{a.tag}_train")
    turns_tr = pd.read_parquet(config.TURNS_TRAIN)
    d = runlog.start(a.run, vars(a), [config.TURNS_TRAIN])
    t0 = time.time()

    children, reports, oof_cols = {}, [], {}
    for role in ROLES:
        sub = lab[lab.role == role]
        if sub.empty:
            raise SystemExit(f"no {role} labels for tag {a.tag!r}")
        df = turns_tr[turns_tr[FLAG_COL[role]]].merge(
            sub[["session_id", "turn", "code"]], on=["session_id", "turn"], how="inner")
        if a.limit_sessions:
            keep = set(list(dict.fromkeys(df.session_id))[: a.limit_sessions])
            df = df[df.session_id.isin(keep)]
        df = df.reset_index(drop=True)
        codes = T.codes(role)
        Y = pd.DataFrame({c: (df.code == c).astype(int) for c in codes})
        print(f"[{role}] {len(df):,} labelled turns over {df.session_id.nunique():,} sessions, "
              f"{len(codes)} codes", flush=True)

        oof, child, rep = distil.fit_children(df[TEXT_COL[role]].to_numpy(), Y,
                                              df.session_id.to_numpy(), role,
                                              fit_cap=a.fit_cap)
        children[role] = child
        reports += [r.as_row() for r in rep.values()]
        oof_cols[role] = (df[["session_id", "turn"]], oof)
        passed = sum(r.status == "pass" for r in rep.values())
        print(f"[{role}] gates: {passed}/{len(codes)} pass  ({time.time() - t0:.0f}s)",
              flush=True)

    gates = pd.DataFrame(reports)
    gates.to_csv(d / "gates.csv", index=False)
    failed = gates[gates.status != "pass"]
    print(f"\ngates: {len(gates) - len(failed)}/{len(gates)} codes pass")
    if len(failed):
        print(failed[["code", "role", "n_pos", "auroc", "p_sd", "status"]]
              .to_string(index=False))
        print("⚠️  A failing code's column must not be read as evidence about its concept "
              "(spec §6.2): merge it into its parent, or report it as unestimable.")

    # ── score every turn of both splits with the all-data heads ──
    for split, path, out in (("train", config.TURNS_TRAIN, config.TURN_PROBS_TRAIN),
                             ("test", config.TURNS_TEST, config.TURN_PROBS_TEST)):
        if split == "test" and a.no_test:
            continue
        if not path.exists():
            print(f"skipping {split}: {path} missing (run stage 01)")
            continue
        turns = pd.read_parquet(path)
        frames = [turns[["session_id", "turn", "pos", "n_turns", "has_teacher", "has_student"]]
                  .reset_index(drop=True)]
        for role in ROLES:
            P = pd.DataFrame(np.nan, index=turns.index, columns=T.codes(role))
            m = turns[FLAG_COL[role]].to_numpy()
            scored = distil.score_turns(children[role], turns.loc[m, TEXT_COL[role]].to_numpy())
            P.loc[m, :] = scored.to_numpy()
            if split == "train":
                # labelled turns get their out-of-fold value instead of an in-sample one
                key, oof = oof_cols[role]
                oof = pd.concat([key.reset_index(drop=True), oof.reset_index(drop=True)], axis=1)
                idx = turns.reset_index().merge(oof, on=["session_id", "turn"], how="inner")
                P.loc[idx["index"].to_numpy(), T.codes(role)] = idx[T.codes(role)].to_numpy()
                print(f"  [{split}/{role}] {len(idx):,} labelled turns use out-of-fold values")
            # per-turn probability columns are `t_<CODE>` / `s_<CODE>`; stage 05 strips the
            # prefix when it aggregates, so a code name appears exactly once per role axis
            frames.append(P.add_prefix(f"{role[0]}_"))
        wide = pd.concat(frames, axis=1)
        wide.to_parquet(out, index=False)
        print(f"wrote {out} ({len(wide):,} turns)", flush=True)

    # ── held-out fidelity: the children against the 122B on the test split ──
    fid = []
    if a.fidelity_tag:
        from sklearn.metrics import roc_auc_score
        tlab = _load_labels(f"{a.fidelity_tag}_test")
        tturns = pd.read_parquet(config.TURNS_TEST)
        meta = data.load_test()[["session_id", "data_source"]].drop_duplicates()
        for role in ROLES:
            axis = tturns[tturns[FLAG_COL[role]]][["session_id", "turn", TEXT_COL[role]]]
            truth = tlab[tlab.role == role][["session_id", "turn", "code"]]
            m = axis.merge(truth, on=["session_id", "turn"], how="inner").merge(
                meta, on="session_id", how="left")
            P = distil.score_turns(children[role], m[TEXT_COL[role]].to_numpy())
            codes = list(P.columns)
            argmax = P.to_numpy().argmax(axis=1)
            pred = pd.Series([codes[i] for i in argmax], index=m.index)
            for src in ("tsl", "eedi", "all"):
                sel = (m.data_source == src) if src != "all" else pd.Series(True, index=m.index)
                if not sel.any():
                    continue
                agree = float((pred[sel] == m.code[sel]).mean())
                print(f"  [fidelity {role}/{src}] n={int(sel.sum()):,} "
                      f"argmax agreement with the 122B {agree:.3f}")
                for c in codes:
                    y = (m.code[sel] == c).astype(int).to_numpy()
                    if y.sum() < 20 or y.sum() == len(y):
                        continue
                    fid.append({"code": c, "role": role, "source": src,
                                "n_pos": int(y.sum()), "n": int(sel.sum()),
                                "auroc": round(float(roc_auc_score(y, P.loc[sel, c])), 4),
                                "argmax_agreement_overall": round(agree, 4)})
        if fid:
            pd.DataFrame(fid).to_csv(d / "fidelity_heldout.csv", index=False)
            print(f"wrote {d / 'fidelity_heldout.csv'} — the fast annotator measured against "
                  f"the 122B on held-out data, by provider")

    joblib.dump(children, config.CHILDREN)
    runlog.finish(d, {"run": a.run, "tag": a.tag, "fidelity": fid,
                      "n_codes": len(gates), "n_pass": int((gates.status == "pass").sum()),
                      "gates": gates.to_dict("records"),
                      "seconds": round(time.time() - t0)})
    print(f"wrote {config.CHILDREN}")


def _load_labels(tag: str) -> pd.DataFrame:
    parts = []
    for role in ROLES:
        files = sorted(config.LABELS_V3.glob(f"v3_{tag}_{role}_shard*.csv"))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f, dtype={"session_id": str}) for f in files],
                       ignore_index=True)
        df["role"] = role
        parts.append(df)
    if not parts:
        raise SystemExit(f"no label files v3_{tag}_*_shard*.csv in {config.LABELS_V3} — "
                         "run stage 02 first.")
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["role", "session_id", "turn"])


if __name__ == "__main__":
    main()
