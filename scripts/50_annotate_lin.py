#!/usr/bin/env python
"""Stage 50 — the Lin-scheme baseline's annotation pass (vLLM, guided JSON). GPU stage.

Tags the transcripts with the dialogue-act scheme of Lin et al. (2022), from
`docs/lin_scheme.md` via :mod:`ttapub.taxonomy_lin`. This is a baseline using Lin's scheme with
our annotator, not a replication.

Based on `02_annotate_v3.py`, and kept as close to it as the scheme allows. The same: model,
greedy decode, guided JSON, both pools in one `generate()`, 2 turns of leading context, both
speakers rendered, session order (imported from stage 02, so a pilot of N is the same N sessions
stage 02 would have taken), resume, and refusal to exit 0 on a degraded run. Different:

* **The unit** is the sentence by default (`--unit turn` is the fallback; see
  `docs/lin_scheme.md` §4.1 and §6). Target sentences are numbered through the session per role,
  and a label is `(session_id, turn, sent, n, code)`.
* **A chunk is capped by sentences as well as turns.** A session has about 2.6 times as many
  sentences as turns, and one turn reached 79, so a 24-turn window alone could outrun
  `--max-tokens`. A long turn may therefore span two chunks; its whole text is rendered in both.
* **No `also` field**: Lin coded one tag per sentence.
* **Its own output directory and file names**, `data/derived/labels_lin/lin_{tag}_{role}_shard{i}.csv`,
  so it cannot touch `data/derived/labels_v3`.
* Each shard writes `lin_{tag}_shard{i}_summary.json` with chunk, failure, vocabulary and timing
  counts, so the pilot report reads them instead of scraping a log.

    python scripts/50_annotate_lin.py --dry-run 2 --tag check
    python scripts/50_annotate_lin.py --sessions 50 --tag pilot
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from ttapub import anno_json as aj
from ttapub import config, data, dialogue
from ttapub import taxonomy_lin as L

DEFAULT_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
LABELS = config.DERIVED / "labels_lin"
FIELD = {"tutor": "teacher", "student": "student"}
WORD = {"teacher": "Tutor", "student": "Student"}


def _stage02():
    """Stage 02 as a module: its session order and chat rendering are reused, never copied."""
    spec = importlib.util.spec_from_file_location(
        "stage02", Path(__file__).with_name("02_annotate_v3.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_units(text: str, unit: str) -> list[str]:
    if unit == "turn":
        return [text.strip()] if text.strip() else []
    return L.segment(text)


def number_units(turns: list[dict], role: str, unit: str) -> list[dict]:
    """Every target unit of one session, numbered through the session for this role."""
    out = []
    for i, t in enumerate(turns):
        for s, text in enumerate(split_units(t[FIELD[role]], unit)):
            out.append({"n": len(out) + 1, "i": i, "turn": t["turn"], "sent": s, "text": text})
    return out


def chunk_dialogue(sid: str, turns: list[dict], role: str, unit: str, max_turns: int,
                   max_units: int):
    """Yield ((session_id, {n: (turn, sent)}), user_prompt), per window.

    A chunk closes when it reaches `max_units` target units or spans `max_turns` turns. The
    window adds 2 turns of leading context, and only the target role's units carry numbers.
    """
    units = number_units(turns, role, unit)
    by_i: dict[int, list[dict]] = {}
    for u in units:
        by_i.setdefault(u["i"], []).append(u)
    pos = 0
    while pos < len(units):
        first_i = units[pos]["i"]
        end = pos
        while (end < len(units) and end - pos < max_units
               and units[end]["i"] - first_i < max_turns):
            end += 1
        take = units[pos:end]
        last_i = take[-1]["i"]
        lines = []
        for i in range(max(0, first_i - 2), last_i + 1):
            t = turns[i]
            for field in ("teacher", "student"):
                if not t[field]:
                    continue
                if field == FIELD[role]:
                    body = " ".join(f"[{u['n']}] {u['text']}" for u in by_i.get(i, []))
                else:
                    body = t[field]
                lines.append(f"{WORD[field]} Turn {t['turn']}: {body}")
        want = {u["n"]: (u["turn"], u["sent"]) for u in take}
        noun = "sentences" if unit == "sentence" else "turns"
        user = ("[DIALOGUE CHUNK]\n" + "\n".join(lines) +
                f"\n[END CHUNK]\n\nTag {role} {noun} numbered "
                f"{take[0]['n']} to {take[-1]['n']}. Return EXACTLY {len(take)} objects, one "
                f"for every number, in order. Do not skip any, including [unclear] ones.")
        yield (sid, want), user
        pos = end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--roles", default="tutor,student")
    ap.add_argument("--unit", default="sentence", choices=["sentence", "turn"])
    ap.add_argument("--sessions", type=int, default=0, help="total across shards; 0 = the split")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--chunk-turns", type=int, default=24)
    ap.add_argument("--chunk-units", type=int, default=40,
                    help="at most this many target units per chunk: ~20 tokens per object, so "
                         "40 sits well under --max-tokens")
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--fail-rate", type=float, default=0.05)
    ap.add_argument("--dry-run", type=int, default=0,
                    help="N: build prompts for N sessions, print one per role, load no model")
    a = ap.parse_args()

    stage02 = _stage02()
    roles = [r for r in a.roles.split(",") if r]
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    outdir = Path(a.outdir) if a.outdir else LABELS
    if outdir.resolve() == config.LABELS_V3.resolve():
        raise SystemExit("refusing to write Lin-scheme labels into data/derived/labels_v3")
    outdir.mkdir(parents=True, exist_ok=True)
    out = {r: outdir / f"lin_{a.tag}_{r}_shard{shard_i}.csv" for r in roles}

    specs = {}
    for r in roles:
        prompt = L.build_prompt(r, a.unit)
        n = L.verify_verbatim(r, prompt)
        specs[r] = (prompt, L.schema(r), L.codes(r))
        print(f"[{r}] {len(L.codes(r))} tags, {n} definitions verified verbatim against "
              f"{L.SPEC.name}", flush=True)

    done = None
    for r in roles:
        have = set(pd.read_csv(out[r], dtype={"session_id": str}).session_id.unique()) \
            if out[r].exists() else set()
        done = have if done is None else (done & have)
    order = stage02.session_order(a.split)
    if a.offset:
        order = order[a.offset:]
    if a.sessions:
        order = order[: a.sessions]
    order = [s for i, s in enumerate(order) if i % shard_n == shard_i]
    order = [s for s in order if s not in (done or set())]
    print(f"[lin/{a.tag} shard {shard_i}/{shard_n}] model={a.model} unit={a.unit} roles={roles} "
          f"split={a.split} sessions to annotate: {len(order):,}", flush=True)
    if not order:
        return

    store = data.store(a.split)

    if a.dry_run:
        n_units = n_prompts = 0
        shown: set[str] = set()
        biggest = 0
        for sid, tx in store.iter(order[: a.dry_run]):
            turns = dialogue.build_turns(tx)
            for role in roles:
                for (_, want), user in chunk_dialogue(sid, turns, role, a.unit, a.chunk_turns,
                                                      a.chunk_units):
                    n_prompts += 1
                    n_units += len(want)
                    biggest = max(biggest, len(specs[role][0]) + len(user))
                    if role not in shown:
                        print("=" * 72 + f"\n[SYSTEM: {role}]\n" + specs[role][0] +
                              f"\n{'=' * 72}\n[USER]\n{user}\n" + "=" * 72, flush=True)
                        shown.add(role)
        print(f"dry run: {a.dry_run} sessions, {n_units:,} {a.unit}s, {n_prompts:,} prompts "
              f"({n_prompts / max(1, a.dry_run):.1f} per session), largest prompt "
              f"{biggest:,} characters", flush=True)
        return

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype=a.dtype,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              max_num_seqs=a.max_num_seqs, **aj.engine_kwargs())
    sp = {r: aj.make_sampling_params(specs[r][1], a.max_tokens) for r in roles}
    audit = {r: aj.VocabAudit(specs[r][2]) for r in roles}
    n_chunks = {r: 0 for r in roles}
    n_fail = {r: 0 for r in roles}
    n_want = {r: 0 for r in roles}
    n_got = {r: 0 for r in roles}
    t0 = time.time()

    for bstart in range(0, len(order), a.block):
        sids = order[bstart:bstart + a.block]
        dialogues = {}
        for sid, tx in store.iter(sids):
            turns = dialogue.build_turns(tx)
            if turns:
                dialogues[sid] = turns

        prompts, metas, sps = [], [], []
        for role in roles:
            system = specs[role][0]
            for sid, turns in dialogues.items():
                for meta, user in chunk_dialogue(sid, turns, role, a.unit, a.chunk_turns,
                                                 a.chunk_units):
                    prompts.append(stage02._render(tok, system, user))
                    metas.append((role,) + meta)
                    sps.append(sp[role])
        outs = llm.generate(prompts, sps) if prompts else []

        rows = {r: [] for r in roles}
        for (role, sid, want), o in zip(metas, outs):
            n_chunks[role] += 1
            n_want[role] += len(want)
            obj = aj.extract_json(o.outputs[0].text)
            try:
                if obj is None:
                    raise ValueError("unparseable or truncated")
                seen: set[int] = set()
                for lab in (obj if isinstance(obj, list) else obj["labels"]):
                    n = int(lab["n"])
                    if n not in want or n in seen:
                        continue
                    keep = audit[role].check([lab["code"]])
                    if not keep:
                        continue
                    seen.add(n)
                    rows[role].append({"session_id": sid, "turn": want[n][0],
                                       "sent": want[n][1], "n": n, "code": keep[0]})
                n_got[role] += len(seen)
            except Exception:
                n_fail[role] += 1

        for role, rr in rows.items():
            if rr:
                pd.DataFrame(rr).to_csv(out[role], mode="a", header=not out[role].exists(),
                                        index=False)
        print(f"  sessions {min(bstart + a.block, len(order)):,}/{len(order):,}  "
              f"{time.time() - t0:.0f}s  " +
              "  ".join(f"{r}: chunks {n_chunks[r]:,} fail {n_fail[r]} "
                        f"labelled {n_got[r]:,}/{n_want[r]:,}" for r in roles), flush=True)

    seconds = time.time() - t0
    summary = {"tag": a.tag, "shard": a.shard, "split": a.split, "unit": a.unit, "model": a.model,
               "sessions": len(order), "seconds": round(seconds, 1),
               "seconds_per_session": round(seconds / max(1, len(order)), 3),
               "roles": {r: {"chunks": n_chunks[r], "parse_fail": n_fail[r],
                             "units_requested": n_want[r], "units_labelled": n_got[r],
                             "codes_seen": audit[r].seen, "off_vocabulary": audit[r].bad.total()}
                         for r in roles}}
    (outdir / f"lin_{a.tag}_shard{shard_i}_summary.json").write_text(json.dumps(summary, indent=1))

    bad = 0
    for role in roles:
        rate = n_fail[role] / max(1, n_chunks[role])
        print(f"DONE {role} shard {shard_i}/{shard_n}: chunks {n_chunks[role]:,}, "
              f"parse_fail {n_fail[role]} ({rate:.2%}), labelled "
              f"{n_got[role]:,}/{n_want[role]:,} -> {out[role]}", flush=True)
        bad += audit[role].report(f" [{role} shard {shard_i}]") + (rate > a.fail_rate)
    if bad:
        raise SystemExit("🛑 RUN IS NOT SHIPPABLE: parse failures over the threshold or "
                         "systematic vocabulary drift. Do not build features from this output.")


if __name__ == "__main__":
    main()
