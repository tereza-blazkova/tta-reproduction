#!/usr/bin/env python
"""Stage 02 — the v3 LLM annotation pass (vLLM, guided JSON). **This is the GPU-box stage.**

Labels every tutor turn with one of the 41 tutor codes and every student turn with one of the
16 student codes, from `docs/taxonomy_v3.md` via :mod:`ttapub.taxonomy_v3`. Design choices that
are load-bearing rather than stylistic:

* **Both pools in one `generate()` call.** vLLM accepts a per-request `SamplingParams`, so the
  two schemas coexist in one batch. Every checkpoint therefore holds both pools for every
  session it covers. Running the passes back to back instead leaves an interrupted block with
  tutor labels and no student twin, which is unusable for any paired analysis — and it halves
  the batch width, so it is slower too.
* **24-turn chunks with 2 turns of leading context**, and both speakers rendered into every
  window: the student pass sees the tutor's turns as context and vice versa, because a turn's
  function is only visible in the exchange.
* **Greedy decode.** The annotation is a measurement instrument; a sampled label would make the
  same turn get a different code on a re-run and put decoder noise into every rate.
* **Deterministic session order** (objective-diverse first, then a seeded shuffle), so the
  pilot's N sessions are a prefix of the full run's order and the pilot is not a different
  sample from the corpus.
* **Resume** counts a session as done only when *every* requested role has it.
* **Refuses to exit 0** on a degraded run (parse failures over the threshold, or systematic
  off-vocabulary output), so a half-labelled corpus cannot quietly reach the distiller.

Pilot, then the full pass (spec §7):

    # 1. pilot: ~50 sessions, for per-code prevalence, human agreement and `also` pairs
    python scripts/02_annotate_v3.py --model Qwen/Qwen3.5-122B --sessions 50 --tag pilot --tp 4

    # 2. full pass, sharded over GPUs; one process per shard
    python scripts/02_annotate_v3.py --model Qwen/Qwen3.5-122B --sessions 3000 \
        --tag full --shard 0/4 --tp 2

Out: `data/derived/labels_v3/v3_{tag}_{role}_shard{i}.csv` with
`session_id, turn, code, also` — the join key is `(session_id, turn)` of the stage-01 table.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from ttapub import anno_json as aj
from ttapub import config, data, dialogue
from ttapub import taxonomy_v3 as T

SESSION_ORDER_SEED = 221


def session_order(split: str, seed: int = SESSION_ORDER_SEED) -> list[str]:
    """Objective-diverse first, then a seeded shuffle of the rest.

    One session per objective comes first so that even a truncated run covers the curriculum;
    the remainder is shuffled with a fixed seed so the order is identical on every machine and
    a pilot of N sessions is a prefix of the full run.
    """
    feats = data.load_train() if split == "train" else data.load_test()
    first = feats.groupby("learning_objective_id", group_keys=False).head(1)
    sids = list(dict.fromkeys(first.session_id))
    rest = [s for s in dict.fromkeys(feats.session_id) if s not in set(sids)]
    random.Random(seed).shuffle(rest)
    return sids + rest


def _render(tok, system: str, user: str) -> str:
    """Apply the tokenizer's chat template, with the reasoning toggle off where it exists.

    `enable_thinking=False` suppresses Qwen3's reasoning block, which the annotation does not
    want (it would spend the token budget on prose and risk truncating the JSON). Other
    families' templates have no such argument and some reject unknown kwargs, so the panel
    models fall back to a plain render — and the fallback is tried per call rather than
    assumed, because a template that silently ignores the flag is indistinguishable from one
    that honours it.
    """
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def chunk_dialogue(sid: str, turns: list[dict], role: str, chunk: int,
                   enforce_coverage: bool = False):
    """Yield ((session_id, wanted_turns), user_prompt) per <=`chunk`-turn window."""
    field = "teacher" if role == "tutor" else "student"
    for s in range(0, len(turns), chunk):
        lo = max(0, s - 2)                      # 2 turns of leading context
        window = turns[lo:s + chunk]
        want = [t["turn"] for t in turns[s:s + chunk] if t.get(field, "").strip()]
        if not want:
            continue
        lines = []
        for t in window:
            if t["teacher"]:
                lines.append(f"Tutor Turn {t['turn']}: {t['teacher']}")
            if t["student"]:
                lines.append(f"Student Turn {t['turn']}: {t['student']}")
        user = ("[DIALOGUE CHUNK]\n" + "\n".join(lines) +
                f"\n[END CHUNK]\n\nClassify {role} turns {want}.")
        if enforce_coverage:
            # Measured on Mistral-Small-24B: under the plain instruction it returned valid JSON
            # with objects for only 52% of the requested tutor turns — a silent omission the
            # chunk-level parse check cannot see. Qwen3 models need no such reminder, so this is
            # OFF by default: it would otherwise change the prompt for every annotator and make
            # the panel members incomparable.
            user += (f" Return EXACTLY {len(want)} objects — one for every listed turn, in "
                     f"order, including backchannels and [unclear] turns. Do not skip any turn "
                     f"and do not merge turns.")
        yield (sid, set(want)), user


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("ANNO_MODEL", "Qwen/Qwen3-32B"))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--roles", default="tutor,student")
    ap.add_argument("--sessions", type=int, default=0,
                    help="total sessions to cover across all shards (0 = the whole split); "
                         "this shard takes every Nth of them")
    ap.add_argument("--shard", default="0/1", help="i/N: this process takes every Nth session")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip this many sessions of the deterministic order first. Round 2 of "
                         "the pilot needs a FRESH sample (spec §7), so it runs with "
                         "--offset <round-1 size>; the full pass uses offset 0.")
    ap.add_argument("--tag", default="full", help="pilot | full | an enrichment name")
    ap.add_argument("--chunk-turns", type=int, default=24)
    ap.add_argument("--block", type=int, default=48, help="sessions per generate() call")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=512,
                    help="Qwen3.5 is hybrid-Mamba: vLLM's default 1024 exceeds the ~882 Mamba "
                         "cache blocks that fit beside FP8 weights on one B200 and the engine "
                         "dies at init. 512 is the known-good value.")
    ap.add_argument("--no-also", action="store_true",
                    help="drop the optional second code (the full pass may not need it)")
    ap.add_argument("--no-null-also", action="store_true",
                    help="use a NONE sentinel instead of a nullable union, for backends that "
                         "will not compile the union")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--fail-rate", type=float, default=0.05)
    ap.add_argument("--enforce-coverage", action="store_true",
                    help="append an explicit one-object-per-turn demand to each chunk. OFF by "
                         "default: it changes the prompt, so a run using it is not comparable "
                         "with one that does not. Use it to test whether a model's turn "
                         "omissions are a prompt-portability problem or a capability one.")
    ap.add_argument("--dry-run", type=int, default=0,
                    help="N: build prompts for N sessions and print the first one, without "
                         "loading vLLM. Use this before shipping to the GPU box.")
    a = ap.parse_args()

    roles = [r for r in a.roles.split(",") if r]
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    outdir = Path(a.outdir) if a.outdir else config.LABELS_V3
    outdir.mkdir(parents=True, exist_ok=True)
    out = {r: outdir / f"v3_{a.tag}_{r}_shard{shard_i}.csv" for r in roles}

    # Prompts and schemas, verified against the spec before the model is loaded: a drifted
    # prompt is worth catching in the first second, not after an hour of GPU time.
    specs = {}
    for r in roles:
        prompt = T.build_prompt(r, also=not a.no_also)
        n = T.verify_verbatim(r, prompt)
        schema = T.schema(r, also=not a.no_also, null_also=not a.no_null_also)
        specs[r] = (prompt, schema, T.codes(r))
        print(f"[{r}] {len(T.codes(r))} codes, {n} definitions verified verbatim "
              f"against {T.SPEC.name}", flush=True)

    done = None
    for r in roles:
        have = set(pd.read_csv(out[r]).session_id.unique()) if out[r].exists() else set()
        done = have if done is None else (done & have)
    # truncate the corpus FIRST, then shard it: `--sessions 3000` means 3,000 sessions in
    # total whatever the GPU count, so the covered set does not change when a shard is added
    # or a card is lost.
    order = session_order(a.split)
    if a.offset:
        order = order[a.offset:]
    if a.sessions:
        order = order[: a.sessions]
    order = [s for i, s in enumerate(order) if i % shard_n == shard_i]
    order = [s for s in order if s not in (done or set())]
    print(f"[v3/{a.tag} shard {shard_i}/{shard_n}] model={a.model} roles={roles} split={a.split} "
          f"sessions to annotate: {len(order):,}", flush=True)
    if not order:
        return

    if a.dry_run:
        store = data.store(a.split)
        n_turns = n_prompts = 0
        shown = False
        for sid, tx in store.iter(order[: a.dry_run]):
            turns = dialogue.build_turns(tx)
            n_turns += len(turns)
            for role in roles:
                for (_, want), user in chunk_dialogue(sid, turns, role, a.chunk_turns,
                                                      a.enforce_coverage):
                    n_prompts += 1
                    if not shown:
                        print("=" * 72 + f"\n[SYSTEM: {role}]\n" + specs[role][0] +
                              f"\n{'=' * 72}\n[USER]\n{user}\n" + "=" * 72, flush=True)
                        shown = True
        print(f"dry run: {a.dry_run} sessions, {n_turns:,} paired turns, {n_prompts:,} prompts "
              f"({n_prompts / max(1, a.dry_run):.1f} per session, both pools)", flush=True)
        return

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(a.model)
    # one GPU per shard (tp=1) is the shape every corpus-scale pass here has used: the model
    # fits on one card, so GPUs buy throughput by sharding, not capacity.
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype=a.dtype,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              max_num_seqs=a.max_num_seqs, **aj.engine_kwargs())
    sp = {r: aj.make_sampling_params(specs[r][1], a.max_tokens) for r in roles}
    audit = {r: aj.VocabAudit(specs[r][2]) for r in roles}

    store = data.store(a.split)
    n_chunks = {r: 0 for r in roles}
    n_fail = {r: 0 for r in roles}
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
                for meta, user in chunk_dialogue(sid, turns, role, a.chunk_turns,
                                                 a.enforce_coverage):
                    prompts.append(_render(tok, system, user))
                    metas.append((role,) + meta)
                    sps.append(sp[role])
        outs = llm.generate(prompts, sps) if prompts else []

        rows = {r: [] for r in roles}
        for (role, sid, want), o in zip(metas, outs):
            n_chunks[role] += 1
            obj = aj.extract_json(o.outputs[0].text)
            try:
                if obj is None:
                    raise ValueError("unparseable or truncated")
                for lab in (obj if isinstance(obj, list) else obj["labels"]):
                    turn = int(lab["turn"])
                    if turn not in want:
                        continue                 # a label for a turn we did not ask about
                    keep = audit[role].check([lab["code"]])
                    if not keep:
                        continue                 # unknown primary code => no usable row
                    also = lab.get("also")
                    also = ("" if not also or also == T.NONE_SENTINEL
                            else (audit[role].check([also]) or [""])[0])
                    rows[role].append({"session_id": sid, "turn": turn,
                                       "code": keep[0], "also": also})
            except Exception:
                n_fail[role] += 1

        # both pools written together, so a checkpoint is never half a session
        for role, rr in rows.items():
            if rr:
                pd.DataFrame(rr).to_csv(out[role], mode="a", header=not out[role].exists(),
                                        index=False)
        print(f"  sessions {min(bstart + a.block, len(order)):,}/{len(order):,}  "
              f"{time.time() - t0:.0f}s  " +
              "  ".join(f"{r}: chunks {n_chunks[r]:,} fail {n_fail[r]}" for r in roles),
              flush=True)

    bad = 0
    for role in roles:
        rate = n_fail[role] / max(1, n_chunks[role])
        print(f"DONE {role} shard {shard_i}/{shard_n}: chunks {n_chunks[role]:,}, "
              f"parse_fail {n_fail[role]} ({rate:.2%}) -> {out[role]}", flush=True)
        bad += audit[role].report(f" [{role} shard {shard_i}]") + (rate > a.fail_rate)
    if bad:
        raise SystemExit("🛑 RUN IS NOT SHIPPABLE: parse failures over the threshold or "
                         "systematic vocabulary drift. Do not distil this output.")


if __name__ == "__main__":
    main()
