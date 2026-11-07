#!/usr/bin/env python3
"""Replay reproduce/runs.tsv in dependency order, several runs at a time.

    python reproduce/run_all.py                    # every run the paper depends on (paper == 1)
    python reproduce/run_all.py --jobs 8           # up to 8 runs at once, each as soon as its inputs exist
    python reproduce/run_all.py --all              # every recorded run, exploratory ones included
    python reproduce/run_all.py --cpu-only         # skip GPU rows
    python reproduce/run_all.py --only arm-both    # one run (its inputs must already exist)
    python reproduce/run_all.py --dry-run          # print the commands, run nothing
    python reproduce/run_all.py --force            # re-run rows whose metrics.json already exists
    python reproduce/run_all.py --keep-going       # a failed run skips its dependants, not everything
    python reproduce/run_all.py --venvs .venv39 .venv312   # each row in the environment it was made in
    python reproduce/run_all.py --platform here    # only rows made on this OS and CPU architecture
    python reproduce/run_all.py --gpus 0,1,2,3     # one GPU row per card, CPU rows alongside

The interpreter that runs this script runs every command (``python`` in a command is replaced by
it), unless --venvs names environments: then each row runs in the one whose python (major.minor), numpy,
pandas, scikit-learn, scipy and xgboost versions equal the row's ``versions_used``, and in the
closest one when none does exactly (scikit-learn and xgboost count most, since fitted models
are pickled with them), listed as "no matching environment" in the summary. A run whose output (runs.tsv's last column) exists is skipped unless --force, so an
interrupted replay resumes. Logs: results/_replay_logs/<run>.log. Many runs take ``--n-jobs``
themselves, so ``--jobs`` times that is the number of cores in use.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform as _platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def done(row: dict) -> bool:
    """A row is finished when its output file exists (runs.tsv's `output` column)."""
    run = row["run"]
    if run == "stage02-annotate":
        return any((ROOT / "data/derived/labels_v3").glob("v3_full_*_shard*.csv"))
    if run == "stage50-annotate-lin":
        return any((ROOT / "data/derived/labels_lin").glob("*.csv"))
    return (ROOT / (row.get("output") or f"results/{run}/metrics.json")).exists()


CORE = ("python", "numpy", "pandas", "scikit-learn", "scipy", "xgboost")
#: matched too for GPU rows; CPU rows often record torch only because it was installed
GPU_KEYS = ("torch", "tabpfn")
_PROBE = ("import json, platform, importlib.metadata as m\n"
          "out = {'python': platform.python_version()}\n"
          "for k in %r:\n"
          "    try: out[k] = m.version(k)\n"
          "    except m.PackageNotFoundError: pass\n"
          "print(json.dumps(out))" % (CORE[1:] + GPU_KEYS,))


def env_versions(py: str) -> dict:
    out = subprocess.run([py, "-c", _PROBE], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def row_versions(row: dict) -> dict:
    kv = dict(x.split("=", 1) for x in row.get("versions_used", "").split() if "=" in x)
    keys = CORE + (GPU_KEYS if row.get("device") == "gpu" else ())
    return {k: kv[k] for k in keys if k in kv}


def pick_python(row: dict, envs: dict[str, dict], default: str) -> tuple[str, bool]:
    """The interpreter whose core versions equal the row's; (default, False) if none does."""
    want = row_versions(row)
    if not want:
        return default, True  # stages with no record (data check, turn tables): any environment
    def same(k: str, have: str | None, v: str) -> bool:  # python: major.minor is enough
        if k == "python" and have:
            return have.split(".")[:2] == v.split(".")[:2]
        if k == "torch" and have:  # the CUDA build suffix (+cu128) is not always recorded
            return have.split("+")[0] == v.split("+")[0]
        return have == v
    for py, have in envs.items():
        if all(same(k, have.get(k), v) for k, v in want.items()):
            return py, True
    # no exact match: the closest environment, scikit-learn and xgboost weighted most because
    # fitted models are pickled with them and do not load across their versions
    weight = {"scikit-learn": 3, "xgboost": 3}
    score = {py: sum(weight.get(k, 1) for k, v in want.items() if same(k, have.get(k), v))
             for py, have in envs.items()}
    best = max(score, key=score.get) if score else default
    return (best if score.get(best, 0) > 0 else default), False


def this_platform() -> str:
    """This machine as runs.tsv's `platform` column writes it."""
    s, m = _platform.system(), _platform.machine().lower()
    arch = "arm64" if m in ("arm64", "aarch64") else "x86_64" if m in ("x86_64", "amd64") else m
    return f"{'macOS' if s == 'Darwin' else s}-{arch}"


def shell_line(row: dict, py: str) -> str:
    """The row's command with every `python` that starts a command (after &&, ;, `do`) set to
    `py`; the command itself starts without the interpreter."""
    q = shlex.quote(py)
    cmd = re.sub(r"(&& |; |\bdo )python ", lambda m: f"{m.group(1)}{q} ", row["command"])
    return f"{row['env'] + ' ' if row['env'] else ''}{q} {cmd}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--gpus", default="",
                    help="comma-separated CUDA device ids; each GPU row gets one to itself")
    ap.add_argument("--platform", help="only rows recorded on this platform (runs.tsv's "
                    "`platform` column), e.g. macOS-arm64; 'here' means this machine's")
    ap.add_argument("--venvs", nargs="*", default=[],
                    help="virtual environments (directories) to choose from per row")
    a = ap.parse_args()
    envs = {}
    for v in a.venvs:
        py = str(Path(v).resolve() / "bin" / "python")
        envs[py] = env_versions(py)
        print(f"-- environment {v}: " + " ".join(f"{k}={envs[py].get(k, '-')}" for k in CORE))

    rows = list(csv.DictReader(open(ROOT / "reproduce/runs.tsv"), delimiter="\t"))
    by = {r["run"]: r for r in rows}
    if a.only:
        todo = [by[a.only]]
    else:
        todo = [r for r in rows if a.all or r["paper"] == "1"]
    if a.platform:
        want = this_platform() if a.platform == "here" else a.platform
        todo = [r for r in todo if r.get("platform") == want]
        print(f"-- {len(todo)} rows recorded on {want}; their other inputs must already exist")
    # with --cpu-only, a GPU row that has not run is unavailable, and so is every run that
    # reads it (directly or not): those are skipped, never run on missing inputs
    unavailable = set()
    if a.cpu_only:
        unavailable = {r["run"] for r in todo if r["device"] == "gpu" and not done(r)}
        for n in sorted(unavailable):
            print(f"-- skip (gpu) {n}")
        todo = [r for r in todo if r["run"] not in unavailable]

    pending = {}
    for r in todo:
        if not a.force and done(r):
            print(f"-- done {r['run']}")
            continue
        if r["command"].startswith("see README"):
            if a.dry_run:
                print(f"!! {r['run']} needs the GPU annotation first: {r['command']}")
                continue
            sys.exit(f"!! {r['run']} needs the GPU annotation: {r['command']}")
        pending[r["run"]] = r
    in_set = set(pending) | unavailable
    deps = {n: {d for d in r["depends_on"].split(",") if d in in_set} for n, r in pending.items()}
    # an input outside this replay (--only, --platform, or already done) must exist on disk
    missing = {n: sorted(d for d in filter(None, r["depends_on"].split(","))
                         if d not in in_set and d in by and not done(by[d]))
               for n, r in pending.items()}
    for n in [n for n, m in missing.items() if m]:
        print(f"!! skip {n}: input not built: {', '.join(missing[n][:4])}")
        pending.pop(n)
        unavailable.add(n)

    chosen = {n: pick_python(r, envs, sys.executable) if envs else (sys.executable, True)
              for n, r in pending.items()}
    unmatched = sorted(n for n, (_, ok) in chosen.items() if not ok)
    if envs:
        print(f"-- {len(pending) - len(unmatched)} rows in a matching environment, "
              f"{len(unmatched)} in the closest one")

    if a.dry_run:
        for n, r in pending.items():
            tag = "" if chosen[n][1] else f"   [no matching environment; closest: {chosen[n][0]}]"
            print(f"== [{r['order']}] {n}{tag}\n   {shell_line(r, chosen[n][0])}")
        return

    logs = ROOT / "results/_replay_logs"
    logs.mkdir(parents=True, exist_ok=True)
    finished, failed, blocked = set(), set(), set(unavailable)
    running: dict[str, tuple[subprocess.Popen, float]] = {}
    free_gpus = [g for g in a.gpus.split(",") if g]
    holds: dict[str, str] = {}  # run -> the card it holds
    t0 = time.time()
    while pending or running:
        for n in [n for n in pending if deps[n] & (failed | blocked)]:
            blocked.add(n)
            pending.pop(n)
            why = "needs a GPU run" if deps[n] & unavailable else "an input run failed"
            print(f"!! skip {n}: {why}")
        ready = [n for n in pending if deps[n] <= finished]
        if free_gpus or holds:  # with --gpus, a GPU row waits for a free card
            ready = [n for n in ready if pending[n]["device"] != "gpu" or free_gpus]
        while ready and len(running) < a.jobs and not (failed and not a.keep_going):
            n = ready.pop(0)
            if pending[n]["device"] == "gpu" and (free_gpus or holds):
                if not free_gpus:
                    continue
                holds[n] = free_gpus.pop(0)
            r = pending.pop(n)
            line = shell_line(r, chosen[n][0])
            if n in holds:
                line = f"CUDA_VISIBLE_DEVICES={holds[n]} {line}"
            print(f"== [{r['order']}] {n}  ({time.time() - t0:.0f}s)", flush=True)
            fh = open(logs / f"{n}.log", "w")
            fh.write(line + "\n\n")
            fh.flush()
            running[n] = (subprocess.Popen(["bash", "-c", line], cwd=ROOT, stdout=fh,
                                           stderr=subprocess.STDOUT), time.time())
        if failed and not a.keep_going and not running:
            break
        if not running and pending and not [n for n in pending if deps[n] <= finished]:
            sys.exit(f"!! nothing runnable; waiting on {sorted(pending)[:5]}")
        time.sleep(1)
        for n, (p, ts) in list(running.items()):
            if p.poll() is None:
                continue
            running.pop(n)
            if n in holds:
                free_gpus.append(holds.pop(n))
            if p.returncode == 0:
                finished.add(n)
                print(f"   ok {n} ({time.time() - ts:.0f}s)", flush=True)
            else:
                failed.add(n)
                print(f"!! {n} failed (exit {p.returncode}); see {logs / (n + '.log')}", flush=True)
    print(f"\n{len(finished)} ok, {len(failed)} failed, {len(blocked - unavailable)} skipped (input failed or GPU), "
          f"{len(pending)} not started; {time.time() - t0:.0f}s")
    if unmatched:
        print(f"{len(unmatched)} rows ran with no matching environment; expect small differences:",
              *unmatched, sep="\n  ")
    if failed:
        print("failed:", *sorted(failed), sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
