"""Run directories and provenance, so every number in the manuscript traces to a commit.

A number may enter the paper only when it comes from a committed run in this repo, and the
manuscript cites the run directory. That is only enforceable if each run records what produced
it, so :func:`start` writes `provenance.json` (argv, git commit and dirty flag, host, python,
key package versions, input file digests) before any fitting happens, and :func:`finish` writes
`metrics.json` beside it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import config


def git_state() -> dict:
    def q(*a):
        try:
            return subprocess.run(a, cwd=config.ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return ""
    return {"commit": q("git", "rev-parse", "HEAD"),
            "branch": q("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(q("git", "status", "--porcelain"))}


def digest(path: Path, limit: int = 64 << 20) -> str:
    """sha256 of a file, or of its first `limit` bytes for a large one (prefixed `partial:`)."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
            if n >= limit:
                return "partial:" + h.hexdigest()[:32]
    return h.hexdigest()[:32]


def start(name: str, args: dict | None = None, inputs: list[Path] | None = None) -> Path:
    """Create `results/<name>/`, write provenance.json, return the directory."""
    d = config.RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    prov = {"run": name, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "argv": sys.argv, "args": args or {}, "git": git_state(),
            "annotator": config.ANNOTATOR or "122B",
            "host": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "packages": _versions(),
            "inputs": {str(p.relative_to(config.ROOT)) if p.is_absolute() else str(p):
                       digest(p) for p in (inputs or []) if Path(p).exists()}}
    (d / "provenance.json").write_text(json.dumps(prov, indent=1))
    if prov["git"]["dirty"]:
        print("⚠️  working tree is dirty: this run is not reproducible from a commit", flush=True)
    return d


def _versions() -> dict:
    out = {}
    # the last ones exist only where a foundation-model baseline runs; absent ones are skipped
    for mod in ("numpy", "pandas", "scikit-learn", "xgboost", "scipy", "torch", "tabpfn"):
        try:
            from importlib.metadata import version
            out[mod] = version(mod)
        except Exception:
            pass
    return out


def finish(d: Path, metrics: dict) -> Path:
    p = d / "metrics.json"
    p.write_text(json.dumps(metrics, indent=1, default=float))
    print(f"wrote {p}", flush=True)
    return p
