#!/usr/bin/env bash
# Replay reproduce/runs.tsv in dependency order. A thin wrapper: the work is in run_all.py, and
# every option passes through (--jobs N, --all, --cpu-only, --only RUN, --dry-run, --force,
# --keep-going). PY selects the interpreter (default: python).
#
#   bash reproduce/run_all.sh --jobs 8
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PY:-python}" reproduce/run_all.py "$@"
