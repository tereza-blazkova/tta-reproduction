#!/bin/sh
# Launch one annotation shard per GPU, detached, each pinned to its own card.
#
#   sh scripts/launch_shards.sh <N_GPUS> <TAG> <SESSIONS> [MODEL] [SPLITS] [EXTRA_ARGS]
#   sh scripts/launch_shards.sh 3 full 0 "" "train test"     # 0 = every session of each split
#   sh scripts/launch_shards.sh 2 panelm4full 0 mistralai/Mistral-Small-4-119B-2603 "train test" --enforce-coverage
#
# EXTRA_ARGS is passed verbatim to 02_annotate_v3.py (e.g. --enforce-coverage, which the
# Mistral-Small-4 pilot needed to label every turn: 18%/53% -> 96.8%/99.0% coverage).
#
# Splits run SEQUENTIALLY within each shard, so one launch covers both and nothing needs
# babysitting between them. Each split gets its own tag ("<TAG>_train", "<TAG>_test") because
# the label file is keyed by tag: sharing one tag would make the resume logic treat test
# sessions as already-done train sessions.
#
# Why one process per GPU with tp=1 rather than one tensor-parallel job: Qwen3.5-122B-A10B-FP8
# fits on a single B200 (183 GB), so GPUs buy throughput by sharding the session list, not
# capacity.
# Tensor parallelism would add collective overhead and, at odd GPU counts, would not even be
# valid (tp must divide the attention-head count).
#
# Each shard writes its own CSV, so shards never contend, and `--shard i/N` takes every Nth
# session from one deterministic order: the union is the same set a single process would cover.
# Resume is per shard — re-running the same command picks up where a killed shard stopped.
set -e
N=${1:?usage: launch_shards.sh N_GPUS TAG SESSIONS [MODEL]}
TAG=${2:?}
SESSIONS=${3:?}
MODEL=${4:-Qwen/Qwen3.5-122B-A10B-FP8}
MODEL=${MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}
SPLITS=${5:-train test}
EXTRA=${6:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOGS=$ROOT/logs
mkdir -p "$LOGS"

# Applies to every Qwen3.5 vLLM job: without spawn, EngineCore forks and dies with
# "Cannot re-initialize CUDA".
# Every one of these is a no-nvcc workaround measured on this image:
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # else EngineCore forks: "Cannot re-initialize CUDA"
export VLLM_USE_DEEP_GEMM=0                 # DeepGEMM JITs FP8 MoE kernels; needs a toolkit
export VLLM_MOE_USE_DEEP_GEMM=0

i=0
while [ "$i" -lt "$N" ]; do
  CUDA_VISIBLE_DEVICES=$i nohup sh -c '
    for SPLIT in '"$SPLITS"'; do
      echo "=== shard '"$i"'/'"$N"' starting split $SPLIT at $(date) ==="
      python "'"$ROOT"'/scripts/02_annotate_v3.py"         --model "'"$MODEL"'" --tag "'"$TAG"'_$SPLIT" --split "$SPLIT"         --sessions '"$SESSIONS"' --shard "'"$i"'/'"$N"'" --tp 1 '"$EXTRA"' || exit 1
    done
    echo "=== shard '"$i"' finished all splits at $(date) ==="
  ' > "$LOGS/anno_${TAG}_shard${i}.log" 2>&1 &
  echo "shard $i/$N -> GPU $i, splits [$SPLITS], log $LOGS/anno_${TAG}_shard${i}.log (pid $!)"
  i=$((i + 1))
done
echo
echo "progress:  tail -f $LOGS/anno_${TAG}_shard*.log"
echo "labels so far:  wc -l $ROOT/data/derived/labels_v3/v3_${TAG}_*_shard*.csv"
echo "NOTE: SESSIONS is the TOTAL across shards, so the covered set is the same at any GPU"
echo "      count: 3000 means 3,000 sessions whether you run 2 cards or 5."
