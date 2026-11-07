#!/bin/sh
# One Lin-scheme annotation shard per GPU, detached. launch_shards.sh with stage 50 in place of
# stage 02, so the v3 launcher stays exactly as it was. See its header.
#
#   sh scripts/launch_shards_lin.sh <N_GPUS> <TAG> <SESSIONS> [SPLITS] [EXTRA_ARGS]
#   sh scripts/launch_shards_lin.sh 3 pilot 50 train
#   sh scripts/launch_shards_lin.sh 3 full 0 "train test"
#
# Each split gets its own tag, lin_<TAG>_<split>_<role>_shard<i>.csv, because the resume logic
# reads the label file for that tag. Re-running the same command resumes each shard.
set -e
N=${1:?usage: launch_shards_lin.sh N_GPUS TAG SESSIONS [SPLITS] [EXTRA_ARGS]}
TAG=${2:?}
SESSIONS=${3:?}
SPLITS=${4:-train test}
EXTRA=${5:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOGS=$ROOT/logs
mkdir -p "$LOGS"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

i=0
while [ "$i" -lt "$N" ]; do
  CUDA_VISIBLE_DEVICES=$i nohup sh -c '
    for SPLIT in '"$SPLITS"'; do
      echo "=== shard '"$i"'/'"$N"' starting split $SPLIT at $(date) ==="
      python "'"$ROOT"'/scripts/50_annotate_lin.py" --tag "'"$TAG"'_$SPLIT" --split "$SPLIT" --sessions '"$SESSIONS"' --shard "'"$i"'/'"$N"'" --tp 1 '"$EXTRA"' || exit 1
    done
    echo "=== shard '"$i"' finished all splits at $(date) ==="
  ' > "$LOGS/lin_${TAG}_shard${i}.log" 2>&1 &
  echo "shard $i/$N -> GPU $i, splits [$SPLITS], log $LOGS/lin_${TAG}_shard${i}.log (pid $!)"
  i=$((i + 1))
done
