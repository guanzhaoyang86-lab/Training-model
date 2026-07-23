#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$ROOT_DIR/submit_qwen3_vl_train_eval.sh}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
CONSTRAINT="${CONSTRAINT:-quest12&sxm}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
QOS="${QOS:-}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SPLIT="${SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DRY_RUN="${DRY_RUN:-0}"

MEMORY_2B="${MEMORY_2B:-140G}"
MEMORY_4B="${MEMORY_4B:-180G}"
MEMORY_8B="${MEMORY_8B:-240G}"
TIME_LIMIT_2B="${TIME_LIMIT_2B:-12:00:00}"
TIME_LIMIT_4B="${TIME_LIMIT_4B:-16:00:00}"
TIME_LIMIT_8B="${TIME_LIMIT_8B:-24:00:00}"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Submit script not found: $SUBMIT_SCRIPT" >&2
  exit 1
fi

submit_one() {
  local model_key="$1"
  local memory_var="$2"
  local time_var="$3"

  echo "==== submitting $model_key ===="
  ACCOUNT="$ACCOUNT" \
  PARTITION="$PARTITION" \
  CONSTRAINT="$CONSTRAINT" \
  GRES="$GRES" \
  CPUS="$CPUS" \
  QOS="$QOS" \
  PYTHON_BIN="$PYTHON_BIN" \
  SOURCE_ROOT="$SOURCE_ROOT" \
  BF16="$BF16" \
  FP16="$FP16" \
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  SPLIT="$SPLIT" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  MEMORY="$memory_var" \
  TIME_LIMIT="$time_var" \
  MODEL_KEY="$model_key" \
  DRY_RUN="$DRY_RUN" \
  bash "$SUBMIT_SCRIPT"
}

submit_one 2b "$MEMORY_2B" "$TIME_LIMIT_2B"
submit_one 4b "$MEMORY_4B" "$TIME_LIMIT_4B"
submit_one 8b "$MEMORY_8B" "$TIME_LIMIT_8B"
