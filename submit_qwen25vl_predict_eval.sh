#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$ROOT_DIR/run_predict_eval_qwen.sbatch}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
CONSTRAINT="${CONSTRAINT:-quest12&sxm}"
JOB_NAME="${JOB_NAME:-tsg-qwen25vl-eval}"
GRES="${GRES:-gpu:1}"
MEMORY="${MEMORY:-180G}"
CPUS="${CPUS:-8}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
QOS="${QOS:-}"
DEPENDENCY="${DEPENDENCY:-}"

PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/outputs/qwen25vl_7b_single_a100_20260411_155921}"
SPLIT="${SPLIT:-test}"
CONFIG_PATH="${CONFIG_PATH:-$RUN_DIR/resolved_config.yaml}"
MODEL_PATH="${MODEL_PATH:-$RUN_DIR/model}"
DATASET_JSONL="${DATASET_JSONL:-$RUN_DIR/dataset_cache/$SPLIT.jsonl}"
EVAL_DIR="${EVAL_DIR:-$RUN_DIR/eval}"
PREDICTIONS_PATH="${PREDICTIONS_PATH:-$EVAL_DIR/${SPLIT}_predictions.jsonl}"
METRICS_PATH="${METRICS_PATH:-$EVAL_DIR/${SPLIT}_metrics.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DRY_RUN="${DRY_RUN:-0}"
PARSABLE="${PARSABLE:-0}"
ALLOW_MISSING_RUN_ARTIFACTS="${ALLOW_MISSING_RUN_ARTIFACTS:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$RUN_DIR" && "$ALLOW_MISSING_RUN_ARTIFACTS" != "1" ]]; then
  echo "RUN_DIR not found: $RUN_DIR" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" && "$ALLOW_MISSING_RUN_ARTIFACTS" != "1" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" && "$ALLOW_MISSING_RUN_ARTIFACTS" != "1" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$DATASET_JSONL" && "$ALLOW_MISSING_RUN_ARTIFACTS" != "1" ]]; then
  echo "Dataset jsonl not found: $DATASET_JSONL" >&2
  exit 1
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Slurm script not found: $SBATCH_SCRIPT" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/outputs" "$EVAL_DIR"
cd "$ROOT_DIR"

env_kv=(
  "PROJECT_ROOT=$ROOT_DIR"
  "PYTHON_BIN=$PYTHON_BIN"
  "RUN_DIR=$RUN_DIR"
  "SPLIT=$SPLIT"
  "CONFIG_PATH=$CONFIG_PATH"
  "MODEL_PATH=$MODEL_PATH"
  "DATASET_JSONL=$DATASET_JSONL"
  "EVAL_DIR=$EVAL_DIR"
  "PREDICTIONS_PATH=$PREDICTIONS_PATH"
  "METRICS_PATH=$METRICS_PATH"
  "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
)

sbatch_cmd=(
  sbatch
  "--account=$ACCOUNT"
  "--partition=$PARTITION"
  "--job-name=$JOB_NAME"
  "--gres=$GRES"
  "--mem=$MEMORY"
  "--cpus-per-task=$CPUS"
  "--time=$TIME_LIMIT"
)

if [[ "$PARSABLE" == "1" ]]; then
  sbatch_cmd+=("--parsable")
fi

if [[ -n "$CONSTRAINT" ]]; then
  sbatch_cmd+=("--constraint=$CONSTRAINT")
fi

if [[ -n "$QOS" ]]; then
  sbatch_cmd+=("--qos=$QOS")
fi

if [[ -n "$DEPENDENCY" ]]; then
  sbatch_cmd+=("--dependency=$DEPENDENCY")
fi

sbatch_cmd+=("$SBATCH_SCRIPT")

if [[ "$PARSABLE" != "1" ]]; then
  echo "Submitting VLM predict+eval job with:"
  echo "  ACCOUNT=$ACCOUNT"
  echo "  PARTITION=$PARTITION"
  echo "  CONSTRAINT=$CONSTRAINT"
  echo "  JOB_NAME=$JOB_NAME"
  echo "  GRES=$GRES"
  echo "  MEMORY=$MEMORY"
  echo "  CPUS=$CPUS"
  echo "  TIME_LIMIT=$TIME_LIMIT"
  echo "  DEPENDENCY=$DEPENDENCY"
  echo "  PYTHON_BIN=$PYTHON_BIN"
  echo "  RUN_DIR=$RUN_DIR"
  echo "  SPLIT=$SPLIT"
  echo "  CONFIG_PATH=$CONFIG_PATH"
  echo "  MODEL_PATH=$MODEL_PATH"
  echo "  DATASET_JSONL=$DATASET_JSONL"
  echo "  EVAL_DIR=$EVAL_DIR"
  echo "  PREDICTIONS_PATH=$PREDICTIONS_PATH"
  echo "  METRICS_PATH=$METRICS_PATH"
  echo "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1, not submitting."
  printf 'env'
  for kv in "${env_kv[@]}"; do
    printf ' %q' "$kv"
  done
  for arg in "${sbatch_cmd[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  exit 0
fi

env "${env_kv[@]}" "${sbatch_cmd[@]}"
