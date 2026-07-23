#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$ROOT_DIR/run_train_and_eval_qwen.sbatch}"

MODEL_KEY="${MODEL_KEY:-8b}"

case "$MODEL_KEY" in
  2b)
    DEFAULT_JOB_NAME="tsg-qwen3vl-2b"
    DEFAULT_MEMORY="140G"
    DEFAULT_TIME_LIMIT="12:00:00"
    DEFAULT_CONFIG_PATH="$ROOT_DIR/configs/vlm_qwen3_vl_2b_single_a100_try.yaml"
    DEFAULT_MODEL_PATH="$WORKSPACE_ROOT/models/Qwen3-VL-2B-Instruct"
    DEFAULT_RUN_PREFIX="qwen3_vl_2b_single_a100"
    ;;
  4b)
    DEFAULT_JOB_NAME="tsg-qwen3vl-4b"
    DEFAULT_MEMORY="180G"
    DEFAULT_TIME_LIMIT="16:00:00"
    DEFAULT_CONFIG_PATH="$ROOT_DIR/configs/vlm_qwen3_vl_4b_single_a100_try.yaml"
    DEFAULT_MODEL_PATH="$WORKSPACE_ROOT/models/Qwen3-VL-4B-Instruct"
    DEFAULT_RUN_PREFIX="qwen3_vl_4b_single_a100"
    ;;
  8b)
    DEFAULT_JOB_NAME="tsg-qwen3vl-8b"
    DEFAULT_MEMORY="240G"
    DEFAULT_TIME_LIMIT="24:00:00"
    DEFAULT_CONFIG_PATH="$ROOT_DIR/configs/vlm_qwen3_vl_8b_single_a100_try.yaml"
    DEFAULT_MODEL_PATH="$WORKSPACE_ROOT/models/Qwen3-VL-8B-Instruct"
    DEFAULT_RUN_PREFIX="qwen3_vl_8b_single_a100"
    ;;
  *)
    echo "Unsupported MODEL_KEY: $MODEL_KEY (expected one of: 2b, 4b, 8b)" >&2
    exit 1
    ;;
esac

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
CONSTRAINT="${CONSTRAINT:-quest12&sxm}"
JOB_NAME="${JOB_NAME:-$DEFAULT_JOB_NAME}"
GRES="${GRES:-gpu:1}"
MEMORY="${MEMORY:-$DEFAULT_MEMORY}"
CPUS="${CPUS:-8}"
TIME_LIMIT="${TIME_LIMIT:-$DEFAULT_TIME_LIMIT}"
QOS="${QOS:-}"
DEPENDENCY="${DEPENDENCY:-}"

PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$DEFAULT_CONFIG_PATH}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
RUN_TAG="${RUN_TAG:-${DEFAULT_RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_TAG}"
MODE="${MODE:-train}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
CHECKPOINT_OUTPUT_DIR="${CHECKPOINT_OUTPUT_DIR:-}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SPLIT="${SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DRY_RUN="${DRY_RUN:-0}"
PARSABLE="${PARSABLE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Dataset path not found: $SOURCE_ROOT" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Slurm script not found: $SBATCH_SCRIPT" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/outputs"
cd "$ROOT_DIR"

env_kv=(
  "PROJECT_ROOT=$ROOT_DIR"
  "CONFIG_PATH=$CONFIG_PATH"
  "OUTPUT_DIR=$OUTPUT_DIR"
  "MODE=$MODE"
  "PYTHON_BIN=$PYTHON_BIN"
  "MODEL_PATH=$MODEL_PATH"
  "SOURCE_ROOT=$SOURCE_ROOT"
  "BF16=$BF16"
  "FP16=$FP16"
  "LAUNCHER=python"
  "NUM_PROCS_PER_NODE=1"
  "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
  "SPLIT=$SPLIT"
  "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  env_kv+=("RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT")
fi

if [[ -n "$CHECKPOINT_OUTPUT_DIR" ]]; then
  env_kv+=("CHECKPOINT_OUTPUT_DIR=$CHECKPOINT_OUTPUT_DIR")
fi

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
  echo "Submitting train+test job with:"
  echo "  MODEL_KEY=$MODEL_KEY"
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
  echo "  MODEL_PATH=$MODEL_PATH"
  echo "  SOURCE_ROOT=$SOURCE_ROOT"
  echo "  CONFIG_PATH=$CONFIG_PATH"
  echo "  OUTPUT_DIR=$OUTPUT_DIR"
  echo "  SPLIT=$SPLIT"
  echo "  MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
  echo "  BF16=$BF16"
  echo "  FP16=$FP16"
  echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
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
