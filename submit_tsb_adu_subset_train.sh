#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$ROOT_DIR/run_full_train_qwen.sbatch}"

SOURCE_DATASET_FILTER="${1:-${SOURCE_DATASET_FILTER:-}}"
if [[ -z "$SOURCE_DATASET_FILTER" ]]; then
  echo "Usage: bash submit_tsb_adu_subset_train.sh <SOURCE_DATASET>" >&2
  echo "Example: bash submit_tsb_adu_subset_train.sh NAB" >&2
  exit 1
fi

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
CONSTRAINT="${CONSTRAINT:-quest12&sxm}"
JOB_NAME="${JOB_NAME:-tsg-tsb-adu-${SOURCE_DATASET_FILTER,,}}"
GRES="${GRES:-gpu:1}"
MEMORY="${MEMORY:-96G}"
CPUS="${CPUS:-8}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
DEPENDENCY="${DEPENDENCY:-}"

PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_tsb_adu_windowed_indexed_text.yaml}"
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/p33222/ybq9740/models/Qwen2.5-VL-7B-Instruct}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
SOURCE_ROOT="${SOURCE_ROOT:-$SOURCE_ROOT_BASE/$SOURCE_DATASET_FILTER}"
RUN_TAG="${RUN_TAG:-qwen25vl_7b_tsb_adu_${SOURCE_DATASET_FILTER,,}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$RUN_TAG}"
MODE="${MODE:-train}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
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

mkdir -p "$OUTPUT_ROOT"
cd "$ROOT_DIR"

env_kv=(
  "PROJECT_ROOT=$ROOT_DIR"
  "CONFIG_PATH=$CONFIG_PATH"
  "OUTPUT_DIR=$OUTPUT_DIR"
  "MODE=$MODE"
  "PYTHON_BIN=$PYTHON_BIN"
  "MODEL_PATH=$MODEL_PATH"
  "SOURCE_ROOT=$SOURCE_ROOT"
  "TS_GROUNDER_SOURCE_DATASET_FILTER=$SOURCE_DATASET_FILTER"
  "BF16=$BF16"
  "FP16=$FP16"
  "LAUNCHER=python"
  "NUM_PROCS_PER_NODE=1"
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

if [[ -n "$DEPENDENCY" ]]; then
  sbatch_cmd+=("--dependency=$DEPENDENCY")
fi

sbatch_cmd+=("$SBATCH_SCRIPT")

if [[ "$PARSABLE" != "1" ]]; then
  echo "Submitting TSB-AD-U subset training job with:"
  echo "  SOURCE_DATASET_FILTER=$SOURCE_DATASET_FILTER"
  echo "  JOB_NAME=$JOB_NAME"
  echo "  SOURCE_ROOT=$SOURCE_ROOT"
  echo "  CONFIG_PATH=$CONFIG_PATH"
  echo "  OUTPUT_DIR=$OUTPUT_DIR"
  echo "  DEPENDENCY=$DEPENDENCY"
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
