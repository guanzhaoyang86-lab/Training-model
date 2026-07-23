#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$ROOT_DIR/run_full_train_qwen.sbatch}"

# =========================
# Slurm resources
# =========================
ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
CONSTRAINT="${CONSTRAINT:-quest12&sxm}"
JOB_NAME="${JOB_NAME:-tsg-qwen25vl-7b-fsdp}"
GRES="${GRES:-gpu:2}"
MEMORY="${MEMORY:-400G}"
CPUS="${CPUS:-16}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
QOS="${QOS:-}"

# =========================
# Distributed training
# =========================
LAUNCHER="${LAUNCHER:-torchrun}"
NUM_PROCS_PER_NODE="${NUM_PROCS_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29600}"

# =========================
# Training settings
# =========================
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_full_fsdp_2gpu.yaml}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE_ROOT/models/Qwen2.5-VL-7B-Instruct}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
RUN_TAG="${RUN_TAG:-qwen25vl_7b_fsdp_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_TAG}"
MODE="${MODE:-train}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
CHECKPOINT_OUTPUT_DIR="${CHECKPOINT_OUTPUT_DIR:-}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DRY_RUN="${DRY_RUN:-0}"

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
  "LAUNCHER=$LAUNCHER"
  "NUM_PROCS_PER_NODE=$NUM_PROCS_PER_NODE"
  "MASTER_PORT=$MASTER_PORT"
  "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
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

if [[ -n "$CONSTRAINT" ]]; then
  sbatch_cmd+=("--constraint=$CONSTRAINT")
fi

if [[ -n "$QOS" ]]; then
  sbatch_cmd+=("--qos=$QOS")
fi

sbatch_cmd+=("$SBATCH_SCRIPT")

echo "Submitting 7B FSDP training job with:"
echo "  ACCOUNT=$ACCOUNT"
echo "  PARTITION=$PARTITION"
echo "  CONSTRAINT=$CONSTRAINT"
echo "  JOB_NAME=$JOB_NAME"
echo "  GRES=$GRES"
echo "  MEMORY=$MEMORY"
echo "  CPUS=$CPUS"
echo "  TIME_LIMIT=$TIME_LIMIT"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  SOURCE_ROOT=$SOURCE_ROOT"
echo "  CONFIG_PATH=$CONFIG_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MODE=$MODE"
echo "  LAUNCHER=$LAUNCHER"
echo "  NUM_PROCS_PER_NODE=$NUM_PROCS_PER_NODE"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  BF16=$BF16"
echo "  FP16=$FP16"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

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
