#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
SBATCH_SCRIPT="$ROOT_DIR/run_full_train_qwen.sbatch"
LOG_DIR="$ROOT_DIR/logs"

# =========================
# Slurm resources
# =========================
ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
JOB_NAME="${JOB_NAME:-tsg-qwen25vl-72b}"
GRES="${GRES:-gpu:1}"
MEMORY="${MEMORY:-128G}"
CPUS="${CPUS:-8}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

# Optional Slurm settings
QOS="${QOS:-}"
CONSTRAINT="${CONSTRAINT:-}"

# =========================
# Training settings
# =========================
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_full.yaml}"
MODEL_PATH="${MODEL_PATH:-$WORKSPACE_ROOT/models/Qwen2.5-VL-72B-Instruct}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
RUN_TAG="${RUN_TAG:-qwen25vl_72b_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_TAG}"
MODE="${MODE:-train}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
CHECKPOINT_OUTPUT_DIR="${CHECKPOINT_OUTPUT_DIR:-}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"

# Cache / temp paths
HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"

# Safety / debugging
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

mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/.cache" "$ROOT_DIR/.tmp" "$LOG_DIR"
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
  "HF_HOME=$HF_HOME"
  "TMPDIR=$TMPDIR"
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
  "--output=$LOG_DIR/%x-%j.out"
  "--error=$LOG_DIR/%x-%j.err"
)

if [[ -n "$QOS" ]]; then
  sbatch_cmd+=("--qos=$QOS")
fi

if [[ -n "$CONSTRAINT" ]]; then
  sbatch_cmd+=("--constraint=$CONSTRAINT")
fi

sbatch_cmd+=("$SBATCH_SCRIPT")

echo "Submitting Qwen2.5-VL-72B training job with:"
echo "  ACCOUNT=$ACCOUNT"
echo "  PARTITION=$PARTITION"
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
echo "  BF16=$BF16"
echo "  FP16=$FP16"
echo "  LOG_DIR=$LOG_DIR"

cat <<'EOF'
Note:
  The current repository uses single-process Hugging Face Trainer.
  Changing GRES to request multiple GPUs only reserves more GPUs; it does not
  by itself enable distributed training, tensor parallelism, LoRA, or QLoRA.
  Full-parameter 72B training is therefore very likely to hit GPU memory limits
  unless the training code is refactored for distributed or parameter-efficient training.
EOF

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
