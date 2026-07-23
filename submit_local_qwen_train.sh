#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CHOICE="${1:-7b}"

case "${MODEL_CHOICE,,}" in
  7b|qwen7b|qwen2.5-vl-7b)
    MODEL_PATH_DEFAULT="/gpfs/projects/p33222/ybq9740/models/Qwen2.5-VL-7B-Instruct"
    JOB_NAME_DEFAULT="tsg-qwen25vl-7b-ground-idx"
    ;;
  72b|qwen72b|qwen2.5-vl-72b)
    cat >&2 <<'EOF'
The current repository trains with single-process full-parameter HF Trainer.
It does not use LoRA, QLoRA, DeepSpeed, or FSDP.
Submitting the local 72B model with this script is very likely to fail on GPU memory.
If you want, I can help you build a separate 72B inference script or refactor training to LoRA/distributed mode.
EOF
    exit 1
    ;;
  *)
    echo "Usage: bash submit_local_qwen_train.sh [7b]" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_anomaly_db_indexed_text_plain_image.yaml}"
MODEL_PATH="${MODEL_PATH:-$MODEL_PATH_DEFAULT}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
RUN_TAG="${RUN_TAG:-qwen25vl_${MODEL_CHOICE}_grounding_indexed_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_TAG}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$ROOT_DIR/run_full_train_qwen.sbatch}"
MODE="${MODE:-train}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
CHECKPOINT_OUTPUT_DIR="${CHECKPOINT_OUTPUT_DIR:-}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"

PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:1}"
MEMORY="${MEMORY:-64G}"
CPUS="${CPUS:-8}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
JOB_NAME="${JOB_NAME:-$JOB_NAME_DEFAULT}"
CONSTRAINT="${CONSTRAINT:-}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Local model path not found: $MODEL_PATH" >&2
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
)

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  env_kv+=("RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT")
fi

if [[ -n "$CHECKPOINT_OUTPUT_DIR" ]]; then
  env_kv+=("CHECKPOINT_OUTPUT_DIR=$CHECKPOINT_OUTPUT_DIR")
fi

sbatch_cmd=(
  sbatch
  "--job-name=$JOB_NAME"
  "--partition=$PARTITION"
  "--gres=$GRES"
  "--mem=$MEMORY"
  "--cpus-per-task=$CPUS"
  "--time=$TIME_LIMIT"
  "$SBATCH_SCRIPT"
)
if [[ -n "$CONSTRAINT" ]]; then
  sbatch_cmd=(
    sbatch
    "--job-name=$JOB_NAME"
    "--partition=$PARTITION"
    "--constraint=$CONSTRAINT"
    "--gres=$GRES"
    "--mem=$MEMORY"
    "--cpus-per-task=$CPUS"
    "--time=$TIME_LIMIT"
    "$SBATCH_SCRIPT"
  )
fi

echo "Submitting training job with:"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  SOURCE_ROOT=$SOURCE_ROOT"
echo "  CONFIG_PATH=$CONFIG_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  PARTITION=$PARTITION"
echo "  GRES=$GRES"
echo "  CONSTRAINT=${CONSTRAINT:-none}"
echo "  MEMORY=$MEMORY"
echo "  CPUS=$CPUS"
echo "  TIME_LIMIT=$TIME_LIMIT"
echo "  BF16=$BF16"
echo "  FP16=$FP16"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
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
