#!/bin/bash
#SBATCH --account=p33222
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=tsg_qwen25vl_7b_ground_idx
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ===== User-configurable options =====
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/p33222/ybq9740/models/Qwen2.5-VL-7B-Instruct}"
CONFIG_PATH="${CONFIG_PATH:-configs/vlm_7b_anomaly_db_indexed_text_plain_image.yaml}"
SOURCE_ROOT="${SOURCE_ROOT:-dataset/anomaly_db_v1}"
RUN_TAG="${RUN_TAG:-qwen25vl_7b_grounding_indexed_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$RUN_TAG}"
MODE="${MODE:-train}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
CHECKPOINT_OUTPUT_DIR="${CHECKPOINT_OUTPUT_DIR:-}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"

# Switching MODEL_PATH to the local 72B checkpoint will likely require
# a very different resource setup and probably code changes such as LoRA
# or distributed training. The current pipeline is single-process full
# parameter training.

ROOT_DIR="/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong"

cd "$ROOT_DIR"
mkdir -p logs outputs .cache .tmp

module purge || true

if command -v module >/dev/null 2>&1; then
  module load anaconda3 >/dev/null 2>&1 || module load miniconda3 >/dev/null 2>&1 || true
fi

set +u
source ~/.bashrc || true
set -u

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"
export PYTHONUNBUFFERED=1

export CONFIG_PATH="$ROOT_DIR/$CONFIG_PATH"
export OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
export MODE
export PYTHON_BIN
export MODEL_PATH
export SOURCE_ROOT="$ROOT_DIR/$SOURCE_ROOT"
export RESUME_FROM_CHECKPOINT
export BF16
export FP16

if [[ -n "$CHECKPOINT_OUTPUT_DIR" ]]; then
  export CHECKPOINT_OUTPUT_DIR
fi

echo "Job info:"
echo "  ROOT_DIR=$ROOT_DIR"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  CONFIG_PATH=$CONFIG_PATH"
echo "  SOURCE_ROOT=$SOURCE_ROOT"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  MODE=$MODE"
echo "  BF16=$BF16"
echo "  FP16=$FP16"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

bash "$ROOT_DIR/run_full_train_gpu.sh"
