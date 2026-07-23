#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsg-q3vl8b-exam
#SBATCH --partition=gengpu
#SBATCH --constraint=quest12&sxm
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

DEFAULT_ROOT_DIR="/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong"
ROOT_DIR="${PROJECT_ROOT:-}"
if [[ -z "$ROOT_DIR" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/run_train_then_timeseries_exam_gpu.sh" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  else
    ROOT_DIR="$DEFAULT_ROOT_DIR"
  fi
fi

if [[ ! -f "$ROOT_DIR/run_train_then_timeseries_exam_gpu.sh" ]]; then
  echo "PROJECT_ROOT does not look like Training-model-yilong: $ROOT_DIR" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/.cache" "$ROOT_DIR/.tmp"

export CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_qwen3_vl_8b_anomaly_db_indexed_text_plain_image.yaml}"
export MODEL_PATH="${MODEL_PATH:-/gpfs/projects/p33222/ybq9740/models/Qwen3-VL-8B-Instruct}"
export SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
export PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
export BF16="${BF16:-1}"
export FP16="${FP16:-0}"
export MODE="${MODE:-train}"
export EXAM_DATA="${EXAM_DATA:-$ROOT_DIR/../TimeSeriesExam-main/output/round_3_folder/qa_dataset.json}"
export EXAM_SEED="${EXAM_SEED:-2026}"
export EXAM_MAX_NEW_TOKENS="${EXAM_MAX_NEW_TOKENS:-64}"
export EXAM_TEMPERATURE="${EXAM_TEMPERATURE:-0.0}"
export EXAM_DEVICE_MAP="${EXAM_DEVICE_MAP:-auto}"
export EXAM_TORCH_DTYPE="${EXAM_TORCH_DTYPE:-bfloat16}"
export EXAM_MAX_PIXELS="${EXAM_MAX_PIXELS:-131072}"

bash "$ROOT_DIR/run_train_then_timeseries_exam_gpu.sh"
