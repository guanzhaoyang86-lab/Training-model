#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsg-qwen-full
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-}"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/.cache" "$ROOT_DIR/.tmp"

export CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_anomaly_db_indexed_text_plain_image.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/vlm_qwen25vl_7b_grounding_indexed_run}"
export MODE="${MODE:-train}"
export PYTHON_BIN="${PYTHON_BIN:-}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

bash "$ROOT_DIR/run_full_train_gpu.sh"
