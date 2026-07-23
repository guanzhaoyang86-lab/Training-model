#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsg-qwen-train-eval
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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

export CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_full.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/vlm_qwen_train_eval_run}"
export MODE="${MODE:-train}"
export PYTHON_BIN="${PYTHON_BIN:-}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export SPLIT="${SPLIT:-test}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

bash "$ROOT_DIR/run_train_and_eval_gpu.sh"
