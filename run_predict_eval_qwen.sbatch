#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsg-qwen-eval
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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

export PYTHON_BIN="${PYTHON_BIN:-}"
export RUN_DIR="${RUN_DIR:-$ROOT_DIR/outputs/qwen25vl_7b_single_a100_20260411_155921}"
export SPLIT="${SPLIT:-test}"
export CONFIG_PATH="${CONFIG_PATH:-$RUN_DIR/resolved_config.yaml}"
export MODEL_PATH="${MODEL_PATH:-$RUN_DIR/model}"
export DATASET_JSONL="${DATASET_JSONL:-$RUN_DIR/dataset_cache/$SPLIT.jsonl}"
export EVAL_DIR="${EVAL_DIR:-$RUN_DIR/eval}"
export PREDICTIONS_PATH="${PREDICTIONS_PATH:-$EVAL_DIR/${SPLIT}_predictions.jsonl}"
export METRICS_PATH="${METRICS_PATH:-$EVAL_DIR/${SPLIT}_metrics.json}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

bash "$ROOT_DIR/run_predict_eval_gpu.sh"
