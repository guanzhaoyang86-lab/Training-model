#!/bin/bash
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-}"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/vlm_qwen_train_eval_run}"
SPLIT="${SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

echo "==== Stage 1/2: train ===="
bash "$ROOT_DIR/run_full_train_gpu.sh"

echo "==== Stage 2/2: predict + eval ===="
export RUN_DIR="$OUTPUT_DIR"
export CONFIG_PATH="$OUTPUT_DIR/resolved_config.yaml"
export MODEL_PATH="$OUTPUT_DIR/model"
export DATASET_JSONL="$OUTPUT_DIR/dataset_cache/$SPLIT.jsonl"
export EVAL_DIR="$OUTPUT_DIR/eval"
export PREDICTIONS_PATH="$EVAL_DIR/${SPLIT}_predictions.jsonl"
export METRICS_PATH="$EVAL_DIR/${SPLIT}_metrics.json"
export MAX_NEW_TOKENS

bash "$ROOT_DIR/run_predict_eval_gpu.sh"

echo "Train + predict + eval finished."
echo "Run dir: $OUTPUT_DIR"
echo "Predictions: $PREDICTIONS_PATH"
echo "Metrics: $METRICS_PATH"
