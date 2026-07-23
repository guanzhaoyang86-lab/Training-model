#!/bin/bash
set -euo pipefail

DEFAULT_ROOT_DIR="/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong"
ROOT_DIR="${PROJECT_ROOT:-}"
if [[ -z "$ROOT_DIR" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/train.py" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  elif [[ -f "$DEFAULT_ROOT_DIR/train.py" ]]; then
    ROOT_DIR="$DEFAULT_ROOT_DIR"
  else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
fi

if [[ ! -f "$ROOT_DIR/train.py" ]]; then
  echo "PROJECT_ROOT does not look like Training-model-yilong: $ROOT_DIR" >&2
  exit 1
fi

pick_python() {
  local candidate
  local candidates=(
    "$ROOT_DIR/.venv/bin/python"
    "$ROOT_DIR/.venv_smoke/bin/python"
    "/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python"
    "/gpfs/projects/p33222/ybq9740/envs/anomllm/bin/python"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  return 1
}

cd "$ROOT_DIR"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_qwen3_vl_8b_anomaly_db_indexed_text_plain_image.yaml}"
RUN_TAG="${RUN_TAG:-qwen3_vl_8b_anomaly_db_indexed_text_plain_image_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/$RUN_TAG}"
MODE="${MODE:-train}"
PYTHON_BIN="${PYTHON_BIN:-}"
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/p33222/ybq9740/models/Qwen3-VL-8B-Instruct}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

EXAM_DATA="${EXAM_DATA:-$ROOT_DIR/../TimeSeriesExam-main/output/round_3_folder/qa_dataset.json}"
EXAM_OUTPUT="${EXAM_OUTPUT:-$OUTPUT_DIR/timeseries_exam_round3_predictions.json}"
EXAM_IMAGE_CACHE_DIR="${EXAM_IMAGE_CACHE_DIR:-$OUTPUT_DIR/timeseries_exam_round3_images}"
EXAM_SEED="${EXAM_SEED:-2026}"
EXAM_MAX_NEW_TOKENS="${EXAM_MAX_NEW_TOKENS:-64}"
EXAM_TEMPERATURE="${EXAM_TEMPERATURE:-0.0}"
EXAM_DEVICE_MAP="${EXAM_DEVICE_MAP:-auto}"
EXAM_TORCH_DTYPE="${EXAM_TORCH_DTYPE:-bfloat16}"
EXAM_MAX_PIXELS="${EXAM_MAX_PIXELS:-131072}"
EXAM_LIMIT="${EXAM_LIMIT:-}"
EXAM_ADD_QUESTION_HINT="${EXAM_ADD_QUESTION_HINT:-0}"
EXAM_ADD_CONCEPTS="${EXAM_ADD_CONCEPTS:-0}"

if [[ -z "$PYTHON_BIN" ]]; then
  if ! PYTHON_BIN="$(pick_python)"; then
    echo "No usable Python found. Set PYTHON_BIN explicitly." >&2
    exit 1
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

export PROJECT_ROOT="$ROOT_DIR"
export CONFIG_PATH
export OUTPUT_DIR
export MODE
export PYTHON_BIN
export MODEL_PATH
export SOURCE_ROOT
export BF16
export FP16
export PYTORCH_CUDA_ALLOC_CONF
export SKIP_TRAIN

if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "==== Stage 1/2: use existing anomaly grounding model ===="
  echo "Skipping training and reusing model: $OUTPUT_DIR/model"
  if [[ ! -f "$OUTPUT_DIR/model/config.json" ]]; then
    echo "Existing model directory is missing or incomplete: $OUTPUT_DIR/model" >&2
    exit 1
  fi
else
  echo "==== Stage 1/2: train anomaly grounding model ===="
  bash "$ROOT_DIR/run_full_train_gpu.sh"
fi

echo "==== Stage 2/2: answer TimeSeriesExam round 3 ===="
exam_cmd=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/evaluate_timeseries_exam.py"
  "--data-file" "$EXAM_DATA"
  "--model-path" "$OUTPUT_DIR/model"
  "--output" "$EXAM_OUTPUT"
  "--image-cache-dir" "$EXAM_IMAGE_CACHE_DIR"
  "--seed" "$EXAM_SEED"
  "--max-new-tokens" "$EXAM_MAX_NEW_TOKENS"
  "--temperature" "$EXAM_TEMPERATURE"
  "--device-map" "$EXAM_DEVICE_MAP"
  "--torch-dtype" "$EXAM_TORCH_DTYPE"
  "--max-pixels" "$EXAM_MAX_PIXELS"
  "--indexed-series-text"
)

if [[ -n "$EXAM_LIMIT" ]]; then
  exam_cmd+=("--limit" "$EXAM_LIMIT")
fi
if [[ "$EXAM_ADD_QUESTION_HINT" == "1" ]]; then
  exam_cmd+=("--add-question-hint")
fi
if [[ "$EXAM_ADD_CONCEPTS" == "1" ]]; then
  exam_cmd+=("--add-concepts")
fi

"${exam_cmd[@]}"

echo "Train + TimeSeriesExam evaluation finished."
echo "Run dir: $OUTPUT_DIR"
echo "Exam predictions: $EXAM_OUTPUT"
echo "Exam metrics: ${EXAM_OUTPUT%.json}.metrics.json"
