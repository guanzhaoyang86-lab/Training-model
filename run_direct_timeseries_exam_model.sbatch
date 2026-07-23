#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsqa-direct
#SBATCH --partition=gengpu
#SBATCH --constraint=quest12&sxm
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

DEFAULT_ROOT_DIR="/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong"
ROOT_DIR="${PROJECT_ROOT:-$DEFAULT_ROOT_DIR}"
MODELS_DIR="${MODELS_DIR:-/gpfs/projects/p33222/ybq9740/models}"

if [[ ! -f "$ROOT_DIR/scripts/evaluate_timeseries_exam.py" ]]; then
  echo "PROJECT_ROOT does not look like Training-model-yilong: $ROOT_DIR" >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the model directory name under MODELS_DIR.}"
MODEL_PATH="${MODEL_PATH:-$MODELS_DIR/$MODEL_NAME}"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model path not found: $MODEL_PATH" >&2
  exit 1
fi

safe_model_name="${MODEL_NAME//\//__}"
safe_model_name="${safe_model_name// /_}"

BATCH_ID="${BATCH_ID:-direct_timeseries_exam_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/direct_timeseries_exam_models/$BATCH_ID}"
RUN_DIR="${RUN_DIR:-$OUTPUT_ROOT/$safe_model_name}"
EXAM_OUTPUT="${EXAM_OUTPUT:-$RUN_DIR/timeseries_exam_round3_predictions.json}"
EXAM_IMAGE_CACHE_DIR="${EXAM_IMAGE_CACHE_DIR:-$RUN_DIR/timeseries_exam_round3_images}"
STATUS_FILE="$RUN_DIR/status.tsv"

PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
EXAM_DATA="${EXAM_DATA:-$ROOT_DIR/../TimeSeriesExam-main/output/round_3_folder/qa_dataset.json}"
EXAM_SEED="${EXAM_SEED:-2026}"
EXAM_MAX_NEW_TOKENS="${EXAM_MAX_NEW_TOKENS:-64}"
EXAM_TEMPERATURE="${EXAM_TEMPERATURE:-0.0}"
EXAM_DEVICE_MAP="${EXAM_DEVICE_MAP:-auto}"
EXAM_TORCH_DTYPE="${EXAM_TORCH_DTYPE:-bfloat16}"
EXAM_MAX_PIXELS="${EXAM_MAX_PIXELS:-131072}"
EXAM_LIMIT="${EXAM_LIMIT:-}"
EXAM_INDEXED_SERIES_TEXT="${EXAM_INDEXED_SERIES_TEXT:-1}"
EXAM_ADD_QUESTION_HINT="${EXAM_ADD_QUESTION_HINT:-0}"
EXAM_ADD_CONCEPTS="${EXAM_ADD_CONCEPTS:-0}"
HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$EXAM_DATA" ]]; then
  echo "EXAM_DATA not found: $EXAM_DATA" >&2
  exit 1
fi

mkdir -p "$RUN_DIR" "$EXAM_IMAGE_CACHE_DIR" "$ROOT_DIR/logs" "$HF_HOME" "$TMPDIR"

start_time="$(date +%Y-%m-%dT%H:%M:%S%z)"
write_status() {
  local exit_code="$1"
  local state="completed"
  if [[ "$exit_code" != "0" ]]; then
    state="failed"
  fi
  {
    printf 'model_name\t%s\n' "$MODEL_NAME"
    printf 'model_path\t%s\n' "$MODEL_PATH"
    printf 'run_dir\t%s\n' "$RUN_DIR"
    printf 'predictions\t%s\n' "$EXAM_OUTPUT"
    printf 'metrics\t%s\n' "${EXAM_OUTPUT%.json}.metrics.json"
    printf 'slurm_job_id\t%s\n' "${SLURM_JOB_ID:-local}"
    printf 'state\t%s\n' "$state"
    printf 'exit_code\t%s\n' "$exit_code"
    printf 'start_time\t%s\n' "$start_time"
    printf 'end_time\t%s\n' "$(date +%Y-%m-%dT%H:%M:%S%z)"
  } > "$STATUS_FILE"
}
trap 'code=$?; write_status "$code"; exit "$code"' EXIT

cd "$ROOT_DIR"
export PROJECT_ROOT="$ROOT_DIR"
export HF_HOME
export TMPDIR
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "==== Direct TimeSeriesExam QA ===="
echo "Model name: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Exam data: $EXAM_DATA"
echo "Run dir: $RUN_DIR"
echo "Predictions: $EXAM_OUTPUT"
echo "Metrics: ${EXAM_OUTPUT%.json}.metrics.json"
echo "Slurm job id: ${SLURM_JOB_ID:-local}"

exam_cmd=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/evaluate_timeseries_exam.py"
  "--data-file" "$EXAM_DATA"
  "--model-path" "$MODEL_PATH"
  "--output" "$EXAM_OUTPUT"
  "--image-cache-dir" "$EXAM_IMAGE_CACHE_DIR"
  "--seed" "$EXAM_SEED"
  "--max-new-tokens" "$EXAM_MAX_NEW_TOKENS"
  "--temperature" "$EXAM_TEMPERATURE"
  "--device-map" "$EXAM_DEVICE_MAP"
  "--torch-dtype" "$EXAM_TORCH_DTYPE"
  "--max-pixels" "$EXAM_MAX_PIXELS"
)

if [[ "$EXAM_INDEXED_SERIES_TEXT" == "1" ]]; then
  exam_cmd+=("--indexed-series-text")
else
  exam_cmd+=("--no-indexed-series-text")
fi
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

echo "Direct TimeSeriesExam QA finished for $MODEL_NAME."
