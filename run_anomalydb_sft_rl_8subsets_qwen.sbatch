#!/bin/bash
#SBATCH --account=p33222
#SBATCH --job-name=tsg-adb-8sub
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=sxm
#SBATCH --mem=180G
#SBATCH --time=1-18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:-}"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"

cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/.cache" "$ROOT_DIR/.tmp"

MODEL_KEY="${MODEL_KEY:-7b}"
case "${MODEL_KEY,,}" in
  7b|qwen25vl_7b|qwen2.5vl_7b)
    RUN_LABEL_DEFAULT="qwen25vl_7b"
    BASE_MODEL_PATH_DEFAULT="$WORKSPACE_ROOT/models/Qwen2.5-VL-7B-Instruct"
    ;;
  2b|qwen3_vl_2b)
    RUN_LABEL_DEFAULT="qwen3_vl_2b"
    BASE_MODEL_PATH_DEFAULT="$WORKSPACE_ROOT/models/Qwen3-VL-2B-Instruct"
    ;;
  4b|qwen3_vl_4b)
    RUN_LABEL_DEFAULT="qwen3_vl_4b"
    BASE_MODEL_PATH_DEFAULT="$WORKSPACE_ROOT/models/Qwen3-VL-4B-Instruct"
    ;;
  8b|8n|qwen3_vl_8b)
    RUN_LABEL_DEFAULT="qwen3_vl_8b"
    BASE_MODEL_PATH_DEFAULT="$WORKSPACE_ROOT/models/Qwen3-VL-8B-Instruct"
    ;;
  *)
    echo "Unsupported MODEL_KEY=$MODEL_KEY. Expected one of: 7b, 2b, 4b, 8b." >&2
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_qwen3_vl_8b_anomaly_db_indexed_text_plain_image.yaml}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$BASE_MODEL_PATH_DEFAULT}"
RUN_LABEL="${RUN_LABEL:-$RUN_LABEL_DEFAULT}"
ANOMALY_SOURCE_ROOT="${ANOMALY_SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
TSB_SOURCE_ROOT_BASE="${TSB_SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
EVAL_SUBSETS="${EVAL_SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/${RUN_LABEL}_anomalydb_sft_rl_8subsets_${SLURM_JOB_ID:-manual}}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$OUTPUT_DIR/rl}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
RL_NUM_GENERATIONS="${RL_NUM_GENERATIONS:-4}"
RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-256}"
RL_TEMPERATURE="${RL_TEMPERATURE:-0.7}"
RL_TOP_P="${RL_TOP_P:-0.9}"
RL_LEARNING_RATE="${RL_LEARNING_RATE:-1e-6}"
RL_NUM_TRAIN_EPOCHS="${RL_NUM_TRAIN_EPOCHS:-1}"
RL_KL_COEF="${RL_KL_COEF:-0.02}"
RL_REWARD_EVENT_F1_WEIGHT="${RL_REWARD_EVENT_F1_WEIGHT:-0.45}"
RL_REWARD_BOUNDARY_IOU_WEIGHT="${RL_REWARD_BOUNDARY_IOU_WEIGHT:-0.35}"
RL_REWARD_HALLUCINATION_PENALTY_WEIGHT="${RL_REWARD_HALLUCINATION_PENALTY_WEIGHT:-0.0}"
RL_SAVE_STEPS="${RL_SAVE_STEPS:-0}"
RL_OPTIMIZER="${RL_OPTIMIZER:-adafactor}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "CONFIG_PATH does not exist: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -d "$BASE_MODEL_PATH" ]]; then
  echo "BASE_MODEL_PATH does not exist: $BASE_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$ANOMALY_SOURCE_ROOT" ]]; then
  echo "ANOMALY_SOURCE_ROOT does not exist: $ANOMALY_SOURCE_ROOT" >&2
  exit 1
fi
for split in train val test; do
  if [[ ! -f "$ANOMALY_SOURCE_ROOT/$split.json" ]]; then
    echo "Missing anomaly_db_v1 $split.json under $ANOMALY_SOURCE_ROOT" >&2
    exit 1
  fi
done

read -r -a SELECTED_SUBSETS <<< "$EVAL_SUBSETS"
for subset in "${SELECTED_SUBSETS[@]}"; do
  subset_root="$TSB_SOURCE_ROOT_BASE/$subset"
  if [[ ! -d "$subset_root" ]]; then
    echo "Missing TSB subset dir: $subset_root" >&2
    exit 1
  fi
  for split in train val test; do
    if [[ ! -f "$subset_root/$split.json" ]]; then
      echo "Missing $split.json under $subset_root" >&2
      exit 1
    fi
  done
done

echo "=== AnomalyDB SFT+RL then 8-subset TSB test workflow ==="
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-manual}"
echo "MODEL_KEY=$MODEL_KEY"
echo "RUN_LABEL=$RUN_LABEL"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "ANOMALY_SOURCE_ROOT=$ANOMALY_SOURCE_ROOT"
echo "TSB_SOURCE_ROOT_BASE=$TSB_SOURCE_ROOT_BASE"
echo "EVAL_SUBSETS=${SELECTED_SUBSETS[*]}"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "RL_OUTPUT_DIR=$RL_OUTPUT_DIR"
echo "RL weights: event=$RL_REWARD_EVENT_F1_WEIGHT boundary=$RL_REWARD_BOUNDARY_IOU_WEIGHT hallucination=$RL_REWARD_HALLUCINATION_PENALTY_WEIGHT"
nvidia-smi || true

echo "=== Stage 1: SFT on anomaly_db_v1 train.json ==="
PROJECT_ROOT="$ROOT_DIR" \
CONFIG_PATH="$CONFIG_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
MODE=train \
PYTHON_BIN="$PYTHON_BIN" \
MODEL_PATH="$BASE_MODEL_PATH" \
SOURCE_ROOT="$ANOMALY_SOURCE_ROOT" \
BF16="$BF16" \
FP16="$FP16" \
LAUNCHER=python \
NUM_PROCS_PER_NODE=1 \
bash "$ROOT_DIR/run_full_train_gpu.sh"

echo "=== Stage 2: prepare TSB test JSONL and evaluate SFT model ==="
sft_eval_manifest="$OUTPUT_DIR/sft_eval_8subsets/metrics_manifest.tsv"
mkdir -p "$(dirname "$sft_eval_manifest")"
printf 'subset\tdataset_jsonl\tpredictions\tmetrics\n' > "$sft_eval_manifest"

for subset in "${SELECTED_SUBSETS[@]}"; do
  subset_root="$TSB_SOURCE_ROOT_BASE/$subset"
  eval_prepare_dir="$OUTPUT_DIR/eval_datasets/$subset"
  eval_dir="$OUTPUT_DIR/sft_eval_8subsets/$subset"

  echo "=== Prepare eval dataset: $subset ==="
  PROJECT_ROOT="$ROOT_DIR" \
  CONFIG_PATH="$OUTPUT_DIR/resolved_config.yaml" \
  OUTPUT_DIR="$eval_prepare_dir" \
  MODE=prepare_only \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL_PATH="$BASE_MODEL_PATH" \
  SOURCE_ROOT="$subset_root" \
  BF16="$BF16" \
  FP16="$FP16" \
  LAUNCHER=python \
  NUM_PROCS_PER_NODE=1 \
  bash "$ROOT_DIR/run_full_train_gpu.sh"

  dataset_jsonl="$eval_prepare_dir/dataset_cache/test.jsonl"
  predictions_path="$eval_dir/test_predictions.jsonl"
  metrics_path="$eval_dir/test_metrics.json"

  echo "=== Predict+eval SFT model subset=$subset split=test ==="
  PROJECT_ROOT="$ROOT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  RUN_DIR="$OUTPUT_DIR" \
  SPLIT=test \
  CONFIG_PATH="$OUTPUT_DIR/resolved_config.yaml" \
  MODEL_PATH="$OUTPUT_DIR/model" \
  DATASET_JSONL="$dataset_jsonl" \
  EVAL_DIR="$eval_dir" \
  PREDICTIONS_PATH="$predictions_path" \
  METRICS_PATH="$metrics_path" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  bash "$ROOT_DIR/run_predict_eval_gpu.sh"

  printf '%s\t%s\t%s\t%s\n' "$subset" "$dataset_jsonl" "$predictions_path" "$metrics_path" >> "$sft_eval_manifest"
done

echo "=== Stage 3: RL on the same prepared anomaly_db_v1 train.jsonl ==="
export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" \
"$PYTHON_BIN" -m ts_grounder.rl_train_grpo \
  --model_name_or_path "$OUTPUT_DIR/model" \
  --reference_model_path "$OUTPUT_DIR/model" \
  --train_file "$OUTPUT_DIR/dataset_cache/train.jsonl" \
  --image_root "$ROOT_DIR" \
  --output_dir "$RL_OUTPUT_DIR" \
  --num_generations "$RL_NUM_GENERATIONS" \
  --max_new_tokens "$RL_MAX_NEW_TOKENS" \
  --temperature "$RL_TEMPERATURE" \
  --top_p "$RL_TOP_P" \
  --learning_rate "$RL_LEARNING_RATE" \
  --num_train_epochs "$RL_NUM_TRAIN_EPOCHS" \
  --kl_coef "$RL_KL_COEF" \
  --reward_event_f1_weight "$RL_REWARD_EVENT_F1_WEIGHT" \
  --reward_boundary_iou_weight "$RL_REWARD_BOUNDARY_IOU_WEIGHT" \
  --reward_hallucination_penalty_weight "$RL_REWARD_HALLUCINATION_PENALTY_WEIGHT" \
  --save_steps "$RL_SAVE_STEPS" \
  --optimizer "$RL_OPTIMIZER"

echo "=== Stage 4: evaluate RL model on the same TSB test JSONL ==="
eval_manifest="$RL_OUTPUT_DIR/eval_8subsets/metrics_manifest.tsv"
mkdir -p "$(dirname "$eval_manifest")"
printf 'subset\tdataset_jsonl\tpredictions\tmetrics\n' > "$eval_manifest"

for subset in "${SELECTED_SUBSETS[@]}"; do
  eval_prepare_dir="$OUTPUT_DIR/eval_datasets/$subset"
  eval_dir="$RL_OUTPUT_DIR/eval_8subsets/$subset"
  dataset_jsonl="$eval_prepare_dir/dataset_cache/test.jsonl"
  predictions_path="$eval_dir/test_predictions.jsonl"
  metrics_path="$eval_dir/test_metrics.json"

  if [[ ! -f "$dataset_jsonl" ]]; then
    echo "Missing prepared eval dataset for $subset: $dataset_jsonl" >&2
    exit 1
  fi

  echo "=== Predict+eval RL model subset=$subset split=test ==="
  PROJECT_ROOT="$ROOT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  RUN_DIR="$RL_OUTPUT_DIR" \
  SPLIT=test \
  CONFIG_PATH="$OUTPUT_DIR/resolved_config.yaml" \
  MODEL_PATH="$RL_OUTPUT_DIR/model" \
  DATASET_JSONL="$dataset_jsonl" \
  EVAL_DIR="$eval_dir" \
  PREDICTIONS_PATH="$predictions_path" \
  METRICS_PATH="$metrics_path" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  bash "$ROOT_DIR/run_predict_eval_gpu.sh"

  printf '%s\t%s\t%s\t%s\n' "$subset" "$dataset_jsonl" "$predictions_path" "$metrics_path" >> "$eval_manifest"
done

echo "=== Workflow finished ==="
echo "SFT model retained at: $OUTPUT_DIR/model"
echo "RL model retained at: $RL_OUTPUT_DIR/model"
echo "SFT 8-subset eval manifest: $sft_eval_manifest"
echo "RL 8-subset eval manifest: $eval_manifest"
