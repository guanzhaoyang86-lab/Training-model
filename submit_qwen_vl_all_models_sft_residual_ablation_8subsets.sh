#!/bin/bash
set -euo pipefail

# Quest submitter for the full model matrix:
#   Models: Qwen2.5-VL-7B, Qwen3-VL-2B, Qwen3-VL-4B, Qwen3-VL-8B
#   Stage 1: synthetic SFT, 1 epoch
#   Stage 2: real-subset SFT, 3 epochs on each of the 8 TSB-AD-U subsets
#   Full branch: residual mining + boundary-aware GRPO, 1 epoch, then test
#   Ablation branch: no residual mining, direct boundary-aware GRPO, 1 epoch, then test
#
# The script removes model/checkpoint weights after evaluation and keeps metrics,
# logs, manifests, and summaries.

ROOT_DIR="${PROJECT_ROOT:-/gpfs/projects/p33222/ybq9740/Thesis/Training-model-new}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/gpfs/projects/p33222/ybq9740}"

PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/envs/anomamind/bin/python}"
SYNTHETIC_SOURCE_ROOT="${SYNTHETIC_SOURCE_ROOT:-/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong/dataset/anomaly_db_v1}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"

MODELS="${MODELS:-7b 2b 4b 8b}"
SUBSETS="${SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"
VARIANTS="${VARIANTS:-full_residual no_residual}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
CPUS="${CPUS:-8}"
QOS="${QOS:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
STAGE1_MEMORY="${STAGE1_MEMORY:-140G}"
SUBSET_MEMORY="${SUBSET_MEMORY:-140G}"
STAGE1_TIME_LIMIT="${STAGE1_TIME_LIMIT:-06:00:00}"
CLEANUP_TIME_LIMIT="${CLEANUP_TIME_LIMIT:-00:10:00}"
CLEANUP_PARTITION="${CLEANUP_PARTITION:-$PARTITION}"
CLEANUP_GRES="${CLEANUP_GRES:-$GRES}"
CLEANUP_CONSTRAINT="${CLEANUP_CONSTRAINT:-$CONSTRAINT}"

TINY_TRAIN_LIMIT="${TINY_TRAIN_LIMIT:-30}"
SMALL_TRAIN_LIMIT="${SMALL_TRAIN_LIMIT:-80}"
MEDIUM_TRAIN_LIMIT="${MEDIUM_TRAIN_LIMIT:-200}"
LARGE_TRAIN_LIMIT="${LARGE_TRAIN_LIMIT:-500}"
TINY_TIME_LIMIT="${TINY_TIME_LIMIT:-01:00:00}"
SMALL_TIME_LIMIT="${SMALL_TIME_LIMIT:-02:00:00}"
MEDIUM_TIME_LIMIT="${MEDIUM_TIME_LIMIT:-04:00:00}"
LARGE_TIME_LIMIT="${LARGE_TIME_LIMIT:-08:00:00}"
XL_TIME_LIMIT="${XL_TIME_LIMIT:-16:00:00}"

STAGE1_SFT_NUM_TRAIN_EPOCHS="${STAGE1_SFT_NUM_TRAIN_EPOCHS:-1}"
STAGE2_SFT_NUM_TRAIN_EPOCHS="${STAGE2_SFT_NUM_TRAIN_EPOCHS:-3}"
RL_NUM_TRAIN_EPOCHS="${RL_NUM_TRAIN_EPOCHS:-1}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
RESIDUAL_BATCH_SIZE="${RESIDUAL_BATCH_SIZE:-1}"
RESIDUAL_MAX_SAMPLES="${RESIDUAL_MAX_SAMPLES:-}"
TAU_MATCH="${TAU_MATCH:-0.1}"
TAU_GOOD="${TAU_GOOD:-0.5}"
SERIES_LENGTH="${SERIES_LENGTH:-256}"
MAX_INDEX="${MAX_INDEX:-255}"

RL_NUM_GENERATIONS="${RL_NUM_GENERATIONS:-4}"
RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-256}"
RL_TEMPERATURE="${RL_TEMPERATURE:-0.7}"
RL_TOP_P="${RL_TOP_P:-0.9}"
RL_LEARNING_RATE="${RL_LEARNING_RATE:-1e-6}"
RL_KL_COEF="${RL_KL_COEF:-0.02}"
RL_SAVE_STEPS="${RL_SAVE_STEPS:-0}"
RL_OPTIMIZER="${RL_OPTIMIZER:-adafactor}"
RL_MAX_SAMPLES="${RL_MAX_SAMPLES:-}"
RL_RESIDUAL_POOL_EPOCH_SIZE="${RL_RESIDUAL_POOL_EPOCH_SIZE:-}"
RESIDUAL_POOL_SAMPLING_RATIOS="${RESIDUAL_POOL_SAMPLING_RATIOS:-false_negative=0.3,boundary_error=0.3,false_positive=0.2,correct_abnormal=0.1,correct_normal=0.1}"

REWARD_EVENT_WEIGHT="${REWARD_EVENT_WEIGHT:-0.45}"
REWARD_IOU_WEIGHT="${REWARD_IOU_WEIGHT:-0.35}"
REWARD_BOUNDARY_WEIGHT="${REWARD_BOUNDARY_WEIGHT:-0.15}"
REWARD_TYPE_WEIGHT="${REWARD_TYPE_WEIGHT:-0.05}"

SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
KEEP_RESIDUAL_POOL="${KEEP_RESIDUAL_POOL:-0}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
DRY_RUN="${DRY_RUN:-0}"

model_label() {
  case "${1,,}" in
    7b|qwen25vl_7b|qwen2.5vl_7b) echo "qwen25vl_7b" ;;
    2b|qwen3_vl_2b) echo "qwen3_vl_2b" ;;
    4b|qwen3_vl_4b) echo "qwen3_vl_4b" ;;
    8b|qwen3_vl_8b) echo "qwen3_vl_8b" ;;
    *) echo "Unsupported model key: $1" >&2; return 1 ;;
  esac
}

model_path() {
  case "${1,,}" in
    7b|qwen25vl_7b|qwen2.5vl_7b) echo "$WORKSPACE_ROOT/models/Qwen2.5-VL-7B-Instruct" ;;
    2b|qwen3_vl_2b) echo "$WORKSPACE_ROOT/models/Qwen3-VL-2B-Instruct" ;;
    4b|qwen3_vl_4b) echo "$WORKSPACE_ROOT/models/Qwen3-VL-4B-Instruct" ;;
    8b|qwen3_vl_8b) echo "$WORKSPACE_ROOT/models/Qwen3-VL-8B-Instruct" ;;
    *) echo "Unsupported model key: $1" >&2; return 1 ;;
  esac
}

count_split_samples() {
  local split_file="$1"
  "$PYTHON_BIN" - "$split_file" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
if not isinstance(data, list):
    raise SystemExit(f"{path} must contain a JSON list")
print(len(data))
PY
}

time_limit_for_train_count() {
  local train_count="$1"
  if (( train_count <= TINY_TRAIN_LIMIT )); then
    echo "$TINY_TIME_LIMIT"
  elif (( train_count <= SMALL_TRAIN_LIMIT )); then
    echo "$SMALL_TIME_LIMIT"
  elif (( train_count <= MEDIUM_TRAIN_LIMIT )); then
    echo "$MEDIUM_TIME_LIMIT"
  elif (( train_count <= LARGE_TRAIN_LIMIT )); then
    echo "$LARGE_TIME_LIMIT"
  else
    echo "$XL_TIME_LIMIT"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing directory: $path" >&2
    exit 1
  fi
}

require_dir "$ROOT_DIR"
require_dir "$SYNTHETIC_SOURCE_ROOT"
require_dir "$SOURCE_ROOT_BASE"
require_file "$ROOT_DIR/run_full_train_gpu.sh"
require_file "$ROOT_DIR/run_predict_eval_gpu.sh"
require_file "$ROOT_DIR/scripts/build_residual_pool.py"
require_file "$ROOT_DIR/scripts/eval_vlm_grounder.py"
require_file "$ROOT_DIR/src/ts_grounder/rl_train_grpo.py"
require_file "$ROOT_DIR/train.py"
require_file "$ROOT_DIR/predict.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
for split in train val test; do
  require_file "$SYNTHETIC_SOURCE_ROOT/$split.json"
done

read -r -a SELECTED_MODELS <<< "$MODELS"
read -r -a SELECTED_SUBSETS <<< "$SUBSETS"
read -r -a SELECTED_VARIANTS <<< "$VARIANTS"

for model_key in "${SELECTED_MODELS[@]}"; do
  require_dir "$(model_path "$model_key")"
done
for subset in "${SELECTED_SUBSETS[@]}"; do
  subset_root="$SOURCE_ROOT_BASE/$subset"
  require_dir "$subset_root"
  for split in train val test; do
    require_file "$subset_root/$split.json"
  done
done

batch_root="$OUTPUT_ROOT/qwen_vl_all_models_sft_residual_ablation_8subsets_$BATCH_TAG"
generated_dir="$batch_root/generated"
mkdir -p "$generated_dir"

subset_job_script="$generated_dir/run_subset_variant_sft_rl_test.sh"
cleanup_job_script="$generated_dir/cleanup_model_weights.sh"
submission_manifest="$batch_root/submission_manifest.tsv"

cat > "$subset_job_script" <<'JOB'
#!/bin/bash
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:?PROJECT_ROOT is required}"
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
SUBSET="${SOURCE_DATASET:?SOURCE_DATASET is required}"
VARIANT="${VARIANT:?VARIANT is required}"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
CONFIG_PATH="${CONFIG_PATH:?CONFIG_PATH is required}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:?BASE_MODEL_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
PYTHON_BIN="${PYTHON_BIN:?PYTHON_BIN is required}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-$OUTPUT_DIR/rl}"

mkdir -p "$OUTPUT_DIR" "$RL_OUTPUT_DIR" "$ROOT_DIR/.cache" "$ROOT_DIR/.tmp"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-$ROOT_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TMPDIR="${TMPDIR:-$ROOT_DIR/.tmp}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Stage 2: real-subset SFT ==="
echo "model=$MODEL_LABEL subset=$SUBSET variant=$VARIANT"
echo "source_root=$SOURCE_ROOT"
echo "base stage-1 synthetic SFT model=$BASE_MODEL_PATH"
PROJECT_ROOT="$ROOT_DIR" \
CONFIG_PATH="$CONFIG_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
MODE=train \
PYTHON_BIN="$PYTHON_BIN" \
MODEL_PATH="$BASE_MODEL_PATH" \
SOURCE_ROOT="$SOURCE_ROOT" \
SFT_NUM_TRAIN_EPOCHS="${STAGE2_SFT_NUM_TRAIN_EPOCHS:-3}" \
BF16="${BF16:-1}" \
FP16="${FP16:-0}" \
LAUNCHER=python \
NUM_PROCS_PER_NODE=1 \
bash "$ROOT_DIR/run_full_train_gpu.sh"

RESIDUAL_POOL_PATH="${RESIDUAL_POOL_PATH:-$OUTPUT_DIR/residual_pool/residual_pool.jsonl}"
RESIDUAL_SUMMARY_PATH="${RESIDUAL_SUMMARY_PATH:-$OUTPUT_DIR/residual_pool/summary.json}"
mkdir -p "$(dirname "$RESIDUAL_POOL_PATH")"

case "$VARIANT" in
  full_residual)
    echo "=== Stage 3: residual error mining on real train split only ==="
    residual_cmd=(
      "$PYTHON_BIN" "$ROOT_DIR/scripts/build_residual_pool.py"
      --model-path "$OUTPUT_DIR/model"
      --real-train-data "$SOURCE_ROOT/train.json"
      --output "$RESIDUAL_POOL_PATH"
      --summary-output "$RESIDUAL_SUMMARY_PATH"
      --config "$OUTPUT_DIR/resolved_config.yaml"
      --source-root "$SOURCE_ROOT"
      --dataset-name "$SUBSET"
      --batch-size "${RESIDUAL_BATCH_SIZE:-1}"
      --max-new-tokens "${MAX_NEW_TOKENS:-256}"
      --tau-match "${TAU_MATCH:-0.1}"
      --tau-good "${TAU_GOOD:-0.5}"
      --series-length "${SERIES_LENGTH:-256}"
      --max-index "${MAX_INDEX:-255}"
      --prediction-schema evidence
      --torch-dtype bfloat16
    )
    if [[ -n "${RESIDUAL_MAX_SAMPLES:-}" ]]; then
      residual_cmd+=(--max-samples "$RESIDUAL_MAX_SAMPLES")
    fi
    PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" "${residual_cmd[@]}"
    ;;
  no_residual)
    echo "=== Stage 3 skipped: no residual mining ablation ==="
    ;;
  *)
    echo "Unsupported VARIANT=$VARIANT. Expected full_residual or no_residual." >&2
    exit 1
    ;;
esac

echo "=== Stage 4: boundary-aware GRPO ==="
grpo_cmd=(
  "$PYTHON_BIN" -m ts_grounder.rl_train_grpo
  --model_name_or_path "$OUTPUT_DIR/model"
  --reference_model_path "$OUTPUT_DIR/model"
  --train_file "$OUTPUT_DIR/dataset_cache/train.jsonl"
  --use_boundary_aware_reward
  --image_root "$SOURCE_ROOT"
  --output_dir "$RL_OUTPUT_DIR"
  --num_generations "${RL_NUM_GENERATIONS:-4}"
  --max_new_tokens "${RL_MAX_NEW_TOKENS:-256}"
  --temperature "${RL_TEMPERATURE:-0.7}"
  --top_p "${RL_TOP_P:-0.9}"
  --learning_rate "${RL_LEARNING_RATE:-1e-6}"
  --num_train_epochs "${RL_NUM_TRAIN_EPOCHS:-1}"
  --kl_coef "${RL_KL_COEF:-0.02}"
  --tau_match "${TAU_MATCH:-0.1}"
  --tau_good "${TAU_GOOD:-0.5}"
  --series_length "${SERIES_LENGTH:-256}"
  --max_index "${MAX_INDEX:-255}"
  --prediction_schema evidence
  --reward_event_weight "${REWARD_EVENT_WEIGHT:-0.45}"
  --reward_iou_weight "${REWARD_IOU_WEIGHT:-0.35}"
  --reward_boundary_weight "${REWARD_BOUNDARY_WEIGHT:-0.15}"
  --reward_type_weight "${REWARD_TYPE_WEIGHT:-0.05}"
  --save_steps "${RL_SAVE_STEPS:-0}"
  --optimizer "${RL_OPTIMIZER:-adafactor}"
)
if [[ "$VARIANT" == "full_residual" ]]; then
  grpo_cmd+=(
    --residual_pool_file "$RESIDUAL_POOL_PATH"
    --use_residual_pool
    --residual_pool_sampling_ratios "${RESIDUAL_POOL_SAMPLING_RATIOS:-false_negative=0.3,boundary_error=0.3,false_positive=0.2,correct_abnormal=0.1,correct_normal=0.1}"
  )
  if [[ -n "${RL_RESIDUAL_POOL_EPOCH_SIZE:-}" ]]; then
    grpo_cmd+=(--residual_pool_epoch_size "$RL_RESIDUAL_POOL_EPOCH_SIZE")
  fi
fi
if [[ -n "${RL_MAX_SAMPLES:-}" ]]; then
  grpo_cmd+=(--max_samples "$RL_MAX_SAMPLES")
fi
PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" "${grpo_cmd[@]}"

echo "=== Stage 5: final test ==="
PROJECT_ROOT="$ROOT_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
RUN_DIR="$RL_OUTPUT_DIR" \
SPLIT=test \
CONFIG_PATH="$OUTPUT_DIR/resolved_config.yaml" \
MODEL_PATH="$RL_OUTPUT_DIR/model" \
DATASET_JSONL="$OUTPUT_DIR/dataset_cache/test.jsonl" \
EVAL_DIR="$RL_OUTPUT_DIR/eval" \
PREDICTIONS_PATH="$RL_OUTPUT_DIR/eval/test_predictions.jsonl" \
METRICS_PATH="$RL_OUTPUT_DIR/eval/test_metrics.json" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}" \
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}" \
bash "$ROOT_DIR/run_predict_eval_gpu.sh"

echo "=== Cleanup: removing model weights/checkpoints and training caches ==="
rm -rf "$OUTPUT_DIR/model" "$OUTPUT_DIR/hf_checkpoints"
rm -rf "$RL_OUTPUT_DIR/model" "$RL_OUTPUT_DIR/checkpoints" "$RL_OUTPUT_DIR/hf_checkpoints"
rm -rf "$OUTPUT_DIR/dataset_cache"
rm -f "$RL_OUTPUT_DIR/rl_rollouts.jsonl"
if [[ "${KEEP_RESIDUAL_POOL:-0}" != "1" ]]; then
  rm -f "$RESIDUAL_POOL_PATH"
fi

echo "=== Subset variant workflow finished ==="
echo "model=$MODEL_LABEL subset=$SUBSET variant=$VARIANT"
echo "workspace=$OUTPUT_DIR"
echo "test_metrics=$RL_OUTPUT_DIR/eval/test_metrics.json"
JOB
chmod +x "$subset_job_script"

cat > "$cleanup_job_script" <<'JOB'
#!/bin/bash
set -euo pipefail

MODEL_RUN_ROOT="${MODEL_RUN_ROOT:?MODEL_RUN_ROOT is required}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:?STAGE1_OUTPUT_DIR is required}"

echo "Cleaning model weights under: $MODEL_RUN_ROOT"
rm -rf "$STAGE1_OUTPUT_DIR/model" "$STAGE1_OUTPUT_DIR/hf_checkpoints" "$STAGE1_OUTPUT_DIR/dataset_cache"
find "$MODEL_RUN_ROOT" -type d \( -name model -o -name hf_checkpoints -o -name checkpoints \) -prune -print -exec rm -rf {} +
find "$MODEL_RUN_ROOT" -type f \( -name 'rl_rollouts.jsonl' -o -name 'test_predictions.jsonl' \) -print -delete
echo "Cleanup complete. Metrics and logs are kept."
JOB
chmod +x "$cleanup_job_script"

printf 'model_key\tmodel_label\tvariant\tsubset\ttrain_count\tval_count\ttest_count\ttime_limit\tstage1_job_id\tjob_id\tcleanup_job_id\toutput_dir\ttest_metrics\n' > "$submission_manifest"

echo "Submitting all-model dual-SFT residual/ablation workflow on Quest."
echo "ROOT_DIR=$ROOT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "SYNTHETIC_SOURCE_ROOT=$SYNTHETIC_SOURCE_ROOT"
echo "SOURCE_ROOT_BASE=$SOURCE_ROOT_BASE"
echo "MODELS=${SELECTED_MODELS[*]}"
echo "SUBSETS=${SELECTED_SUBSETS[*]}"
echo "VARIANTS=${SELECTED_VARIANTS[*]}"
echo "STAGE1_SFT_NUM_TRAIN_EPOCHS=$STAGE1_SFT_NUM_TRAIN_EPOCHS"
echo "STAGE2_SFT_NUM_TRAIN_EPOCHS=$STAGE2_SFT_NUM_TRAIN_EPOCHS"
echo "RL_NUM_TRAIN_EPOCHS=$RL_NUM_TRAIN_EPOCHS"
echo "EXCLUDE_NODES=${EXCLUDE_NODES:-none}"
echo "BATCH_ROOT=$batch_root"

for model_key in "${SELECTED_MODELS[@]}"; do
  label="$(model_label "$model_key")"
  base_model_path="$(model_path "$model_key")"
  model_run_root="$batch_root/$label"
  stage1_output_dir="$model_run_root/stage1/synthetic_sft"
  config_path="$model_run_root/generated/${label}_indexed_plain_image_config.json"
  mkdir -p "$stage1_output_dir" "$model_run_root/generated"

  cat > "$config_path" <<JSON
{
  "seed": 2026,
  "output_dir": "$stage1_output_dir",
  "data": {
    "source_root": "$SYNTHETIC_SOURCE_ROOT",
    "image_subdir": "images_plain_768x384",
    "include_grounding_records": true,
    "include_qa_pairs": false,
    "include_indexed_series_text": true,
    "indexed_series_precision": 4,
    "indexed_series_compact": false,
    "series_normalization": {
      "enabled": true,
      "method": "robust_zscore",
      "clip": 32.0,
      "normalize_text": true,
      "normalize_image": true,
      "fit_on_original_window": true,
      "image_subdir": "images_normalized_768x384"
    },
    "max_train_samples": null,
    "max_val_samples": null,
    "max_test_samples": null,
    "balanced_by_type": true
  },
  "prompt": {
    "system_prompt": "You are a vision-language anomaly grounding model for time-series plots. Read the plain PNG plot and the indexed time-series values, then return exactly one JSON object. The JSON object must contain two keys: evidence and summary. evidence must be a list. Each item in evidence must contain start, end, type, strength, and direction. type must be one of point, freq, trend, range. Return at most 5 evidence items. Merge adjacent or nearby anomalous indices into continuous intervals before output. Do not list many isolated single-point fluctuations; if many anomalous points appear, summarize them using the smallest covering intervals. Keep the summary to one short sentence. If no anomaly is detected, return {\"evidence\": [], \"summary\": \"No anomaly is detected.\"}. Do not output markdown, explanations, or extra text.",
    "user_prompt": "Inspect this plain time-series window plot and the accompanying indexed values. Ground the anomaly in the current window and return one JSON object with structured evidence and one English summary sentence."
  },
  "model": {
    "model_name_or_path": "$base_model_path",
    "trust_remote_code": true,
    "processor_kwargs": {
      "max_pixels": 131072
    },
    "from_pretrained_kwargs": {
      "torch_dtype": "bfloat16",
      "attn_implementation": "sdpa"
    }
  },
  "training": {
    "mode": "train",
    "load_best_model_at_end": false,
    "save_total_limit": 1,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 0.00001,
    "num_train_epochs": $STAGE1_SFT_NUM_TRAIN_EPOCHS,
    "logging_steps": 10,
    "save_steps": 1000,
    "eval_steps": 1000,
    "evaluation_strategy": "no",
    "save_strategy": "no",
    "gradient_checkpointing": true,
    "bf16": true,
    "fp16": false,
    "optim": "adafactor",
    "memory_optimized_loss": true,
    "generation_eval_after_train": false,
    "generation_eval_splits": ["val"],
    "generation_eval_task_types": ["grounding"],
    "generation_eval_max_new_tokens": 256,
    "generation_eval_max_samples": null
  }
}
JSON

  stage1_sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--job-name=tsg-${label//_/-}-sft1"
    "--gres=$GRES"
    "--mem=$STAGE1_MEMORY"
    "--cpus-per-task=$CPUS"
    "--time=$STAGE1_TIME_LIMIT"
    "--output=$stage1_output_dir/slurm-%x-%j.out"
    "--error=$stage1_output_dir/slurm-%x-%j.err"
  )
  if [[ -n "$CONSTRAINT" ]]; then
    stage1_sbatch_cmd+=("--constraint=$CONSTRAINT")
  fi
  if [[ -n "$QOS" ]]; then
    stage1_sbatch_cmd+=("--qos=$QOS")
  fi
  if [[ -n "$EXCLUDE_NODES" ]]; then
    stage1_sbatch_cmd+=("--exclude=$EXCLUDE_NODES")
  fi
  stage1_sbatch_cmd+=("$ROOT_DIR/run_full_train_gpu.sh")

  stage1_env_cmd=(
    env
    "PROJECT_ROOT=$ROOT_DIR"
    "CONFIG_PATH=$config_path"
    "OUTPUT_DIR=$stage1_output_dir"
    "MODE=train"
    "PYTHON_BIN=$PYTHON_BIN"
    "MODEL_PATH=$base_model_path"
    "SOURCE_ROOT=$SYNTHETIC_SOURCE_ROOT"
    "SFT_NUM_TRAIN_EPOCHS=$STAGE1_SFT_NUM_TRAIN_EPOCHS"
    "BF16=$BF16"
    "FP16=$FP16"
    "LAUNCHER=python"
    "NUM_PROCS_PER_NODE=1"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] stage1 model=%s command=' "$label"
    printf ' %q' "${stage1_env_cmd[@]}" "${stage1_sbatch_cmd[@]}"
    printf '\n'
    stage1_job_id="DRY_RUN_${label}_STAGE1"
  else
    stage1_job_id="$("${stage1_env_cmd[@]}" "${stage1_sbatch_cmd[@]}")"
    echo "stage1 model=$label job_id=$stage1_job_id output_dir=$stage1_output_dir"
  fi

  model_subset_job_ids=()
  for variant in "${SELECTED_VARIANTS[@]}"; do
    for subset in "${SELECTED_SUBSETS[@]}"; do
      source_root="$SOURCE_ROOT_BASE/$subset"
      train_count="$(count_split_samples "$source_root/train.json")"
      val_count="$(count_split_samples "$source_root/val.json")"
      test_count="$(count_split_samples "$source_root/test.json")"
      time_limit="$(time_limit_for_train_count "$train_count")"
      output_dir="$model_run_root/$variant/$subset"
      rl_output_dir="$output_dir/rl"
      metrics_path="$rl_output_dir/eval/test_metrics.json"
      mkdir -p "$output_dir"

      subset_sbatch_cmd=(
        sbatch
        --parsable
        "--account=$ACCOUNT"
        "--partition=$PARTITION"
        "--job-name=tsg-${label//_/-}-${variant//_/-}-${subset,,}"
        "--gres=$GRES"
        "--mem=$SUBSET_MEMORY"
        "--cpus-per-task=$CPUS"
        "--time=$time_limit"
        "--output=$output_dir/slurm-%x-%j.out"
        "--error=$output_dir/slurm-%x-%j.err"
      )
      if [[ -n "$CONSTRAINT" ]]; then
        subset_sbatch_cmd+=("--constraint=$CONSTRAINT")
      fi
      if [[ -n "$QOS" ]]; then
        subset_sbatch_cmd+=("--qos=$QOS")
      fi
      if [[ -n "$EXCLUDE_NODES" ]]; then
        subset_sbatch_cmd+=("--exclude=$EXCLUDE_NODES")
      fi
      if [[ "$DRY_RUN" != "1" ]]; then
        subset_sbatch_cmd+=("--dependency=afterok:$stage1_job_id")
      fi
      subset_sbatch_cmd+=("$subset_job_script")

      subset_env_cmd=(
        env
        "PROJECT_ROOT=$ROOT_DIR"
        "MODEL_LABEL=$label"
        "VARIANT=$variant"
        "SOURCE_DATASET=$subset"
        "SOURCE_ROOT=$source_root"
        "CONFIG_PATH=$stage1_output_dir/resolved_config.yaml"
        "BASE_MODEL_PATH=$stage1_output_dir/model"
        "OUTPUT_DIR=$output_dir"
        "PYTHON_BIN=$PYTHON_BIN"
        "STAGE2_SFT_NUM_TRAIN_EPOCHS=$STAGE2_SFT_NUM_TRAIN_EPOCHS"
        "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
        "RESIDUAL_BATCH_SIZE=$RESIDUAL_BATCH_SIZE"
        "RESIDUAL_MAX_SAMPLES=$RESIDUAL_MAX_SAMPLES"
        "TAU_MATCH=$TAU_MATCH"
        "TAU_GOOD=$TAU_GOOD"
        "SERIES_LENGTH=$SERIES_LENGTH"
        "MAX_INDEX=$MAX_INDEX"
        "RL_OUTPUT_DIR=$rl_output_dir"
        "RL_NUM_GENERATIONS=$RL_NUM_GENERATIONS"
        "RL_MAX_NEW_TOKENS=$RL_MAX_NEW_TOKENS"
        "RL_TEMPERATURE=$RL_TEMPERATURE"
        "RL_TOP_P=$RL_TOP_P"
        "RL_LEARNING_RATE=$RL_LEARNING_RATE"
        "RL_NUM_TRAIN_EPOCHS=$RL_NUM_TRAIN_EPOCHS"
        "RL_KL_COEF=$RL_KL_COEF"
        "RL_SAVE_STEPS=$RL_SAVE_STEPS"
        "RL_OPTIMIZER=$RL_OPTIMIZER"
        "RL_MAX_SAMPLES=$RL_MAX_SAMPLES"
        "RL_RESIDUAL_POOL_EPOCH_SIZE=$RL_RESIDUAL_POOL_EPOCH_SIZE"
        "RESIDUAL_POOL_SAMPLING_RATIOS=$RESIDUAL_POOL_SAMPLING_RATIOS"
        "REWARD_EVENT_WEIGHT=$REWARD_EVENT_WEIGHT"
        "REWARD_IOU_WEIGHT=$REWARD_IOU_WEIGHT"
        "REWARD_BOUNDARY_WEIGHT=$REWARD_BOUNDARY_WEIGHT"
        "REWARD_TYPE_WEIGHT=$REWARD_TYPE_WEIGHT"
        "SAVE_PREDICTIONS=$SAVE_PREDICTIONS"
        "KEEP_RESIDUAL_POOL=$KEEP_RESIDUAL_POOL"
        "BF16=$BF16"
        "FP16=$FP16"
      )

      if [[ "$DRY_RUN" == "1" ]]; then
        printf '[DRY_RUN] model=%s variant=%s subset=%s train=%s val=%s test=%s time_limit=%s command=' \
          "$label" "$variant" "$subset" "$train_count" "$val_count" "$test_count" "$time_limit"
        printf ' %q' "${subset_env_cmd[@]}" "${subset_sbatch_cmd[@]}"
        printf '\n'
        job_id="DRY_RUN_${label}_${variant}_${subset}"
      else
        job_id="$("${subset_env_cmd[@]}" "${subset_sbatch_cmd[@]}")"
        echo "model=$label variant=$variant subset=$subset train=$train_count val=$val_count test=$test_count time_limit=$time_limit job_id=$job_id"
        model_subset_job_ids+=("$job_id")
      fi

      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t%s\t%s\n' \
        "$model_key" "$label" "$variant" "$subset" "$train_count" "$val_count" "$test_count" \
        "$time_limit" "$stage1_job_id" "$job_id" "$output_dir" "$metrics_path" \
        >> "$submission_manifest"
    done
  done

  cleanup_job_id=""
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] cleanup model=$label after all subset variant jobs"
    cleanup_job_id="DRY_RUN_${label}_CLEANUP"
  elif (( ${#model_subset_job_ids[@]} > 0 )); then
    dependency="afterany:$(IFS=:; echo "${model_subset_job_ids[*]}")"
    cleanup_output_dir="$model_run_root/cleanup"
    mkdir -p "$cleanup_output_dir"
    cleanup_env_cmd=(
      env
      "MODEL_RUN_ROOT=$model_run_root"
      "STAGE1_OUTPUT_DIR=$stage1_output_dir"
    )
    cleanup_sbatch_cmd=(
      sbatch
      --parsable
      "--account=$ACCOUNT"
      "--partition=$CLEANUP_PARTITION"
      "--job-name=tsg-${label//_/-}-cleanup"
      "--mem=8G"
      "--cpus-per-task=1"
      "--time=$CLEANUP_TIME_LIMIT"
      "--dependency=$dependency"
      "--output=$cleanup_output_dir/slurm-%x-%j.out"
      "--error=$cleanup_output_dir/slurm-%x-%j.err"
    )
    if [[ -n "$CLEANUP_GRES" ]]; then
      cleanup_sbatch_cmd+=("--gres=$CLEANUP_GRES")
    fi
    if [[ -n "$CLEANUP_CONSTRAINT" ]]; then
      cleanup_sbatch_cmd+=("--constraint=$CLEANUP_CONSTRAINT")
    fi
    cleanup_sbatch_cmd+=("$cleanup_job_script")
    cleanup_job_id="$("${cleanup_env_cmd[@]}" "${cleanup_sbatch_cmd[@]}")"
    echo "cleanup model=$label job_id=$cleanup_job_id dependency=$dependency"
  fi

  if [[ -n "$cleanup_job_id" ]]; then
    tmp_manifest="$submission_manifest.tmp"
    awk -v model="$label" -v cleanup="$cleanup_job_id" 'BEGIN{FS=OFS="\t"} NR==1{print; next} $2==model{$11=cleanup} {print}' \
      "$submission_manifest" > "$tmp_manifest"
    mv "$tmp_manifest" "$submission_manifest"
  fi
done

echo "Done. submission_manifest=$submission_manifest"
