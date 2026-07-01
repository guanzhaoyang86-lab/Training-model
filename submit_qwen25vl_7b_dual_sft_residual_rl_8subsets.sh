#!/bin/bash
set -euo pipefail

# Quest submitter for:
#   1) synthetic SFT on anomaly_db_v1
#   2) per-subset real SFT on the 8 TSB-AD-U subsets
#   3) per-subset residual error mining on each real train split
#   4) per-subset boundary-aware GRPO
#   5) per-subset test evaluation
#
# This script intentionally defaults to Quest paths, not local workstation paths.

ROOT_DIR="${PROJECT_ROOT:-/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/gpfs/projects/p33222/ybq9740}"

PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/envs/anomamind/bin/python}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$WORKSPACE_ROOT/models/Qwen2.5-VL-7B-Instruct}"
SYNTHETIC_SOURCE_ROOT="${SYNTHETIC_SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
SUBSETS="${SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
CPUS="${CPUS:-8}"
QOS="${QOS:-}"
STAGE1_MEMORY="${STAGE1_MEMORY:-120G}"
SUBSET_MEMORY="${SUBSET_MEMORY:-120G}"
STAGE1_TIME_LIMIT="${STAGE1_TIME_LIMIT:-06:00:00}"

TINY_TRAIN_LIMIT="${TINY_TRAIN_LIMIT:-30}"
SMALL_TRAIN_LIMIT="${SMALL_TRAIN_LIMIT:-80}"
MEDIUM_TRAIN_LIMIT="${MEDIUM_TRAIN_LIMIT:-200}"
LARGE_TRAIN_LIMIT="${LARGE_TRAIN_LIMIT:-500}"
TINY_TIME_LIMIT="${TINY_TIME_LIMIT:-02:00:00}"
SMALL_TIME_LIMIT="${SMALL_TIME_LIMIT:-04:00:00}"
MEDIUM_TIME_LIMIT="${MEDIUM_TIME_LIMIT:-08:00:00}"
LARGE_TIME_LIMIT="${LARGE_TIME_LIMIT:-12:00:00}"
XL_TIME_LIMIT="${XL_TIME_LIMIT:-1-00:00:00}"

SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-1}"
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
RL_NUM_TRAIN_EPOCHS="${RL_NUM_TRAIN_EPOCHS:-1}"
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
CLEANUP_SFT_AFTER_RL="${CLEANUP_SFT_AFTER_RL:-0}"
CLEANUP_FINAL_MODEL_AFTER_EVAL="${CLEANUP_FINAL_MODEL_AFTER_EVAL:-1}"
CLEANUP_RL_ARTIFACTS_AFTER_EVAL="${CLEANUP_RL_ARTIFACTS_AFTER_EVAL:-1}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
DRY_RUN="${DRY_RUN:-0}"

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
require_dir "$BASE_MODEL_PATH"
require_dir "$SYNTHETIC_SOURCE_ROOT"
require_dir "$SOURCE_ROOT_BASE"
require_file "$ROOT_DIR/run_full_train_gpu.sh"
require_file "$ROOT_DIR/run_predict_eval_gpu.sh"
require_file "$ROOT_DIR/scripts/build_residual_pool.py"
require_file "$ROOT_DIR/scripts/eval_vlm_grounder.py"
require_file "$ROOT_DIR/src/ts_grounder/rl_train_grpo.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
for split in train val test; do
  require_file "$SYNTHETIC_SOURCE_ROOT/$split.json"
done

read -r -a SELECTED_SUBSETS <<< "$SUBSETS"
for subset in "${SELECTED_SUBSETS[@]}"; do
  subset_root="$SOURCE_ROOT_BASE/$subset"
  require_dir "$subset_root"
  for split in train val test; do
    require_file "$subset_root/$split.json"
  done
done

batch_root="$OUTPUT_ROOT/qwen25vl_7b_dual_sft_residual_rl_8subsets_$BATCH_TAG"
stage1_output_dir="$batch_root/stage1/synthetic_sft"
subset_root_dir="$batch_root/subsets"
mkdir -p "$stage1_output_dir" "$subset_root_dir" "$batch_root/generated"

config_path="$batch_root/generated/qwen25vl_7b_indexed_plain_image_config.json"
subset_job_script="$batch_root/generated/run_subset_sft_residual_rl_test.sh"
submission_manifest="$batch_root/submission_manifest.tsv"

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
    "model_name_or_path": "$BASE_MODEL_PATH",
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
    "num_train_epochs": $SFT_NUM_TRAIN_EPOCHS,
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

cat > "$subset_job_script" <<'JOB'
#!/bin/bash
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:?PROJECT_ROOT is required}"
SUBSET="${SOURCE_DATASET:?SOURCE_DATASET is required}"
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
echo "subset=$SUBSET"
echo "source_root=$SOURCE_ROOT"
echo "base synthetic SFT model=$BASE_MODEL_PATH"
PROJECT_ROOT="$ROOT_DIR" \
CONFIG_PATH="$CONFIG_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
MODE=train \
PYTHON_BIN="$PYTHON_BIN" \
MODEL_PATH="$BASE_MODEL_PATH" \
SOURCE_ROOT="$SOURCE_ROOT" \
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-1}" \
BF16="${BF16:-1}" \
FP16="${FP16:-0}" \
LAUNCHER=python \
NUM_PROCS_PER_NODE=1 \
bash "$ROOT_DIR/run_full_train_gpu.sh"

echo "=== Stage 3a: residual error mining on real train split only ==="
RESIDUAL_POOL_PATH="${RESIDUAL_POOL_PATH:-$OUTPUT_DIR/residual_pool/residual_pool.jsonl}"
RESIDUAL_SUMMARY_PATH="${RESIDUAL_SUMMARY_PATH:-$OUTPUT_DIR/residual_pool/summary.json}"
mkdir -p "$(dirname "$RESIDUAL_POOL_PATH")"

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

echo "=== Stage 3b: boundary-aware GRPO from residual pool ==="
grpo_cmd=(
  "$PYTHON_BIN" -m ts_grounder.rl_train_grpo
  --model_name_or_path "$OUTPUT_DIR/model"
  --reference_model_path "$OUTPUT_DIR/model"
  --train_file "$OUTPUT_DIR/dataset_cache/train.jsonl"
  --residual_pool_file "$RESIDUAL_POOL_PATH"
  --use_residual_pool
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
  --residual_pool_sampling_ratios "${RESIDUAL_POOL_SAMPLING_RATIOS:-false_negative=0.3,boundary_error=0.3,false_positive=0.2,correct_abnormal=0.1,correct_normal=0.1}"
  --save_steps "${RL_SAVE_STEPS:-0}"
  --optimizer "${RL_OPTIMIZER:-adafactor}"
)
if [[ -n "${RL_RESIDUAL_POOL_EPOCH_SIZE:-}" ]]; then
  grpo_cmd+=(--residual_pool_epoch_size "$RL_RESIDUAL_POOL_EPOCH_SIZE")
fi
if [[ -n "${RL_MAX_SAMPLES:-}" ]]; then
  grpo_cmd+=(--max_samples "$RL_MAX_SAMPLES")
fi
PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}" "${grpo_cmd[@]}"

if [[ "${CLEANUP_SFT_AFTER_RL:-0}" == "1" ]]; then
  echo "=== Cleanup: removing stage-2 SFT model/checkpoints after RL ==="
  rm -rf "$OUTPUT_DIR/model" "$OUTPUT_DIR/hf_checkpoints"
fi

echo "=== Stage 4: final test on the same real subset test split ==="
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

if [[ "${CLEANUP_FINAL_MODEL_AFTER_EVAL:-1}" == "1" ]]; then
  echo "=== Cleanup: removing final RL model after test ==="
  rm -rf "$RL_OUTPUT_DIR/model" "$RL_OUTPUT_DIR/checkpoints" "$RL_OUTPUT_DIR/hf_checkpoints"
fi
if [[ "${CLEANUP_RL_ARTIFACTS_AFTER_EVAL:-1}" == "1" ]]; then
  echo "=== Cleanup: removing rollout cache only; residual pool and metrics are kept ==="
  rm -f "$RL_OUTPUT_DIR/rl_rollouts.jsonl"
fi

echo "=== Subset workflow finished ==="
echo "subset=$SUBSET"
echo "workspace=$OUTPUT_DIR"
echo "residual_pool=$RESIDUAL_POOL_PATH"
echo "residual_summary=$RESIDUAL_SUMMARY_PATH"
echo "test_metrics=$RL_OUTPUT_DIR/eval/test_metrics.json"
JOB
chmod +x "$subset_job_script"

printf 'subset\ttrain_count\tval_count\ttest_count\ttime_limit\tjob_id\tstage1_job_id\toutput_dir\tresidual_pool\ttest_metrics\n' > "$submission_manifest"

echo "Submitting Qwen2.5-VL-7B dual-SFT residual-RL workflow on Quest."
echo "ROOT_DIR=$ROOT_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "SYNTHETIC_SOURCE_ROOT=$SYNTHETIC_SOURCE_ROOT"
echo "SOURCE_ROOT_BASE=$SOURCE_ROOT_BASE"
echo "SUBSETS=${SELECTED_SUBSETS[*]}"
echo "BATCH_ROOT=$batch_root"
echo "CONFIG_PATH=$config_path"
echo "SUBSET_JOB_SCRIPT=$subset_job_script"

stage1_sbatch_cmd=(
  sbatch
  --parsable
  "--account=$ACCOUNT"
  "--partition=$PARTITION"
  "--job-name=tsg-q25vl7b-sft1-synth"
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
stage1_sbatch_cmd+=("$ROOT_DIR/run_full_train_gpu.sh")

stage1_env_cmd=(
  env
  "PROJECT_ROOT=$ROOT_DIR"
  "CONFIG_PATH=$config_path"
  "OUTPUT_DIR=$stage1_output_dir"
  "MODE=train"
  "PYTHON_BIN=$PYTHON_BIN"
  "MODEL_PATH=$BASE_MODEL_PATH"
  "SOURCE_ROOT=$SYNTHETIC_SOURCE_ROOT"
  "SFT_NUM_TRAIN_EPOCHS=$SFT_NUM_TRAIN_EPOCHS"
  "BF16=$BF16"
  "FP16=$FP16"
  "LAUNCHER=python"
  "NUM_PROCS_PER_NODE=1"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[DRY_RUN] stage1 command='
  printf ' %q' "${stage1_env_cmd[@]}" "${stage1_sbatch_cmd[@]}"
  printf '\n'
  stage1_job_id="DRY_RUN_STAGE1"
else
  stage1_job_id="$("${stage1_env_cmd[@]}" "${stage1_sbatch_cmd[@]}")"
  echo "stage1_job_id=$stage1_job_id stage1_output_dir=$stage1_output_dir"
fi

for subset in "${SELECTED_SUBSETS[@]}"; do
  source_root="$SOURCE_ROOT_BASE/$subset"
  train_count="$(count_split_samples "$source_root/train.json")"
  val_count="$(count_split_samples "$source_root/val.json")"
  test_count="$(count_split_samples "$source_root/test.json")"
  time_limit="$(time_limit_for_train_count "$train_count")"
  output_dir="$subset_root_dir/$subset"
  rl_output_dir="$output_dir/rl"
  residual_pool="$output_dir/residual_pool/residual_pool.jsonl"
  metrics_path="$rl_output_dir/eval/test_metrics.json"
  mkdir -p "$output_dir"

  subset_sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--job-name=tsg-q25vl7b-${subset,,}-resrl"
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
  if [[ "$DRY_RUN" != "1" ]]; then
    subset_sbatch_cmd+=("--dependency=afterok:$stage1_job_id")
  fi
  subset_sbatch_cmd+=("$subset_job_script")

  subset_env_cmd=(
    env
    "PROJECT_ROOT=$ROOT_DIR"
    "SOURCE_DATASET=$subset"
    "SOURCE_ROOT=$source_root"
    "CONFIG_PATH=$stage1_output_dir/resolved_config.yaml"
    "BASE_MODEL_PATH=$stage1_output_dir/model"
    "OUTPUT_DIR=$output_dir"
    "PYTHON_BIN=$PYTHON_BIN"
    "SFT_NUM_TRAIN_EPOCHS=$SFT_NUM_TRAIN_EPOCHS"
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
    "CLEANUP_SFT_AFTER_RL=$CLEANUP_SFT_AFTER_RL"
    "CLEANUP_FINAL_MODEL_AFTER_EVAL=$CLEANUP_FINAL_MODEL_AFTER_EVAL"
    "CLEANUP_RL_ARTIFACTS_AFTER_EVAL=$CLEANUP_RL_ARTIFACTS_AFTER_EVAL"
    "BF16=$BF16"
    "FP16=$FP16"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] subset=%s train=%s val=%s test=%s time_limit=%s command=' \
      "$subset" "$train_count" "$val_count" "$test_count" "$time_limit"
    printf ' %q' "${subset_env_cmd[@]}" "${subset_sbatch_cmd[@]}"
    printf '\n'
    job_id="DRY_RUN_${subset}"
  else
    job_id="$("${subset_env_cmd[@]}" "${subset_sbatch_cmd[@]}")"
    echo "subset=$subset train=$train_count val=$val_count test=$test_count time_limit=$time_limit job_id=$job_id"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$subset" "$train_count" "$val_count" "$test_count" "$time_limit" "$job_id" \
    "$stage1_job_id" "$output_dir" "$residual_pool" "$metrics_path" \
    >> "$submission_manifest"
done

echo "Done. submission_manifest=$submission_manifest"
