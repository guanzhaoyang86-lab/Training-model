#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$ROOT_DIR/run_tsb_adu_subset_train_val_test_qwen.sbatch}"

SUBSETS="${SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"
BASE_SFT_ROOT="${BASE_SFT_ROOT:-$ROOT_DIR/outputs/dual_stage_sft_8subsets_dual_stage_sft_7b_clip32_20260526_161502/stage1/qwen25vl_7b}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$BASE_SFT_ROOT/model}"
CONFIG_PATH="${CONFIG_PATH:-$BASE_SFT_ROOT/resolved_config.yaml}"
ALLOW_PENDING_BASE_SFT="${ALLOW_PENDING_BASE_SFT:-0}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
BATCH_PREFIX="${BATCH_PREFIX:-sft2_rl_8subsets_from_synthetic_sft}"
WORKFLOW_DESCRIPTION="${WORKFLOW_DESCRIPTION:-second-stage real-subset SFT+RL jobs from the synthetic SFT model}"
BASE_MODEL_DESCRIPTION="${BASE_MODEL_DESCRIPTION:-Base synthetic SFT model}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-tsg-sft2rl-q25vl7b}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
MEMORY="${MEMORY:-120G}"
CPUS="${CPUS:-8}"
QOS="${QOS:-}"
DEPENDENCY="${DEPENDENCY:-}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/envs/anomamind/bin/python}"

TINY_TRAIN_LIMIT="${TINY_TRAIN_LIMIT:-30}"
SMALL_TRAIN_LIMIT="${SMALL_TRAIN_LIMIT:-80}"
MEDIUM_TRAIN_LIMIT="${MEDIUM_TRAIN_LIMIT:-200}"
LARGE_TRAIN_LIMIT="${LARGE_TRAIN_LIMIT:-500}"
TINY_TIME_LIMIT="${TINY_TIME_LIMIT:-00:30:00}"
SMALL_TIME_LIMIT="${SMALL_TIME_LIMIT:-01:00:00}"
MEDIUM_TIME_LIMIT="${MEDIUM_TIME_LIMIT:-02:00:00}"
LARGE_TIME_LIMIT="${LARGE_TIME_LIMIT:-03:00:00}"
XL_TIME_LIMIT="${XL_TIME_LIMIT:-04:00:00}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
EVAL_SPLITS="${EVAL_SPLITS:-test}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-1}"
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
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
CLEANUP_SFT_AFTER_RL="${CLEANUP_SFT_AFTER_RL:-0}"
CLEANUP_FINAL_MODEL_AFTER_EVAL="${CLEANUP_FINAL_MODEL_AFTER_EVAL:-1}"
CLEANUP_RL_ARTIFACTS_AFTER_EVAL="${CLEANUP_RL_ARTIFACTS_AFTER_EVAL:-1}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$BASE_MODEL_PATH" && "$ALLOW_PENDING_BASE_SFT" != "1" ]]; then
  echo "BASE_MODEL_PATH not found: $BASE_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" && "$ALLOW_PENDING_BASE_SFT" != "1" ]]; then
  echo "CONFIG_PATH not found: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_ROOT_BASE" ]]; then
  echo "SOURCE_ROOT_BASE not found: $SOURCE_ROOT_BASE" >&2
  exit 1
fi
if [[ "$CLEANUP_FINAL_MODEL_AFTER_EVAL" != "1" ]]; then
  echo "This experiment is meant to remove final RL weights after test; set CLEANUP_FINAL_MODEL_AFTER_EVAL=1." >&2
  exit 1
fi

count_split_samples() {
  local split_file="$1"
  grep -o '"image_path"' "$split_file" | wc -l
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

read -r -a SELECTED_SUBSETS <<< "$SUBSETS"
batch_root="$OUTPUT_ROOT/${BATCH_PREFIX}_$BATCH_TAG"
mkdir -p "$batch_root"
submission_manifest="$batch_root/submission_manifest.tsv"
printf 'subset\ttrain_count\tval_count\ttest_count\tjob_id\tmemory\ttime_limit\tsft_num_train_epochs\trl_num_train_epochs\tbase_sft_model\tconfig_path\toutput_dir\tsft2_model_path\trl_output_dir\ttest_metrics_path\tcleanup_sft_after_rl\tcleanup_final_model_after_eval\tcleanup_rl_artifacts_after_eval\n' > "$submission_manifest"

echo "Submitting $WORKFLOW_DESCRIPTION."
echo "$BASE_MODEL_DESCRIPTION: $BASE_MODEL_PATH"
echo "Config: $CONFIG_PATH"
echo "Subsets: ${SELECTED_SUBSETS[*]}"
echo "Eval splits: $EVAL_SPLITS"
echo "SFT epochs: $SFT_NUM_TRAIN_EPOCHS"
echo "RL epochs: $RL_NUM_TRAIN_EPOCHS"
echo "Save predictions: $SAVE_PREDICTIONS"
if [[ "$CLEANUP_SFT_AFTER_RL" == "1" ]]; then
  echo "Retain second-stage SFT model: no"
else
  echo "Retain second-stage SFT model: yes"
fi
echo "Remove final RL weights after test: yes"
echo "Cleanup rollout and prepared cache after test: $CLEANUP_RL_ARTIFACTS_AFTER_EVAL"
echo "Batch root: $batch_root"

for subset in "${SELECTED_SUBSETS[@]}"; do
  source_root="$SOURCE_ROOT_BASE/$subset"
  if [[ ! -d "$source_root" ]]; then
    echo "Missing subset dir: $source_root" >&2
    exit 1
  fi
  for split in train val test; do
    if [[ ! -f "$source_root/$split.json" ]]; then
      echo "Missing $split.json under $source_root" >&2
      exit 1
    fi
  done

  train_count="$(count_split_samples "$source_root/train.json")"
  val_count="$(count_split_samples "$source_root/val.json")"
  test_count="$(count_split_samples "$source_root/test.json")"
  time_limit="$(time_limit_for_train_count "$train_count")"
  output_dir="$batch_root/$subset"
  sft2_model_path="$output_dir/model"
  rl_output_dir="$output_dir/rl"
  metrics_path="$rl_output_dir/eval/test_metrics.json"
  job_name="$JOB_NAME_PREFIX-${subset,,}"
  mkdir -p "$output_dir"

  sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--job-name=$job_name"
    "--gres=$GRES"
    "--mem=$MEMORY"
    "--cpus-per-task=$CPUS"
    "--time=$time_limit"
    "--output=$output_dir/slurm-%x-%j.out"
    "--error=$output_dir/slurm-%x-%j.err"
  )
  if [[ -n "$CONSTRAINT" ]]; then
    sbatch_cmd+=("--constraint=$CONSTRAINT")
  fi
  if [[ -n "$QOS" ]]; then
    sbatch_cmd+=("--qos=$QOS")
  fi
  if [[ -n "$DEPENDENCY" ]]; then
    sbatch_cmd+=("--dependency=$DEPENDENCY")
  fi
  sbatch_cmd+=("$SBATCH_SCRIPT")

  env_cmd=(
    env
    "PROJECT_ROOT=$ROOT_DIR"
    "SOURCE_DATASET=$subset"
    "SOURCE_ROOT_BASE=$SOURCE_ROOT_BASE"
    "SOURCE_ROOT=$source_root"
    "CONFIG_PATH=$CONFIG_PATH"
    "BASE_MODEL_PATH=$BASE_MODEL_PATH"
    "OUTPUT_DIR=$output_dir"
    "PYTHON_BIN=$PYTHON_BIN"
    "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
    "EVAL_SPLITS=$EVAL_SPLITS"
    "SFT_NUM_TRAIN_EPOCHS=$SFT_NUM_TRAIN_EPOCHS"
    "ENABLE_RL=1"
    "SKIP_SFT_BEFORE_RL=0"
    "RL_OUTPUT_DIR=$rl_output_dir"
    "RL_NUM_GENERATIONS=$RL_NUM_GENERATIONS"
    "RL_MAX_NEW_TOKENS=$RL_MAX_NEW_TOKENS"
    "RL_TEMPERATURE=$RL_TEMPERATURE"
    "RL_TOP_P=$RL_TOP_P"
    "RL_LEARNING_RATE=$RL_LEARNING_RATE"
    "RL_NUM_TRAIN_EPOCHS=$RL_NUM_TRAIN_EPOCHS"
    "RL_KL_COEF=$RL_KL_COEF"
    "RL_REWARD_EVENT_F1_WEIGHT=$RL_REWARD_EVENT_F1_WEIGHT"
    "RL_REWARD_BOUNDARY_IOU_WEIGHT=$RL_REWARD_BOUNDARY_IOU_WEIGHT"
    "RL_REWARD_HALLUCINATION_PENALTY_WEIGHT=$RL_REWARD_HALLUCINATION_PENALTY_WEIGHT"
    "RL_SAVE_STEPS=$RL_SAVE_STEPS"
    "RL_OPTIMIZER=$RL_OPTIMIZER"
    "SAVE_PREDICTIONS=$SAVE_PREDICTIONS"
    "CLEANUP_SFT_AFTER_RL=$CLEANUP_SFT_AFTER_RL"
    "CLEANUP_FINAL_MODEL_AFTER_EVAL=$CLEANUP_FINAL_MODEL_AFTER_EVAL"
    "CLEANUP_RL_ARTIFACTS_AFTER_EVAL=$CLEANUP_RL_ARTIFACTS_AFTER_EVAL"
    "BF16=$BF16"
    "FP16=$FP16"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] subset=%s train=%s val=%s test=%s time_limit=%s output_dir=%s command=' \
      "$subset" "$train_count" "$val_count" "$test_count" "$time_limit" "$output_dir"
    printf ' %q' "${env_cmd[@]}" "${sbatch_cmd[@]}"
    printf '\n'
    job_id="DRY_RUN"
  else
    job_id="$("${env_cmd[@]}" "${sbatch_cmd[@]}")"
    echo "subset=$subset train=$train_count val=$val_count test=$test_count job_id=$job_id time_limit=$time_limit sft2_model=$sft2_model_path metrics=$metrics_path"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$subset" "$train_count" "$val_count" "$test_count" "$job_id" "$MEMORY" "$time_limit" \
    "$SFT_NUM_TRAIN_EPOCHS" "$RL_NUM_TRAIN_EPOCHS" "$BASE_MODEL_PATH" "$CONFIG_PATH" \
    "$output_dir" "$sft2_model_path" "$rl_output_dir" \
    "$metrics_path" "$CLEANUP_SFT_AFTER_RL" "$CLEANUP_FINAL_MODEL_AFTER_EVAL" \
    "$CLEANUP_RL_ARTIFACTS_AFTER_EVAL" \
    >> "$submission_manifest"
done

echo "Done. submission_manifest=$submission_manifest"
