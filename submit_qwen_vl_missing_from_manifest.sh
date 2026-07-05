#!/bin/bash
set -euo pipefail

# Resubmit missing subset jobs from a batch submission_manifest.tsv.
# Run this after the queue has drained for a batch. It detects rows whose
# test_metrics.json is absent, resubmits those exact model/setting/subset jobs
# into the same output directories, then submits a cleanup job after the retries
# complete successfully.

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/batch_root" >&2
  exit 2
fi

BATCH_ROOT="$(cd "$1" && pwd)"
ROOT_DIR="${PROJECT_ROOT:-/gpfs/projects/p33222/ybq9740/Thesis/Training-model-new}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/gpfs/projects/p33222/ybq9740}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/envs/anomamind/bin/python}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-/gpfs/projects/p33222/ybq9740/Thesis/Training-model-yilong/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
CPUS="${CPUS:-8}"
QOS="${QOS:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-qgpu2014}"
SUBSET_MEMORY="${SUBSET_MEMORY:-140G}"
CLEANUP_TIME_LIMIT="${CLEANUP_TIME_LIMIT:-00:10:00}"
CLEANUP_PARTITION="${CLEANUP_PARTITION:-$PARTITION}"
CLEANUP_GRES="${CLEANUP_GRES:-$GRES}"
CLEANUP_CONSTRAINT="${CLEANUP_CONSTRAINT:-$CONSTRAINT}"

STAGE2_SFT_NUM_TRAIN_EPOCHS="${STAGE2_SFT_NUM_TRAIN_EPOCHS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
RESIDUAL_BATCH_SIZE="${RESIDUAL_BATCH_SIZE:-1}"
RESIDUAL_MAX_SAMPLES="${RESIDUAL_MAX_SAMPLES:-}"
TAU_MATCH="${TAU_MATCH:-0.1}"
TAU_GOOD="${TAU_GOOD:-0.5}"
SERIES_LENGTH="${SERIES_LENGTH:-256}"
MAX_INDEX="${MAX_INDEX:-255}"
RL_NUM_GENERATIONS="${RL_NUM_GENERATIONS:-4}"
RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-512}"
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
RESIDUAL_V2_POOL_SAMPLING_RATIOS="${RESIDUAL_V2_POOL_SAMPLING_RATIOS:-false_negative=0.60,boundary_error=0.20,false_positive=0.10,correct_abnormal=0.10,correct_normal=0.00}"
RESIDUAL_V2_RL_NUM_GENERATIONS="${RESIDUAL_V2_RL_NUM_GENERATIONS:-6}"
RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS="${RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS:-2}"
RESIDUAL_V2_RL_LEARNING_RATE="${RESIDUAL_V2_RL_LEARNING_RATE:-2e-6}"
RESIDUAL_V2_RL_KL_COEF="${RESIDUAL_V2_RL_KL_COEF:-0.03}"
REWARD_POINT_WEIGHT="${REWARD_POINT_WEIGHT:-0}"
REWARD_EVENT_WEIGHT="${REWARD_EVENT_WEIGHT:-0.45}"
REWARD_IOU_WEIGHT="${REWARD_IOU_WEIGHT:-0.40}"
REWARD_BOUNDARY_WEIGHT="${REWARD_BOUNDARY_WEIGHT:-0.15}"
REWARD_TYPE_WEIGHT="${REWARD_TYPE_WEIGHT:-0}"
RESIDUAL_V2_REWARD_POINT_WEIGHT="${RESIDUAL_V2_REWARD_POINT_WEIGHT:-0.60}"
RESIDUAL_V2_REWARD_EVENT_WEIGHT="${RESIDUAL_V2_REWARD_EVENT_WEIGHT:-0.00}"
RESIDUAL_V2_REWARD_IOU_WEIGHT="${RESIDUAL_V2_REWARD_IOU_WEIGHT:-0.25}"
RESIDUAL_V2_REWARD_BOUNDARY_WEIGHT="${RESIDUAL_V2_REWARD_BOUNDARY_WEIGHT:-0.15}"
RESIDUAL_V2_REWARD_TYPE_WEIGHT="${RESIDUAL_V2_REWARD_TYPE_WEIGHT:-0.00}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
KEEP_RESIDUAL_POOL="${KEEP_RESIDUAL_POOL:-0}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
DRY_RUN="${DRY_RUN:-0}"

MANIFEST="$BATCH_ROOT/submission_manifest.tsv"
SUBSET_JOB_SCRIPT="$BATCH_ROOT/generated/run_subset_variant_sft_rl_test.sh"
CLEANUP_JOB_SCRIPT="$BATCH_ROOT/generated/cleanup_model_weights.sh"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST" >&2
  exit 1
fi
if [[ ! -x "$SUBSET_JOB_SCRIPT" ]]; then
  echo "Missing or non-executable subset job script: $SUBSET_JOB_SCRIPT" >&2
  exit 1
fi

seed_for_model() {
  local label="$1"
  local config="$BATCH_ROOT/$label/generated/${label}_indexed_plain_image_config.json"
  if [[ -f "$config" ]]; then
    "$PYTHON_BIN" - "$config" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text()).get("seed", 2026))
PY
  else
    echo "${EXPERIMENT_SEED:-2026}"
  fi
}

retry_job_ids=()
missing_count=0

while IFS=$'\t' read -r model_key label variant subset train_count val_count test_count time_limit stage1_job_id old_job_id cleanup_job_id output_dir metrics_path; do
  [[ "$model_key" == "model_key" ]] && continue
  [[ -z "${model_key:-}" ]] && continue
  if [[ -f "$metrics_path" ]]; then
    continue
  fi

  missing_count=$((missing_count + 1))
  source_root="$SOURCE_ROOT_BASE/$subset"
  stage1_output_dir="$BATCH_ROOT/$label/stage1/synthetic_sft"
  config_path="$stage1_output_dir/resolved_config.yaml"
  base_model_path="$stage1_output_dir/model"
  rl_output_dir="$output_dir/rl"
  retry_output_dir="$output_dir/retry_logs"
  mkdir -p "$retry_output_dir"

  if [[ ! -d "$base_model_path" ]]; then
    echo "Cannot resubmit $label $variant $subset: missing stage1 model $base_model_path" >&2
    continue
  fi

  experiment_seed="$(seed_for_model "$label")"
  subset_sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--job-name=retry-${label//_/-}-${variant//_/-}-${subset,,}"
    "--gres=$GRES"
    "--mem=$SUBSET_MEMORY"
    "--cpus-per-task=$CPUS"
    "--time=$time_limit"
    "--output=$retry_output_dir/slurm-%x-%j.out"
    "--error=$retry_output_dir/slurm-%x-%j.err"
  )
  if [[ -n "$CONSTRAINT" ]]; then subset_sbatch_cmd+=("--constraint=$CONSTRAINT"); fi
  if [[ -n "$QOS" ]]; then subset_sbatch_cmd+=("--qos=$QOS"); fi
  if [[ -n "$EXCLUDE_NODES" ]]; then subset_sbatch_cmd+=("--exclude=$EXCLUDE_NODES"); fi
  subset_sbatch_cmd+=("$SUBSET_JOB_SCRIPT")

  subset_env_cmd=(
    env
    "PROJECT_ROOT=$ROOT_DIR"
    "MODEL_LABEL=$label"
    "VARIANT=$variant"
    "SOURCE_DATASET=$subset"
    "SOURCE_ROOT=$source_root"
    "CONFIG_PATH=$config_path"
    "BASE_MODEL_PATH=$base_model_path"
    "OUTPUT_DIR=$output_dir"
    "PYTHON_BIN=$PYTHON_BIN"
    "EXPERIMENT_SEED=$experiment_seed"
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
    "RESIDUAL_V2_POOL_SAMPLING_RATIOS=$RESIDUAL_V2_POOL_SAMPLING_RATIOS"
    "RESIDUAL_V2_RL_NUM_GENERATIONS=$RESIDUAL_V2_RL_NUM_GENERATIONS"
    "RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS=$RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS"
    "RESIDUAL_V2_RL_LEARNING_RATE=$RESIDUAL_V2_RL_LEARNING_RATE"
    "RESIDUAL_V2_RL_KL_COEF=$RESIDUAL_V2_RL_KL_COEF"
    "REWARD_POINT_WEIGHT=$REWARD_POINT_WEIGHT"
    "REWARD_EVENT_WEIGHT=$REWARD_EVENT_WEIGHT"
    "REWARD_IOU_WEIGHT=$REWARD_IOU_WEIGHT"
    "REWARD_BOUNDARY_WEIGHT=$REWARD_BOUNDARY_WEIGHT"
    "REWARD_TYPE_WEIGHT=$REWARD_TYPE_WEIGHT"
    "RESIDUAL_V2_REWARD_POINT_WEIGHT=$RESIDUAL_V2_REWARD_POINT_WEIGHT"
    "RESIDUAL_V2_REWARD_EVENT_WEIGHT=$RESIDUAL_V2_REWARD_EVENT_WEIGHT"
    "RESIDUAL_V2_REWARD_IOU_WEIGHT=$RESIDUAL_V2_REWARD_IOU_WEIGHT"
    "RESIDUAL_V2_REWARD_BOUNDARY_WEIGHT=$RESIDUAL_V2_REWARD_BOUNDARY_WEIGHT"
    "RESIDUAL_V2_REWARD_TYPE_WEIGHT=$RESIDUAL_V2_REWARD_TYPE_WEIGHT"
    "SAVE_PREDICTIONS=$SAVE_PREDICTIONS"
    "KEEP_RESIDUAL_POOL=$KEEP_RESIDUAL_POOL"
    "BF16=$BF16"
    "FP16=$FP16"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] resubmit model=%s variant=%s subset=%s command=' "$label" "$variant" "$subset"
    printf ' %q' "${subset_env_cmd[@]}" "${subset_sbatch_cmd[@]}"
    printf '\n'
  else
    new_job_id="$("${subset_env_cmd[@]}" "${subset_sbatch_cmd[@]}")"
    retry_job_ids+=("$new_job_id")
    echo "resubmitted model=$label variant=$variant subset=$subset job_id=$new_job_id"
  fi
done < "$MANIFEST"

if (( missing_count == 0 )); then
  echo "No missing metrics under $BATCH_ROOT"
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] missing_count=$missing_count; cleanup would be submitted after retry jobs"
  exit 0
fi

if (( ${#retry_job_ids[@]} == 0 )); then
  echo "Missing metrics were found, but no retry jobs were submitted. Stage1 models may already be cleaned." >&2
  exit 1
fi

if [[ -x "$CLEANUP_JOB_SCRIPT" ]]; then
  dependency="afterok:$(IFS=:; echo "${retry_job_ids[*]}")"
  cleanup_output_dir="$BATCH_ROOT/retry_cleanup"
  mkdir -p "$cleanup_output_dir"
  cleanup_sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$CLEANUP_PARTITION"
    "--job-name=retry-cleanup"
    "--mem=8G"
    "--cpus-per-task=1"
    "--time=$CLEANUP_TIME_LIMIT"
    "--dependency=$dependency"
    "--output=$cleanup_output_dir/slurm-%x-%j.out"
    "--error=$cleanup_output_dir/slurm-%x-%j.err"
  )
  if [[ -n "$CLEANUP_GRES" ]]; then cleanup_sbatch_cmd+=("--gres=$CLEANUP_GRES"); fi
  if [[ -n "$CLEANUP_CONSTRAINT" ]]; then cleanup_sbatch_cmd+=("--constraint=$CLEANUP_CONSTRAINT"); fi
  cleanup_sbatch_cmd+=("$CLEANUP_JOB_SCRIPT")
  cleanup_job_id="$(env "MODEL_RUN_ROOT=$BATCH_ROOT" "STAGE1_OUTPUT_DIR=$BATCH_ROOT/__all_stage1__" "${cleanup_sbatch_cmd[@]}")"
  echo "retry cleanup job_id=$cleanup_job_id dependency=$dependency"
else
  echo "Cleanup script is missing; retry jobs submitted but weights will remain under $BATCH_ROOT" >&2
fi
