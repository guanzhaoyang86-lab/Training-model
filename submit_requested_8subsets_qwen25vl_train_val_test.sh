#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$ROOT_DIR/run_tsb_adu_subset_train_val_test_qwen.sbatch}"

DEFAULT_SUBSETS=(
  Daphnet
  MSL
  NEK
  Power
  SED
  TAO
  TODS
  YAHOO
)
if [[ -n "${SUBSETS:-}" ]]; then
  read -r -a SELECTED_SUBSETS <<< "$SUBSETS"
else
  SELECTED_SUBSETS=("${DEFAULT_SUBSETS[@]}")
fi

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
MEMORY="${MEMORY:-80G}"
CPUS="${CPUS:-8}"
TIME_LIMIT="${TIME_LIMIT:-}"
DEFAULT_TIME_LIMIT="${DEFAULT_TIME_LIMIT:-24:00:00}"
SMALL_SUBSET_TIME_LIMIT="${SMALL_SUBSET_TIME_LIMIT:-00:45:00}"
TAO_TIME_LIMIT="${TAO_TIME_LIMIT:-01:30:00}"
TODS_TIME_LIMIT="${TODS_TIME_LIMIT:-02:30:00}"
YAHOO_TIME_LIMIT="${YAHOO_TIME_LIMIT:-06:00:00}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_tsb_adu_windowed_indexed_text.yaml}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/gpfs/projects/p33222/ybq9740/models/Qwen2.5-VL-7B-Instruct}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_LABEL="${RUN_LABEL:-qwen25vl_7b}"
JOB_PREFIX="${JOB_PREFIX:-tsg-q25rl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
EVAL_SPLITS="${EVAL_SPLITS:-val test}"
ENABLE_RL="${ENABLE_RL:-1}"
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
RL_SAVE_STEPS="${RL_SAVE_STEPS:-100}"
RL_OPTIMIZER="${RL_OPTIMIZER:-adafactor}"
CLEANUP_SFT_AFTER_RL="${CLEANUP_SFT_AFTER_RL:-1}"
CLEANUP_FINAL_MODEL_AFTER_EVAL="${CLEANUP_FINAL_MODEL_AFTER_EVAL:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$BASE_MODEL_PATH" ]]; then
  echo "BASE_MODEL_PATH not found: $BASE_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "CONFIG_PATH not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
WORKFLOW_TAG="sft_rl"
if [[ "$ENABLE_RL" == "0" ]]; then
  WORKFLOW_TAG="sft"
fi
manifest_dir="$OUTPUT_ROOT/${RUN_LABEL}_requested_8subsets_${WORKFLOW_TAG}_$BATCH_TAG"
mkdir -p "$manifest_dir"
submission_manifest="$manifest_dir/submission_manifest.tsv"
printf 'subset\tjob_id\tenable_rl\ttime_limit\tcleanup_sft_after_rl\tcleanup_final_model_after_eval\toutput_dir\trl_output_dir\n' > "$submission_manifest"

if [[ "$ENABLE_RL" == "1" ]]; then
  echo "Submitting requested subsets with SFT -> RL -> final eval: ${SELECTED_SUBSETS[*]}"
else
  echo "Submitting requested subsets with SFT -> final eval: ${SELECTED_SUBSETS[*]}"
fi
echo "submission_manifest=$submission_manifest"

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

  run_tag="${RUN_LABEL}_${subset,,}_${WORKFLOW_TAG}_train_val_test_$BATCH_TAG"
  output_dir="$OUTPUT_ROOT/$run_tag"
  rl_output_dir="$output_dir/rl"
  job_name="${JOB_PREFIX}-${subset,,}"
  subset_time_limit="$TIME_LIMIT"
  if [[ -z "$subset_time_limit" ]]; then
    case "${subset,,}" in
      yahoo) subset_time_limit="$YAHOO_TIME_LIMIT" ;;
      tods) subset_time_limit="$TODS_TIME_LIMIT" ;;
      tao) subset_time_limit="$TAO_TIME_LIMIT" ;;
      daphnet|nek|power|msl|sed) subset_time_limit="$SMALL_SUBSET_TIME_LIMIT" ;;
      *) subset_time_limit="$DEFAULT_TIME_LIMIT" ;;
    esac
  fi

  sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--job-name=$job_name"
    "--gres=$GRES"
    "--mem=$MEMORY"
    "--cpus-per-task=$CPUS"
    "--time=$subset_time_limit"
  )
  if [[ -n "$CONSTRAINT" ]]; then
    sbatch_cmd+=("--constraint=$CONSTRAINT")
  fi
  sbatch_cmd+=(
    "--output=$output_dir/slurm-%x-%j.out"
    "--error=$output_dir/slurm-%x-%j.err"
    "$SBATCH_SCRIPT"
  )

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
    "ENABLE_RL=$ENABLE_RL"
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
    "CLEANUP_SFT_AFTER_RL=$CLEANUP_SFT_AFTER_RL"
    "CLEANUP_FINAL_MODEL_AFTER_EVAL=$CLEANUP_FINAL_MODEL_AFTER_EVAL"
  )

  mkdir -p "$output_dir"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] subset=%s enable_rl=%s time_limit=%s cleanup_sft_after_rl=%s cleanup_final_model_after_eval=%s output_dir=%s rl_output_dir=%s command=' "$subset" "$ENABLE_RL" "$subset_time_limit" "$CLEANUP_SFT_AFTER_RL" "$CLEANUP_FINAL_MODEL_AFTER_EVAL" "$output_dir" "$rl_output_dir"
    printf ' %q' "${env_cmd[@]}" "${sbatch_cmd[@]}"
    printf '\n'
    job_id="DRY_RUN"
  else
    job_id="$("${env_cmd[@]}" "${sbatch_cmd[@]}")"
    echo "subset=$subset job_id=$job_id enable_rl=$ENABLE_RL time_limit=$subset_time_limit cleanup_sft_after_rl=$CLEANUP_SFT_AFTER_RL cleanup_final_model_after_eval=$CLEANUP_FINAL_MODEL_AFTER_EVAL output_dir=$output_dir rl_output_dir=$rl_output_dir"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$subset" "$job_id" "$ENABLE_RL" "$subset_time_limit" "$CLEANUP_SFT_AFTER_RL" "$CLEANUP_FINAL_MODEL_AFTER_EVAL" "$output_dir" "$rl_output_dir" >> "$submission_manifest"
done

echo "Done. submission_manifest=$submission_manifest"
