#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SUBMIT_INDEX="${TRAIN_SUBMIT_INDEX:-$ROOT_DIR/submit_tsb_adu_subset_train.sh}"
TRAIN_SUBMIT_NO_INDEX="${TRAIN_SUBMIT_NO_INDEX:-$ROOT_DIR/submit_tsb_adu_subset_train_no_index.sh}"
EVAL_SUBMIT="${EVAL_SUBMIT:-$ROOT_DIR/submit_qwen25vl_predict_eval.sh}"
SUMMARY_RUN_SCRIPT="${SUMMARY_RUN_SCRIPT:-$ROOT_DIR/run_tsb_adu_subset_summary.sh}"

SUBSETS=(
  Daphnet
  MSL
  NEK
  Power
  SED
  TAO
  TODS
  YAHOO
)

if [[ -n "${SUBSETS_OVERRIDE:-}" ]]; then
  read -r -a SUBSETS <<< "$SUBSETS_OVERRIDE"
fi

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
CONSTRAINT="${CONSTRAINT:-quest12&sxm}"
MODEL_PATH="${MODEL_PATH:-/gpfs/projects/p33222/ybq9740/models/Qwen2.5-VL-7B-Instruct}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
INDEX_CONFIG_PATH="${INDEX_CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_tsb_adu_windowed_indexed_text_plain_image.yaml}"
NO_INDEX_CONFIG_PATH="${NO_INDEX_CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_tsb_adu_windowed_plain_no_index.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT_DIR/outputs}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
EVAL_MEMORY="${EVAL_MEMORY:-96G}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-08:00:00}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TRAIN_DEPENDENCY="${TRAIN_DEPENDENCY:-}"
SUMMARY_MEMORY="${SUMMARY_MEMORY:-16G}"
SUMMARY_TIME_LIMIT="${SUMMARY_TIME_LIMIT:-04:00:00}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

for required_file in "$TRAIN_SUBMIT_INDEX" "$TRAIN_SUBMIT_NO_INDEX" "$EVAL_SUBMIT" "$SUMMARY_RUN_SCRIPT" "$INDEX_CONFIG_PATH" "$NO_INDEX_CONFIG_PATH"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 1
  fi
done

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Dataset root not found: $SOURCE_ROOT" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_BASE"

submit_condition() {
  local condition="$1"
  local config_path="$2"
  local train_submit="$3"
  local train_job_suffix="$4"
  local eval_job_suffix="$5"
  local summary_job_name="$6"

  local condition_root="$OUTPUT_BASE/$condition"
  local summary_root="$condition_root/_summary"
  local submission_manifest="$summary_root/submission_manifest.tsv"
  local eval_job_ids=()

  mkdir -p "$condition_root" "$summary_root"
  printf 'subset\ttrain_job_id\teval_job_id\trun_dir\n' > "$submission_manifest"

  echo "Submitting condition=$condition for subsets: ${SUBSETS[*]}"

  for subset in "${SUBSETS[@]}"; do
    local output_dir="$condition_root/$subset"
    local train_job_id=""
    local eval_job_id=""

    mkdir -p "$output_dir"

    if [[ "$DRY_RUN" == "1" ]]; then
      env -u SOURCE_ROOT \
        ACCOUNT="$ACCOUNT" \
        PARTITION="$PARTITION" \
        CONSTRAINT="$CONSTRAINT" \
        MODEL_PATH="$MODEL_PATH" \
        OUTPUT_ROOT="$condition_root" \
        SOURCE_ROOT_BASE="$SOURCE_ROOT" \
        CONFIG_PATH="$config_path" \
        MEMORY="$MEMORY" \
        TIME_LIMIT="$TIME_LIMIT" \
        DEPENDENCY="$TRAIN_DEPENDENCY" \
        RUN_TAG="qwen25vl_7b_tsb_adu_${subset,,}_${condition}_${RUN_STAMP}" \
        OUTPUT_DIR="$output_dir" \
        JOB_NAME="tsg-tsb-adu-${subset,,}${train_job_suffix}" \
        PARSABLE=0 \
        DRY_RUN=1 \
        bash "$train_submit" "$subset"
      train_job_id="dryrun-train-${subset}"

      ACCOUNT="$ACCOUNT" \
      PARTITION="$PARTITION" \
      CONSTRAINT="$CONSTRAINT" \
      MEMORY="$EVAL_MEMORY" \
      TIME_LIMIT="$EVAL_TIME_LIMIT" \
      RUN_DIR="$output_dir" \
      CONFIG_PATH="$output_dir/resolved_config.yaml" \
      MODEL_PATH="$output_dir/model" \
      DATASET_JSONL="$output_dir/dataset_cache/test.jsonl" \
      EVAL_DIR="$output_dir/eval" \
      PREDICTIONS_PATH="$output_dir/eval/test_predictions.jsonl" \
      METRICS_PATH="$output_dir/eval/test_metrics.json" \
      JOB_NAME="tsg-tsb-adu-${subset,,}${eval_job_suffix}" \
      DEPENDENCY="afterok:$train_job_id" \
      MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
      ALLOW_MISSING_RUN_ARTIFACTS=1 \
      PARSABLE=0 \
      DRY_RUN=1 \
      bash "$EVAL_SUBMIT"
      eval_job_id="dryrun-eval-${subset}"
    else
      train_job_id="$(
        env -u SOURCE_ROOT \
          ACCOUNT="$ACCOUNT" \
          PARTITION="$PARTITION" \
          CONSTRAINT="$CONSTRAINT" \
          MODEL_PATH="$MODEL_PATH" \
          OUTPUT_ROOT="$condition_root" \
          SOURCE_ROOT_BASE="$SOURCE_ROOT" \
          CONFIG_PATH="$config_path" \
          MEMORY="$MEMORY" \
          TIME_LIMIT="$TIME_LIMIT" \
          DEPENDENCY="$TRAIN_DEPENDENCY" \
          RUN_TAG="qwen25vl_7b_tsb_adu_${subset,,}_${condition}_${RUN_STAMP}" \
          OUTPUT_DIR="$output_dir" \
          JOB_NAME="tsg-tsb-adu-${subset,,}${train_job_suffix}" \
          PARSABLE=1 \
          DRY_RUN=0 \
          bash "$train_submit" "$subset"
      )"

      eval_job_id="$(
        ACCOUNT="$ACCOUNT" \
        PARTITION="$PARTITION" \
        CONSTRAINT="$CONSTRAINT" \
        MEMORY="$EVAL_MEMORY" \
        TIME_LIMIT="$EVAL_TIME_LIMIT" \
        RUN_DIR="$output_dir" \
        CONFIG_PATH="$output_dir/resolved_config.yaml" \
        MODEL_PATH="$output_dir/model" \
        DATASET_JSONL="$output_dir/dataset_cache/test.jsonl" \
        EVAL_DIR="$output_dir/eval" \
        PREDICTIONS_PATH="$output_dir/eval/test_predictions.jsonl" \
        METRICS_PATH="$output_dir/eval/test_metrics.json" \
        JOB_NAME="tsg-tsb-adu-${subset,,}${eval_job_suffix}" \
        DEPENDENCY="afterok:$train_job_id" \
        MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        ALLOW_MISSING_RUN_ARTIFACTS=1 \
        PARSABLE=1 \
        DRY_RUN=0 \
        bash "$EVAL_SUBMIT"
      )"
    fi

    printf '%s\t%s\t%s\t%s\n' "$subset" "$train_job_id" "$eval_job_id" "$output_dir" >> "$submission_manifest"
    eval_job_ids+=("$eval_job_id")
    echo "condition=$condition subset=$subset train_job_id=$train_job_id eval_job_id=$eval_job_id run_dir=$output_dir"
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] condition=$condition submission_manifest=$submission_manifest summary_root=$summary_root"
    return 0
  fi

  local dependency="afterany:$(IFS=:; echo "${eval_job_ids[*]}")"
  local summary_sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--job-name=$summary_job_name"
    --gres=gpu:1
    "--mem=$SUMMARY_MEMORY"
    --cpus-per-task=2
    "--time=$SUMMARY_TIME_LIMIT"
    "--dependency=$dependency"
    "--output=$summary_root/summary_%j.out"
    "--error=$summary_root/summary_%j.err"
  )

  if [[ -n "$CONSTRAINT" ]]; then
    summary_sbatch_cmd+=("--constraint=$CONSTRAINT")
  fi

  summary_sbatch_cmd+=(
    "--wrap=PROJECT_ROOT=$ROOT_DIR PYTHON_BIN=$PYTHON_BIN SUBMISSION_MANIFEST=$submission_manifest SUMMARY_OUTPUT_DIR=$summary_root bash $SUMMARY_RUN_SCRIPT"
  )

  local summary_job_id
  summary_job_id="$("${summary_sbatch_cmd[@]}")"
  echo "condition=$condition summary_job_id=$summary_job_id submission_manifest=$submission_manifest summary_root=$summary_root"
}

submit_condition \
  index \
  "$INDEX_CONFIG_PATH" \
  "$TRAIN_SUBMIT_INDEX" \
  "-idx" \
  "-idx-eval" \
  "tsg-tsb-adu-index-summary"

submit_condition \
  no_index \
  "$NO_INDEX_CONFIG_PATH" \
  "$TRAIN_SUBMIT_NO_INDEX" \
  "-noidx" \
  "-noidx-eval" \
  "tsg-tsb-adu-noindex-summary"
