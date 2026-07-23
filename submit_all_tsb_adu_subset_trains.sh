#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SUBMIT="${TRAIN_SUBMIT:-$ROOT_DIR/submit_tsb_adu_subset_train.sh}"
EVAL_SUBMIT="${EVAL_SUBMIT:-$ROOT_DIR/submit_qwen25vl_predict_eval.sh}"
SUMMARY_RUN_SCRIPT="${SUMMARY_RUN_SCRIPT:-$ROOT_DIR/run_tsb_adu_subset_summary.sh}"

SUBSETS=(
  Exathlon
  IOPS
  LTDB
  MSL
  NAB
  NEK
  OPPORTUNITY
  SMAP
  SMD
  SVDB
  UCR
  WSD
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
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_tsb_adu_windowed_indexed_text.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
MEMORY="${MEMORY:-96G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
EVAL_MEMORY="${EVAL_MEMORY:-96G}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-08:00:00}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TRAIN_DEPENDENCY="${TRAIN_DEPENDENCY:-}"
DRY_RUN="${DRY_RUN:-0}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY_MEMORY="${SUMMARY_MEMORY:-16G}"
SUMMARY_TIME_LIMIT="${SUMMARY_TIME_LIMIT:-04:00:00}"
TRAIN_JOB_NAME_PREFIX="${TRAIN_JOB_NAME_PREFIX:-tsg-tsb-adu-}"
TRAIN_JOB_NAME_SUFFIX="${TRAIN_JOB_NAME_SUFFIX:-}"
EVAL_JOB_NAME_PREFIX="${EVAL_JOB_NAME_PREFIX:-tsg-tsb-adu-}"
EVAL_JOB_NAME_SUFFIX="${EVAL_JOB_NAME_SUFFIX:--eval}"
SUMMARY_JOB_NAME="${SUMMARY_JOB_NAME:-tsg-tsb-adu-summary}"

if [[ ! -x "$TRAIN_SUBMIT" ]]; then
  echo "Missing train submit script: $TRAIN_SUBMIT" >&2
  exit 1
fi

if [[ ! -f "$EVAL_SUBMIT" ]]; then
  echo "Missing eval submit script: $EVAL_SUBMIT" >&2
  exit 1
fi

if [[ ! -f "$SUMMARY_RUN_SCRIPT" ]]; then
  echo "Missing summary run script: $SUMMARY_RUN_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

summary_root="$OUTPUT_ROOT/tsb_adu_subset_batch_$BATCH_TAG"
mkdir -p "$summary_root"
submission_manifest="$summary_root/submission_manifest.tsv"
summary_output_dir="$summary_root/summary"
printf 'subset\ttrain_job_id\teval_job_id\trun_dir\n' > "$submission_manifest"
eval_job_ids=()

echo "Submitting TSB-AD-U subset jobs for: ${SUBSETS[*]}"

for subset in "${SUBSETS[@]}"; do
  run_tag="qwen25vl_7b_tsb_adu_${subset,,}_$BATCH_TAG"
  output_dir="$OUTPUT_ROOT/$run_tag"
  mkdir -p "$output_dir"

  env_cmd=(env)
  source_root_env=()
  if [[ -f "$SOURCE_ROOT/train.json" ]]; then
    source_root_env=("SOURCE_ROOT=$SOURCE_ROOT")
  elif [[ -f "$SOURCE_ROOT/$subset/train.json" ]]; then
    env_cmd+=(-u SOURCE_ROOT)
    source_root_env=("SOURCE_ROOT_BASE=$SOURCE_ROOT")
  else
    echo "Dataset layout for subset '$subset' not found under SOURCE_ROOT=$SOURCE_ROOT" >&2
    exit 1
  fi

  train_job_id="$(
    "${env_cmd[@]}" \
      "ACCOUNT=$ACCOUNT" \
      "PARTITION=$PARTITION" \
      "CONSTRAINT=$CONSTRAINT" \
      "MODEL_PATH=$MODEL_PATH" \
      "OUTPUT_ROOT=$OUTPUT_ROOT" \
      "${source_root_env[@]}" \
      "CONFIG_PATH=$CONFIG_PATH" \
      "MEMORY=$MEMORY" \
      "TIME_LIMIT=$TIME_LIMIT" \
      "DEPENDENCY=$TRAIN_DEPENDENCY" \
      "RUN_TAG=$run_tag" \
      "OUTPUT_DIR=$output_dir" \
      "JOB_NAME=${TRAIN_JOB_NAME_PREFIX}${subset,,}${TRAIN_JOB_NAME_SUFFIX}" \
      "PARSABLE=1" \
      "DRY_RUN=$DRY_RUN" \
      bash "$TRAIN_SUBMIT" "$subset"
  )"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] subset=$subset train_job_id=$train_job_id"
    continue
  fi

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
    JOB_NAME="${EVAL_JOB_NAME_PREFIX}${subset,,}${EVAL_JOB_NAME_SUFFIX}" \
    DEPENDENCY="afterok:$train_job_id" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    ALLOW_MISSING_RUN_ARTIFACTS=1 \
    PARSABLE=1 \
    bash "$EVAL_SUBMIT"
  )"

  printf '%s\t%s\t%s\t%s\n' "$subset" "$train_job_id" "$eval_job_id" "$output_dir" >> "$submission_manifest"
  eval_job_ids+=("$eval_job_id")
  echo "subset=$subset train_job_id=$train_job_id eval_job_id=$eval_job_id run_dir=$output_dir"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] submission_manifest=$submission_manifest"
  exit 0
fi

if [[ ${#eval_job_ids[@]} -eq 0 ]]; then
  echo "No eval jobs were submitted; skipping summary job." >&2
  exit 1
fi

dependency="afterok:$(IFS=:; echo "${eval_job_ids[*]}")"
summary_sbatch_cmd=(
  sbatch
  --parsable
  "--account=$ACCOUNT"
  "--partition=$PARTITION"
  "--job-name=$SUMMARY_JOB_NAME"
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
  "--wrap=PROJECT_ROOT=$ROOT_DIR PYTHON_BIN=/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python SUBMISSION_MANIFEST=$submission_manifest SUMMARY_OUTPUT_DIR=$summary_output_dir bash $SUMMARY_RUN_SCRIPT"
)

summary_job_id="$("${summary_sbatch_cmd[@]}")"

echo "summary_job_id=$summary_job_id summary_root=$summary_root submission_manifest=$submission_manifest"
