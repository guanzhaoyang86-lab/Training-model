#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$ROOT_DIR/submit_qwen3_vl_train_eval.sh}"
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

MODEL_KEYS=(
  2b
  4b
  8b
)

if [[ -n "${SUBSETS_OVERRIDE:-}" ]]; then
  read -r -a SUBSETS <<< "$SUBSETS_OVERRIDE"
fi

if [[ -n "${MODEL_KEYS_OVERRIDE:-}" ]]; then
  read -r -a MODEL_KEYS <<< "$MODEL_KEYS_OVERRIDE"
fi

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
CONSTRAINT="${CONSTRAINT:-quest12&sxm}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
QOS="${QOS:-}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_qwen3_vl_tsb_adu_indexed_text_plain_image.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT_DIR/outputs/qwen3_vl_tsb_adu_index_plain_image}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SPLIT="${SPLIT:-test}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
SUBMIT_SUMMARY="${SUBMIT_SUMMARY:-1}"
SUMMARY_MEMORY="${SUMMARY_MEMORY:-16G}"
SUMMARY_TIME_LIMIT="${SUMMARY_TIME_LIMIT:-04:00:00}"
SUMMARY_GRES="${SUMMARY_GRES:-gpu:1}"

MEMORY_2B="${MEMORY_2B:-140G}"
MEMORY_4B="${MEMORY_4B:-180G}"
MEMORY_8B="${MEMORY_8B:-240G}"
TIME_LIMIT_2B="${TIME_LIMIT_2B:-12:00:00}"
TIME_LIMIT_4B="${TIME_LIMIT_4B:-16:00:00}"
TIME_LIMIT_8B="${TIME_LIMIT_8B:-24:00:00}"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Submit script not found: $SUBMIT_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$SUMMARY_RUN_SCRIPT" ]]; then
  echo "Summary script not found: $SUMMARY_RUN_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_ROOT_BASE" ]]; then
  echo "Dataset root not found: $SOURCE_ROOT_BASE" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

memory_for_model() {
  case "$1" in
    2b) echo "$MEMORY_2B" ;;
    4b) echo "$MEMORY_4B" ;;
    8b) echo "$MEMORY_8B" ;;
    *)
      echo "Unsupported MODEL_KEY: $1" >&2
      return 1
      ;;
  esac
}

time_for_model() {
  case "$1" in
    2b) echo "$TIME_LIMIT_2B" ;;
    4b) echo "$TIME_LIMIT_4B" ;;
    8b) echo "$TIME_LIMIT_8B" ;;
    *)
      echo "Unsupported MODEL_KEY: $1" >&2
      return 1
      ;;
  esac
}

mkdir -p "$OUTPUT_BASE"

batch_root="$OUTPUT_BASE/batch_$RUN_STAMP"
mkdir -p "$batch_root"
combined_manifest="$batch_root/submission_manifest_all.tsv"
printf 'model_key\tsubset\ttrain_job_id\teval_job_id\trun_dir\n' > "$combined_manifest"

echo "Submitting Qwen3-VL TSB-AD-U indexed-text/plain-image runs"
echo "  models: ${MODEL_KEYS[*]}"
echo "  subsets: ${SUBSETS[*]}"
echo "  batch_root: $batch_root"
echo "  config: $CONFIG_PATH"

for model_key in "${MODEL_KEYS[@]}"; do
  memory="$(memory_for_model "$model_key")"
  time_limit="$(time_for_model "$model_key")"
  model_root="$batch_root/qwen3_vl_${model_key}"
  summary_root="$model_root/_summary"
  submission_manifest="$summary_root/submission_manifest.tsv"
  job_ids=()

  mkdir -p "$model_root" "$summary_root"
  printf 'subset\ttrain_job_id\teval_job_id\trun_dir\n' > "$submission_manifest"

  echo "==== model=$model_key memory=$memory time=$time_limit ===="

  for subset in "${SUBSETS[@]}"; do
    source_root="$SOURCE_ROOT_BASE/$subset"
    output_dir="$model_root/$subset"
    job_name="tsg-q3vl-${model_key}-${subset,,}-idx"
    run_tag="qwen3_vl_${model_key}_tsb_adu_${subset,,}_index_plain_image_$RUN_STAMP"

    if [[ ! -f "$source_root/train.json" ]]; then
      echo "Subset dataset not found: $source_root" >&2
      exit 1
    fi

    mkdir -p "$output_dir"

    if [[ "$DRY_RUN" == "1" ]]; then
      ACCOUNT="$ACCOUNT" \
      PARTITION="$PARTITION" \
      CONSTRAINT="$CONSTRAINT" \
      GRES="$GRES" \
      CPUS="$CPUS" \
      QOS="$QOS" \
      PYTHON_BIN="$PYTHON_BIN" \
      CONFIG_PATH="$CONFIG_PATH" \
      SOURCE_ROOT="$source_root" \
      OUTPUT_DIR="$output_dir" \
      RUN_TAG="$run_tag" \
      MODEL_KEY="$model_key" \
      JOB_NAME="$job_name" \
      MEMORY="$memory" \
      TIME_LIMIT="$time_limit" \
      BF16="$BF16" \
      FP16="$FP16" \
      PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
      SPLIT="$SPLIT" \
      MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
      PARSABLE=0 \
      DRY_RUN=1 \
      bash "$SUBMIT_SCRIPT"
      job_id="dryrun-${model_key}-${subset}"
    else
      job_id="$(
        ACCOUNT="$ACCOUNT" \
        PARTITION="$PARTITION" \
        CONSTRAINT="$CONSTRAINT" \
        GRES="$GRES" \
        CPUS="$CPUS" \
        QOS="$QOS" \
        PYTHON_BIN="$PYTHON_BIN" \
        CONFIG_PATH="$CONFIG_PATH" \
        SOURCE_ROOT="$source_root" \
        OUTPUT_DIR="$output_dir" \
        RUN_TAG="$run_tag" \
        MODEL_KEY="$model_key" \
        JOB_NAME="$job_name" \
        MEMORY="$memory" \
        TIME_LIMIT="$time_limit" \
        BF16="$BF16" \
        FP16="$FP16" \
        PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
        SPLIT="$SPLIT" \
        MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        PARSABLE=1 \
        DRY_RUN=0 \
        bash "$SUBMIT_SCRIPT"
      )"
      job_ids+=("$job_id")
    fi

    printf '%s\t%s\t%s\t%s\n' "$subset" "$job_id" "$job_id" "$output_dir" >> "$submission_manifest"
    printf '%s\t%s\t%s\t%s\t%s\n' "$model_key" "$subset" "$job_id" "$job_id" "$output_dir" >> "$combined_manifest"
    echo "model=$model_key subset=$subset job_id=$job_id run_dir=$output_dir"
  done

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] model=$model_key submission_manifest=$submission_manifest"
    continue
  fi

  if [[ "$SUBMIT_SUMMARY" != "1" ]]; then
    echo "model=$model_key summary skipped; submission_manifest=$submission_manifest"
    continue
  fi

  dependency="afterany:$(IFS=:; echo "${job_ids[*]}")"
  summary_job_name="tsg-q3vl-${model_key}-idx-summary"
  summary_sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--job-name=$summary_job_name"
    "--mem=$SUMMARY_MEMORY"
    --cpus-per-task=2
    "--time=$SUMMARY_TIME_LIMIT"
    "--dependency=$dependency"
    "--output=$summary_root/summary_%j.out"
    "--error=$summary_root/summary_%j.err"
  )
  if [[ -n "$SUMMARY_GRES" ]]; then
    summary_sbatch_cmd+=("--gres=$SUMMARY_GRES")
  fi
  if [[ -n "$CONSTRAINT" ]]; then
    summary_sbatch_cmd+=("--constraint=$CONSTRAINT")
  fi
  if [[ -n "$QOS" ]]; then
    summary_sbatch_cmd+=("--qos=$QOS")
  fi
  summary_sbatch_cmd+=(
    "--wrap=PROJECT_ROOT=$ROOT_DIR PYTHON_BIN=$PYTHON_BIN SUBMISSION_MANIFEST=$submission_manifest SUMMARY_OUTPUT_DIR=$summary_root bash $SUMMARY_RUN_SCRIPT"
  )
  summary_job_id="$("${summary_sbatch_cmd[@]}")"
  echo "model=$model_key summary_job_id=$summary_job_id submission_manifest=$submission_manifest summary_root=$summary_root"
done

echo "combined_manifest=$combined_manifest"
