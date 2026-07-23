#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$ROOT_DIR/run_tsb_adu_subset_train_val_test_qwen.sbatch}"

FIRST_ROUND_ROOT="${FIRST_ROUND_ROOT:-$ROOT_DIR/outputs/anomalydb_sft_rl_8subsets_20260518_233832}"
MODEL_KEYS="${MODEL_KEYS:-7b}"
SUBSETS="${SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
CPUS="${CPUS:-8}"
QOS="${QOS:-}"
DEPENDENCY="${DEPENDENCY:-}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/envs/anomamind/bin/python}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
EVAL_SPLITS="${EVAL_SPLITS:-test}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-}"
ENABLE_RL="${ENABLE_RL:-0}"
CLEANUP_FINAL_MODEL_AFTER_EVAL="${CLEANUP_FINAL_MODEL_AFTER_EVAL:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
ALLOW_PENDING_FIRST_ROUND="${ALLOW_PENDING_FIRST_ROUND:-0}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
DRY_RUN="${DRY_RUN:-0}"

MEMORY_7B="${MEMORY_7B:-80G}"
MEMORY_2B="${MEMORY_2B:-80G}"
MEMORY_4B="${MEMORY_4B:-80G}"
MEMORY_8B="${MEMORY_8B:-80G}"

TINY_SAMPLE_LIMIT="${TINY_SAMPLE_LIMIT:-40}"
SMALL_SAMPLE_LIMIT="${SMALL_SAMPLE_LIMIT:-80}"
MEDIUM_SAMPLE_LIMIT="${MEDIUM_SAMPLE_LIMIT:-120}"
LARGE_SAMPLE_LIMIT="${LARGE_SAMPLE_LIMIT:-250}"
XL_SAMPLE_LIMIT="${XL_SAMPLE_LIMIT:-500}"
TINY_TIME_LIMIT="${TINY_TIME_LIMIT:-01:00:00}"
SMALL_TIME_LIMIT="${SMALL_TIME_LIMIT:-01:30:00}"
MEDIUM_TIME_LIMIT="${MEDIUM_TIME_LIMIT:-02:00:00}"
LARGE_TIME_LIMIT="${LARGE_TIME_LIMIT:-04:00:00}"
XL_TIME_LIMIT="${XL_TIME_LIMIT:-04:00:00}"
XXL_TIME_LIMIT="${XXL_TIME_LIMIT:-08:00:00}"

if [[ "$ENABLE_RL" != "0" ]]; then
  echo "This submitter is for second-round SFT-only; set ENABLE_RL=0." >&2
  exit 1
fi
if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$FIRST_ROUND_ROOT" && "$ALLOW_PENDING_FIRST_ROUND" != "1" ]]; then
  echo "FIRST_ROUND_ROOT not found: $FIRST_ROUND_ROOT" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_ROOT_BASE" ]]; then
  echo "SOURCE_ROOT_BASE not found: $SOURCE_ROOT_BASE" >&2
  exit 1
fi

model_run_label() {
  case "${1,,}" in
    7b|qwen25vl_7b|qwen2.5vl_7b) echo "qwen25vl_7b" ;;
    2b|qwen3_vl_2b) echo "qwen3_vl_2b" ;;
    4b|qwen3_vl_4b) echo "qwen3_vl_4b" ;;
    8b|8n|qwen3_vl_8b) echo "qwen3_vl_8b" ;;
    *) echo "Unsupported MODEL_KEY=$1" >&2; return 1 ;;
  esac
}

model_memory() {
  case "${1,,}" in
    7b|qwen25vl_7b|qwen2.5vl_7b) echo "$MEMORY_7B" ;;
    2b|qwen3_vl_2b) echo "$MEMORY_2B" ;;
    4b|qwen3_vl_4b) echo "$MEMORY_4B" ;;
    8b|8n|qwen3_vl_8b) echo "$MEMORY_8B" ;;
    *) echo "Unsupported MODEL_KEY=$1" >&2; return 1 ;;
  esac
}

count_split_samples() {
  local split_file="$1"
  grep -o '"image_path"' "$split_file" | wc -l
}

subset_sample_count() {
  local subset_root="$1"
  local total=0
  local split_count
  for split in train val test; do
    split_count="$(count_split_samples "$subset_root/$split.json")"
    total=$((total + split_count))
  done
  echo "$total"
}

time_limit_for_sample_count() {
  local sample_count="$1"
  if (( sample_count <= TINY_SAMPLE_LIMIT )); then
    echo "$TINY_TIME_LIMIT"
  elif (( sample_count <= SMALL_SAMPLE_LIMIT )); then
    echo "$SMALL_TIME_LIMIT"
  elif (( sample_count <= MEDIUM_SAMPLE_LIMIT )); then
    echo "$MEDIUM_TIME_LIMIT"
  elif (( sample_count <= LARGE_SAMPLE_LIMIT )); then
    echo "$LARGE_TIME_LIMIT"
  elif (( sample_count <= XL_SAMPLE_LIMIT )); then
    echo "$XL_TIME_LIMIT"
  else
    echo "$XXL_TIME_LIMIT"
  fi
}

read -r -a SELECTED_MODELS <<< "$MODEL_KEYS"
read -r -a SELECTED_SUBSETS <<< "$SUBSETS"

batch_root="$OUTPUT_ROOT/sft2_8subsets_from_anomalydb_sft_$BATCH_TAG"
mkdir -p "$batch_root"
submission_manifest="$batch_root/submission_manifest.tsv"
printf 'model_key\tmodel_label\tsubset\tsample_count\tjob_id\tmemory\ttime_limit\tsft_num_train_epochs\tbase_sft_model\tconfig_path\toutput_dir\tcleanup_final_model_after_eval\tsave_predictions\n' > "$submission_manifest"

echo "Submitting second-round SFT-only jobs from first-round anomaly_db SFT models."
echo "First-round root: $FIRST_ROUND_ROOT"
echo "Models: ${SELECTED_MODELS[*]}"
echo "Subsets: ${SELECTED_SUBSETS[*]}"
echo "Eval splits: $EVAL_SPLITS"
echo "SFT epochs: ${SFT_NUM_TRAIN_EPOCHS:-config default}"
echo "Save predictions: $SAVE_PREDICTIONS"
echo "Cleanup final model after eval: $CLEANUP_FINAL_MODEL_AFTER_EVAL"
echo "Allow pending first round: $ALLOW_PENDING_FIRST_ROUND"
echo "Batch root: $batch_root"

for model_key in "${SELECTED_MODELS[@]}"; do
  run_label="$(model_run_label "$model_key")"
  memory="$(model_memory "$model_key")"
  base_sft_model="$FIRST_ROUND_ROOT/$run_label/model"
  config_path="$FIRST_ROUND_ROOT/$run_label/resolved_config.yaml"

  if [[ ! -d "$base_sft_model" && "$ALLOW_PENDING_FIRST_ROUND" != "1" ]]; then
    echo "Missing first-round SFT model for $model_key: $base_sft_model" >&2
    exit 1
  fi
  if [[ ! -f "$config_path" && "$ALLOW_PENDING_FIRST_ROUND" != "1" ]]; then
    echo "Missing first-round resolved config for $model_key: $config_path" >&2
    exit 1
  fi

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

    sample_count="$(subset_sample_count "$source_root")"
    output_dir="$batch_root/$run_label/$subset"
    job_name="tsg-sft2-${run_label//_/-}-${subset,,}"
    time_limit="$(time_limit_for_sample_count "$sample_count")"
    mkdir -p "$output_dir"

    sbatch_cmd=(
      sbatch
      --parsable
      "--account=$ACCOUNT"
      "--partition=$PARTITION"
      "--job-name=$job_name"
      "--gres=$GRES"
      "--mem=$memory"
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
      "CONFIG_PATH=$config_path"
      "BASE_MODEL_PATH=$base_sft_model"
      "OUTPUT_DIR=$output_dir"
      "PYTHON_BIN=$PYTHON_BIN"
      "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
      "EVAL_SPLITS=$EVAL_SPLITS"
      "SFT_NUM_TRAIN_EPOCHS=$SFT_NUM_TRAIN_EPOCHS"
      "ENABLE_RL=0"
      "CLEANUP_FINAL_MODEL_AFTER_EVAL=$CLEANUP_FINAL_MODEL_AFTER_EVAL"
      "SAVE_PREDICTIONS=$SAVE_PREDICTIONS"
      "BF16=$BF16"
      "FP16=$FP16"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
      printf '[DRY_RUN] model=%s subset=%s sample_count=%s time_limit=%s memory=%s output_dir=%s command=' "$run_label" "$subset" "$sample_count" "$time_limit" "$memory" "$output_dir"
      printf ' %q' "${env_cmd[@]}" "${sbatch_cmd[@]}"
      printf '\n'
      job_id="DRY_RUN"
    else
      job_id="$("${env_cmd[@]}" "${sbatch_cmd[@]}")"
      echo "model=$run_label subset=$subset sample_count=$sample_count job_id=$job_id memory=$memory time_limit=$time_limit output_dir=$output_dir"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$model_key" "$run_label" "$subset" "$sample_count" "$job_id" "$memory" "$time_limit" \
      "${SFT_NUM_TRAIN_EPOCHS:-config default}" "$base_sft_model" "$config_path" "$output_dir" \
      "$CLEANUP_FINAL_MODEL_AFTER_EVAL" "$SAVE_PREDICTIONS" \
      >> "$submission_manifest"
  done
done

echo "Done. submission_manifest=$submission_manifest"
