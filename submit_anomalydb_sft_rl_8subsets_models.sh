#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-$ROOT_DIR/run_anomalydb_sft_rl_8subsets_qwen.sbatch}"

MODEL_KEYS="${MODEL_KEYS:-7b 2b 4b 8b}"
ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
CPUS="${CPUS:-8}"
QOS="${QOS:-}"
DEPENDENCY="${DEPENDENCY:-}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_qwen3_vl_8b_anomaly_db_indexed_text_plain_image.yaml}"
ANOMALY_SOURCE_ROOT="${ANOMALY_SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
TSB_SOURCE_ROOT_BASE="${TSB_SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
EVAL_SUBSETS="${EVAL_SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"

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
DRY_RUN="${DRY_RUN:-0}"

MEMORY_7B="${MEMORY_7B:-120G}"
MEMORY_2B="${MEMORY_2B:-96G}"
MEMORY_4B="${MEMORY_4B:-120G}"
MEMORY_8B="${MEMORY_8B:-160G}"
TIME_LIMIT_7B="${TIME_LIMIT_7B:-1-12:00:00}"
TIME_LIMIT_2B="${TIME_LIMIT_2B:-18:00:00}"
TIME_LIMIT_4B="${TIME_LIMIT_4B:-1-00:00:00}"
TIME_LIMIT_8B="${TIME_LIMIT_8B:-1-18:00:00}"

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
  echo "Missing sbatch script: $SBATCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "CONFIG_PATH not found: $CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -d "$ANOMALY_SOURCE_ROOT" ]]; then
  echo "ANOMALY_SOURCE_ROOT not found: $ANOMALY_SOURCE_ROOT" >&2
  exit 1
fi
if [[ ! -d "$TSB_SOURCE_ROOT_BASE" ]]; then
  echo "TSB_SOURCE_ROOT_BASE not found: $TSB_SOURCE_ROOT_BASE" >&2
  exit 1
fi

batch_root="$OUTPUT_ROOT/anomalydb_sft_rl_8subsets_$BATCH_TAG"
mkdir -p "$batch_root"
submission_manifest="$batch_root/submission_manifest.tsv"
printf 'model_key\tjob_id\tmemory\ttime_limit\toutput_dir\trl_output_dir\n' > "$submission_manifest"

model_memory() {
  case "${1,,}" in
    7b|qwen25vl_7b|qwen2.5vl_7b) echo "$MEMORY_7B" ;;
    2b|qwen3_vl_2b) echo "$MEMORY_2B" ;;
    4b|qwen3_vl_4b) echo "$MEMORY_4B" ;;
    8b|8n|qwen3_vl_8b) echo "$MEMORY_8B" ;;
    *) echo "Unsupported MODEL_KEY=$1" >&2; return 1 ;;
  esac
}

model_time_limit() {
  case "${1,,}" in
    7b|qwen25vl_7b|qwen2.5vl_7b) echo "$TIME_LIMIT_7B" ;;
    2b|qwen3_vl_2b) echo "$TIME_LIMIT_2B" ;;
    4b|qwen3_vl_4b) echo "$TIME_LIMIT_4B" ;;
    8b|8n|qwen3_vl_8b) echo "$TIME_LIMIT_8B" ;;
    *) echo "Unsupported MODEL_KEY=$1" >&2; return 1 ;;
  esac
}

model_run_label() {
  case "${1,,}" in
    7b|qwen25vl_7b|qwen2.5vl_7b) echo "qwen25vl_7b" ;;
    2b|qwen3_vl_2b) echo "qwen3_vl_2b" ;;
    4b|qwen3_vl_4b) echo "qwen3_vl_4b" ;;
    8b|8n|qwen3_vl_8b) echo "qwen3_vl_8b" ;;
    *) echo "Unsupported MODEL_KEY=$1" >&2; return 1 ;;
  esac
}

model_path() {
  case "${1,,}" in
    7b|qwen25vl_7b|qwen2.5vl_7b) echo "$WORKSPACE_ROOT/models/Qwen2.5-VL-7B-Instruct" ;;
    2b|qwen3_vl_2b) echo "$WORKSPACE_ROOT/models/Qwen3-VL-2B-Instruct" ;;
    4b|qwen3_vl_4b) echo "$WORKSPACE_ROOT/models/Qwen3-VL-4B-Instruct" ;;
    8b|8n|qwen3_vl_8b) echo "$WORKSPACE_ROOT/models/Qwen3-VL-8B-Instruct" ;;
    *) echo "Unsupported MODEL_KEY=$1" >&2; return 1 ;;
  esac
}

read -r -a SELECTED_MODELS <<< "$MODEL_KEYS"
echo "Submitting anomaly_db_v1 SFT+RL once per model, then testing subsets: $EVAL_SUBSETS"
echo "Models: ${SELECTED_MODELS[*]}"
echo "Submission manifest: $submission_manifest"

for model_key in "${SELECTED_MODELS[@]}"; do
  run_label="$(model_run_label "$model_key")"
  memory="$(model_memory "$model_key")"
  time_limit="$(model_time_limit "$model_key")"
  base_model_path="$(model_path "$model_key")"
  output_dir="$batch_root/$run_label"
  rl_output_dir="$output_dir/rl"
  job_name="tsg-adb-${run_label//_/-}"

  if [[ ! -d "$base_model_path" ]]; then
    echo "Model path not found for $model_key: $base_model_path" >&2
    exit 1
  fi

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
    "MODEL_KEY=$model_key"
    "RUN_LABEL=$run_label"
    "BASE_MODEL_PATH=$base_model_path"
    "CONFIG_PATH=$CONFIG_PATH"
    "OUTPUT_DIR=$output_dir"
    "RL_OUTPUT_DIR=$rl_output_dir"
    "PYTHON_BIN=$PYTHON_BIN"
    "ANOMALY_SOURCE_ROOT=$ANOMALY_SOURCE_ROOT"
    "TSB_SOURCE_ROOT_BASE=$TSB_SOURCE_ROOT_BASE"
    "EVAL_SUBSETS=$EVAL_SUBSETS"
    "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
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
    "BF16=$BF16"
    "FP16=$FP16"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] model=%s output_dir=%s command=' "$model_key" "$output_dir"
    printf ' %q' "${env_cmd[@]}" "${sbatch_cmd[@]}"
    printf '\n'
    job_id="DRY_RUN"
  else
    job_id="$("${env_cmd[@]}" "${sbatch_cmd[@]}")"
    echo "model=$model_key job_id=$job_id output_dir=$output_dir rl_output_dir=$rl_output_dir"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model_key" "$job_id" "$memory" "$time_limit" "$output_dir" "$rl_output_dir" >> "$submission_manifest"
done

echo "Done. submission_manifest=$submission_manifest"
