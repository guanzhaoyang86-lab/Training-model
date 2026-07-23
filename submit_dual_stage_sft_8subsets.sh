#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"

STAGE1_SBATCH_SCRIPT="${STAGE1_SBATCH_SCRIPT:-$ROOT_DIR/run_full_train_qwen.sbatch}"
STAGE2_SUBMITTER="${STAGE2_SUBMITTER:-$ROOT_DIR/submit_sft2_8subsets_from_anomalydb_sft.sh}"

MODEL_KEYS="${MODEL_KEYS:-7b}"
SUBSETS="${SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
CPUS="${CPUS:-8}"
QOS="${QOS:-}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_anomaly_db_indexed_text_plain_image.yaml}"
ANOMALY_SOURCE_ROOT="${ANOMALY_SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
EVAL_SPLITS="${EVAL_SPLITS:-test}"
STAGE1_SFT_NUM_TRAIN_EPOCHS="${STAGE1_SFT_NUM_TRAIN_EPOCHS:-1}"
STAGE2_SFT_NUM_TRAIN_EPOCHS="${STAGE2_SFT_NUM_TRAIN_EPOCHS:-}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
CLEANUP_FINAL_MODEL_AFTER_EVAL="${CLEANUP_FINAL_MODEL_AFTER_EVAL:-1}"
BF16="${BF16:-1}"
FP16="${FP16:-0}"
DRY_RUN="${DRY_RUN:-0}"

MEMORY_7B="${MEMORY_7B:-120G}"
MEMORY_2B="${MEMORY_2B:-96G}"
MEMORY_4B="${MEMORY_4B:-120G}"
MEMORY_8B="${MEMORY_8B:-160G}"
TIME_LIMIT_7B="${TIME_LIMIT_7B:-04:00:00}"
TIME_LIMIT_2B="${TIME_LIMIT_2B:-18:00:00}"
TIME_LIMIT_4B="${TIME_LIMIT_4B:-1-00:00:00}"
TIME_LIMIT_8B="${TIME_LIMIT_8B:-1-18:00:00}"

if [[ ! -f "$STAGE1_SBATCH_SCRIPT" ]]; then
  echo "Missing stage-1 sbatch script: $STAGE1_SBATCH_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$STAGE2_SUBMITTER" ]]; then
  echo "Missing stage-2 submitter: $STAGE2_SUBMITTER" >&2
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
if [[ ! -d "$SOURCE_ROOT_BASE" ]]; then
  echo "SOURCE_ROOT_BASE not found: $SOURCE_ROOT_BASE" >&2
  exit 1
fi
for split in train val test; do
  if [[ ! -f "$ANOMALY_SOURCE_ROOT/$split.json" ]]; then
    echo "Missing synthetic split: $ANOMALY_SOURCE_ROOT/$split.json" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
data = cfg.get("data", {})
errors = []
if data.get("include_qa_pairs", False):
    errors.append("include_qa_pairs must be false")
if str(data.get("split_file_suffix", "")):
    errors.append("split_file_suffix must be empty")
if not data.get("include_indexed_series_text", False):
    errors.append("include_indexed_series_text must be true")
normalization = data.get("series_normalization", {})
if not isinstance(normalization, dict) or not normalization.get("enabled", False):
    errors.append("series_normalization.enabled must be true")
if not normalization.get("normalize_text", False):
    errors.append("series_normalization.normalize_text must be true")
if not normalization.get("normalize_image", False):
    errors.append("series_normalization.normalize_image must be true")
if errors:
    raise SystemExit("Config preflight failed: " + "; ".join(errors))
print("Config preflight ok: no QA, indexed text enabled, numeric normalization enabled.")
PY

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

read -r -a SELECTED_MODELS <<< "$MODEL_KEYS"
batch_root="$OUTPUT_ROOT/dual_stage_sft_8subsets_$BATCH_TAG"
stage1_root="$batch_root/stage1"
stage2_root="$batch_root/stage2"
mkdir -p "$stage1_root" "$stage2_root"

submission_manifest="$batch_root/submission_manifest.tsv"
printf 'model_key\tmodel_label\tstage1_job_id\tstage1_output_dir\tstage2_manifest\n' > "$submission_manifest"

echo "Submitting dual-stage SFT workflow."
echo "Stage 1: synthetic anomaly_db SFT."
echo "Stage 2: real 8-subset SFT, then test metrics only."
echo "Config: $CONFIG_PATH"
echo "Synthetic source: $ANOMALY_SOURCE_ROOT"
echo "Real subset root: $SOURCE_ROOT_BASE"
echo "Stage 1 SFT epochs: $STAGE1_SFT_NUM_TRAIN_EPOCHS"
echo "Stage 2 SFT epochs: ${STAGE2_SFT_NUM_TRAIN_EPOCHS:-config default}"
echo "Models: ${SELECTED_MODELS[*]}"
echo "Subsets: $SUBSETS"
echo "Save predictions: $SAVE_PREDICTIONS"
echo "Cleanup final subset model after eval: $CLEANUP_FINAL_MODEL_AFTER_EVAL"
echo "Batch root: $batch_root"

for model_key in "${SELECTED_MODELS[@]}"; do
  run_label="$(model_run_label "$model_key")"
  base_model_path="$(model_path "$model_key")"
  memory="$(model_memory "$model_key")"
  time_limit="$(model_time_limit "$model_key")"
  stage1_output_dir="$stage1_root/$run_label"
  stage2_tag="${BATCH_TAG}_${run_label}"
  stage2_manifest="$stage2_root/sft2_8subsets_from_anomalydb_sft_$stage2_tag/submission_manifest.tsv"

  if [[ ! -d "$base_model_path" ]]; then
    echo "Model path not found for $model_key: $base_model_path" >&2
    exit 1
  fi
  mkdir -p "$stage1_output_dir"

  stage1_sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--job-name=tsg-sft1-${run_label//_/-}-synthetic"
    "--gres=$GRES"
    "--mem=$memory"
    "--cpus-per-task=$CPUS"
    "--time=$time_limit"
    "--output=$stage1_output_dir/slurm-%x-%j.out"
    "--error=$stage1_output_dir/slurm-%x-%j.err"
  )
  if [[ -n "$CONSTRAINT" ]]; then
    stage1_sbatch_cmd+=("--constraint=$CONSTRAINT")
  fi
  if [[ -n "$QOS" ]]; then
    stage1_sbatch_cmd+=("--qos=$QOS")
  fi
  stage1_sbatch_cmd+=("$STAGE1_SBATCH_SCRIPT")

  stage1_env_cmd=(
    env
    "PROJECT_ROOT=$ROOT_DIR"
    "CONFIG_PATH=$CONFIG_PATH"
    "OUTPUT_DIR=$stage1_output_dir"
    "MODE=train"
    "PYTHON_BIN=$PYTHON_BIN"
    "MODEL_PATH=$base_model_path"
    "SOURCE_ROOT=$ANOMALY_SOURCE_ROOT"
    "SFT_NUM_TRAIN_EPOCHS=$STAGE1_SFT_NUM_TRAIN_EPOCHS"
    "BF16=$BF16"
    "FP16=$FP16"
    "LAUNCHER=python"
    "NUM_PROCS_PER_NODE=1"
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] stage1 model=%s output_dir=%s command=' "$run_label" "$stage1_output_dir"
    printf ' %q' "${stage1_env_cmd[@]}" "${stage1_sbatch_cmd[@]}"
    printf '\n'
    stage1_job_id="DRY_RUN_${run_label}"
  else
    stage1_job_id="$("${stage1_env_cmd[@]}" "${stage1_sbatch_cmd[@]}")"
    echo "stage1 model=$run_label job_id=$stage1_job_id output_dir=$stage1_output_dir"
  fi

  stage2_env_cmd=(
    env
    "FIRST_ROUND_ROOT=$stage1_root"
    "MODEL_KEYS=$model_key"
    "SUBSETS=$SUBSETS"
    "OUTPUT_ROOT=$stage2_root"
    "BATCH_TAG=$stage2_tag"
    "DEPENDENCY=afterok:$stage1_job_id"
    "ALLOW_PENDING_FIRST_ROUND=1"
    "SOURCE_ROOT_BASE=$SOURCE_ROOT_BASE"
    "PYTHON_BIN=$PYTHON_BIN"
    "ACCOUNT=$ACCOUNT"
    "PARTITION=$PARTITION"
    "GRES=$GRES"
    "CONSTRAINT=$CONSTRAINT"
    "CPUS=$CPUS"
    "QOS=$QOS"
    "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
    "EVAL_SPLITS=$EVAL_SPLITS"
    "SFT_NUM_TRAIN_EPOCHS=$STAGE2_SFT_NUM_TRAIN_EPOCHS"
    "ENABLE_RL=0"
    "SAVE_PREDICTIONS=$SAVE_PREDICTIONS"
    "CLEANUP_FINAL_MODEL_AFTER_EVAL=$CLEANUP_FINAL_MODEL_AFTER_EVAL"
    "BF16=$BF16"
    "FP16=$FP16"
    "DRY_RUN=$DRY_RUN"
  )

  echo "Submitting stage2 jobs for model=$run_label with dependency=afterok:$stage1_job_id"
  "${stage2_env_cmd[@]}" "$STAGE2_SUBMITTER"

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$model_key" "$run_label" "$stage1_job_id" "$stage1_output_dir" "$stage2_manifest" \
    >> "$submission_manifest"
done

echo "Done. submission_manifest=$submission_manifest"
