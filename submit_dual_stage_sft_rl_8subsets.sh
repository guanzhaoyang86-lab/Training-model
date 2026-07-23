#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"

STAGE1_SBATCH_SCRIPT="${STAGE1_SBATCH_SCRIPT:-$ROOT_DIR/run_full_train_qwen.sbatch}"
STAGE2_SUBMITTER="${STAGE2_SUBMITTER:-$ROOT_DIR/submit_sft2_rl_8subsets_from_synthetic_sft.sh}"

MODEL_KEY="${MODEL_KEY:-7b}"
SUBSETS="${SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
CPUS="${CPUS:-8}"
QOS="${QOS:-}"
PYTHON_BIN="${PYTHON_BIN:-$WORKSPACE_ROOT/envs/anomamind/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_anomaly_db_indexed_text_plain_image.yaml}"
SYNTHETIC_SOURCE_ROOT="${SYNTHETIC_SOURCE_ROOT:-$ROOT_DIR/dataset/anomaly_db_v1}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs}"
BATCH_TAG="${BATCH_TAG:-$(date +%Y%m%d_%H%M%S)}"

STAGE1_MEMORY="${STAGE1_MEMORY:-120G}"
STAGE1_TIME_LIMIT="${STAGE1_TIME_LIMIT:-04:00:00}"
STAGE2_MEMORY="${STAGE2_MEMORY:-120G}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
EVAL_SPLITS="${EVAL_SPLITS:-test}"
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

case "${MODEL_KEY,,}" in
  7b|qwen25vl_7b|qwen2.5vl_7b)
    RUN_LABEL="qwen25vl_7b"
    BASE_MODEL_PATH="$WORKSPACE_ROOT/models/Qwen2.5-VL-7B-Instruct"
    ;;
  *)
    echo "This workflow is scoped to the 7B model. Set MODEL_KEY=7b." >&2
    exit 1
    ;;
esac

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
if [[ ! -d "$SYNTHETIC_SOURCE_ROOT" ]]; then
  echo "SYNTHETIC_SOURCE_ROOT not found: $SYNTHETIC_SOURCE_ROOT" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_ROOT_BASE" ]]; then
  echo "SOURCE_ROOT_BASE not found: $SOURCE_ROOT_BASE" >&2
  exit 1
fi
if [[ ! -d "$BASE_MODEL_PATH" ]]; then
  echo "Base 7B model path not found: $BASE_MODEL_PATH" >&2
  exit 1
fi
for split in train val test; do
  if [[ ! -f "$SYNTHETIC_SOURCE_ROOT/$split.json" ]]; then
    echo "Missing synthetic split: $SYNTHETIC_SOURCE_ROOT/$split.json" >&2
    exit 1
  fi
done
if [[ "$CLEANUP_SFT_AFTER_RL" != "0" ]]; then
  echo "This experiment is meant to retain second-stage SFT models; set CLEANUP_SFT_AFTER_RL=0." >&2
  exit 1
fi
if [[ "$CLEANUP_FINAL_MODEL_AFTER_EVAL" != "1" ]]; then
  echo "This experiment is meant to remove final RL weights after test; set CLEANUP_FINAL_MODEL_AFTER_EVAL=1." >&2
  exit 1
fi

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

batch_root="$OUTPUT_ROOT/dual_stage_sft_rl_8subsets_$BATCH_TAG"
stage1_root="$batch_root/stage1"
stage2_root="$batch_root/stage2"
stage1_output_dir="$stage1_root/$RUN_LABEL"
stage2_tag="${BATCH_TAG}_${RUN_LABEL}"
stage2_manifest="$stage2_root/sft2_rl_8subsets_from_synthetic_sft_$stage2_tag/submission_manifest.tsv"
mkdir -p "$stage1_output_dir" "$stage2_root"

submission_manifest="$batch_root/submission_manifest.tsv"
printf 'model_key\tmodel_label\tstage1_job_id\tstage1_output_dir\tstage2_manifest\n' > "$submission_manifest"

echo "Submitting dual-stage SFT+RL workflow."
echo "Stage 1: synthetic SFT on $SYNTHETIC_SOURCE_ROOT"
echo "Stage 2: real-subset SFT2 + RL, then test metrics."
echo "Model: $RUN_LABEL"
echo "Subsets: $SUBSETS"
echo "Save predictions: $SAVE_PREDICTIONS"
echo "Retain second-stage SFT model: yes"
echo "Remove final RL weights after test: yes"
echo "Batch root: $batch_root"

stage1_sbatch_cmd=(
  sbatch
  --parsable
  "--account=$ACCOUNT"
  "--partition=$PARTITION"
  "--job-name=tsg-sft1-${RUN_LABEL//_/-}-synthetic"
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
stage1_sbatch_cmd+=("$STAGE1_SBATCH_SCRIPT")

stage1_env_cmd=(
  env
  "PROJECT_ROOT=$ROOT_DIR"
  "CONFIG_PATH=$CONFIG_PATH"
  "OUTPUT_DIR=$stage1_output_dir"
  "MODE=train"
  "PYTHON_BIN=$PYTHON_BIN"
  "MODEL_PATH=$BASE_MODEL_PATH"
  "SOURCE_ROOT=$SYNTHETIC_SOURCE_ROOT"
  "BF16=$BF16"
  "FP16=$FP16"
  "LAUNCHER=python"
  "NUM_PROCS_PER_NODE=1"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[DRY_RUN] stage1 model=%s output_dir=%s command=' "$RUN_LABEL" "$stage1_output_dir"
  printf ' %q' "${stage1_env_cmd[@]}" "${stage1_sbatch_cmd[@]}"
  printf '\n'
  stage1_job_id="DRY_RUN_${RUN_LABEL}"
else
  stage1_job_id="$("${stage1_env_cmd[@]}" "${stage1_sbatch_cmd[@]}")"
  echo "stage1 model=$RUN_LABEL job_id=$stage1_job_id output_dir=$stage1_output_dir"
fi

stage2_env_cmd=(
  env
  "BASE_SFT_ROOT=$stage1_output_dir"
  "BASE_MODEL_PATH=$stage1_output_dir/model"
  "CONFIG_PATH=$stage1_output_dir/resolved_config.yaml"
  "ALLOW_PENDING_BASE_SFT=1"
  "SUBSETS=$SUBSETS"
  "SOURCE_ROOT_BASE=$SOURCE_ROOT_BASE"
  "OUTPUT_ROOT=$stage2_root"
  "BATCH_TAG=$stage2_tag"
  "DEPENDENCY=afterok:$stage1_job_id"
  "PYTHON_BIN=$PYTHON_BIN"
  "ACCOUNT=$ACCOUNT"
  "PARTITION=$PARTITION"
  "GRES=$GRES"
  "CONSTRAINT=$CONSTRAINT"
  "MEMORY=$STAGE2_MEMORY"
  "CPUS=$CPUS"
  "QOS=$QOS"
  "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
  "EVAL_SPLITS=$EVAL_SPLITS"
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
  "DRY_RUN=$DRY_RUN"
)

echo "Submitting stage2 jobs with dependency=afterok:$stage1_job_id"
"${stage2_env_cmd[@]}" "$STAGE2_SUBMITTER"

printf '%s\t%s\t%s\t%s\t%s\n' \
  "$MODEL_KEY" "$RUN_LABEL" "$stage1_job_id" "$stage1_output_dir" "$stage2_manifest" \
  >> "$submission_manifest"

echo "Done. submission_manifest=$submission_manifest"
