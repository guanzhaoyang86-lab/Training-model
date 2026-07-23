#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"

STAGE1_SBATCH_SCRIPT="${STAGE1_SBATCH_SCRIPT:-$ROOT_DIR/run_full_train_qwen.sbatch}"
SFT_ONLY_SUBMITTER="${SFT_ONLY_SUBMITTER:-$ROOT_DIR/submit_sft2_8subsets_from_anomalydb_sft.sh}"
SFT_RL_SUBMITTER="${SFT_RL_SUBMITTER:-$ROOT_DIR/submit_sft2_rl_8subsets_from_synthetic_sft.sh}"
CLEANUP_SBATCH_SCRIPT="${CLEANUP_SBATCH_SCRIPT:-$ROOT_DIR/cleanup_experiment_metrics_only.sbatch}"

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
BATCH_TAG="${BATCH_TAG:-paired_s1e1_s2e3_rl1_$(date +%Y%m%d_%H%M%S)}"

STAGE1_MEMORY="${STAGE1_MEMORY:-120G}"
STAGE1_TIME_LIMIT="${STAGE1_TIME_LIMIT:-05:00:00}"
SFT_ONLY_MEMORY="${SFT_ONLY_MEMORY:-80G}"
SFT_RL_MEMORY="${SFT_RL_MEMORY:-120G}"
STAGE1_SFT_NUM_TRAIN_EPOCHS="${STAGE1_SFT_NUM_TRAIN_EPOCHS:-1}"
STAGE2_SFT_NUM_TRAIN_EPOCHS="${STAGE2_SFT_NUM_TRAIN_EPOCHS:-3}"

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

BASE_MODEL_PATH="$WORKSPACE_ROOT/models/Qwen2.5-VL-7B-Instruct"
RUN_LABEL="qwen25vl_7b"

for required_file in \
  "$STAGE1_SBATCH_SCRIPT" \
  "$SFT_ONLY_SUBMITTER" \
  "$SFT_RL_SUBMITTER" \
  "$CLEANUP_SBATCH_SCRIPT" \
  "$CONFIG_PATH"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi
for required_dir in "$BASE_MODEL_PATH" "$SYNTHETIC_SOURCE_ROOT" "$SOURCE_ROOT_BASE"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Missing required directory: $required_dir" >&2
    exit 1
  fi
done
for split in train val test; do
  if [[ ! -f "$SYNTHETIC_SOURCE_ROOT/$split.json" ]]; then
    echo "Missing synthetic split: $SYNTHETIC_SOURCE_ROOT/$split.json" >&2
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
print("Config preflight ok.")
PY

batch_root="$OUTPUT_ROOT/paired_dual_stage_sft_vs_rl_8subsets_$BATCH_TAG"
stage1_root="$batch_root/stage1"
stage1_output_dir="$stage1_root/$RUN_LABEL"
sft_only_root="$batch_root/sft_only"
sft_rl_root="$batch_root/sft_rl"
mkdir -p "$stage1_output_dir" "$sft_only_root" "$sft_rl_root"

submission_manifest="$batch_root/submission_manifest.tsv"
printf 'batch_tag\tstage1_job_id\tstage1_output_dir\tsft_only_manifest\tsft_rl_manifest\tcleanup_job_id\n' \
  > "$submission_manifest"

echo "Submitting paired SFT-only versus SFT+RL experiment."
echo "Shared Stage 1: synthetic SFT, epochs=$STAGE1_SFT_NUM_TRAIN_EPOCHS"
echo "Both branches: real-subset SFT, epochs=$STAGE2_SFT_NUM_TRAIN_EPOCHS"
echo "RL branch: RL epochs=$RL_NUM_TRAIN_EPOCHS"
echo "Subsets: $SUBSETS"
echo "Predictions: removed"
echo "All weights and prepared caches: removed after all 16 jobs succeed"
echo "Batch root: $batch_root"

stage1_sbatch_cmd=(
  sbatch
  --parsable
  "--account=$ACCOUNT"
  "--partition=$PARTITION"
  "--job-name=tsg-pair-sft1-qwen25vl-7b"
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
  "SFT_NUM_TRAIN_EPOCHS=$STAGE1_SFT_NUM_TRAIN_EPOCHS"
  "BF16=$BF16"
  "FP16=$FP16"
  "LAUNCHER=python"
  "NUM_PROCS_PER_NODE=1"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[DRY_RUN] shared stage1 command='
  printf ' %q' "${stage1_env_cmd[@]}" "${stage1_sbatch_cmd[@]}"
  printf '\n'
  stage1_job_id="DRY_RUN_STAGE1"
else
  stage1_job_id="$("${stage1_env_cmd[@]}" "${stage1_sbatch_cmd[@]}")"
  echo "Shared Stage1 job_id=$stage1_job_id"
fi

sft_only_tag="${BATCH_TAG}_shared_stage1"
sft_only_manifest="$sft_only_root/sft2_8subsets_from_anomalydb_sft_$sft_only_tag/submission_manifest.tsv"
env \
  "FIRST_ROUND_ROOT=$stage1_root" \
  "MODEL_KEYS=7b" \
  "SUBSETS=$SUBSETS" \
  "OUTPUT_ROOT=$sft_only_root" \
  "BATCH_TAG=$sft_only_tag" \
  "DEPENDENCY=afterok:$stage1_job_id" \
  "ALLOW_PENDING_FIRST_ROUND=1" \
  "SOURCE_ROOT_BASE=$SOURCE_ROOT_BASE" \
  "PYTHON_BIN=$PYTHON_BIN" \
  "ACCOUNT=$ACCOUNT" \
  "PARTITION=$PARTITION" \
  "GRES=$GRES" \
  "CONSTRAINT=$CONSTRAINT" \
  "CPUS=$CPUS" \
  "QOS=$QOS" \
  "MEMORY_7B=$SFT_ONLY_MEMORY" \
  "MAX_NEW_TOKENS=$MAX_NEW_TOKENS" \
  "EVAL_SPLITS=test" \
  "SFT_NUM_TRAIN_EPOCHS=$STAGE2_SFT_NUM_TRAIN_EPOCHS" \
  "ENABLE_RL=0" \
  "SAVE_PREDICTIONS=0" \
  "CLEANUP_FINAL_MODEL_AFTER_EVAL=1" \
  "TINY_SAMPLE_LIMIT=50" \
  "SMALL_SAMPLE_LIMIT=100" \
  "MEDIUM_SAMPLE_LIMIT=250" \
  "LARGE_SAMPLE_LIMIT=500" \
  "XL_SAMPLE_LIMIT=1000" \
  "TINY_TIME_LIMIT=01:00:00" \
  "SMALL_TIME_LIMIT=01:30:00" \
  "MEDIUM_TIME_LIMIT=03:00:00" \
  "LARGE_TIME_LIMIT=04:00:00" \
  "XL_TIME_LIMIT=05:00:00" \
  "XXL_TIME_LIMIT=06:00:00" \
  "BF16=$BF16" \
  "FP16=$FP16" \
  "DRY_RUN=$DRY_RUN" \
  "$SFT_ONLY_SUBMITTER"

sft_rl_tag="${BATCH_TAG}_shared_stage1"
sft_rl_manifest="$sft_rl_root/sft2_rl_8subsets_from_synthetic_sft_$sft_rl_tag/submission_manifest.tsv"
env \
  "BASE_SFT_ROOT=$stage1_output_dir" \
  "BASE_MODEL_PATH=$stage1_output_dir/model" \
  "CONFIG_PATH=$stage1_output_dir/resolved_config.yaml" \
  "ALLOW_PENDING_BASE_SFT=1" \
  "SUBSETS=$SUBSETS" \
  "SOURCE_ROOT_BASE=$SOURCE_ROOT_BASE" \
  "OUTPUT_ROOT=$sft_rl_root" \
  "BATCH_TAG=$sft_rl_tag" \
  "BATCH_PREFIX=sft2_rl_8subsets_from_synthetic_sft" \
  "DEPENDENCY=afterok:$stage1_job_id" \
  "PYTHON_BIN=$PYTHON_BIN" \
  "ACCOUNT=$ACCOUNT" \
  "PARTITION=$PARTITION" \
  "GRES=$GRES" \
  "CONSTRAINT=$CONSTRAINT" \
  "MEMORY=$SFT_RL_MEMORY" \
  "CPUS=$CPUS" \
  "QOS=$QOS" \
  "MAX_NEW_TOKENS=$MAX_NEW_TOKENS" \
  "EVAL_SPLITS=test" \
  "SFT_NUM_TRAIN_EPOCHS=$STAGE2_SFT_NUM_TRAIN_EPOCHS" \
  "RL_NUM_GENERATIONS=$RL_NUM_GENERATIONS" \
  "RL_MAX_NEW_TOKENS=$RL_MAX_NEW_TOKENS" \
  "RL_TEMPERATURE=$RL_TEMPERATURE" \
  "RL_TOP_P=$RL_TOP_P" \
  "RL_LEARNING_RATE=$RL_LEARNING_RATE" \
  "RL_NUM_TRAIN_EPOCHS=$RL_NUM_TRAIN_EPOCHS" \
  "RL_KL_COEF=$RL_KL_COEF" \
  "RL_REWARD_EVENT_F1_WEIGHT=$RL_REWARD_EVENT_F1_WEIGHT" \
  "RL_REWARD_BOUNDARY_IOU_WEIGHT=$RL_REWARD_BOUNDARY_IOU_WEIGHT" \
  "RL_REWARD_HALLUCINATION_PENALTY_WEIGHT=$RL_REWARD_HALLUCINATION_PENALTY_WEIGHT" \
  "RL_SAVE_STEPS=$RL_SAVE_STEPS" \
  "RL_OPTIMIZER=$RL_OPTIMIZER" \
  "SAVE_PREDICTIONS=0" \
  "CLEANUP_SFT_AFTER_RL=1" \
  "CLEANUP_FINAL_MODEL_AFTER_EVAL=1" \
  "CLEANUP_RL_ARTIFACTS_AFTER_EVAL=1" \
  "TINY_TRAIN_LIMIT=30" \
  "SMALL_TRAIN_LIMIT=80" \
  "MEDIUM_TRAIN_LIMIT=200" \
  "LARGE_TRAIN_LIMIT=500" \
  "TINY_TIME_LIMIT=01:00:00" \
  "SMALL_TIME_LIMIT=01:30:00" \
  "MEDIUM_TIME_LIMIT=03:00:00" \
  "LARGE_TIME_LIMIT=04:00:00" \
  "XL_TIME_LIMIT=06:00:00" \
  "BF16=$BF16" \
  "FP16=$FP16" \
  "DRY_RUN=$DRY_RUN" \
  "$SFT_RL_SUBMITTER"

cleanup_job_id="DRY_RUN_CLEANUP"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[DRY_RUN] cleanup waits for all 16 Stage2 jobs and removes weights/caches under $batch_root"
else
  job_ids=()
  while IFS=$'\t' read -r _ _ _ _ job_id _; do
    [[ "$job_id" == "job_id" ]] && continue
    job_ids+=("$job_id")
  done < "$sft_only_manifest"
  while IFS=$'\t' read -r _ _ _ _ job_id _; do
    [[ "$job_id" == "job_id" ]] && continue
    job_ids+=("$job_id")
  done < "$sft_rl_manifest"

  if [[ "${#job_ids[@]}" -ne 16 ]]; then
    echo "Expected 16 Stage2 job IDs, found ${#job_ids[@]}." >&2
    exit 1
  fi

  cleanup_dependency="afterok:$(IFS=:; echo "${job_ids[*]}")"
  cleanup_job_id="$(
    env \
      "PROJECT_ROOT=$ROOT_DIR" \
      "CLEANUP_ROOT=$batch_root" \
      sbatch \
        --parsable \
        "--account=$ACCOUNT" \
        "--dependency=$cleanup_dependency" \
        "--output=$batch_root/slurm-tsg-clean-metrics-only-%j.out" \
        "--error=$batch_root/slurm-tsg-clean-metrics-only-%j.err" \
        "$CLEANUP_SBATCH_SCRIPT"
  )"
  echo "Cleanup job_id=$cleanup_job_id dependency=$cleanup_dependency"
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$BATCH_TAG" "$stage1_job_id" "$stage1_output_dir" \
  "$sft_only_manifest" "$sft_rl_manifest" "$cleanup_job_id" \
  >> "$submission_manifest"

echo "Done. submission_manifest=$submission_manifest"
