#!/bin/bash
set -euo pipefail

# Submit the main ablation with three random seeds:
#   1) sft_only: SFT1 + SFT2, then test
#   2) no_residual: SFT1 + SFT2 + boundary-aware GRPO on the regular train cache
#   3) full_residual: SFT1 + SFT2 + residual-guided boundary-aware GRPO
#
# Defaults keep the current dataset/model matrix and turn off type reward for real
# TSB-AD-U data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUN_SEEDS="${RUN_SEEDS:-2026 2027 2028}"
BATCH_PREFIX="${BATCH_PREFIX:-qwen_vl_three_setting_3seed}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"

export VARIANTS="${VARIANTS:-sft_only no_residual full_residual}"
export STAGE1_SFT_NUM_TRAIN_EPOCHS="${STAGE1_SFT_NUM_TRAIN_EPOCHS:-1}"
export STAGE2_SFT_NUM_TRAIN_EPOCHS="${STAGE2_SFT_NUM_TRAIN_EPOCHS:-3}"
export RL_NUM_TRAIN_EPOCHS="${RL_NUM_TRAIN_EPOCHS:-1}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-512}"
export REWARD_EVENT_WEIGHT="${REWARD_EVENT_WEIGHT:-0.45}"
export REWARD_IOU_WEIGHT="${REWARD_IOU_WEIGHT:-0.40}"
export REWARD_BOUNDARY_WEIGHT="${REWARD_BOUNDARY_WEIGHT:-0.15}"
export REWARD_TYPE_WEIGHT="${REWARD_TYPE_WEIGHT:-0}"
export CLEANUP_DEPENDENCY_MODE="${CLEANUP_DEPENDENCY_MODE:-afterok}"
export EXCLUDE_NODES="${EXCLUDE_NODES:-qgpu2014}"

AUTO_RESUBMIT_MISSING="${AUTO_RESUBMIT_MISSING:-1}"
AUTO_RESUBMIT_TIME_LIMIT="${AUTO_RESUBMIT_TIME_LIMIT:-00:30:00}"
AUTO_RESUBMIT_MEMORY="${AUTO_RESUBMIT_MEMORY:-8G}"
AUTO_RESUBMIT_CPUS="${AUTO_RESUBMIT_CPUS:-1}"
AUTO_RESUBMIT_PARTITION="${AUTO_RESUBMIT_PARTITION:-$PARTITION}"
AUTO_RESUBMIT_GRES="${AUTO_RESUBMIT_GRES:-$GRES}"
AUTO_RESUBMIT_CONSTRAINT="${AUTO_RESUBMIT_CONSTRAINT:-$CONSTRAINT}"

echo "Submitting three-setting three-seed ablation."
echo "RUN_SEEDS=$RUN_SEEDS"
echo "VARIANTS=$VARIANTS"
echo "MODELS=${MODELS:-7b 2b 4b 8b}"
echo "SUBSETS=${SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "RL_MAX_NEW_TOKENS=$RL_MAX_NEW_TOKENS"
echo "reward_weights=event:$REWARD_EVENT_WEIGHT iou:$REWARD_IOU_WEIGHT boundary:$REWARD_BOUNDARY_WEIGHT type:$REWARD_TYPE_WEIGHT"
echo "CLEANUP_DEPENDENCY_MODE=$CLEANUP_DEPENDENCY_MODE"
echo "AUTO_RESUBMIT_MISSING=$AUTO_RESUBMIT_MISSING"

for seed in $RUN_SEEDS; do
  export EXPERIMENT_SEED="$seed"
  export BATCH_TAG="${BATCH_PREFIX}_seed${seed}_${TIMESTAMP}"
  batch_root="$OUTPUT_ROOT/qwen_vl_all_models_sft_residual_ablation_8subsets_$BATCH_TAG"
  echo
  echo "=== Submitting seed=$seed BATCH_TAG=$BATCH_TAG ==="
  bash "$SCRIPT_DIR/submit_qwen_vl_all_models_sft_residual_ablation_8subsets.sh"

  if [[ "$AUTO_RESUBMIT_MISSING" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
    manifest="$batch_root/submission_manifest.tsv"
    if [[ ! -f "$manifest" ]]; then
      echo "Cannot submit auto-resubmit job: missing manifest $manifest" >&2
      continue
    fi
    mapfile -t subset_job_ids < <(awk 'BEGIN{FS="\t"} NR>1 && $10 != "" {print $10}' "$manifest")
    if (( ${#subset_job_ids[@]} == 0 )); then
      echo "Cannot submit auto-resubmit job: no subset job ids in $manifest" >&2
      continue
    fi
    dependency="afterany:$(IFS=:; echo "${subset_job_ids[*]}")"
    auto_log_dir="$batch_root/auto_resubmit"
    mkdir -p "$auto_log_dir"
    auto_sbatch_cmd=(
      sbatch
      --parsable
      "--account=$ACCOUNT"
      "--partition=$AUTO_RESUBMIT_PARTITION"
      "--job-name=ensure-seed-$seed"
      "--mem=$AUTO_RESUBMIT_MEMORY"
      "--cpus-per-task=$AUTO_RESUBMIT_CPUS"
      "--time=$AUTO_RESUBMIT_TIME_LIMIT"
      "--dependency=$dependency"
      "--output=$auto_log_dir/slurm-%x-%j.out"
      "--error=$auto_log_dir/slurm-%x-%j.err"
    )
    if [[ -n "$AUTO_RESUBMIT_GRES" ]]; then
      auto_sbatch_cmd+=("--gres=$AUTO_RESUBMIT_GRES")
    fi
    if [[ -n "$AUTO_RESUBMIT_CONSTRAINT" ]]; then
      auto_sbatch_cmd+=("--constraint=$AUTO_RESUBMIT_CONSTRAINT")
    fi
    if [[ -n "${QOS:-}" ]]; then
      auto_sbatch_cmd+=("--qos=$QOS")
    fi
    auto_sbatch_cmd+=("$SCRIPT_DIR/submit_qwen_vl_missing_from_manifest.sh" "$batch_root")
    auto_job_id="$("${auto_sbatch_cmd[@]}")"
    echo "auto_resubmit seed=$seed job_id=$auto_job_id dependency=$dependency"
  fi
done

echo
echo "Submitted all requested seeds."
echo "After all jobs finish, summarize mean/std with:"
echo "  $PYTHON_BIN $SCRIPT_DIR/scripts/summarize_three_seed_ablation.py --root $OUTPUT_ROOT --prefix qwen_vl_all_models_sft_residual_ablation_8subsets_${BATCH_PREFIX}"
