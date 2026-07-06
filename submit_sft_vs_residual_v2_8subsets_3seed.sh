#!/bin/bash
set -euo pipefail

# Full 8-subset confirmation run for the optimized residual-GRPO v2 recipe.
# It submits three seeds for:
#   1) sft_only: synthetic SFT -> real-subset SFT -> test
#   2) residual_v2: synthetic SFT -> real-subset SFT -> residual mining
#      -> point-F1 weighted residual-GRPO -> test
#
# Expected final metrics:
#   3 seeds x 4 models x 2 settings x 8 subsets = 192 test_metrics.json files.
#
# The underlying workflow keeps metrics/logs/summaries and deletes model weights
# after successful evaluation. If a subset job fails before producing metrics,
# an automatic missing-metrics check resubmits that exact model/setting/subset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUN_SEEDS="${RUN_SEEDS:-2026 2027 2028}"
BATCH_PREFIX="${BATCH_PREFIX:-sft_vs_residual_v2_8subsets_3seed}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"

export MODELS="${MODELS:-7b 2b 4b 8b}"
export SUBSETS="${SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"
export VARIANTS="${VARIANTS:-sft_only residual_v2}"

export STAGE1_SFT_NUM_TRAIN_EPOCHS="${STAGE1_SFT_NUM_TRAIN_EPOCHS:-1}"
export STAGE2_SFT_NUM_TRAIN_EPOCHS="${STAGE2_SFT_NUM_TRAIN_EPOCHS:-3}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-512}"

# residual_v2: use the optimized recipe validated on hard residual subsets.
export RESIDUAL_V2_POOL_SAMPLING_RATIOS="${RESIDUAL_V2_POOL_SAMPLING_RATIOS:-false_negative=0.60,boundary_error=0.20,false_positive=0.10,correct_abnormal=0.10,correct_normal=0.00}"
export RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS="${RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS:-2}"
export RESIDUAL_V2_RL_NUM_GENERATIONS="${RESIDUAL_V2_RL_NUM_GENERATIONS:-6}"
export RESIDUAL_V2_RL_LEARNING_RATE="${RESIDUAL_V2_RL_LEARNING_RATE:-2e-6}"
export RESIDUAL_V2_RL_KL_COEF="${RESIDUAL_V2_RL_KL_COEF:-0.03}"
export RESIDUAL_V2_REWARD_POINT_WEIGHT="${RESIDUAL_V2_REWARD_POINT_WEIGHT:-0.60}"
export RESIDUAL_V2_REWARD_EVENT_WEIGHT="${RESIDUAL_V2_REWARD_EVENT_WEIGHT:-0.00}"
export RESIDUAL_V2_REWARD_IOU_WEIGHT="${RESIDUAL_V2_REWARD_IOU_WEIGHT:-0.25}"
export RESIDUAL_V2_REWARD_BOUNDARY_WEIGHT="${RESIDUAL_V2_REWARD_BOUNDARY_WEIGHT:-0.15}"
export RESIDUAL_V2_REWARD_TYPE_WEIGHT="${RESIDUAL_V2_REWARD_TYPE_WEIGHT:-0.00}"

# These base RL values are kept for compatibility with the main workflow.
# They do not affect sft_only; residual_v2 overrides them through the variables above.
export RL_NUM_TRAIN_EPOCHS="${RL_NUM_TRAIN_EPOCHS:-2}"
export RL_NUM_GENERATIONS="${RL_NUM_GENERATIONS:-6}"
export RL_LEARNING_RATE="${RL_LEARNING_RATE:-2e-6}"
export RL_KL_COEF="${RL_KL_COEF:-0.03}"
export REWARD_POINT_WEIGHT="${REWARD_POINT_WEIGHT:-0.60}"
export REWARD_EVENT_WEIGHT="${REWARD_EVENT_WEIGHT:-0.00}"
export REWARD_IOU_WEIGHT="${REWARD_IOU_WEIGHT:-0.25}"
export REWARD_BOUNDARY_WEIGHT="${REWARD_BOUNDARY_WEIGHT:-0.15}"
export REWARD_TYPE_WEIGHT="${REWARD_TYPE_WEIGHT:-0.00}"
export RESIDUAL_POOL_SAMPLING_RATIOS="${RESIDUAL_POOL_SAMPLING_RATIOS:-$RESIDUAL_V2_POOL_SAMPLING_RATIOS}"

export SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
export KEEP_RESIDUAL_POOL="${KEEP_RESIDUAL_POOL:-0}"
export CLEANUP_DEPENDENCY_MODE="${CLEANUP_DEPENDENCY_MODE:-afterok}"
export EXCLUDE_NODES="${EXCLUDE_NODES:-qgpu2014}"

AUTO_RESUBMIT_MISSING="${AUTO_RESUBMIT_MISSING:-1}"
AUTO_RESUBMIT_TIME_LIMIT="${AUTO_RESUBMIT_TIME_LIMIT:-00:30:00}"
AUTO_RESUBMIT_MEMORY="${AUTO_RESUBMIT_MEMORY:-8G}"
AUTO_RESUBMIT_CPUS="${AUTO_RESUBMIT_CPUS:-1}"
AUTO_RESUBMIT_PARTITION="${AUTO_RESUBMIT_PARTITION:-$PARTITION}"
AUTO_RESUBMIT_GRES="${AUTO_RESUBMIT_GRES:-$GRES}"
AUTO_RESUBMIT_CONSTRAINT="${AUTO_RESUBMIT_CONSTRAINT:-$CONSTRAINT}"

echo "Submitting full 8-subset SFT-only vs residual_v2 experiment."
echo "RUN_SEEDS=$RUN_SEEDS"
echo "MODELS=$MODELS"
echo "SUBSETS=$SUBSETS"
echo "VARIANTS=$VARIANTS"
echo "stage1_sft_epochs=$STAGE1_SFT_NUM_TRAIN_EPOCHS"
echo "stage2_sft_epochs=$STAGE2_SFT_NUM_TRAIN_EPOCHS"
echo "residual_v2_rl=epochs:$RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS generations:$RESIDUAL_V2_RL_NUM_GENERATIONS lr:$RESIDUAL_V2_RL_LEARNING_RATE kl:$RESIDUAL_V2_RL_KL_COEF"
echo "residual_v2_reward=point:$RESIDUAL_V2_REWARD_POINT_WEIGHT event:$RESIDUAL_V2_REWARD_EVENT_WEIGHT iou:$RESIDUAL_V2_REWARD_IOU_WEIGHT boundary:$RESIDUAL_V2_REWARD_BOUNDARY_WEIGHT type:$RESIDUAL_V2_REWARD_TYPE_WEIGHT"
echo "residual_v2_sampling=$RESIDUAL_V2_POOL_SAMPLING_RATIOS"
echo "cleanup_dependency_mode=$CLEANUP_DEPENDENCY_MODE"
echo "auto_resubmit_missing=$AUTO_RESUBMIT_MISSING"

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
      "--job-name=ensure-v2-seed-$seed"
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
echo "Expected final metrics per seed: 4 models x 2 settings x 8 subsets = 64 test_metrics.json files."
echo "Expected final metrics total: 3 seeds x 64 = 192 test_metrics.json files."
echo "After all jobs finish, summarize mean/std with:"
echo "  $PYTHON_BIN $SCRIPT_DIR/scripts/summarize_three_seed_ablation.py --root $OUTPUT_ROOT --prefix qwen_vl_all_models_sft_residual_ablation_8subsets_${BATCH_PREFIX}"
