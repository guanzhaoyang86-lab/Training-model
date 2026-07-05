#!/bin/bash
set -euo pipefail

# Focused one-seed experiment for residual-GRPO v2.
# It compares:
#   1) sft_only: SFT1 + SFT2
#   2) full_residual: current residual-GRPO
#   3) residual_v2: point-F1 weighted reward + pool-aware residual sampling
# on hard subsets where residual pools contain useful error signal.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs}"

export MODELS="${MODELS:-7b 2b 4b 8b}"
export SUBSETS="${SUBSETS:-MSL NEK TODS Power}"
export VARIANTS="${VARIANTS:-sft_only full_residual residual_v2}"
export EXPERIMENT_SEED="${EXPERIMENT_SEED:-2026}"
export BATCH_TAG="${BATCH_TAG:-residual_v2_hard_subsets_seed${EXPERIMENT_SEED}_${TIMESTAMP}}"

export STAGE1_SFT_NUM_TRAIN_EPOCHS="${STAGE1_SFT_NUM_TRAIN_EPOCHS:-1}"
export STAGE2_SFT_NUM_TRAIN_EPOCHS="${STAGE2_SFT_NUM_TRAIN_EPOCHS:-3}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-512}"

# Current full_residual baseline: keep the previous conservative RL recipe.
export RL_NUM_TRAIN_EPOCHS="${RL_NUM_TRAIN_EPOCHS:-1}"
export RL_NUM_GENERATIONS="${RL_NUM_GENERATIONS:-4}"
export RL_LEARNING_RATE="${RL_LEARNING_RATE:-1e-6}"
export RL_KL_COEF="${RL_KL_COEF:-0.02}"
export REWARD_POINT_WEIGHT="${REWARD_POINT_WEIGHT:-0}"
export REWARD_EVENT_WEIGHT="${REWARD_EVENT_WEIGHT:-0.45}"
export REWARD_IOU_WEIGHT="${REWARD_IOU_WEIGHT:-0.40}"
export REWARD_BOUNDARY_WEIGHT="${REWARD_BOUNDARY_WEIGHT:-0.15}"
export REWARD_TYPE_WEIGHT="${REWARD_TYPE_WEIGHT:-0}"
export RESIDUAL_POOL_SAMPLING_RATIOS="${RESIDUAL_POOL_SAMPLING_RATIOS:-false_negative=0.3,boundary_error=0.3,false_positive=0.2,correct_abnormal=0.1,correct_normal=0.1}"

# residual_v2: align reward with final point-F1 and oversample real hard errors.
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

export CLEANUP_DEPENDENCY_MODE="${CLEANUP_DEPENDENCY_MODE:-afterok}"
export EXCLUDE_NODES="${EXCLUDE_NODES:-qgpu2014}"
export AUTO_RESUBMIT_MISSING="${AUTO_RESUBMIT_MISSING:-1}"
export AUTO_RESUBMIT_TIME_LIMIT="${AUTO_RESUBMIT_TIME_LIMIT:-00:30:00}"
export AUTO_RESUBMIT_MEMORY="${AUTO_RESUBMIT_MEMORY:-8G}"
export AUTO_RESUBMIT_CPUS="${AUTO_RESUBMIT_CPUS:-1}"
export AUTO_RESUBMIT_PARTITION="${AUTO_RESUBMIT_PARTITION:-$PARTITION}"
export AUTO_RESUBMIT_GRES="${AUTO_RESUBMIT_GRES:-$GRES}"
export AUTO_RESUBMIT_CONSTRAINT="${AUTO_RESUBMIT_CONSTRAINT:-$CONSTRAINT}"

batch_root="$OUTPUT_ROOT/qwen_vl_all_models_sft_residual_ablation_8subsets_$BATCH_TAG"

echo "Submitting residual-GRPO v2 hard-subsets experiment."
echo "MODELS=$MODELS"
echo "SUBSETS=$SUBSETS"
echo "VARIANTS=$VARIANTS"
echo "EXPERIMENT_SEED=$EXPERIMENT_SEED"
echo "BATCH_TAG=$BATCH_TAG"
echo "BATCH_ROOT=$batch_root"
echo "current_reward=point:$REWARD_POINT_WEIGHT event:$REWARD_EVENT_WEIGHT iou:$REWARD_IOU_WEIGHT boundary:$REWARD_BOUNDARY_WEIGHT type:$REWARD_TYPE_WEIGHT"
echo "v2_reward=point:$RESIDUAL_V2_REWARD_POINT_WEIGHT event:$RESIDUAL_V2_REWARD_EVENT_WEIGHT iou:$RESIDUAL_V2_REWARD_IOU_WEIGHT boundary:$RESIDUAL_V2_REWARD_BOUNDARY_WEIGHT type:$RESIDUAL_V2_REWARD_TYPE_WEIGHT"
echo "v2_sampling=$RESIDUAL_V2_POOL_SAMPLING_RATIOS"
echo "v2_rl=epochs:$RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS generations:$RESIDUAL_V2_RL_NUM_GENERATIONS lr:$RESIDUAL_V2_RL_LEARNING_RATE kl:$RESIDUAL_V2_RL_KL_COEF"

bash "$SCRIPT_DIR/submit_qwen_vl_all_models_sft_residual_ablation_8subsets.sh"

if [[ "$AUTO_RESUBMIT_MISSING" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
  manifest="$batch_root/submission_manifest.tsv"
  if [[ ! -f "$manifest" ]]; then
    echo "Cannot submit auto-resubmit job: missing manifest $manifest" >&2
    exit 0
  fi
  mapfile -t subset_job_ids < <(awk 'BEGIN{FS="\t"} NR>1 && $10 != "" {print $10}' "$manifest")
  if (( ${#subset_job_ids[@]} == 0 )); then
    echo "Cannot submit auto-resubmit job: no subset job ids in $manifest" >&2
    exit 0
  fi
  dependency="afterany:$(IFS=:; echo "${subset_job_ids[*]}")"
  auto_log_dir="$batch_root/auto_resubmit"
  mkdir -p "$auto_log_dir"
  auto_sbatch_cmd=(
    sbatch
    --parsable
    "--account=$ACCOUNT"
    "--partition=$AUTO_RESUBMIT_PARTITION"
    "--job-name=ensure-v2-hard"
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
  echo "auto_resubmit job_id=$auto_job_id dependency=$dependency"
fi

echo "Submitted. Expected final metrics: 4 models x 3 settings x 4 subsets = 48 test_metrics.json files."
echo "Summarize after completion with:"
echo "  /gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python $SCRIPT_DIR/scripts/summarize_three_seed_ablation.py --root $OUTPUT_ROOT --prefix qwen_vl_all_models_sft_residual_ablation_8subsets_residual_v2_hard_subsets"
