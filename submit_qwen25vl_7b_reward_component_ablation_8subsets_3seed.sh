#!/bin/bash
set -euo pipefail

# Reward-component ablation for residual-GRPO v2.
# Fixed setup:
#   model: Qwen2.5-VL-7B
#   subsets: all 8 real TSB-AD-U subsets
#   seeds: 2026, 2027, 2028
#   Stage 1 SFT: synthetic, epoch=1
#   Stage 2 SFT: real subset, epoch=3
#   RL: residual_v2 only, epoch=2, generations=6, lr=2e-6, KL=0.03
#
# Reward settings:
#   point_only:         R = 1.00 * R_point_f1
#   point_iou:          R = 0.60 * R_point_f1 + 0.40 * R_iou
#   point_iou_boundary: R = 0.45 * R_point_f1 + 0.40 * R_iou + 0.15 * R_boundary
#
# Each reward setting/seed is submitted as an independent batch. The existing
# missing-metrics checker is scheduled after every batch and resubmits any
# subset that did not produce test_metrics.json. The underlying workflow keeps
# metrics/logs/manifests/summaries and deletes model/checkpoint weights after
# evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUN_SEEDS="${RUN_SEEDS:-2026 2027 2028}"
BATCH_PREFIX="${BATCH_PREFIX:-qwen25vl_7b_reward_component_ablation_8subsets_3seed}"

ACCOUNT="${ACCOUNT:-p33222}"
PARTITION="${PARTITION:-gengpu}"
GRES="${GRES:-gpu:a100:1}"
CONSTRAINT="${CONSTRAINT:-sxm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs}"
PYTHON_BIN="${PYTHON_BIN:-/gpfs/projects/p33222/ybq9740/envs/anomamind/bin/python}"

export MODELS="${MODELS:-7b}"
export SUBSETS="${SUBSETS:-Daphnet MSL NEK Power SED TAO TODS YAHOO}"
export VARIANTS="${VARIANTS:-residual_v2}"

export STAGE1_SFT_NUM_TRAIN_EPOCHS="${STAGE1_SFT_NUM_TRAIN_EPOCHS:-1}"
export STAGE2_SFT_NUM_TRAIN_EPOCHS="${STAGE2_SFT_NUM_TRAIN_EPOCHS:-3}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-512}"

export RESIDUAL_V2_POOL_SAMPLING_RATIOS="${RESIDUAL_V2_POOL_SAMPLING_RATIOS:-false_negative=0.60,boundary_error=0.20,false_positive=0.10,correct_abnormal=0.10,correct_normal=0.00}"
export RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS="${RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS:-2}"
export RESIDUAL_V2_RL_NUM_GENERATIONS="${RESIDUAL_V2_RL_NUM_GENERATIONS:-6}"
export RESIDUAL_V2_RL_LEARNING_RATE="${RESIDUAL_V2_RL_LEARNING_RATE:-2e-6}"
export RESIDUAL_V2_RL_KL_COEF="${RESIDUAL_V2_RL_KL_COEF:-0.03}"
export RESIDUAL_V2_REWARD_TYPE_WEIGHT="${RESIDUAL_V2_REWARD_TYPE_WEIGHT:-0.00}"

# Format: name:point:event:iou:boundary:type
REWARD_CONFIGS="${REWARD_CONFIGS:-\
point_only:1.00:0.00:0.00:0.00:0.00 \
point_iou:0.60:0.00:0.40:0.00:0.00 \
point_iou_boundary:0.45:0.00:0.40:0.15:0.00}"

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

echo "Submitting Qwen2.5-VL-7B residual_v2 reward-component ablation."
echo "RUN_SEEDS=$RUN_SEEDS"
echo "MODELS=$MODELS"
echo "SUBSETS=$SUBSETS"
echo "VARIANTS=$VARIANTS"
echo "REWARD_CONFIGS=$REWARD_CONFIGS"
echo "stage1_sft_epochs=$STAGE1_SFT_NUM_TRAIN_EPOCHS"
echo "stage2_sft_epochs=$STAGE2_SFT_NUM_TRAIN_EPOCHS"
echo "residual_v2_rl=epochs:$RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS generations:$RESIDUAL_V2_RL_NUM_GENERATIONS lr:$RESIDUAL_V2_RL_LEARNING_RATE kl:$RESIDUAL_V2_RL_KL_COEF"
echo "residual_v2_sampling=$RESIDUAL_V2_POOL_SAMPLING_RATIOS"
echo "cleanup_dependency_mode=$CLEANUP_DEPENDENCY_MODE"
echo "auto_resubmit_missing=$AUTO_RESUBMIT_MISSING"

submitted_batches=0
for config in $REWARD_CONFIGS; do
  IFS=: read -r config_name point_weight event_weight iou_weight boundary_weight type_weight <<< "$config"
  if [[ -z "${config_name:-}" || -z "${point_weight:-}" || -z "${event_weight:-}" || -z "${iou_weight:-}" || -z "${boundary_weight:-}" || -z "${type_weight:-}" ]]; then
    echo "Invalid reward config: $config" >&2
    exit 2
  fi

  export RESIDUAL_V2_REWARD_POINT_WEIGHT="$point_weight"
  export RESIDUAL_V2_REWARD_EVENT_WEIGHT="$event_weight"
  export RESIDUAL_V2_REWARD_IOU_WEIGHT="$iou_weight"
  export RESIDUAL_V2_REWARD_BOUNDARY_WEIGHT="$boundary_weight"
  export RESIDUAL_V2_REWARD_TYPE_WEIGHT="$type_weight"

  # Base values are passed for compatibility; residual_v2 uses the RESIDUAL_V2_*
  # variables above inside the main workflow.
  export RL_NUM_TRAIN_EPOCHS="$RESIDUAL_V2_RL_NUM_TRAIN_EPOCHS"
  export RL_NUM_GENERATIONS="$RESIDUAL_V2_RL_NUM_GENERATIONS"
  export RL_LEARNING_RATE="$RESIDUAL_V2_RL_LEARNING_RATE"
  export RL_KL_COEF="$RESIDUAL_V2_RL_KL_COEF"
  export REWARD_POINT_WEIGHT="$point_weight"
  export REWARD_EVENT_WEIGHT="$event_weight"
  export REWARD_IOU_WEIGHT="$iou_weight"
  export REWARD_BOUNDARY_WEIGHT="$boundary_weight"
  export REWARD_TYPE_WEIGHT="$type_weight"
  export RESIDUAL_POOL_SAMPLING_RATIOS="$RESIDUAL_V2_POOL_SAMPLING_RATIOS"

  for seed in $RUN_SEEDS; do
    export EXPERIMENT_SEED="$seed"
    export BATCH_TAG="${BATCH_PREFIX}_${config_name}_seed${seed}_${TIMESTAMP}"
    batch_root="$OUTPUT_ROOT/qwen_vl_all_models_sft_residual_ablation_8subsets_$BATCH_TAG"
    submitted_batches=$((submitted_batches + 1))

    echo
    echo "=== Submitting config=$config_name seed=$seed BATCH_TAG=$BATCH_TAG ==="
    echo "reward=point:$point_weight event:$event_weight iou:$iou_weight boundary:$boundary_weight type:$type_weight"
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
        "--job-name=ensure-ra-${config_name//_/-}-s${seed}"
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
      echo "auto_resubmit config=$config_name seed=$seed job_id=$auto_job_id dependency=$dependency"
    fi
  done
done

echo
echo "Submitted reward-component ablation batches: $submitted_batches"
echo "Expected final metrics: 3 configs x 3 seeds x 8 subsets = 72 test_metrics.json files."
echo "Batch prefix:"
echo "  qwen_vl_all_models_sft_residual_ablation_8subsets_${BATCH_PREFIX}"
echo "After all jobs finish, summarize with:"
echo "  $PYTHON_BIN $SCRIPT_DIR/scripts/summarize_reward_weight_sweep.py --root $OUTPUT_ROOT --prefix qwen_vl_all_models_sft_residual_ablation_8subsets_${BATCH_PREFIX} --model qwen25vl_7b"
