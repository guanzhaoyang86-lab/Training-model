#!/bin/bash
set -euo pipefail

# Diagnostic submitter for residual mining quality checks on Quest.
#
# This wraps submit_qwen_vl_all_models_sft_residual_ablation_8subsets.sh with
# defaults aimed at inspecting residual_pool invalid_json cases:
#   Models: 7B, 2B, 4B, 8B
#   Subsets: TAO, TODS, MSL, NEK
#   Variants: full_residual and no_residual
#   Stage 1 SFT epochs: 1 on synthetic data
#   Stage 2 SFT epochs: 3 on real subset data
#   GRPO epochs: 1
#   residual_pool.jsonl and residual record cache are kept for debugging

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODELS="${MODELS:-7b 2b 4b 8b}"
export SUBSETS="${SUBSETS:-TAO TODS MSL NEK}"
export VARIANTS="${VARIANTS:-full_residual no_residual}"

export STAGE1_SFT_NUM_TRAIN_EPOCHS="${STAGE1_SFT_NUM_TRAIN_EPOCHS:-1}"
export STAGE2_SFT_NUM_TRAIN_EPOCHS="${STAGE2_SFT_NUM_TRAIN_EPOCHS:-3}"
export RL_NUM_TRAIN_EPOCHS="${RL_NUM_TRAIN_EPOCHS:-1}"

export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-512}"
export KEEP_RESIDUAL_POOL="${KEEP_RESIDUAL_POOL:-1}"
export SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
export EXCLUDE_NODES="${EXCLUDE_NODES:-qgpu2014}"

export BATCH_TAG="${BATCH_TAG:-residual_debug_4subsets_$(date +%Y%m%d_%H%M%S)}"

echo "Submitting residual debug run."
echo "MODELS=$MODELS"
echo "SUBSETS=$SUBSETS"
echo "VARIANTS=$VARIANTS"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "RL_MAX_NEW_TOKENS=$RL_MAX_NEW_TOKENS"
echo "KEEP_RESIDUAL_POOL=$KEEP_RESIDUAL_POOL"
echo "SAVE_PREDICTIONS=$SAVE_PREDICTIONS"
echo "EXCLUDE_NODES=$EXCLUDE_NODES"

exec bash "$SCRIPT_DIR/submit_qwen_vl_all_models_sft_residual_ablation_8subsets.sh"
