#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="${SUBMIT_SCRIPT:-$ROOT_DIR/submit_requested_8subsets_qwen25vl_train_val_test.sh}"

SUBSETS_VALUE="${SUBSETS:-YAHOO}"
read -r -a SELECTED_SUBSETS <<< "$SUBSETS_VALUE"

ABLATIONS_VALUE="${ABLATIONS:-full no_event_f1 no_boundary_iou no_hallucination event_only boundary_only}"
read -r -a SELECTED_ABLATIONS <<< "$ABLATIONS_VALUE"

BATCH_TAG_BASE="${BATCH_TAG:-reward_ablation_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "Missing submit script: $SUBMIT_SCRIPT" >&2
  exit 1
fi

weights_for_ablation() {
  local name="$1"
  case "$name" in
    full)
      echo "0.45 0.35 -0.10"
      ;;
    no_event_f1)
      echo "0.00 0.35 -0.10"
      ;;
    no_boundary_iou)
      echo "0.45 0.00 -0.10"
      ;;
    no_hallucination)
      echo "0.45 0.35 0.00"
      ;;
    event_only)
      echo "1.00 0.00 0.00"
      ;;
    boundary_only)
      echo "0.00 1.00 0.00"
      ;;
    *)
      echo "Unknown ablation: $name" >&2
      return 1
      ;;
  esac
}

echo "Reward ablation subsets: ${SELECTED_SUBSETS[*]}"
echo "Reward ablations: ${SELECTED_ABLATIONS[*]}"
echo "Base batch tag: $BATCH_TAG_BASE"

for ablation in "${SELECTED_ABLATIONS[@]}"; do
  read -r event_w boundary_w hallucination_w <<< "$(weights_for_ablation "$ablation")"
  echo "=== Submitting reward ablation: $ablation ==="
  echo "weights: event_f1=$event_w boundary_iou=$boundary_w hallucination_penalty=$hallucination_w"

  SUBSETS="${SELECTED_SUBSETS[*]}" \
  BATCH_TAG="${BATCH_TAG_BASE}_${ablation}" \
  RUN_LABEL="${RUN_LABEL:-qwen25vl_7b}_${ablation}" \
  JOB_PREFIX="${JOB_PREFIX:-tsg-rabl}-${ablation}" \
  ENABLE_RL=1 \
  RL_REWARD_EVENT_F1_WEIGHT="$event_w" \
  RL_REWARD_BOUNDARY_IOU_WEIGHT="$boundary_w" \
  RL_REWARD_HALLUCINATION_PENALTY_WEIGHT="$hallucination_w" \
  DRY_RUN="$DRY_RUN" \
  bash "$SUBMIT_SCRIPT"
done
