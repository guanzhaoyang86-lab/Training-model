#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/../.." && pwd)"
SUBMITTER="${SUBMITTER:-$ROOT_DIR/submit_sft2_rl_8subsets_from_synthetic_sft.sh}"

if [[ ! -x "$SUBMITTER" ]]; then
  echo "Submitter is not executable: $SUBMITTER" >&2
  exit 1
fi

export BASE_SFT_ROOT="${BASE_SFT_ROOT:-$WORKSPACE_ROOT/models/Qwen2.5-VL-7B-Instruct}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$BASE_SFT_ROOT}"
export CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_anomaly_db_indexed_text_plain_image.yaml}"
export BATCH_PREFIX="${BATCH_PREFIX:-sft_rl_8subsets_no_synthetic_stage1}"
export WORKFLOW_DESCRIPTION="${WORKFLOW_DESCRIPTION:-real-subset SFT+RL jobs directly from the original 7B model, with no synthetic Stage 1 SFT}"
export BASE_MODEL_DESCRIPTION="${BASE_MODEL_DESCRIPTION:-Original 7B base model}"
export JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-tsg-nosyn-sftrl-q25vl7b}"

exec "$SUBMITTER"
