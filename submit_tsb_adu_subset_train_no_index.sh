#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DATASET_FILTER="${1:-${SOURCE_DATASET_FILTER:-}}"

if [[ -z "$SOURCE_DATASET_FILTER" ]]; then
  echo "Usage: bash submit_tsb_adu_subset_train_no_index.sh <SOURCE_DATASET>" >&2
  echo "Example: bash submit_tsb_adu_subset_train_no_index.sh NAB" >&2
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_tsb_adu_windowed_plain_no_index.yaml}"
SOURCE_ROOT_BASE="${SOURCE_ROOT_BASE:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/plain_no_index}"
RUN_TAG="${RUN_TAG:-qwen25vl_7b_tsb_adu_${SOURCE_DATASET_FILTER,,}_plain_no_index_$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-tsg-tsb-adu-${SOURCE_DATASET_FILTER,,}-noidx}"

exec env \
  "CONFIG_PATH=$CONFIG_PATH" \
  "SOURCE_ROOT_BASE=$SOURCE_ROOT_BASE" \
  "OUTPUT_ROOT=$OUTPUT_ROOT" \
  "RUN_TAG=$RUN_TAG" \
  "JOB_NAME=$JOB_NAME" \
  bash "$ROOT_DIR/submit_tsb_adu_subset_train.sh" "$SOURCE_DATASET_FILTER"
