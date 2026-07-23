#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/vlm_7b_tsb_adu_windowed_plain_no_index.yaml}"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT_DIR/dataset/tsb_adu_subset_splits_raw_file_padded_256_128_7_1_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/plain_no_index}"
TRAIN_SUBMIT="${TRAIN_SUBMIT:-$ROOT_DIR/submit_tsb_adu_subset_train_no_index.sh}"
BATCH_TAG="${BATCH_TAG:-plain_no_index_$(date +%Y%m%d_%H%M%S)}"
TRAIN_JOB_NAME_SUFFIX="${TRAIN_JOB_NAME_SUFFIX:--noidx}"
EVAL_JOB_NAME_SUFFIX="${EVAL_JOB_NAME_SUFFIX:--noidx-eval}"
SUMMARY_JOB_NAME="${SUMMARY_JOB_NAME:-tsg-tsb-adu-noidx-summary}"

exec env \
  "CONFIG_PATH=$CONFIG_PATH" \
  "SOURCE_ROOT=$SOURCE_ROOT" \
  "OUTPUT_ROOT=$OUTPUT_ROOT" \
  "TRAIN_SUBMIT=$TRAIN_SUBMIT" \
  "BATCH_TAG=$BATCH_TAG" \
  "TRAIN_JOB_NAME_SUFFIX=$TRAIN_JOB_NAME_SUFFIX" \
  "EVAL_JOB_NAME_SUFFIX=$EVAL_JOB_NAME_SUFFIX" \
  "SUMMARY_JOB_NAME=$SUMMARY_JOB_NAME" \
  bash "$ROOT_DIR/submit_all_tsb_adu_subset_trains.sh"
